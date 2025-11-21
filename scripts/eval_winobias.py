from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from datasets import load_dataset
import torch
import torch.nn.functional as F
import json, os, time, argparse
# from eval_stereoset import load_model_and_tokenizer  # <- replaced with local, quant-safe version


def load_model_and_tokenizer(model_dir):
    """
    Quantization-safe loader:
    - If standard HF weights exist, load normally.
    - If 'quantized_state_dict.pt' exists, build the model, apply dynamic quantization,
      then load the quantized state dict with strict=True (no missing/unexpected keys).
    """
    quant_path = os.path.join(model_dir, "quantized_state_dict.pt")
    has_bin = os.path.exists(os.path.join(model_dir, "pytorch_model.bin"))
    has_safe = os.path.exists(os.path.join(model_dir, "model.safetensors"))

    # tokenizer (always needed)
    tok = AutoTokenizer.from_pretrained(model_dir)

    if os.path.exists(quant_path) and not (has_bin or has_safe):
        # Quantized checkpoint branch
        cfg = AutoConfig.from_pretrained(model_dir)
        m = AutoModelForMaskedLM.from_config(cfg)
        m.eval()

        # Convert modules to quantized dynamic Linear *before* loading the quantized state dict
        m = torch.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)

        q_state = torch.load(quant_path, map_location="cpu")
        # With matching quantized modules, we can load strictly (no missing/unexpected keys)
        m.load_state_dict(q_state, strict=True)
        m.eval()
        return m, tok

    # Standard HF checkpoint branch
    m = AutoModelForMaskedLM.from_pretrained(model_dir)
    m.eval()
    return m, tok


def pll_score(text, tok, model):
    """
    Pseudo log-likelihood (token masking) score for a sentence.
    Average log-prob of each original token, masking one token at a time.
    """
    ids = tok(text, return_tensors="pt")
    input_ids = ids["input_ids"][0]
    attn = ids.get("attention_mask", None)

    with torch.no_grad():
        total = 0.0
        # Skip [CLS] and [SEP]
        for i in range(1, len(input_ids) - 1):
            masked = input_ids.clone()
            masked[i] = tok.mask_token_id
            out = model(input_ids=masked.unsqueeze(0), attention_mask=attn).logits[0, i]
            total += F.log_softmax(out, dim=-1)[input_ids[i]].item()

    denom = max(1, (len(input_ids) - 2))
    return total / denom


def _detok(tokens):
    s = " ".join(tokens)
    # very light detok for punctuation/quotes
    for p in [ " .", " ,", " !", " ?", " ;", " :", " )", " ]", " 's" ]:
        s = s.replace(p, p[1:])
    s = s.replace("( ", "(").replace("[ ", "[")
    return s

def _grab_text(rec):
    # 1) preferred plain-text keys if present
    for k in ("text", "sentence", "sent"):
        if k in rec and isinstance(rec[k], str):
            return rec[k]
    # 2) Hub WinoBias returns tokenized sentences
    if "tokens" in rec and isinstance(rec["tokens"], list):
        return _detok(rec["tokens"])
    # 3) fallback for local jsonl that already has pro/anti strings
    if "pro" in rec or "anti" in rec:
        return rec.get("pro") or rec.get("anti")
    raise KeyError(f"Cannot find a text field in record with keys: {list(rec.keys())}")


def load_winobias(args):
    """
    Returns a list of rows: [{"pro": <pro_sentence>, "anti": <anti_sentence>}, ...]
    Either from a local JSONL (--data_path) or from the Hub (with --config type1|type2).
    """
    if args.data_path:
        rows = []
        with open(args.data_path, "r", encoding="utf-8-sig") as f:  # handle BOM if present
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                rows.append({"pro": d["pro"], "anti": d["anti"]})
        return rows

    if not args.config:
        raise RuntimeError("Provide --config {type1,type2} or --data_path <local_jsonl>.")

    # Hub provides four configs: type1_pro, type1_anti, type2_pro, type2_anti
    pro = load_dataset("wino_bias", f"{args.config}_pro", split="test")
    anti = load_dataset("wino_bias", f"{args.config}_anti", split="test")

    n = min(len(pro), len(anti))
    rows = [{"pro": _grab_text(pro[i]), "anti": _grab_text(anti[i])} for i in range(n)]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Path to baseline/pruned/quantized model dir")
    ap.add_argument("--out", required=True, help="Where to write JSON metrics")
    ap.add_argument("--config", choices=["type1", "type2"],
                    help="WinoBias Hub config family to use (type1 or type2). "
                         "If omitted, you must pass --data_path.")
    ap.add_argument("--data_path", default=None,
                    help="Optional local JSONL with fields 'pro' and 'anti' per line.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional cap on number of pairs (useful for quick tests).")
    ap.add_argument("--progress_every", type=int, default=200,
                    help="Print a progress line every N examples (0 to disable).")
    args = ap.parse_args()

    rows = load_winobias(args)
    if args.limit is not None:
        rows = rows[: args.limit]
    model, tok = load_model_and_tokenizer(args.model_dir)

    n = len(rows)
    t0 = time.time()
    cnt = {"pro": 0, "anti": 0}

    for i, r in enumerate(rows, 1):
        s_pro = pll_score(r["pro"], tok, model)
        s_anti = pll_score(r["anti"], tok, model)
        if s_pro >= s_anti:
            cnt["pro"] += 1
        else:
            cnt["anti"] += 1

        if args.progress_every and (i % args.progress_every == 0):
            elapsed = time.time() - t0
            print(f"[eval] {i}/{n} pairs processed in {elapsed:.1f}s")

    dt = time.time() - t0
    total_bias = cnt["pro"] + cnt["anti"]
    stereo_rate = cnt["pro"] / total_bias if total_bias else 0.0

    metrics = {
        "model": os.path.basename(args.model_dir.rstrip("/\\")),
        "config": args.config or "local",
        "n": n,
        "counts": cnt,
        "proportions": {k: v / n for k, v in cnt.items()},
        "stereotype_rate": stereo_rate,  # pro / (pro + anti)
        "avg_time_per_example_sec": dt / n if n else 0.0,
        "total_time_sec": dt,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(metrics)


if __name__ == "__main__":
    main()
