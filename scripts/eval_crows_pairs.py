from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch, torch.nn.functional as F
import json, os, time, argparse
from eval_stereoset import load_model_and_tokenizer  # re-use your loader

def pll_score(text, tok, model):
    ids = tok(text, return_tensors="pt")
    input_ids = ids["input_ids"][0]
    attn = ids.get("attention_mask", None)
    with torch.no_grad():
        total = 0.0
        # mask each token once
        for i in range(1, len(input_ids)-1):  # skip [CLS], [SEP]
            masked = input_ids.clone()
            masked[i] = tok.mask_token_id
            out = model(input_ids=masked.unsqueeze(0), attention_mask=attn).logits[0, i]
            total += F.log_softmax(out, dim=-1)[input_ids[i]].item()
    return total / max(1, (len(input_ids)-2))

def load_crows_pairs(args):
    if args.data_path:
        import json
        rows = []
        with open(args.data_path, "r", encoding="utf-8-sig") as f:  # <- handles BOM
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                rows.append({"stereo": d["sent_more"], "anti": d["sent_less"]})
        return rows

    # try modern Hub ID instead of deprecated dataset script???
    try:
        ds = load_dataset("nyu-mll/crows-pairs", split="test")
        rows = [{"stereo": e["sent_more"], "anti": e["sent_less"]} for e in ds]
        return rows
    except Exception as e:
        raise RuntimeError(
            "Could not load CrowS-Pairs from the Hub. "
            "Run again with --data_path <local_jsonl> where each line has "
            'keys "sent_more" (stereotype) and "sent_less" (anti).'
        ) from e

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data_path", default=None, help="optional local CrowS-Pairs jsonl")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N examples")  # CHANGE 1: add limit
    args = ap.parse_args()

    rows = load_crows_pairs(args)
    if args.limit:                       # CHANGE 1: apply limit
        rows = rows[:args.limit]

    model, tok = load_model_and_tokenizer(args.model_dir)

    n = len(rows)
    t0 = time.time()
    cnt = {"stereotype": 0, "anti": 0}

    for i, r in enumerate(rows, 1):
        s_st = pll_score(r["stereo"], tok, model)
        s_an = pll_score(r["anti"], tok, model)
        if s_st >= s_an:
            cnt["stereotype"] += 1
        else:
            cnt["anti"] += 1

        if i % 25 == 0 or i == n:       # CHANGE 2: progress heartbeat every 25 examples
            print(f"[CrowS-Pairs] {i}/{n} done")

    dt = time.time() - t0

    total_bias = cnt["stereotype"] + cnt["anti"]
    stereo_rate = cnt["stereotype"] / total_bias if total_bias else 0.0
    metrics = {
        "model": os.path.basename(args.model_dir.rstrip("/\\")),
        "n": n,
        "counts": cnt,
        "proportions": {k: v / n for k, v in cnt.items()},
        "stereotype_rate": stereo_rate,
        "avg_time_per_example_sec": dt / n if n else 0.0,
        "total_time_sec": dt,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(metrics)


if __name__ == "__main__":
    main()
