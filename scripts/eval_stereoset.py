from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
import torch, torch.nn.functional as F
import json, os, time, argparse

def load_model_and_tokenizer(model_dir):
    """
    Handles:
    - Standard HF checkpoints (baseline, pruned), which include pytorch_model.bin / safetensors
    - Quantized checkpoints (int8 dynamic), which include:
        config.json
        quantized_state_dict.pt
        tokenizer files
    """
    quantized_path = os.path.join(model_dir, "quantized_state_dict.pt")

    # tokenizer always present
    tok = AutoTokenizer.from_pretrained(model_dir)

    if os.path.exists(quantized_path):
        # -------- QUANTIZED BRANCH --------
        print("[eval] Detected quantized checkpoint")

        # 1. load config manually
        cfg = AutoConfig.from_pretrained(model_dir)

        # 2. init an empty model from that config (no weights yet)
        m = AutoModelForMaskedLM.from_config(cfg)
        m.eval()

        # 3. load quantized weights
        q_state = torch.load(quantized_path, map_location="cpu")

        missing, unexpected = m.load_state_dict(q_state, strict=False)
        print("[eval] load_state_dict strict=False")
        print("       missing keys:", len(missing))
        print("       unexpected keys:", len(unexpected))

        m.eval()
        return m, tok

    else:
        # -------- NORMAL HF BRANCH --------
        print("[eval] Detected standard HF checkpoint")
        m = AutoModelForMaskedLM.from_pretrained(model_dir)
        m.eval()
        return m, tok

def score(ctx, opt, m, t):
    toks = t.tokenize(opt)
    masked = ctx.replace("BLANK", " ".join(["[MASK]"] * len(toks)))

    x = t(masked, return_tensors="pt")

    with torch.no_grad():
        y = m(**x).logits[0]

    # positions of the mask tokens
    pos = (x["input_ids"][0] == t.mask_token_id).nonzero(as_tuple=True)[0]

    # convert option tokens -> ids
    ids = t.convert_tokens_to_ids(toks)

    # average log-prob assigned to those tokens in those mask slots
    total = 0.0
    for p, i in zip(pos, ids):
        total += torch.log_softmax(y[p], dim=-1)[i].item()
    avg = total / len(ids)

    return avg

def eval_full(ds, m, t):
    # maps numeric gold_label to label string
    lbl_map = {0: "stereotype", 1: "anti", 2: "unrelated"}
    cnt = {"stereotype": 0, "anti": 0, "unrelated": 0}

    dsplit = ds["validation"]
    n = len(dsplit)

    t0 = time.time()

    for i in range(n):
        e = dsplit[i]
        c = e["context"]
        opts = e["sentences"]["sentence"]

        sc = [score(c, o, m, t) for o in opts]
        pick = sc.index(max(sc))

        gold = e["sentences"]["gold_label"][pick]
        lbl = lbl_map[gold]
        cnt[lbl] += 1

    dt = time.time() - t0

    # stereotype vs anti only
    total_bias_choices = cnt["stereotype"] + cnt["anti"]
    if total_bias_choices > 0:
        stereo_rate = cnt["stereotype"] / total_bias_choices
    else:
        stereo_rate = 0.0

    return {
        "n": n,
        "counts": cnt,
        "proportions": {k: v / n for k, v in cnt.items()},
        "stereotype_rate_given_bias_pair": stereo_rate,
        "avg_time_per_example_sec": dt / n,
        "total_time_sec": dt,
    }

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True,
                    help="Path to model dir (baseline/pruned/quantized)")
    ap.add_argument("--out", required=True,
                    help="Where to save JSON results")
    args = ap.parse_args()

    # load dataset
    ds = load_dataset("stereoset", "intrasentence")

    # load model (this handles both HF-style and quantized)
    m, t = load_model_and_tokenizer(args.model_dir)

    # evaluate
    metrics = eval_full(ds, m, t)
    print(metrics)

    # save
    save_json(metrics, args.out)

if __name__ == "__main__":
    main()
