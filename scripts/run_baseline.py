from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch, torch.nn.functional as F
import json, os, time

def score(ctx, opt, m, t):
    toks = t.tokenize(opt)
    masked = ctx.replace("BLANK", " ".join(["[MASK]"] * len(toks)))
    x = t(masked, return_tensors="pt")
    with torch.no_grad():
        y = m(**x).logits[0]
    pos = (x["input_ids"][0] == t.mask_token_id).nonzero(as_tuple=True)[0]
    ids = t.convert_tokens_to_ids(toks)
    return sum(F.log_softmax(y[p], dim=-1)[i].item() for p, i in zip(pos, ids)) / len(ids)

def run(ds, m, t, n=5):
    d = ds["validation"]
    for i in range(min(n, len(d))):
        e = d[i]  # fetch a single example dict
        c, opts = e["context"], e["sentences"]["sentence"]
        sc = [score(c, o, m, t) for o in opts]
        print(f"\n{c}")
        for o, s in zip(opts, sc):
            print(f"  {o:25s} {s:.3f}")
        print(f"→ {opts[sc.index(max(sc))]}")

def eval_full(ds, m, t):
    lbl_map = {0: "stereotype", 1: "anti", 2: "unrelated"}  # StereoSet label assumption
    cnt = {"stereotype": 0, "anti": 0, "unrelated": 0}
    n = len(ds["validation"])
    t0 = time.time()
    for i in range(n):
        e = ds["validation"][i]
        c, opts = e["context"], e["sentences"]["sentence"]
        sc = [score(c, o, m, t) for o in opts]
        pick = sc.index(max(sc))
        gold = e["sentences"]["gold_label"][pick]  # label for the chosen option
        cnt[lbl_map[gold]] += 1
    dt = time.time() - t0
    return {
        "n": n,
        "counts": cnt,
        "proportions": {k: v / n for k, v in cnt.items()},
        "avg_time_per_example_sec": dt / n,
        "total_time_sec": dt,
    }

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        
def main():
    ds = load_dataset("stereoset", "intrasentence")
    t = AutoTokenizer.from_pretrained("bert-base-uncased")
    m = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    m.eval()
    
    m.save_pretrained("models/base/bert-base-uncased")
    t.save_pretrained("models/base/bert-base-uncased")
    
    run(ds, m, t)
    
    metrics = eval_full(ds, m, t)
    print("\n== baseline (intrasentence) ==")
    print(metrics)
    save_json(metrics, "results/baseline/stereoset_intra_metrics.json")

if __name__ == "__main__":
    main()
