from transformers import AutoTokenizer, AutoModelForMaskedLM
from datasets import load_dataset
import torch, torch.nn.functional as F
import os, json, time, argparse
from eval_stereoset import load_model_and_tokenizer 

def pll_score(text, tok, model):
    ids = tok(text, return_tensors="pt")
    input_ids = ids["input_ids"][0]
    attn = ids.get("attention_mask")

    with torch.no_grad():
        total = 0.0
        for i in range(1, len(input_ids) - 1):
            masked = input_ids.clone()
            masked[i] = tok.mask_token_id
            out = model(input_ids=masked.unsqueeze(0), attention_mask=attn).logits[0, i]
            total += F.log_softmax(out, dim=-1)[input_ids[i]].item()

    return total / max(1, len(input_ids) - 2)

def load_bias_in_bios(args):
    # Load the LabHC dataset
    ds = load_dataset("LabHC/bias_in_bios", split="test")

    groups = {}

    for ex in ds:
        bio = ex["hard_text"]
        gender = ex["gender"]
        job = ex["profession"]

        # Normalize gender values
        if isinstance(gender, int):
            gender = "male" if gender == 0 else "female"
        else:
            gender = gender.lower()
            if gender.startswith("m"):
                gender = "male"
            elif gender.startswith("f"):
                gender = "female"
            else:
                continue  # skip unknown labels

        if job not in groups:
            groups[job] = {"male": [], "female": []}

        groups[job][gender].append(bio)

    # Create male/female paired examples
    rows = []
    for job, g in groups.items():
        males = g["male"]
        females = g["female"]
        n = min(len(males), len(females))
        for i in range(n):
            rows.append({
                "male": males[i],
                "female": females[i],
                "profession": job
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data_path", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--progress_every", type=int, default=50,
                    help="Print progress every N examples (default=50)")
    args = ap.parse_args()

    rows = load_bias_in_bios(args)
    if args.limit:
        rows = rows[:args.limit]

    model, tok = load_model_and_tokenizer(args.model_dir)

    n = len(rows)
    t0 = time.time()
    cnt = {"female": 0, "male": 0}

    for i, r in enumerate(rows, 1):
        s_f = pll_score(r["female"], tok, model)
        s_m = pll_score(r["male"], tok, model)

        if s_m >= s_f:
            cnt["male"] += 1
        else:
            cnt["female"] += 1

        if args.progress_every > 0 and i % args.progress_every == 0:
            elapsed = time.time() - t0
            print(f"[Bias-in-Bios] {i}/{n} processed (elapsed: {elapsed:.1f}s)")

    dt = time.time() - t0
    total = cnt["male"] + cnt["female"]
    stereo_rate = cnt["male"] / total if total else 0.0

    metrics = {
        "model": os.path.basename(args.model_dir.rstrip('/')),
        "n": n,
        "counts": cnt,
        "proportions": {k: v/n for k,v in cnt.items()},
        "stereotype_rate": stereo_rate,
        "avg_time_per_example_sec": dt/n,
        "total_time_sec": dt,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(metrics)


if __name__ == "__main__":
    main()
