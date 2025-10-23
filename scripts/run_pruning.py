from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch, os, json, math
from torch.nn.utils import prune

BASE = "models/base/bert-base-uncased"
OUTDIR = "models/pruning"
META_DIR = "results/pruning"
AMTS = [0.3, 0.5, 0.7]

def count_params(m):
    return sum(p.numel() for p in m.parameters())

def global_prune_linear(m, amt):
    params = []
    for mod in m.modules():
        if isinstance(mod, torch.nn.Linear):
            params.append((mod, "weight"))
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=amt)
    for mod, _ in params:
        prune.remove(mod, "weight")
    return m

def file_size_mb(path):
    return round(os.path.getsize(path) / (1024*1024), 2)

def save_meta(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f: json.dump(obj, f, indent=2)

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(META_DIR, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(BASE)

    for a in AMTS:
        m = AutoModelForMaskedLM.from_pretrained(BASE)
        before = count_params(m)
        m.eval()
        m = global_prune_linear(m, a)

        tag = f"bert_prune{int(a*100)}"
        out = os.path.join(OUTDIR, tag)
        os.makedirs(out, exist_ok=True)
        m.save_pretrained(out)
        tok.save_pretrained(out)

        # collect efficiency metadata
        safepath = os.path.join(out, "model.safetensors")
        binpath  = os.path.join(out, "pytorch_model.bin")
        fp = safepath if os.path.exists(safepath) else binpath
        meta = {
            "method": "global_unstructured_L1",
            "sparsity": a,
            "params_total": before,
            "artifact_size_mb": file_size_mb(fp) if os.path.exists(fp) else None,
            "base": BASE,
            "tag": tag,
        }
        save_meta(os.path.join(META_DIR, f"{tag}_efficiency.json"), meta)
        print(f"[pruned {a:.0%}] → {out} | size_mb={meta['artifact_size_mb']} | params={before}")

if __name__ == "__main__":
    main()
