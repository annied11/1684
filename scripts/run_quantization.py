import torch, os, json
from transformers import AutoModelForMaskedLM, AutoTokenizer

BASE = "models/base/bert-base-uncased"
OUTDIR = "models/quantization"
META_DIR = "results/quantization"

CONFIGS = [
    {"tag": "bert_int8_dynamic", "method": "dynamic_int8"},
]

def count_params(m):
    return sum(p.numel() for p in m.parameters())

def file_size_mb(path):
    return round(os.path.getsize(path) / (1024*1024), 2)

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def quantize_dynamic_int8(model):
    # quantize Linear layers to int8 dynamically (CPU-friendly)
    qmodel = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    return qmodel

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(META_DIR, exist_ok=True)

    # load base model + tokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    base_model = AutoModelForMaskedLM.from_pretrained(BASE)
    base_model.eval()

    base_params = count_params(base_model)

    for cfg in CONFIGS:
        tag = cfg["tag"]
        method = cfg["method"]

        if method == "dynamic_int8":
            qm = quantize_dynamic_int8(base_model)
        else:
            raise ValueError(f"Unknown method {method}")

        qm.eval()

        # make output dir for this quantized variant
        outdir = os.path.join(OUTDIR, tag)
        os.makedirs(outdir, exist_ok=True)

        base_model.config.save_pretrained(outdir)

        tok.save_pretrained(outdir)

        weight_path = os.path.join(outdir, "quantized_state_dict.pt")
        torch.save(qm.state_dict(), weight_path)

        size_mb = file_size_mb(weight_path)

        meta = {
            "method": method,
            "tag": tag,
            "base": BASE,
            "params_total": base_params,
            "artifact_size_mb": size_mb,
            "weights_file": "quantized_state_dict.pt",
        }

        meta_path = os.path.join(META_DIR, f"{tag}_efficiency.json")
        save_json(meta, meta_path)

        print(f"[quantized {method}] → {outdir}")
        print(f"  saved weights: {weight_path} ({size_mb} MB)")
        print(f"  meta:          {meta_path}")

if __name__ == "__main__":
    main()
