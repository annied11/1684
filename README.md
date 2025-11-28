# Bias in Compressed Models

## Folder Structure
    1684/
    ├── data/            # Datasets and processed files
    ├── models/          # Auto-created locally when scripts run (not pushed to GitHub)
    ├── results/         # Evaluation outputs (kept in repo)
    ├── scripts/         # Training, pruning, and evaluation scripts
    ├── requirements.txt # Python dependencies
    └── README.md

Note: The `models/` folder is generated automatically when running scripts locally and is excluded from GitHub due to large file size.

---

## Project Overview
This project studies how model compression techniques, such as pruning, affect fairness in large language models.

## Proposed Approach

The approach can be divided into three main stages:

1. **Model Selection and Compression**
2. **Bias Evaluation Design**
3. **Comparative Analysis of Efficiency–Fairness Tradeoffs**

### Model Selection and Compression
We will start with pretrained transformer-based models, such as BERT or smaller LLaMA variants to ensure feasibility under compute constraints. Each model will undergo common compression strategies:

- **Pruning:** Gradual magnitude pruning of weights at different sparsity levels.  
- **Quantization:** Post-training quantization (8-bit and 4-bit) for weight and activation reduction.  
- **Knowledge Distillation (Stretch Goal):** Training a smaller student model from a teacher model to study transferred bias patterns.

### Bias Evaluation
We use the **StereoSet** dataset to evaluate social bias (stereotype vs. anti-stereotype completions).  
Models are scored using masked language modeling, where the most likely completion determines bias tendency.  
Metrics include:
- **Stereotype Rate:** fraction of bias-related completions favoring stereotypes.  
- **Proportion Breakdown:** counts of stereotype, anti-stereotype, and unrelated predictions.  
- **Average Inference Time:** per example latency to evaluate efficiency tradeoffs.

### Comparative Analysis
Use information from the Bias Evaluation including the **Stereotype Rate** and more to evaluate the differences between increased compression rates and stereotyped choices. This is shown and discussed in various ways:
 - **Table:** Simple table showing the model and stereotype rate anaylzing what that means in the context
 - **Pearson Correlation Coefficient:** Showing and seeing if there is a relationship between increased compression and stereotype
 - **Visualization:** Use a graph to show and discuss the trade off between increased stereotyping of models and their efficiency 

### Comparative Table (Stereotype Rate by Dataset)

| Model               | StereoSet ↑ | CrowS-Pairs ↑ | WinoBias ↑ (TYPE 1) | WinoBias ↑ (TYPE 2) |Bias-in-Bios ↑ |
|---------------------|------------:|--------------:|--------------------:|--------------------:|--------------:|
| bert-base-uncased   | 0.4194      | 0.6141        | 0.5076              | 0.5833              | 0.51          |
| bert_int8_dynamic   | 0.4940      | 0.5212        | 0.5303              | 0.5808              | 0.45          |
| bert_prune30        | 0.4580      | 0.6008        | 0.5025              | 0.5606              | 0.53          |
| bert_prune50        | 0.4820      | 0.5458        | 0.5076              | 0.4874              | 0.56          |
| bert_prune70        | 0.5326      | 0.4768        | 0.4672              | 0.5050              | 0.44          |

<sub>↑ higher = more stereotyped choices (worse fairness).</sub>


## How to Run
```bash
# install dependencies
pip install -r requirements.txt

# baseline (BERT)
python scripts/run_baseline.py

# pruning (30%, 50%, 70%)
python scripts/run_pruning.py

# quantization (8-bit dynamic)
python scripts/run_quantization.py

# evaluate all models on StereoSet
python scripts/eval_stereoset.py --model_dir models/base/bert-base-uncased --out results/stereoset/bert_base.json
python scripts/eval_stereoset.py --model_dir models/pruning/bert_prune30 --out results/stereoset/bert_prune30.json
python scripts/eval_stereoset.py --model_dir models/pruning/bert_prune50 --out results/stereoset/bert_prune50.json
python scripts/eval_stereoset.py --model_dir models/pruning/bert_prune70 --out results/stereoset/bert_prune70.json
python scripts/eval_stereoset.py --model_dir models/quantization/bert_int8_dynamic --out results/stereoset/bert_int8_dynamic.json

# evaluate all models on CrowS-Pairs
python scripts/eval_crows_pairs.py --model_dir models/base/bert-base-uncased --out results/crows_pairs/bert_base.json --data_path data/crows_pairs.jsonl
python scripts/eval_crows_pairs.py --model_dir models/pruning/bert_prune30 --out results/crows_pairs/bert_prune30.json --data_path data/crows_pairs.jsonl
python scripts/eval_crows_pairs.py --model_dir models/pruning/bert_prune50 --out results/crows_pairs/bert_prune50.json --data_path data/crows_pairs.jsonl
python scripts/eval_crows_pairs.py --model_dir models/pruning/bert_prune70 --out results/crows_pairs/bert_prune70.json --data_path data/crows_pairs.jsonl
python scripts/eval_crows_pairs.py --model_dir models/quantization/bert_int8_dynamic --out results/crows_pairs/bert_int8_dynamic.json --data_path data/crows_pairs.jsonl

# evaluate all models on WinoBias type 1
python scripts/eval_winobias.py --model_dir models/base/bert-base-uncased --out results/winobias/bert_base_type1.json --config type1
python scripts/eval_winobias.py --model_dir models/pruning/bert_prune30 --out results/winobias/bert_prune30_type1.json --config type1
python scripts/eval_winobias.py --model_dir models/pruning/bert_prune50 --out results/winobias/bert_prune50_type1.json --config type1
python scripts/eval_winobias.py --model_dir models/pruning/bert_prune70 --out results/winobias/bert_prune70_type1.json --config type1
python scripts/eval_winobias.py --model_dir models/quantization/bert_int8_dynamic --out results/winobias/bert_int8_dynamic_type1.json --config type1

# evaluate all models on WinoBias type 2
python scripts/eval_winobias.py --model_dir models/base/bert-base-uncased --out results/winobias/bert_base_type2.json --config type2
python scripts/eval_winobias.py --model_dir models/pruning/bert_prune30 --out results/winobias/bert_prune30_type2.json --config type2
python scripts/eval_winobias.py --model_dir models/pruning/bert_prune50 --out results/winobias/bert_prune50_type2.json --config type2
python scripts/eval_winobias.py --model_dir models/pruning/bert_prune70 --out results/winobias/bert_prune70_type2.json --config type2
python scripts/eval_winobias.py --model_dir models/quantization/bert_int8_dynamic --out results/winobias/bert_int8_dynamic_type2.json --config type2

# evaluate Bias-in-Bios
python scripts/eval_bias_in_bios.py --model_dir models/base/bert-base-uncased --out results/bios/bert_base.json --limit 100
python scripts/eval_bias_in_bios.py --model_dir models/pruning/bert_prune30 --out results/bios/bert_prune30.json --limit 100
python scripts/eval_bias_in_bios.py --model_dir models/pruning/bert_prune50 --out results/bios/bert_prune50.json --limit 100
python scripts/eval_bias_in_bios.py --model_dir models/pruning/bert_prune70 --out results/bios/bert_prune70.json --limit 100
python scripts/eval_bias_in_bios.py --model_dir models/quantization/bert_int8_dynamic --out results/bios/bert_int8_dynamic.json --limit 100

