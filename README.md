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

---

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
