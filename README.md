# 1684 Project Bias in Compressed Models

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
- **Quantization:** Post-training quantization (8-bit and 4-bit) for weight and activation reduction. (TBD)
- **Knowledge Distillation (Stretch Goal):** Training a smaller student model from a teacher model to study transferred bias patterns. (TBD)

### Bias Evaluation
TBD

### Comparative Analysis
TBD


---

## How to Run
```bash
pip install -r requirements.txt
python scripts/run_baseline.py
python scripts/run_pruning.py

