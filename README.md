# 1684 Project Bias in Compressed Models

## Folder Structure
    1684/
    ├── data/            # Datasets and processed files
    ├── models/          # Auto-created locally when scripts run (not pushed to GitHub)
    ├── notebooks/       # Experiment notebooks
    ├── results/         # Evaluation outputs (kept in repo)
    ├── scripts/         # Training, pruning, and evaluation scripts
    ├── requirements.txt # Python dependencies
    └── README.md

Note: The `models/` folder is generated automatically when running scripts locally and is excluded from GitHub due to large file size.

---

## Project Overview
This project studies how model compression techniques, such as pruning, affect fairness in large language models.

### Approach
1. Use a pretrained BERT model as the baseline.  
2. Apply pruning at different sparsity levels (30%, 50%, 70%).  
3. Evaluate each version for both performance and fairness metrics.  
4. Analyze trade-offs between model efficiency and fairness.

---

## How to Run
```bash
pip install -r requirements.txt
python scripts/run_baseline.py
python scripts/run_pruning.py

