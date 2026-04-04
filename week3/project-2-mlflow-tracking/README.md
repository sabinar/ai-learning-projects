# Project 2 — MLflow Experiment Tracking

Tracked 3 learning rate experiments for DistilBERT sentiment 
classification using MLflow.

## Experiments

| Learning Rate | F1 Score | Accuracy | Eval Loss |
|---|---|---|---|
| 2e-5 | 92.70% | 92.75% | 0.2259 |
| 5e-5 | 91.73% | 91.75% | 0.2937 |
| 1e-5 | 90.77% | 90.75% | 0.2527 |

**Best run:** lr=2e-5, F1=92.70%

## Key Findings
- lr=2e-5 is the sweet spot for DistilBERT fine-tuning
- lr=5e-5 learns faster but overshoots slightly
- lr=1e-5 too slow — needs more epochs to converge
- MLflow tracked all runs including 2 crashed runs from disk issues

## How to Run
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train.py          # run 3 experiments
python query_results.py  # find best run
mlflow ui --port 5000    # view UI at http://localhost:5000
```

## Tech Stack
- MLflow 3.10.1 — experiment tracking
- DistilBERT — sentiment classification model
- HuggingFace Trainer — training loop
- SQLite — MLflow backend store
