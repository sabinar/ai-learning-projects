# Project 4 — End to End ML Pipeline

Automated ML pipeline chaining training → evaluation → deployment → monitoring.

## Pipeline Stages

### Stage 1 — Training
- Model: DistilBERT fine-tuned on Amazon reviews
- Training samples: 2,000
- Result: Accuracy 93%, F1 92.96%

### Stage 2 — Evaluation Gate
- Min F1 threshold: 85%
- Min Accuracy threshold: 85%
- Compares new model vs previous model
- Rejects if F1 drops >2% vs previous

### Stage 3 — Deployment
- Copies model to serving directory
- Starts FastAPI server on port 8000
- Waits for health check
- Runs smoke tests before accepting traffic
- Rolls back if smoke tests fail

### Stage 4 — Monitoring
- Tracks prediction distribution in 20-request rolling window
- Baseline positive rate: 50%
- Drift threshold: 30%
- Fires alert when drift detected
- Recommends retraining when drift exceeds threshold

## Results
- Full pipeline: 2.2 minutes
- Skip training: 0.1 minutes
- Drift detected at request 45 (50% shift from baseline)

## How to Run
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Full pipeline — train + evaluate + deploy + monitor
python run_pipeline.py

# Skip training — reuse existing model
python run_pipeline.py --skip-training
```

## Tech Stack
- DistilBERT — sentiment model
- HuggingFace Trainer — training
- FastAPI — model serving
- Custom drift monitor — prediction distribution tracking
