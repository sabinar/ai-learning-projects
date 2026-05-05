# Week 3 — MLOps: Serving & Pipelines

Taking fine-tuned models from Week 2 and deploying them as production APIs with containerization and automated ML pipelines.

**Tech:** FastAPI · DistilBERT · Docker · HuggingFace Trainer

---

## Projects

### [project-1-model-api](project-1-model-api/)
Production REST API serving DistilBERT sentiment analysis via FastAPI. Supports single and batch inference with Pydantic validation.
- ~56ms single request, ~22ms/text for batch
- Endpoints: `POST /predict`, `POST /predict/batch`, `GET /health`

### [project-3-docker](project-3-docker/)
Containerized version of the sentiment API using Docker and docker-compose. Reduced image size 82% (8.3GB → 1.5GB) by switching to CPU-only PyTorch.

### [project-4-pipeline](project-4-pipeline/)
End-to-end automated ML pipeline: train → evaluate → deploy → monitor.
- Trains DistilBERT on Amazon reviews (93% accuracy)
- F1 threshold gate blocks bad models from deploying
- Drift monitor tracks prediction distribution over time
