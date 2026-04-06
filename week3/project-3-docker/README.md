# Project 3 — Dockerized Sentiment Analysis API

Containerized the FastAPI sentiment analysis model server using Docker.
Reduced image size by 82% using CPU-only PyTorch.

## Images

| Version | Size | Description |
|---|---|---|
| v1 | 8.34GB | Full PyTorch with CUDA |
| v2 | 1.5GB | CPU-only PyTorch (production) |

## Quick Start
```bash
# Build image
docker build --load -t sentiment-api:v2 .

# Run with docker run
docker run --rm -p 8000:8000 sentiment-api:v2

# Run with docker-compose
docker compose up
```

## Test
```bash
# Health check
curl http://localhost:8000/health

# Predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is absolutely amazing!"}'
```

## Performance
- v1 latency: ~176ms
- v2 latency: ~127ms
- 82% smaller image = faster deploys

## Key Concepts
- Docker layer caching — requirements installed before code copy
- CPU-only torch — 82% size reduction vs full torch
- docker-compose — declarative service definition
- Health checks — Docker monitors /health every 30s
- restart: unless-stopped — auto-recovery from crashes

## Tech Stack
- FastAPI — REST API framework
- DistilBERT — sentiment model
- Docker — containerization
- docker-compose — service orchestration
