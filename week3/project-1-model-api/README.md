# Project 1 — Sentiment Analysis API

A production-ready REST API serving a DistilBERT sentiment analysis model using FastAPI.

## Architecture
Client → FastAPI Server → DistilBERT Model → JSON Response

## Endpoints

| Endpoint | Method | Description |
|---|---|---|
| / | GET | Service info |
| /health | GET | Health check |
| /predict | POST | Single text prediction |
| /predict/batch | POST | Batch prediction (max 32) |
| /docs | GET | Auto-generated Swagger docs |

## Performance
- Single request latency: ~56ms (CPU)
- Batch of 4 latency: ~88ms (~22ms per text)
- Concurrent requests: 71-94ms under 10 concurrent

## Key Features
- Request ID tracking for distributed tracing
- Middleware logging with latency per request
- Input validation with clear error messages
- Batch inference endpoint
- Auto-generated API documentation

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Tech Stack
- FastAPI — REST API framework
- DistilBERT — sentiment classification model
- Uvicorn — ASGI server
- Pydantic — request/response validation
