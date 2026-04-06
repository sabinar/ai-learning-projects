from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from contextlib import asynccontextmanager
from app.model import load_model, predict
from typing import List
import time
import logging
import uuid

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# Request/Response models
class ReviewRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float
    probabilities: dict
    request_id: str
    latency_ms: float

class BatchReviewRequest(BaseModel):
    texts: List[str]
    
class BatchPredictionResponse(BaseModel):
    results: List[dict]
    total_latency_ms: float
    request_id: str    

# Load model on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading model...")
    load_model()
    logger.info("Server ready")
    yield
    logger.info("Shutting down")

# Create FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="Classifies Amazon reviews as POSITIVE or NEGATIVE",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware — logs every request
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id  # share ID with endpoint
    start_time = time.time()
    
    response = await call_next(request)
    
    latency = (time.time() - start_time) * 1000
    logger.info(
        f"request_id={request_id} "
        f"method={request.method} "
        f"path={request.url.path} "
        f"status={response.status_code} "
        f"latency={latency:.2f}ms"
    )
    return response


@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "distilbert-sst2"}

@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(request: ReviewRequest, req: Request):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    if len(request.text) > 5000:
        raise HTTPException(status_code=400, detail="Text too long, max 5000 characters")
    
    # Get shared request ID from middleware
    request_id = getattr(req.state, "request_id", str(uuid.uuid4())[:8])
    
    start = time.time()
    result = predict(request.text)
    latency_ms = (time.time() - start) * 1000
    
    logger.info(
        f"request_id={request_id} "
        f"sentiment={result['sentiment']} "
        f"confidence={result['confidence']} "
        f"latency={latency_ms:.2f}ms "
        f"text_length={len(request.text)}"
    )
    
    return {
        **result,
        "request_id": request_id,
        "latency_ms": round(latency_ms, 2)
    }

@app.get("/")
def root():
    return {
        "service": "Sentiment Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict",
            "docs": "GET /docs"
        }
    }

@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchReviewRequest, req: Request):
    if not request.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")
    
    if len(request.texts) > 32:
        raise HTTPException(status_code=400, detail="Max 32 texts per batch")
    
    request_id = getattr(req.state, "request_id", str(uuid.uuid4())[:8])
    
    start = time.time()
    results = [predict(text) for text in request.texts]
    total_latency = (time.time() - start) * 1000
    
    logger.info(
        f"request_id={request_id} "
        f"batch_size={len(request.texts)} "
        f"total_latency={total_latency:.2f}ms "
        f"avg_latency={total_latency/len(request.texts):.2f}ms"
    )
    
    return {
        "results": results,
        "total_latency_ms": round(total_latency, 2),
        "request_id": request_id
    }