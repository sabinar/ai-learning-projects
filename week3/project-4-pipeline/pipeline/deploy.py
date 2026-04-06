import os
import subprocess
import time
import requests
import json
import shutil

MODEL_DIR     = "./trained_model"
API_DIR       = "./serving"
API_PORT      = 8000
HEALTH_URL    = f"http://localhost:{API_PORT}/health"
PREDICT_URL   = f"http://localhost:{API_PORT}/predict"

def setup_serving_directory():
    """Copy model and create FastAPI server for serving"""
    print("Setting up serving directory...")

    os.makedirs(API_DIR, exist_ok=True)
    os.makedirs(f"{API_DIR}/app", exist_ok=True)
    os.makedirs(f"{API_DIR}/app/model", exist_ok=True)

    # Copy trained model to serving directory
    if os.path.exists(f"{API_DIR}/model"):
        shutil.rmtree(f"{API_DIR}/model")
    shutil.copytree(MODEL_DIR, f"{API_DIR}/model")

    # Write FastAPI server
    server_code = '''
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import time
import uuid

tokenizer = None
model     = None
device    = None

def load_model():
    global tokenizer, model, device
    device    = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained("./model")
    model     = AutoModelForSequenceClassification.from_pretrained("./model")
    model.to(device)
    model.eval()
    print("Model loaded from ./model")

@asynccontextmanager
async def lifespan(app):
    load_model()
    yield

app = FastAPI(lifespan=lifespan)

class ReviewRequest(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "healthy", "model": "pipeline-trained"}

@app.post("/predict")
def predict(request: ReviewRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    start = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    latency = (time.time() - start) * 1000

    probs     = F.softmax(outputs.logits, dim=-1)
    predicted = torch.argmax(probs).item()
    confidence = probs[0][predicted].item()
    labels    = {0: "NEGATIVE", 1: "POSITIVE"}

    return {
        "sentiment":   labels[predicted],
        "confidence":  round(confidence, 4),
        "latency_ms":  round(latency, 2),
        "request_id":  str(uuid.uuid4())[:8]
    }
'''

    with open(f"{API_DIR}/server.py", "w") as f:
        f.write(server_code)

    print(f"Server code written to {API_DIR}/server.py")

def wait_for_server(timeout=60):
    """Wait until server is healthy"""
    print(f"Waiting for server to start...")
    start = time.time()

    while time.time() - start < timeout:
        try:
            response = requests.get(HEALTH_URL, timeout=2)
            if response.status_code == 200:
                print(f"✅ Server is healthy")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(2)

    print(f"❌ Server did not start within {timeout} seconds")
    return False

def smoke_test():
    """Run basic tests against deployed model"""
    print("\nRunning smoke tests...")

    test_cases = [
        ("This product is absolutely amazing!", "POSITIVE"),
        ("Broke after two days, terrible quality", "NEGATIVE"),
    ]

    passed = 0
    for text, expected in test_cases:
        try:
            response = requests.post(
                PREDICT_URL,
                json={"text": text},
                timeout=30
            )
            result = response.json()
            actual = result["sentiment"]

            if actual == expected:
                print(f"  ✅ '{text[:40]}' → {actual}")
                passed += 1
            else:
                print(f"  ❌ '{text[:40]}' → {actual} (expected {expected})")
        except Exception as e:
            print(f"  ❌ Request failed: {e}")

    print(f"\nSmoke tests: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)

def run_deployment():
    print("\n" + "="*50)
    print("STAGE 3 — DEPLOYMENT")
    print("="*50)

    # Check model exists
    if not os.path.exists(MODEL_DIR):
        print("❌ No trained model found")
        return False

    # Setup serving directory
    setup_serving_directory()

    # Start server
    print(f"\nStarting server on port {API_PORT}...")
    server_process = subprocess.Popen(
        ["uvicorn", "server:app",
         "--host", "0.0.0.0",
         "--port", str(API_PORT)],
        cwd=API_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Wait for server to be healthy
    if not wait_for_server():
        server_process.terminate()
        return False

    # Run smoke tests
    if not smoke_test():
        print("❌ Smoke tests failed — rolling back")
        server_process.terminate()
        return False

    print(f"\n✅ DEPLOYMENT SUCCESSFUL")
    print(f"   Server running at http://localhost:{API_PORT}")
    print(f"   Press Ctrl+C to stop")

    # Save server process ID for monitor stage
    with open("server.pid", "w") as f:
        f.write(str(server_process.pid))

    return server_process

if __name__ == "__main__":
    process = run_deployment()
    if process:
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            process.terminate()