import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Model name — using a pretrained sentiment model
# so we don't need to load your Kaggle trained weights
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

# Global model and tokenizer — loaded once at startup
tokenizer = None
model = None
device = None

def load_model():
    global tokenizer, model, device
    
    print(f"Loading model: {MODEL_NAME}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()  # set to inference mode
    
    print("Model loaded successfully")

def predict(text: str):
    if model is None or tokenizer is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Convert to probabilities
    probs = F.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probs).item()
    confidence = probs[0][predicted_class].item()
    
    # Map to label
    labels = {0: "NEGATIVE", 1: "POSITIVE"}
    
    return {
        "sentiment": labels[predicted_class],
        "confidence": round(confidence, 4),
        "probabilities": {
            "NEGATIVE": round(probs[0][0].item(), 4),
            "POSITIVE": round(probs[0][1].item(), 4),
        }
    }