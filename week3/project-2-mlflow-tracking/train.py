import mlflow
import mlflow.pytorch
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
import evaluate

# ── Config ──────────────────────────────────────────────
EXPERIMENT_NAME = "amazon-sentiment-classifier"
MODEL_NAME      = "distilbert-base-uncased"
TRAIN_SAMPLES   = 2000
TEST_SAMPLES    = 400
MAX_LENGTH      = 128

# ── Dataset ─────────────────────────────────────────────
def get_dataset(tokenizer, train_samples, test_samples, max_length):
    print("Loading dataset...")
    dataset = load_dataset("amazon_polarity")
    
    small_train = dataset["train"].select(range(train_samples))
    small_test  = dataset["test"].select(range(test_samples))
    
    def preprocess(examples):
        texts  = [t + ". " + c for t, c in zip(examples["title"], examples["content"])]
        tokens = tokenizer(texts, truncation=True, padding="max_length", max_length=max_length)
        tokens["labels"] = examples["label"]
        return tokens
    
    tokenized_train = small_train.map(preprocess, batched=True)
    tokenized_test  = small_test.map(preprocess, batched=True)
    
    return tokenized_train, tokenized_test

# ── Metrics ─────────────────────────────────────────────
accuracy_metric = evaluate.load("accuracy")
f1_metric       = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels  = eval_pred
    predictions     = np.argmax(logits, axis=-1)
    accuracy        = accuracy_metric.compute(predictions=predictions, references=labels)
    f1              = f1_metric.compute(predictions=predictions, references=labels, average="binary")
    return {**accuracy, **f1}

# ── Training run ─────────────────────────────────────────
def train(learning_rate, num_epochs, batch_size):
    # Set MLflow experiment
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params({
            "model_name":     MODEL_NAME,
            "learning_rate":  learning_rate,
            "num_epochs":     num_epochs,
            "batch_size":     batch_size,
            "train_samples":  TRAIN_SAMPLES,
            "test_samples":   TEST_SAMPLES,
            "max_length":     MAX_LENGTH,
        })
        
        print(f"\nStarting run: lr={learning_rate} epochs={num_epochs} batch={batch_size}")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model     = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=2
        )
        
        # Load data
        tokenized_train, tokenized_test = get_dataset(
            tokenizer, TRAIN_SAMPLES, TEST_SAMPLES, MAX_LENGTH
        )
        
        # Training arguments
        args = TrainingArguments(
            output_dir          = f"./checkpoints/lr{learning_rate}",
            eval_strategy       = "epoch",
            save_strategy       = "epoch",
            num_train_epochs    = num_epochs,
            per_device_train_batch_size = batch_size,
            per_device_eval_batch_size  = batch_size,
            learning_rate       = learning_rate,
            load_best_model_at_end = True,
            logging_steps       = 50,
        )
        
        trainer = Trainer(
            model           = model,
            args            = args,
            train_dataset   = tokenized_train,
            eval_dataset    = tokenized_test,
            compute_metrics = compute_metrics,
        )
        
        # Train
        trainer.train()
        
        # Evaluate
        results = trainer.evaluate()
        print(f"\nResults: {results}")
        
        # Log metrics to MLflow
        mlflow.log_metrics({
            "accuracy":  results["eval_accuracy"],
            "f1":        results["eval_f1"],
            "eval_loss": results["eval_loss"],
        })
        
        # Log model to MLflow
        mlflow.pytorch.log_model(model, "model")
        
        print(f"Run logged to MLflow")
        return results["eval_f1"]

# ── Run 3 experiments ────────────────────────────────────
if __name__ == "__main__":
    experiments = [
        {"learning_rate": 2e-5, "num_epochs": 2, "batch_size": 16},
        {"learning_rate": 5e-5, "num_epochs": 2, "batch_size": 16},
        {"learning_rate": 1e-5, "num_epochs": 2, "batch_size": 16},
    ]
    
    results = []
    for exp in experiments:
        f1 = train(**exp)
        results.append({"params": exp, "f1": f1})
    
    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    for r in results:
        print(f"lr={r['params']['learning_rate']} → F1: {r['f1']:.4f}")
    
    best = max(results, key=lambda x: x["f1"])
    print(f"\nBest run: lr={best['params']['learning_rate']} F1={best['f1']:.4f}")