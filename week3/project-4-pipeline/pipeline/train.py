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
import os
import json

MODEL_NAME     = "distilbert-base-uncased"
TRAIN_SAMPLES  = 2000
TEST_SAMPLES   = 400
MAX_LENGTH     = 128
OUTPUT_DIR     = "./trained_model"

def get_dataset(tokenizer):
    print("Loading dataset...")
    dataset = load_dataset("amazon_polarity")

    small_train = dataset["train"].select(range(TRAIN_SAMPLES))
    small_test  = dataset["test"].select(range(TEST_SAMPLES))

    def preprocess(examples):
        texts  = [t + ". " + c for t, c in zip(examples["title"], examples["content"])]
        tokens = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        )
        tokens["labels"] = examples["label"]
        return tokens

    tokenized_train = small_train.map(preprocess, batched=True)
    tokenized_test  = small_test.map(preprocess, batched=True)

    return tokenized_train, tokenized_test

def run_training():
    print("\n" + "="*50)
    print("STAGE 1 — TRAINING")
    print("="*50)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )

    # Load data
    tokenized_train, tokenized_test = get_dataset(tokenizer)

    # Metrics
    accuracy_metric = evaluate.load("accuracy")
    f1_metric       = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions    = np.argmax(logits, axis=-1)
        accuracy = accuracy_metric.compute(
            predictions=predictions, references=labels)
        f1 = f1_metric.compute(
            predictions=predictions, references=labels, average="binary")
        return {**accuracy, **f1}

    # Training arguments
    args = TrainingArguments(
        output_dir                  = OUTPUT_DIR,
        eval_strategy               = "epoch",
        save_strategy               = "no",
        num_train_epochs            = 2,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size  = 16,
        learning_rate               = 2e-5,
        logging_steps               = 50,
    )

    trainer = Trainer(
        model           = model,
        args            = args,
        train_dataset   = tokenized_train,
        eval_dataset    = tokenized_test,
        compute_metrics = compute_metrics,
    )

    # Train
    print("Training model...")
    trainer.train()

    # Evaluate
    results = trainer.evaluate()
    print(f"\nTraining complete:")
    print(f"  Accuracy: {results['eval_accuracy']:.4f}")
    print(f"  F1:       {results['eval_f1']:.4f}")

    # Save model and tokenizer
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Save metrics
    metrics = {
        "accuracy": results["eval_accuracy"],
        "f1":       results["eval_f1"],
        "loss":     results["eval_loss"],
    }
    with open(f"{OUTPUT_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved to {OUTPUT_DIR}")
    return metrics

if __name__ == "__main__":
    run_training()