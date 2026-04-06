import json
import os

METRICS_PATH      = "./trained_model/metrics.json"
F1_THRESHOLD      = 0.85   # model must achieve this F1 to be deployed
ACCURACY_THRESHOLD = 0.85  # model must achieve this accuracy to be deployed

def run_evaluation():
    print("\n" + "="*50)
    print("STAGE 2 — EVALUATION")
    print("="*50)

    # Check model exists
    if not os.path.exists(METRICS_PATH):
        print("❌ No metrics file found — training may have failed")
        return False

    # Load metrics from training
    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    print(f"Model metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1:       {metrics['f1']:.4f}")
    print(f"  Loss:     {metrics['loss']:.4f}")

    print(f"\nThresholds:")
    print(f"  Min F1:       {F1_THRESHOLD}")
    print(f"  Min Accuracy: {ACCURACY_THRESHOLD}")

    # Gate 1 — F1 check
    if metrics["f1"] < F1_THRESHOLD:
        print(f"\n❌ EVALUATION FAILED")
        print(f"   F1 {metrics['f1']:.4f} is below threshold {F1_THRESHOLD}")
        print(f"   Model will NOT be deployed")
        return False

    # Gate 2 — Accuracy check
    if metrics["accuracy"] < ACCURACY_THRESHOLD:
        print(f"\n❌ EVALUATION FAILED")
        print(f"   Accuracy {metrics['accuracy']:.4f} is below threshold {ACCURACY_THRESHOLD}")
        print(f"   Model will NOT be deployed")
        return False

    print(f"\n✅ EVALUATION PASSED")
    print(f"   F1 {metrics['f1']:.4f} >= threshold {F1_THRESHOLD}")
    print(f"   Accuracy {metrics['accuracy']:.4f} >= threshold {ACCURACY_THRESHOLD}")
    print(f"   Model approved for deployment")
    return True

def run_evaluation_with_comparison():
    """Compare new model against previously deployed model"""
    print("\n" + "="*50)
    print("STAGE 2 — EVALUATION WITH COMPARISON")
    print("="*50)

    # Load new model metrics
    if not os.path.exists(METRICS_PATH):
        print("❌ No metrics file found")
        return False

    with open(METRICS_PATH) as f:
        new_metrics = json.load(f)

    # Load previous model metrics if exists
    prev_metrics_path = "./trained_model/prev_metrics.json"
    if os.path.exists(prev_metrics_path):
        with open(prev_metrics_path) as f:
            prev_metrics = json.load(f)

        print(f"{'Metric':<12} {'Previous':<12} {'New':<12} {'Change'}")
        print("-" * 50)
        for key in ["accuracy", "f1", "loss"]:
            prev = prev_metrics.get(key, 0)
            new  = new_metrics.get(key, 0)
            change = new - prev
            direction = "⬆️" if change > 0 else "⬇️" if change < 0 else "➡️"
            print(f"{key:<12} {prev:<12.4f} {new:<12.4f} {direction} {change:+.4f}")

        # Reject if new model is significantly worse
        if new_metrics["f1"] < prev_metrics["f1"] - 0.02:
            print(f"\n❌ New model F1 dropped by more than 2% vs previous")
            print(f"   Keeping previous model")
            return False
    else:
        print("No previous model found — skipping comparison")

    # Standard threshold check
    passed = run_evaluation()

    # Save current as previous for next run
    if passed:
        import shutil
        shutil.copy(METRICS_PATH, prev_metrics_path)

    return passed

if __name__ == "__main__":
    result = run_evaluation_with_comparison()
    print(f"\nEvaluation result: {'PASS' if result else 'FAIL'}")