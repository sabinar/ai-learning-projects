import sys
import time
from pipeline.train import run_training
from pipeline.evaluate import run_evaluation_with_comparison
from pipeline.deploy import run_deployment
from pipeline.monitor import run_monitoring

def run_pipeline(skip_training=False):
    print("\n" + "="*50)
    print("ML PIPELINE STARTING")
    print("="*50)
    start_time = time.time()

    # ── Stage 1 — Training ───────────────────────────
    if skip_training:
        print("\nSkipping training — using existing model")
    else:
        try:
            metrics = run_training()
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            sys.exit(1)

    # ── Stage 2 — Evaluation ─────────────────────────
    try:
        passed = run_evaluation_with_comparison()
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        sys.exit(1)

    if not passed:
        print("\n❌ PIPELINE ABORTED — model did not pass evaluation")
        print("   Check metrics and retrain with different hyperparameters")
        sys.exit(1)

    # ── Stage 3 — Deployment ─────────────────────────
    try:
        server_process = run_deployment()
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        sys.exit(1)

    if not server_process:
        print("\n❌ PIPELINE ABORTED — deployment failed")
        sys.exit(1)

    # ── Stage 4 — Monitoring ─────────────────────────
    try:
        drift_detected = run_monitoring()
    except Exception as e:
        print(f"\n❌ Monitoring failed: {e}")
        server_process.terminate()
        sys.exit(1)

    # ── Pipeline Summary ─────────────────────────────
    total_time = time.time() - start_time
    print("\n" + "="*50)
    print("PIPELINE COMPLETE")
    print("="*50)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Stages completed: Training → Evaluation → Deployment → Monitoring")

    if drift_detected:
        print(f"\n⚠️  Drift detected — scheduling retraining")
        print(f"   In production: trigger next pipeline run with fresh data")
    else:
        print(f"\n✅ Model healthy — no retraining needed")

    # Cleanup
    print("\nShutting down server...")
    server_process.terminate()
    server_process.wait()
    print("Server stopped")

    return drift_detected

if __name__ == "__main__":
    # Pass --skip-training to reuse existing trained model
    skip = "--skip-training" in sys.argv
    run_pipeline(skip_training=skip)