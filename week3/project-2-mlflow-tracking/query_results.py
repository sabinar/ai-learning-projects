import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "amazon-sentiment-classifier"

client = MlflowClient()

# ── Get experiment ────────────────────────────────────────
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
print(f"Experiment: {experiment.name}")
print(f"Experiment ID: {experiment.experiment_id}")
print(f"Location: {experiment.artifact_location}")

# ── Get all runs ─────────────────────────────────────────
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.f1 DESC"]  # sort by F1 descending
)

print(f"\nTotal runs: {len(runs)}")
print("\n" + "="*65)
print(f"{'Run ID':<12} {'LR':<10} {'F1':<10} {'Accuracy':<12} {'Loss':<10}")
print("="*65)

for run in runs:
    params  = run.data.params
    metrics = run.data.metrics
    print(
        f"{run.info.run_id[:8]:<12} "
        f"{params.get('learning_rate', 'N/A'):<10} "
        f"{metrics.get('f1', 0):.4f}     "
        f"{metrics.get('accuracy', 0):.4f}       "
        f"{metrics.get('eval_loss', 0):.4f}"
    )

# ── Best run ─────────────────────────────────────────────
best_run = runs[0]
print("\n" + "="*65)
print("BEST RUN")
print("="*65)
print(f"Run ID:        {best_run.info.run_id}")
print(f"Learning rate: {best_run.data.params.get('learning_rate')}")
print(f"F1 Score:      {best_run.data.metrics.get('f1'):.4f}")
print(f"Accuracy:      {best_run.data.metrics.get('accuracy'):.4f}")
print(f"Eval Loss:     {best_run.data.metrics.get('eval_loss'):.4f}")
print(f"Status:        {best_run.info.status}")
print(f"Start time:    {best_run.info.start_time}")

# ── All parameters of best run ───────────────────────────
print("\nAll parameters:")
for k, v in best_run.data.params.items():
    print(f"  {k}: {v}")

# ── Tag best run ─────────────────────────────────────────
client.set_tag(best_run.info.run_id, "candidate", "best")
client.set_tag(best_run.info.run_id, "stage", "staging")
print(f"\nTagged run {best_run.info.run_id[:8]} as best candidate")