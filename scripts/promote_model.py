# scripts/promote_model.py
import json
import argparse
import mlflow
from mlflow.tracking import MlflowClient

os.getenv("HF_TOKEN")

def main(threshold: float):
    mlflow.set_tracking_uri("http://ec2-13-61-180-100.eu-north-1.compute.amazonaws.com:5000/")
    client = MlflowClient()

    # En son run'ı bul
    experiment = mlflow.get_experiment_by_name("az_sentiment_nb")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    if not runs:
        raise RuntimeError("No runs found.")

    latest_run = runs[0]
    metrics = latest_run.data.metrics
    test_acc = metrics.get("test_accuracy", 0.0)

    if test_acc >= threshold:
        model_name = "az_sentiment_nb"
        version = None
        # Registered model version'ını bul
        for mv in client.search_model_versions(f"name='{model_name}'"):
            if mv.run_id == latest_run.info.run_id:
                version = mv.version
                break

        if version:
            # Staging'e geç (önce varsa)
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Staging"
            )
            # Production'a geç
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production",
                archive_existing_versions=True
            )
            print(f"✅ Model v{version} promoted to Production (test_acc={test_acc:.4f})")
        else:
            print("⚠️ Model version not found in registry.")
    else:
        print(f"❌ Model not promoted: test_accuracy ({test_acc:.4f}) < threshold ({threshold})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.75)
    args = parser.parse_args()
    main(args.threshold)
