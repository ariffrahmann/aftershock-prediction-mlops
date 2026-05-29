"""Register model lama dari mlruns/ lokal ke MLflow Docker."""
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient

MODEL_NAME = "gempawas-aftershock-classifier"
LOCAL_URI  = "file:./mlruns"
DOCKER_URI = "http://localhost:5000"

print("Memuat model dari mlruns lokal...")
mlflow.set_tracking_uri(LOCAL_URI)
model = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}@champion")
print("Model lokal berhasil dimuat.")

print(f"Register ke MLflow Docker: {DOCKER_URI}")
mlflow.set_tracking_uri(DOCKER_URI)
mlflow.set_experiment("gempawas-aftershock-prediction")

with mlflow.start_run(run_name="register-from-local") as run:
    mlflow.xgboost.log_model(
        model, artifact_path="model",
        registered_model_name=MODEL_NAME,
    )
    print(f"Model ter-log ke run: {run.info.run_id}")

client = MlflowClient(tracking_uri=DOCKER_URI)
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
latest = max(int(v.version) for v in versions)
print(f"Versi terbaru di Docker: {latest}")

try:
    client.transition_model_version_stage(
        name=MODEL_NAME, version=latest, stage="Staging",
    )
    print(f"Versi {latest} -> stage Staging")
except Exception:
    client.set_registered_model_alias(MODEL_NAME, "Staging", latest)
    print(f"Versi {latest} -> alias Staging (MLflow 3.x)")

print("SELESAI. Model siap di MLflow Docker.")
