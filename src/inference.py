import logging
import os
import socket
import time

import mlflow
import mlflow.xgboost
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)
from pydantic import BaseModel

# Logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Konfigurasi
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
MODEL_NAME          = os.getenv("MODEL_NAME", "gempawas-aftershock-classifier")
MODEL_STAGE         = os.getenv("MODEL_STAGE", "Staging")
HOSTNAME            = socket.gethostname()  # dipakai sebagai label replika

FEATURE_COLUMNS = [
    "mainshock_magnitude", "mainshock_depth", "jam_sejak_mainshock",
    "count_susulan_1jam",  "count_susulan_6jam",  "count_susulan_24jam",
    "max_mag_susulan_6jam","max_mag_susulan_24jam","omori_rate_est", "zona_sesar",
]

#  Prometheus Metrics 
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total jumlah HTTP request",
    ["method", "endpoint", "status_code", "replica"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Latensi HTTP request dalam detik",
    ["endpoint", "replica"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
INFERENCE_DURATION = Histogram(
    "model_inference_duration_seconds",
    "Waktu inferensi model murni (tanpa overhead HTTP)",
    ["replica"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
)
PREDICTION_COUNT = Counter(
    "model_predictions_total",
    "Total prediksi model per label kelas",
    ["label", "replica"],
)
PREDICTION_SCORE = Histogram(
    "prediction_score",
    "Distribusi skor probabilitas prediksi (indikator data drift)",
    ["replica"],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
IN_PROGRESS = Gauge(
    "http_requests_in_progress",
    "Jumlah request yang sedang diproses",
    ["endpoint", "replica"],
)
MODEL_INFO = Info("model_serving_info", "Informasi model yang sedang di-serve")

# Model Loading 
model = None

def load_model():
    global model
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    try:
        logger.info(f"Memuat model dari MLflow: {model_uri}")
        model = mlflow.xgboost.load_model(model_uri)
        MODEL_INFO.info({"name": MODEL_NAME, "stage": MODEL_STAGE, "replica": HOSTNAME})
        logger.info("Model berhasil dimuat.")
    except Exception as e:
        logger.error(f"Gagal memuat model: {e}")
        # Fallback: buat model XGBoost sederhana untuk testing
        import xgboost as xgb
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=200, n_features=len(FEATURE_COLUMNS), random_state=42)
        model = xgb.XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
        model.fit(X, y)
        MODEL_INFO.info({"name": "fallback-demo", "stage": "demo", "replica": HOSTNAME})
        logger.warning("Menggunakan fallback demo model.")

# FastAPI App 
app = FastAPI(title="GempaWas Inference API", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    load_model()

# Middleware: catat latensi dan jumlah request
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    endpoint = request.url.path
    IN_PROGRESS.labels(endpoint=endpoint, replica=HOSTNAME).inc()
    start = time.perf_counter()
    try:
        response = await call_next(request)
        status = str(response.status_code)
    except Exception as exc:
        status = "500"
        raise exc
    finally:
        latency = time.perf_counter() - start
        REQUEST_LATENCY.labels(endpoint=endpoint, replica=HOSTNAME).observe(latency)
        REQUEST_COUNT.labels(
            method=request.method, endpoint=endpoint,
            status_code=status, replica=HOSTNAME,
        ).inc()
        IN_PROGRESS.labels(endpoint=endpoint, replica=HOSTNAME).dec()
    return response

# Schemas 
class PredictRequest(BaseModel):
    # Format kompatibel dengan mlflow models serve
    inputs: list[list[float]]

class PredictResponse(BaseModel):
    predictions: list[float]  # skor probabilitas
    labels: list[int]          # 0 = tidak ada aftershock besar, 1 = ada
    replica: str

# Endpoints 
@app.get("/ping")
def ping():
    return {"status": "ok", "replica": HOSTNAME}

@app.post("/invocations", response_model=PredictResponse)
def invocations(payload: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model belum siap.")
    if not payload.inputs:
        raise HTTPException(status_code=400, detail="Input kosong.")

    X = np.array(payload.inputs, dtype=np.float32)

    # Ukur waktu inferensi murni
    t_start = time.perf_counter()
    probabilities = model.predict_proba(X)[:, 1]
    INFERENCE_DURATION.labels(replica=HOSTNAME).observe(time.perf_counter() - t_start)

    labels = (probabilities >= 0.45).astype(int).tolist()
    scores = probabilities.tolist()

    # Catat distribusi skor dan label ke Prometheus
    for score, label in zip(scores, labels):
        PREDICTION_SCORE.labels(replica=HOSTNAME).observe(score)
        PREDICTION_COUNT.labels(label=str(label), replica=HOSTNAME).inc()

    return PredictResponse(predictions=scores, labels=labels, replica=HOSTNAME)

@app.get("/metrics")
def metrics():
    """Endpoint Prometheus — Prometheus scrape dari sini setiap 10-15 detik."""
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )
