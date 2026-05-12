"""
GempaWas — Inference API

REST API serving untuk prediksi gempa susulan.
Run dengan: uvicorn src.inference:app --reload --port 8000
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features import FEATURES, classify_zone, omori_rate

logger = logging.getLogger(__name__)

app = FastAPI(
    title="GempaWas Inference API",
    description="Prediksi probabilitas gempa susulan di Indonesia",
    version="0.1.0",
)


class PredictionRequest(BaseModel):
    """Input untuk prediksi: kondisi sekuens gempa saat ini."""
    mainshock_magnitude: float = Field(..., ge=5.0, le=9.5,
                                       description="Magnitudo gempa utama (≥ 5.0)")
    mainshock_depth: float = Field(..., ge=0, le=700,
                                   description="Kedalaman mainshock (km)")
    mainshock_latitude: float = Field(..., ge=-11.0, le=6.0)
    mainshock_longitude: float = Field(..., ge=95.0, le=141.0)
    jam_sejak_mainshock: float = Field(..., ge=0, le=168,
                                       description="Berapa jam sudah lewat sejak mainshock")
    count_susulan_1jam: int = Field(0, ge=0)
    count_susulan_6jam: int = Field(0, ge=0)
    count_susulan_24jam: int = Field(0, ge=0)
    max_mag_susulan_6jam: float = Field(0.0, ge=0)
    max_mag_susulan_24jam: float = Field(0.0, ge=0)


class PredictionResponse(BaseModel):
    """Output prediksi."""
    probabilitas_susulan_besar: float = Field(...,
        description="Probabilitas susulan ≥ M4.0 dalam 24 jam ke depan")
    kategori_risiko: str = Field(...,
        description="Rendah / Sedang / Tinggi")
    zona_sesar_terdeteksi: int
    omori_baseline: float = Field(...,
        description="Estimasi laju susulan menurut Omori's Law (per hari)")
    model_version: str
    inference_time_utc: str


_model = None


def get_model():
    """Lazy load model dari MLflow Registry."""
    global _model
    if _model is None:
        try:
            _model = mlflow.pyfunc.load_model("models:/gempawas-production/latest")
            logger.info("Model loaded from MLflow Registry")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(503, "Model tidak tersedia, lakukan training dulu")
    return _model


@app.get("/")
def root():
    return {
        "service": "GempaWas",
        "status": "online",
        "endpoints": ["/predict", "/health", "/docs"],
    }


@app.get("/health")
def health():
    return {"status": "healthy", "time": datetime.now(timezone.utc).isoformat()}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    """Prediksi probabilitas susulan signifikan dalam 24 jam ke depan."""
    model = get_model()

    zona = classify_zone(req.mainshock_latitude, req.mainshock_longitude)
    omori_est = omori_rate(req.jam_sejak_mainshock)

    feature_dict = {
        "mainshock_magnitude": req.mainshock_magnitude,
        "mainshock_depth": req.mainshock_depth,
        "jam_sejak_mainshock": req.jam_sejak_mainshock,
        "count_susulan_1jam": req.count_susulan_1jam,
        "count_susulan_6jam": req.count_susulan_6jam,
        "count_susulan_24jam": req.count_susulan_24jam,
        "max_mag_susulan_6jam": req.max_mag_susulan_6jam,
        "max_mag_susulan_24jam": req.max_mag_susulan_24jam,
        "omori_rate_est": omori_est,
        "zona_sesar": zona,
    }
    X = pd.DataFrame([feature_dict])[FEATURES]

    try:
        proba = float(model.predict_proba(X)[0, 1])
    except Exception:
        # Fallback kalau model tidak punya predict_proba
        proba = float(model.predict(X)[0])

    if proba >= 0.7:
        kategori = "Tinggi"
    elif proba >= 0.4:
        kategori = "Sedang"
    else:
        kategori = "Rendah"

    return PredictionResponse(
        probabilitas_susulan_besar=round(proba, 4),
        kategori_risiko=kategori,
        zona_sesar_terdeteksi=zona,
        omori_baseline=round(omori_est, 2),
        model_version="latest",
        inference_time_utc=datetime.now(timezone.utc).isoformat(),
    )
