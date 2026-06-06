"""
Mendemonstrasikan siklus penuh CT secara lokal, end-to-end:
  1. Latih CHAMPION pada data acuan (distribusi lama) → register v1 (Production)
  2. Suntikkan SHIFTED DATA (distribusi sengaja digeser) sebagai "data baru"
  3. DETEKSI: ukur performa champion pada data baru (decay) + jalankan drift gate (PSI)
  4. RETRAIN: latih CHALLENGER pada data acuan + data baru
  5. EVALUASI KOMPARATIF: champion vs challenger pada distribusi baru
  6. PROMOSI: challenger naik ke Production hanya jika lebih baik
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.combine import SMOTETomek
from mlflow.tracking import MlflowClient
from sklearn.metrics import average_precision_score, f1_score, recall_score

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "ct_simulation.log", mode="w"),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("ct-sim")

FEATURES_PATH = Path("data/processed/features.parquet")
TARGET = "label_susulan_besar_24jam"
FEATURE_COLUMNS = [
    "mainshock_magnitude", "mainshock_depth", "jam_sejak_mainshock",
    "count_susulan_1jam", "count_susulan_6jam", "count_susulan_24jam",
    "max_mag_susulan_6jam", "max_mag_susulan_24jam", "omori_rate_est", "zona_sesar",
]
MODEL_NAME = "gempawas-aftershock-classifier"
TRACKING_URI = "sqlite:///mlflow_sim.db"
THRESHOLD = 0.45
RNG = np.random.default_rng(42)

REF_PATH = Path("data/processed/sim_reference.parquet")
CUR_PATH = Path("data/processed/sim_current.parquet")
METADATA = Path("models/model_registry_metadata.json")


def make_shifted_batch(base: pd.DataFrame, n: int = 700) -> pd.DataFrame:
    sample = base.sample(n=n, replace=True, random_state=42).reset_index(drop=True)

    # --- Covariate shift moderat (memicu PSI tapi tidak ekstrem) ---
    sample["mainshock_magnitude"] = np.round(RNG.normal(6.1, 0.55, n), 2)
    sample["mainshock_depth"] = np.round(np.abs(RNG.normal(62.0, 18.0, n)), 2)
    sample["jam_sejak_mainshock"] = np.round(np.abs(RNG.exponential(10.0, n)), 2)
    sample["omori_rate_est"] = np.round(np.abs(RNG.normal(4.2, 1.3, n)), 2)
    # Fitur susulan dijaga overlap dgn data lama → bukan sinyal pemisah baru
    sample["count_susulan_1jam"] = RNG.integers(0, 6, n)
    sample["count_susulan_6jam"] = RNG.integers(0, 16, n)
    sample["count_susulan_24jam"] = RNG.integers(0, 35, n)
    sample["max_mag_susulan_6jam"] = np.round(np.abs(RNG.normal(3.4, 0.9, n)), 2)
    sample["max_mag_susulan_24jam"] = np.round(np.abs(RNG.normal(3.9, 1.0, n)), 2)

    logit = (
        -8.5
        + 1.15 * sample["mainshock_magnitude"]
        + 0.045 * sample["mainshock_depth"]
        - 0.55 * sample["max_mag_susulan_24jam"]
    )
    prob = 1 / (1 + np.exp(-logit))
    sample[TARGET] = (RNG.uniform(0, 1, n) < prob).astype(int)
    return sample


def train_eval(train_df: pd.DataFrame, eval_X: pd.DataFrame, eval_y: pd.Series,
               run_name: str, tag: str) -> tuple[str, dict]:
    """Latih XGBoost (config run2 LK-06) pada train_df, evaluasi pada eval set, log MLflow."""
    Xtr = train_df[FEATURE_COLUMNS]
    ytr = train_df[TARGET].astype(int)
    smt = SMOTETomek(random_state=42)
    Xr, yr = smt.fit_resample(Xtr, ytr)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("ct_role", tag)
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5, gamma=0.5,
            objective="binary:logistic", eval_metric="aucpr",
            random_state=42, verbosity=0,
        )
        model.fit(Xr, yr)
        proba = model.predict_proba(eval_X)[:, 1]
        pred = (proba >= THRESHOLD).astype(int)
        metrics = {
            "pr_auc": float(average_precision_score(eval_y, proba)),
            "f1_score": float(f1_score(eval_y, pred, zero_division=0)),
            "recall": float(recall_score(eval_y, pred, zero_division=0)),
        }
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.log_param("n_train", len(Xr))
        mlflow.log_param("eval_distribution", "current_shifted")
        mlflow.xgboost.log_model(model, artifact_path="model")
        return run.info.run_id, metrics


def banner(t):
    log.info("\n" + "=" * 66)
    log.info("  " + t)
    log.info("=" * 66)


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("gempawas-ct-simulation")
    client = MlflowClient(tracking_uri=TRACKING_URI)

    banner("LANGKAH 0 — Persiapan data acuan & data baru (shifted)")
    base = pd.read_parquet(FEATURES_PATH)
    reference = base.copy()
    reference.to_parquet(REF_PATH)
    log.info("Data acuan (reference): %d baris | positif=%d (%.1f%%)",
             len(reference), reference[TARGET].sum(), 100 * reference[TARGET].mean())

    new_batch = make_shifted_batch(base, n=700)
    new_batch.to_parquet(CUR_PATH)
    log.info("Data baru (shifted)   : %d baris | positif=%d (%.1f%%)",
             len(new_batch), new_batch[TARGET].sum(), 100 * new_batch[TARGET].mean())

    # Split data baru: sebagian untuk retraining, sebagian sebagai "realita baru" (eval)
    new_train = new_batch.iloc[:450].reset_index(drop=True)
    new_eval = new_batch.iloc[450:].reset_index(drop=True)
    eval_X = new_eval[FEATURE_COLUMNS]
    eval_y = new_eval[TARGET].astype(int)
    log.info("Eval set (realita baru): %d baris | positif=%d", len(new_eval), eval_y.sum())

    banner("LANGKAH 1 — Latih CHAMPION pada distribusi LAMA, register v1 → Production")
    champ_run, champ_metrics = train_eval(reference, eval_X, eval_y,
                                          "sim_champion_old_dist", "champion")
    log.info("Champion run %s", champ_run[:12])
    log.info("Champion metrics pada DATA BARU: pr_auc=%.4f f1=%.4f recall=%.4f",
             champ_metrics["pr_auc"], champ_metrics["f1_score"], champ_metrics["recall"])

    # register v1 + champion alias
    mv1 = _create_and_version(client, champ_run)
    client.set_registered_model_alias(MODEL_NAME, "champion", mv1.version)
    log.info("✓ Champion = v%s (alias 'champion')", mv1.version)

    # tulis baseline champion ke metadata (performa pada distribusi baru = yang dipantau)
    METADATA.parent.mkdir(exist_ok=True, parents=True)
    METADATA.write_text(json.dumps({
        "model_name": MODEL_NAME,
        "primary_metric": "pr_auc",
        "production_model": {
            "model_name": MODEL_NAME, "version": mv1.version, "stage": "Production",
            "run_id": champ_run, "metrics": {k: round(v, 4) for k, v in champ_metrics.items()},
        },
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    banner("LANGKAH 2 — DETEKSI DRIFT (PSI) reference vs data baru")
    drift = subprocess.run(
        [sys.executable, "scripts/ct_drift_gate.py",
         "--reference", str(REF_PATH), "--current", str(CUR_PATH)],
        capture_output=True, text=True,
    )
    log.info(drift.stdout.strip())
    if drift.stderr.strip():
        log.info(drift.stderr.strip())

    banner("LANGKAH 3 — RETRAIN CHALLENGER pada data acuan + data baru")
    combined = pd.concat([reference, new_train], ignore_index=True)
    log.info("Dataset retraining: %d baris (acuan %d + baru %d)",
             len(combined), len(reference), len(new_train))
    chal_run, chal_metrics = train_eval(combined, eval_X, eval_y,
                                        "sim_challenger_new_dist", "challenger")
    log.info("Challenger run %s", chal_run[:12])
    log.info("Challenger metrics pada DATA BARU: pr_auc=%.4f f1=%.4f recall=%.4f",
             chal_metrics["pr_auc"], chal_metrics["f1_score"], chal_metrics["recall"])

    banner("LANGKAH 4 — EVALUASI KOMPARATIF champion vs challenger + PROMOSI")
    promote = subprocess.run(
        [sys.executable, "scripts/promote_if_better.py",
         "--model-name", MODEL_NAME,
         "--challenger-run-id", chal_run,
         "--tracking-uri", TRACKING_URI,
         "--register"],
        capture_output=True, text=True, env={**_env(), "CT_TRIGGER": "data_drift"},
    )
    log.info(promote.stdout.strip())
    if promote.stderr.strip():
        log.info(promote.stderr.strip())

    # Ringkasan before/after
    banner("RINGKASAN — Sebelum vs Sesudah Retraining (pada distribusi baru)")
    summary = {
        "champion_before": champ_metrics,
        "challenger_after": chal_metrics,
        "delta": {k: round(chal_metrics[k] - champ_metrics[k], 4) for k in champ_metrics},
        "reference_rows": len(reference),
        "new_batch_rows": len(new_batch),
        "eval_rows": len(new_eval),
        "ref_positive_rate": round(float(reference[TARGET].mean()), 4),
        "new_positive_rate": round(float(new_batch[TARGET].mean()), 4),
    }
    (LOG_DIR / "ct_before_after.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("  %-10s %-12s %-12s %-10s", "Metrik", "Champion", "Challenger", "Δ")
    log.info("  " + "-" * 46)
    for k in ("pr_auc", "f1_score", "recall"):
        log.info("  %-10s %-12.4f %-12.4f %+.4f", k, champ_metrics[k],
                 chal_metrics[k], summary["delta"][k])
    log.info("\n✓ Simulasi selesai. Lihat logs/ct_simulation.log & logs/ct_before_after.json")


def _model_exists(client) -> bool:
    try:
        client.get_registered_model(MODEL_NAME)
        return True
    except Exception:
        return False


def _create_and_version(client, run_id):
    try:
        client.create_registered_model(MODEL_NAME)
    except Exception:
        pass
    return client.create_model_version(MODEL_NAME, f"runs:/{run_id}/model", run_id=run_id)


def _env():
    import os
    return dict(os.environ)


if __name__ == "__main__":
    main()
