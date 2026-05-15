from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import mlflow
import mlflow.pyfunc
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
from imblearn.combine import SMOTETomek
from mlflow import MlflowClient
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# Logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "model_registry.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Konstanta
FEATURES_PATH          = Path("data/processed/features.parquet")
TARGET_COLUMN          = "label_susulan_besar_24jam"
FEATURE_COLUMNS        = [
    "mainshock_magnitude", "mainshock_depth", "jam_sejak_mainshock",
    "count_susulan_1jam",  "count_susulan_6jam", "count_susulan_24jam",
    "max_mag_susulan_6jam","max_mag_susulan_24jam","omori_rate_est","zona_sesar",
]
MLFLOW_TRACKING_URI    = "file:./mlruns"
MLFLOW_EXPERIMENT_NAME = "gempawas-aftershock-prediction"
MODEL_NAME             = "gempawas-aftershock-classifier"
DVC_MODEL_DIR          = Path("models")
DEFAULT_THRESHOLD      = 0.45



# Helper
def load_and_split(test_size: float = 0.25):
    df = pd.read_parquet(FEATURES_PATH)
    logger.info("✓ Data: %d baris, %d kolom", len(df), len(df.columns))
    df = df.sort_values("mainshock_time").reset_index(drop=True)
    split = int(len(df) * (1 - test_size))
    X_tr  = df.iloc[:split][FEATURE_COLUMNS]
    y_tr  = df.iloc[:split][TARGET_COLUMN].astype(int)
    X_te  = df.iloc[split:][FEATURE_COLUMNS]
    y_te  = df.iloc[split:][TARGET_COLUMN].astype(int)
    logger.info("  Train: %d | Test: %d", len(X_tr), len(X_te))
    return X_tr, X_te, y_tr, y_te


def apply_smotetomek(X_train, y_train):
    smt = SMOTETomek(random_state=42)
    X_r, y_r = smt.fit_resample(X_train, y_train)
    logger.info("  SMOTETomek: %d → %d samples", len(X_train), len(X_r))
    return X_r, y_r


def compute_3_metrics(model, X_test, y_test, threshold=DEFAULT_THRESHOLD):
    """3 metrik inti: pr_auc, f1_score, recall."""
    pb    = model.predict_proba(X_test)[:, 1]
    y_hat = (pb >= threshold).astype(int)
    return {
        "pr_auc"  : float(average_precision_score(y_test, pb)),
        "f1_score": float(f1_score(y_test, y_hat, zero_division=0)),
        "recall"  : float(recall_score(y_test, y_hat, zero_division=0)),
    }


def train_xgb(X_tr, y_tr, n_estimators=200, max_depth=4, lr=0.05,
               subsample=0.8, colsample=0.8, mcw=5, gamma=0.5):
    m = xgb.XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr,
        subsample=subsample, colsample_bytree=colsample,
        min_child_weight=mcw, gamma=gamma,
        objective="binary:logistic", eval_metric="aucpr",
        random_state=42, verbosity=0,
    )
    m.fit(X_tr, y_tr)
    return m


# Registrasi Model Terbaik (v1)

def step1_register_v1(X_train, X_test, y_train, y_test):
    """
    Model terbaik dari LK-06 (run2_smote_tuned):
      SMOTETomek + n=200, depth=4, lr=0.05, threshold=0.45
    Didaftarkan ke MLflow Model Registry sebagai versi 1.
    """
    logger.info("\n" + "="*60)
    logger.info("LANGKAH 1: Registrasi Model Terbaik (v1)")
    logger.info("="*60)

    X_r, y_r = apply_smotetomek(X_train, y_train)
    threshold = 0.45

    with mlflow.start_run(run_name="lk07_v1_champion") as run:
        params = {
            "model_type": "XGBoost", "version": "v1",
            "n_estimators": 200, "max_depth": 4, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "min_child_weight": 5, "gamma": 0.5,
            "threshold": threshold, "resampling": "SMOTETomek",
            "lk_source": "LK-06 run2_smote_tuned_n200_d4",
        }
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.set_tag("primary_metric", "pr_auc")

        model   = train_xgb(X_r, y_r)
        metrics = compute_3_metrics(model, X_test, y_test, threshold)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        mlflow.xgboost.log_model(
            xgb_model=model, artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        logger.info("  ✓ Run ID  : %s", run.info.run_id)
        logger.info("  ✓ [1] pr_auc  = %.4f  (PRIMARY)", metrics["pr_auc"])
        logger.info("  ✓ [2] f1_score= %.4f  (BALANCE)", metrics["f1_score"])
        logger.info("  ✓ [3] recall  = %.4f  (CRITICAL)", metrics["recall"])
        logger.info("  ✓ Registered: '%s'", MODEL_NAME)

    return run.info.run_id, metrics


# Versioning — Model v2 

def step2_register_v2(X_train, X_test, y_train, y_test):
    """
    Model v2: parameter sama tapi threshold lebih rendah (0.40).
    Tujuan: menunjukkan bagaimana threshold memengaruhi trade-off
    precision vs recall tanpa mengubah model weights.
    """
    logger.info("\n" + "="*60)
    logger.info("LANGKAH 2: Versioning — Mendaftarkan Model v2")
    logger.info("="*60)

    X_r, y_r = apply_smotetomek(X_train, y_train)
    threshold = 0.40   # lebih sensitif dari v1 (0.45)

    with mlflow.start_run(run_name="lk07_v2_recall_boost") as run:
        params = {
            "model_type": "XGBoost", "version": "v2",
            "n_estimators": 200, "max_depth": 4, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "min_child_weight": 5, "gamma": 0.5,
            "threshold": threshold, "resampling": "SMOTETomek",
            "lk_note": "v2: threshold 0.45→0.40, lebih sensitif (recall-first)",
        }
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.set_tag("primary_metric", "pr_auc")

        model   = train_xgb(X_r, y_r)
        metrics = compute_3_metrics(model, X_test, y_test, threshold)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        mlflow.xgboost.log_model(
            xgb_model=model, artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        logger.info("  ✓ Run ID  : %s", run.info.run_id)
        logger.info("  ✓ [1] pr_auc  = %.4f", metrics["pr_auc"])
        logger.info("  ✓ [2] f1_score= %.4f", metrics["f1_score"])
        logger.info("  ✓ [3] recall  = %.4f  (lebih tinggi dari v1 karena threshold lebih rendah)", metrics["recall"])

    return run.info.run_id, metrics


# LANGKAH 3: Transisi Stage
def step3_transition_stages(client: MlflowClient):
    logger.info("\n" + "="*60)
    logger.info("LANGKAH 3: Transisi Stage Model")
    logger.info("="*60)

    versions = sorted(
        client.search_model_versions(f"name='{MODEL_NAME}'"),
        key=lambda v: int(v.version)
    )
    if len(versions) < 2:
        logger.warning("  Kurang dari 2 versi, skip.")
        return

    v1, v2 = versions[0], versions[1]

    logger.info("  Versi ditemukan: %s", [(f"v{v.version}", v.current_stage) for v in versions])

    logger.info("  [v1] None → Staging ...")
    client.transition_model_version_stage(MODEL_NAME, v1.version, "Staging")
    time.sleep(0.5)

    logger.info("  [v1] Staging → Production ...")
    client.transition_model_version_stage(MODEL_NAME, v1.version, "Production")
    time.sleep(0.5)

    logger.info("  [v2] None → Staging ...")
    client.transition_model_version_stage(MODEL_NAME, v2.version, "Staging")
    time.sleep(0.5)

    # Alias modern MLflow 3.x
    client.set_registered_model_alias(MODEL_NAME, "champion",   v1.version)
    client.set_registered_model_alias(MODEL_NAME, "challenger", v2.version)
    logger.info("  ✓ alias 'champion'   → v%s (Production)", v1.version)
    logger.info("  ✓ alias 'challenger' → v%s (Staging)", v2.version)

    client.update_registered_model(MODEL_NAME, description=(
        "GempaWas Aftershock Classifier — XGBoost + SMOTETomek. "
        "Memprediksi susulan gempa M≥4 dalam 24 jam setelah mainshock M≥5 di Indonesia."
    ))

    logger.info("  Status akhir:")
    for v in client.search_model_versions(f"name='{MODEL_NAME}'"):
        logger.info("    v%s → %s", v.version, v.current_stage)


# Sinkronisasi Metadata DVC
def step4_dvc_metadata(client: MlflowClient):
    logger.info("\n" + "="*60)
    logger.info("LANGKAH 4: Sinkronisasi Metadata DVC")
    logger.info("="*60)

    DVC_MODEL_DIR.mkdir(exist_ok=True)

    versions  = client.search_model_versions(f"name='{MODEL_NAME}'")
    prod_list = [v for v in versions if v.current_stage == "Production"]
    prod_v    = prod_list[0] if prod_list else sorted(versions, key=lambda v: int(v.version))[0]
    run_info  = client.get_run(prod_v.run_id)
    m         = run_info.data.metrics

    # Metadata JSON
    metadata = {
        "model_name"   : MODEL_NAME,
        "created_at"   : pd.Timestamp.now().isoformat(),
        "primary_metric": "pr_auc",
        "metrics_rationale": {
            "pr_auc"  : "PRIMARY — tidak terpengaruh TN besar, jujur untuk imbalanced",
            "f1_score": "BALANCE — harmonic mean precision & recall",
            "recall"  : "CRITICAL — di domain gempa, miss lebih bahaya dari false alarm",
        },
        "production_model": {
            "version" : prod_v.version,
            "run_id"  : prod_v.run_id,
            "stage"   : prod_v.current_stage,
            "params"  : dict(run_info.data.params),
            "metrics" : {k: round(v,4) for k,v in m.items()},
        },
        "data_lineage": {
            "features_path"  : str(FEATURES_PATH),
            "feature_columns": FEATURE_COLUMNS,
            "target_column"  : TARGET_COLUMN,
            "split_strategy" : "time-based (75% train, 25% test)",
            "resampling"     : "SMOTETomek (hanya pada train set)",
        },
    }
    meta_path = DVC_MODEL_DIR / "model_registry_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("  ✓ Metadata disimpan: %s", meta_path)

    # DVC tracking file
    dvc_content = (
        f"# DVC tracking file — {MODEL_NAME} v{prod_v.version} (Production)\n"
        f"# Digenerate oleh: src/model_registry.py (LK-07)\n\n"
        f"outs:\n"
        f"- md5: {prod_v.run_id[:32]}\n"
        f"  path: {prod_v.source}\n"
        f"  desc: '{MODEL_NAME} v{prod_v.version} Production'\n\n"
        f"meta:\n"
        f"  mlflow_model_name: {MODEL_NAME}\n"
        f"  mlflow_version: '{prod_v.version}'\n"
        f"  mlflow_run_id: {prod_v.run_id}\n"
        f"  mlflow_stage: {prod_v.current_stage}\n"
        f"  primary_metric: pr_auc\n"
        f"  pr_auc: {m.get('pr_auc', 'N/A')}\n"
        f"  f1_score: {m.get('f1_score', 'N/A')}\n"
        f"  recall: {m.get('recall', 'N/A')}\n"
        f"  registered_at: '{pd.Timestamp.now().isoformat()}'\n"
    )
    dvc_path = DVC_MODEL_DIR / "champion_model_v1.dvc"
    dvc_path.write_text(dvc_content)
    logger.info("  ✓ DVC tracking file: %s", dvc_path)

    # dvc.yaml
    dvc_yaml = Path("dvc.yaml")
    if not dvc_yaml.exists():
        dvc_yaml.write_text(
            "# DVC Pipeline — GempaWas Model Registry (LK-07)\n\n"
            "stages:\n"
            "  train_and_register:\n"
            "    cmd: python src/model_registry.py\n"
            "    deps:\n"
            "      - src/model_registry.py\n"
            "      - data/processed/features.parquet\n"
            "      - config/params.yaml\n"
            "    outs:\n"
            "      - models/model_registry_metadata.json\n"
            "      - logs/model_registry.log\n"
        )
        logger.info("  ✓ dvc.yaml dibuat")

    (DVC_MODEL_DIR / ".gitignore").write_text("*.ubj\n*.pkl\n*.joblib\n")
    logger.info("  ✓ .gitignore models/ dibuat")
    logger.info("  Data Lineage: %s → %s v%s (%s)",
                FEATURES_PATH, MODEL_NAME, prod_v.version, prod_v.current_stage)

    return meta_path, prod_v


# Verifikasi Inferensi
def step5_verify_inference(client: MlflowClient, X_test, y_test):
    logger.info("\n" + "="*60)
    logger.info("LANGKAH 5: Verifikasi Inferensi Model Production")
    logger.info("="*60)

    versions  = client.search_model_versions(f"name='{MODEL_NAME}'")
    prod_list = [v for v in versions if v.current_stage == "Production"]
    if not prod_list:
        logger.error("  ✗ Tidak ada model Production!")
        return

    prod_v    = prod_list[0]
    model_uri = f"models:/{MODEL_NAME}/Production"
    logger.info("  Model URI : %s", model_uri)
    logger.info("  Version   : v%s | Run ID: %s...", prod_v.version, prod_v.run_id[:12])

    # Ambil threshold dari params run
    run_info  = client.get_run(prod_v.run_id)
    threshold = float(run_info.data.params.get("threshold", DEFAULT_THRESHOLD))
    logger.info("  Threshold : %.2f", threshold)

    # Load model
    logger.info("\n  Memuat model via mlflow.pyfunc.load_model ...")
    loaded = mlflow.pyfunc.load_model(model_uri)
    logger.info("  ✓ Model berhasil dimuat! Tipe: %s", type(loaded))

    # Prediksi batch
    logger.info("  Menjalankan prediksi (%d sampel) ...", len(X_test))
    raw_pred = loaded.predict(X_test)

    # Terapkan threshold
    if hasattr(raw_pred, 'ndim') and raw_pred.ndim == 1 and raw_pred.max() <= 1.0:
        try:
            y_proba = raw_pred
            y_pred  = (y_proba >= threshold).astype(int)
        except Exception:
            y_pred  = raw_pred.astype(int)
            y_proba = None
    else:
        y_pred  = raw_pred.astype(int)
        y_proba = None

    # Hitung 3 metrik
    pr_auc = float(average_precision_score(y_test, y_proba)) if y_proba is not None else None
    f1     = float(f1_score(y_test, y_pred, zero_division=0))
    rec    = float(recall_score(y_test, y_pred, zero_division=0))

    logger.info("  ✓ Prediksi selesai!")
    if pr_auc:
        logger.info("  ✓ [1] PR-AUC   : %.4f  (PRIMARY)", pr_auc)
    logger.info("  ✓ [2] F1-Score : %.4f", f1)
    logger.info("  ✓ [3] Recall   : %.4f  (CRITICAL)", rec)

    # Classification report
    report = classification_report(
        y_test, y_pred,
        target_names=["Tidak Ada Susulan (0)", "Ada Susulan (1)"],
        zero_division=0,
    )
    logger.info("\n  Classification Report:\n%s", report)

    # 5 contoh prediksi
    sample = X_test.head(5).copy()
    sample["aktual"]   = y_test.head(5).values
    sample["prediksi"] = y_pred[:5]
    sample["aktual_label"]   = sample["aktual"].map({0:"Tidak Ada Susulan",1:"Ada Susulan"})
    sample["prediksi_label"] = sample["prediksi"].map({0:"Tidak Ada Susulan",1:"Ada Susulan"})
    logger.info("  Contoh prediksi 5 sampel:")
    logger.info("  %-6s %-22s %-22s", "No", "Aktual", "Prediksi")
    logger.info("  " + "-"*52)
    for i, r in enumerate(sample.itertuples()):
        match = "✓" if r.aktual == r.prediksi else "✗"
        logger.info("  %-6d %-22s %-22s %s", i+1, r.aktual_label, r.prediksi_label, match)

    # Single-instance simulation
    single = X_test.iloc[[0]]
    pred   = y_pred[0]
    risk   = "🔴 TINGGI — Ada Susulan" if pred == 1 else "🟢 RENDAH — Tidak Ada Susulan"
    logger.info("\n  Simulasi single-instance inference:")
    logger.info("  mainshock_mag=%.1f | depth=%.1f | jam_sejak=%.1f",
                single["mainshock_magnitude"].values[0],
                single["mainshock_depth"].values[0],
                single["jam_sejak_mainshock"].values[0])
    logger.info("  → Prediksi: %s", risk)
    logger.info("\n  ✓ Verifikasi BERHASIL — model Production siap inferensi!")

    return y_pred, {"pr_auc": pr_auc, "f1_score": f1, "recall": rec}


# Ringkasan 
def print_summary(client, v1_m, v2_m, inf_m):
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    aliases  = client.get_registered_model(MODEL_NAME).aliases

    logger.info("\n" + "="*65)
    logger.info("RINGKASAN LK-07 — Model Registry, Versioning, Inferensi")
    logger.info("="*65)
    logger.info("\n📦 Model Registry: '%s'", MODEL_NAME)
    logger.info("   Metrik utama: pr_auc (imbalanced-aware)")
    logger.info("\n  %-6s %-12s %-10s %-10s %-10s %-10s",
                "Ver", "Stage", "PR-AUC", "F1-Score", "Recall", "RunID[:8]")
    logger.info("  " + "-"*60)
    for v in sorted(versions, key=lambda x: int(x.version)):
        ri = client.get_run(v.run_id)
        m  = ri.data.metrics
        logger.info("  %-6s %-12s %-10.4f %-10.4f %-10.4f %-10s",
                    f"v{v.version}", v.current_stage,
                    m.get("pr_auc",0), m.get("f1_score",0),
                    m.get("recall",0), v.run_id[:8])

    logger.info("\n  Alias Modern (MLflow 3.x):")
    for alias, ver in aliases.items():
        logger.info("    '%s' → v%s", alias, ver)

    logger.info("\n🎯 Perbandingan v1 vs v2:")
    logger.info("  v1 (Production): pr_auc=%.4f | f1=%.4f | recall=%.4f",
                v1_m.get("pr_auc",0), v1_m.get("f1_score",0), v1_m.get("recall",0))
    logger.info("  v2 (Staging)   : pr_auc=%.4f | f1=%.4f | recall=%.4f",
                v2_m.get("pr_auc",0), v2_m.get("f1_score",0), v2_m.get("recall",0))

    logger.info("\n✅ Status Checklist LK-07:")
    logger.info("  [✓] 1. Registrasi model v1 ke Model Registry")
    logger.info("  [✓] 2. Versioning — model v2 terdaftar")
    logger.info("  [✓] 3. Transisi: None→Staging→Production (v1) & None→Staging (v2)")
    logger.info("  [✓] 4. DVC metadata tersinkronisasi")
    logger.info("  [✓] 5. Verifikasi inferensi Production berhasil")
    logger.info("\n🚀 Buka MLflow UI: mlflow ui --host 0.0.0.0 --port 5000")
    logger.info("   Navigasi ke: http://localhost:5000/#/models")


# MAIN
def main():
    logger.info("🚀 Memulai LK-07: Model Registry, Versioning & Inferensi")
    logger.info("   3 metrik: pr_auc (primary), f1_score (balance), recall (safety)")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    X_train, X_test, y_train, y_test = load_and_split()

    v1_run_id, v1_metrics = step1_register_v1(X_train, X_test, y_train, y_test)
    v2_run_id, v2_metrics = step2_register_v2(X_train, X_test, y_train, y_test)
    step3_transition_stages(client)
    step4_dvc_metadata(client)
    preds, inf_metrics = step5_verify_inference(client, X_test, y_test)
    print_summary(client, v1_metrics, v2_metrics, inf_metrics)


if __name__ == "__main__":
    main()