from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Setup logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "train.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
FEATURES_PATH = Path("data/processed/features.parquet")
TARGET_COLUMN = "label_susulan_besar_24jam"

FEATURE_COLUMNS = [
    "mainshock_magnitude",
    "mainshock_depth",
    "jam_sejak_mainshock",
    "count_susulan_1jam",
    "count_susulan_6jam",
    "count_susulan_24jam",
    "max_mag_susulan_6jam",
    "max_mag_susulan_24jam",
    "omori_rate_est",
    "zona_sesar",
]

MLFLOW_TRACKING_URI = "file:./mlruns"
MLFLOW_EXPERIMENT_NAME = "gempawas-aftershock-prediction"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_features(features_path: Path = FEATURES_PATH) -> pd.DataFrame:
    if not features_path.exists():
        raise FileNotFoundError(
            f"{features_path} tidak ada.\n"
            f"Jalankan dulu:\n"
            f"  1. python src/data/ingest_data.py --hours 17520\n"
            f"  2. python src/data/preprocess.py --all\n"
            f"  3. python src/build_features.py"
        )
    df = pd.read_parquet(features_path)
    logger.info("Loaded features: %d rows, %d cols", len(df), len(df.columns))
    return df


def prepare_train_test(df: pd.DataFrame, test_size: float = 0.25):
    """Time-based split: data lama untuk train, data baru untuk test."""
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom missing dari features.parquet: {missing}")

    if "mainshock_time" in df.columns:
        df = df.sort_values("mainshock_time").reset_index(drop=True)
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        logger.info("TIME-BASED split: train=%d, test=%d", len(train_df), len(test_df))
    else:
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=42,
            stratify=df[TARGET_COLUMN],
        )
        logger.info("STRATIFIED random split: train=%d, test=%d", len(train_df), len(test_df))

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN].astype(int)
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN].astype(int)

    logger.info("Class distribution train: %s", y_train.value_counts().to_dict())
    logger.info("Class distribution test : %s", y_test.value_counts().to_dict())
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Train satu run
# ---------------------------------------------------------------------------
def train_one_run(
    X_train, X_test, y_train, y_test,
    n_estimators: int = 100,
    max_depth: int = 5,
    learning_rate: float = 0.1,
    scale_pos_weight: float | None = None,
    run_name: str = "default",
    register_model: bool = False,
):
    logger.info("\n" + "=" * 60)
    logger.info("RUN: %s", run_name)
    logger.info("=" * 60)

    # Auto scale_pos_weight untuk handle class imbalance
    if scale_pos_weight is None:
        n_neg = int((y_train == 0).sum())
        n_pos = max(int((y_train == 1).sum()), 1)
        scale_pos_weight = n_neg / n_pos

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # ===== 1) LOG PARAMETER (LK-06 #3 bullet 1) =====
        params = {
            "model_type": "XGBoost",
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "scale_pos_weight": round(scale_pos_weight, 3),
            "objective": "binary:logistic",
            "random_state": 42,
            "n_features": X_train.shape[1],
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
        }
        for k, v in params.items():
            mlflow.log_param(k, v)

        # ===== 2) TRAIN MODEL =====
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="aucpr",
            random_state=42,
            verbosity=0,
        )
        model.fit(X_train, y_train)

        # ===== 3) EVALUATE & LOG 3 METRIC INTI (LK-06 #3 bullet 2) =====
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # HANYA 3 metric inti — rapi di dashboard
        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        try:
            auc = float(roc_auc_score(y_test, y_proba))
        except ValueError:
            auc = 0.0  # kalau hanya 1 kelas di test set

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        # ===== 4) FEATURE IMPORTANCE sebagai ARTIFACT (BUKAN metric) =====
        importances = pd.Series(
            model.feature_importances_, index=FEATURE_COLUMNS
        ).sort_values(ascending=False)
        importance_text = "Feature Importance (top to bottom):\n\n"
        for feat, imp in importances.items():
            importance_text += f"  {feat:<25} {imp:.4f}\n"
        fi_path = LOG_DIR / f"feature_importance_{run_name}.txt"
        fi_path.write_text(importance_text)
        mlflow.log_artifact(str(fi_path))

        # ===== 5) CONFUSION MATRIX sebagai ARTIFACT =====
        cm = confusion_matrix(y_test, y_pred)
        cm_text = (
            f"Confusion Matrix (run: {run_name})\n\n"
            f"               Predicted\n"
            f"                No   Yes\n"
            f"Actual No     {cm[0, 0]:>4}  {cm[0, 1] if cm.shape[1] > 1 else 0:>4}\n"
            f"       Yes    {cm[1, 0] if cm.shape[0] > 1 else 0:>4}  "
            f"{cm[1, 1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0:>4}\n"
        )
        cm_path = LOG_DIR / f"cm_{run_name}.txt"
        cm_path.write_text(cm_text)
        mlflow.log_artifact(str(cm_path))

        # ===== 6) LOG MODEL (LK-06 #3 bullet 3) =====
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name="gempawas-aftershock-classifier" if register_model else None,
        )

        # Print summary ke terminal
        logger.info("\n--- Hasil Run: %s ---", run_name)
        logger.info("  accuracy : %.4f", acc)
        logger.info("  f1_score : %.4f", f1)
        logger.info("  roc_auc  : %.4f", auc)
        logger.info("  MLflow Run ID: %s", run_id)
        if register_model:
            logger.info("  ✓ Model di-register sebagai 'gempawas-aftershock-classifier'")

        return run_id, {"accuracy": acc, "f1_score": f1, "roc_auc": auc}


# ---------------------------------------------------------------------------
# 3 variasi
# ---------------------------------------------------------------------------
def run_variations(X_train, X_test, y_train, y_test):
    """3 variasi sesuai requirement LK-06 #4."""
    variations = [
        {"run_name": "run1_shallow_n50_d3_lr0.1",
         "n_estimators": 50,  "max_depth": 3, "learning_rate": 0.1},
        {"run_name": "run2_medium_n100_d5_lr0.1",
         "n_estimators": 100, "max_depth": 5, "learning_rate": 0.1},
        {"run_name": "run3_deep_n200_d7_lr0.05",
         "n_estimators": 200, "max_depth": 7, "learning_rate": 0.05},
    ]

    results = []
    for v in variations:
        run_id, metrics = train_one_run(X_train, X_test, y_train, y_test, **v)
        results.append({"run_name": v["run_name"], "run_id": run_id, **metrics})

    return results


def print_comparison(results):
    logger.info("\n" + "=" * 80)
    logger.info("PERBANDINGAN HASIL 3 RUN — LK-06 #4 & #5")
    logger.info("=" * 80)
    df = pd.DataFrame(results)
    cols = ["run_name", "accuracy", "f1_score", "roc_auc"]
    cols = [c for c in cols if c in df.columns]
    print("\n" + df[cols].to_string(index=False))

    best_idx = df["f1_score"].idxmax()
    best = df.iloc[best_idx]
    logger.info("\n>>> Best Run (by F1-score): %s", best["run_name"])
    logger.info("    Run ID: %s", best["run_id"])
    logger.info("    accuracy=%.4f  f1_score=%.4f  roc_auc=%.4f",
                best["accuracy"], best["f1_score"], best["roc_auc"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="GempaWas LK-06 Training")
    parser.add_argument("--variations", action="store_true")
    parser.add_argument("--run-name", default="manual_run")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--register-model", action="store_true")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--features-path", type=Path, default=FEATURES_PATH)
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger.info("MLflow tracking URI : %s", MLFLOW_TRACKING_URI)
    logger.info("MLflow experiment   : %s", MLFLOW_EXPERIMENT_NAME)

    df = load_features(args.features_path)
    X_train, X_test, y_train, y_test = prepare_train_test(df, test_size=args.test_size)

    if args.variations:
        results = run_variations(X_train, X_test, y_train, y_test)
        print_comparison(results)
    else:
        train_one_run(
            X_train, X_test, y_train, y_test,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            run_name=args.run_name,
            register_model=args.register_model,
        )

    logger.info("\n✓ Selesai. Buka MLflow UI: mlflow ui --host 0.0.0.0 --port 5000")


if __name__ == "__main__":
    main()