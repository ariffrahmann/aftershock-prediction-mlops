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
from imblearn.combine import SMOTETomek
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# Setup Logging
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

# Konfigurasi
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

MLFLOW_TRACKING_URI    = "file:./mlruns"
MLFLOW_EXPERIMENT_NAME = "gempawas-aftershock-prediction"
DEFAULT_THRESHOLD      = 0.45


# Data Loading & Split
def load_features(features_path: Path = FEATURES_PATH) -> pd.DataFrame:
    if not features_path.exists():
        raise FileNotFoundError(
            f"{features_path} tidak ada.\n"
            "Jalankan dulu:\n"
            "  1. python src/data/ingest_data.py --hours 17520\n"
            "  2. python src/data/preprocess.py --all\n"
            "  3. python src/build_features.py"
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
        train_df  = df.iloc[:split_idx]
        test_df   = df.iloc[split_idx:]
        logger.info("TIME-BASED split: train=%d, test=%d", len(train_df), len(test_df))
    else:
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=42,
            stratify=df[TARGET_COLUMN],
        )

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN].astype(int)
    X_test  = test_df[FEATURE_COLUMNS]
    y_test  = test_df[TARGET_COLUMN].astype(int)

    logger.info("Distribusi train → pos=%d (%.1f%%), neg=%d (%.1f%%)",
                (y_train==1).sum(), 100*(y_train==1).mean(),
                (y_train==0).sum(), 100*(y_train==0).mean())
    return X_train, X_test, y_train, y_test


def apply_smotetomek(X_train, y_train):
    logger.info("Menerapkan SMOTETomek...")
    logger.info("  Sebelum: pos=%d, neg=%d", (y_train==1).sum(), (y_train==0).sum())
    smt      = SMOTETomek(random_state=42)
    X_r, y_r = smt.fit_resample(X_train, y_train)
    logger.info("  Sesudah : pos=%d, neg=%d", (y_r==1).sum(), (y_r==0).sum())
    return X_r, y_r


# 3 Metrik Inti
def compute_3_metrics(model, X_test, y_test,
                       threshold: float = DEFAULT_THRESHOLD) -> dict:
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    return {
        "pr_auc"  : float(average_precision_score(y_test, y_proba)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "recall"  : float(recall_score(y_test, y_pred, zero_division=0)),
    }


# Train Satu Run
def train_one_run(
    X_train, X_test, y_train, y_test,
    n_estimators: int       = 200,
    max_depth: int           = 4,
    learning_rate: float     = 0.05,
    subsample: float         = 0.8,
    colsample_bytree: float  = 0.8,
    min_child_weight: int    = 5,
    gamma: float             = 0.5,
    threshold: float         = DEFAULT_THRESHOLD,
    use_smotetomek: bool     = True,
    run_name: str            = "default",
    register_model: bool     = False,
):
    logger.info("\n" + "=" * 65)
    logger.info("RUN: %s", run_name)
    logger.info("=" * 65)

    X_tr, y_tr = apply_smotetomek(X_train, y_train) if use_smotetomek else (X_train, y_train)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # 1) Log Parameter
        params = {
            "model_type"      : "XGBoost",
            "n_estimators"    : n_estimators,
            "max_depth"       : max_depth,
            "learning_rate"   : learning_rate,
            "subsample"       : subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "gamma"           : gamma,
            "objective"       : "binary:logistic",
            "eval_metric"     : "aucpr",
            "threshold"       : threshold,
            "resampling"      : "SMOTETomek" if use_smotetomek else "none",
            "random_state"    : 42,
            "n_train_samples" : len(X_tr),
            "n_test_samples"  : len(X_test),
        }
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.set_tag("primary_metric", "pr_auc")

        # 2) Train
        model = xgb.XGBClassifier(
            n_estimators     = n_estimators,
            max_depth        = max_depth,
            learning_rate    = learning_rate,
            subsample        = subsample,
            colsample_bytree = colsample_bytree,
            min_child_weight = min_child_weight,
            gamma            = gamma,
            objective        = "binary:logistic",
            eval_metric      = "aucpr",
            random_state     = 42,
            verbosity        = 0,
        )
        model.fit(X_tr, y_tr)

        # 3) Log 3 Metrik Inti
        metrics = compute_3_metrics(model, X_test, y_test, threshold=threshold)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # 4) Classification Report sebagai Artifact
        y_proba  = model.predict_proba(X_test)[:, 1]
        y_pred   = (y_proba >= threshold).astype(int)
        report   = classification_report(
            y_test, y_pred,
            target_names=["Tidak Ada Susulan (0)", "Ada Susulan (1)"],
            zero_division=0,
        )
        rp_path = LOG_DIR / f"report_{run_name}.txt"
        rp_path.write_text(
            f"Run: {run_name}\nThreshold: {threshold}\n"
            f"Resampling: {'SMOTETomek' if use_smotetomek else 'None'}\n"
            f"{'='*50}\n\n{report}\n\n"
            f"pr_auc (primary): {metrics['pr_auc']:.4f}\n"
            f"f1_score        : {metrics['f1_score']:.4f}\n"
            f"recall (safety) : {metrics['recall']:.4f}\n"
        )
        mlflow.log_artifact(str(rp_path))

        # 5) Confusion Matrix sebagai Artifact
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (0,0,0,0)
        cm_path = LOG_DIR / f"cm_{run_name}.txt"
        cm_path.write_text(
            f"Confusion Matrix (run: {run_name})\n\n"
            f"               Predicted\n"
            f"                No   Yes\n"
            f"Actual No     {tn:>4}  {fp:>4}\n"
            f"       Yes    {fn:>4}  {tp:>4}\n\n"
            f"→ Missed aftershock (FN): {fn}  ← harus sekecil mungkin!\n"
        )
        mlflow.log_artifact(str(cm_path))

        # 6) Feature Importance sebagai Artifact
        importances = pd.Series(
            model.feature_importances_, index=FEATURE_COLUMNS
        ).sort_values(ascending=False)
        fi_text = "Feature Importance:\n\n"
        for feat, imp in importances.items():
            fi_text += f"  {feat:<25} {imp:.4f}\n"
        fi_path = LOG_DIR / f"feature_importance_{run_name}.txt"
        fi_path.write_text(fi_text)
        mlflow.log_artifact(str(fi_path))

        # 7) Log Model
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name="gempawas-aftershock-classifier" if register_model else None,
        )

        # Ringkasan
        logger.info("\n--- Hasil Run: %s ---", run_name)
        logger.info("  [1] pr_auc    : %.4f  ← PRIMARY", metrics["pr_auc"])
        logger.info("  [2] f1_score  : %.4f  ← BALANCE", metrics["f1_score"])
        logger.info("  [3] recall    : %.4f  ← CRITICAL (safety)", metrics["recall"])
        logger.info("  Threshold     : %.2f", threshold)
        logger.info("  MLflow Run ID : %s", run_id)
        if register_model:
            logger.info("  ✓ Model registered: 'gempawas-aftershock-classifier'")

        return run_id, metrics


# 3 Variasi Run
def run_variations(X_train, X_test, y_train, y_test):
    variations = [
        {   # Run 1: SMOTE ringan → baseline perbandingan
            "run_name"        : "run1_smote_light_n100_d3",
            "n_estimators"    : 100, "max_depth": 3, "learning_rate": 0.1,
            "subsample"       : 1.0, "colsample_bytree": 1.0,
            "min_child_weight": 1,   "gamma": 0.0,
            "threshold"       : 0.45, "use_smotetomek": True,
        },
        {   # Run 2: SMOTE + regularisasi → lebih kuat, lebih presisi
            "run_name"        : "run2_smote_tuned_n200_d4",
            "n_estimators"    : 200, "max_depth": 4, "learning_rate": 0.05,
            "subsample"       : 0.8, "colsample_bytree": 0.8,
            "min_child_weight": 5,   "gamma": 0.5,
            "threshold"       : 0.45, "use_smotetomek": True,
        },
        {   # Run 3: Run2 + threshold lebih rendah → recall lebih tinggi (safety-first)
            "run_name"        : "run3_smotetomek_thresh040",
            "n_estimators"    : 200, "max_depth": 4, "learning_rate": 0.05,
            "subsample"       : 0.8, "colsample_bytree": 0.8,
            "min_child_weight": 5,   "gamma": 0.5,
            "threshold"       : 0.40, "use_smotetomek": True,
        },
    ]
    results = []
    for v in variations:
        run_id, metrics = train_one_run(X_train, X_test, y_train, y_test, **v)
        results.append({"run_name": v["run_name"], "run_id": run_id, **metrics})
    return results


def print_comparison(results):
    logger.info("\n" + "=" * 70)
    logger.info("PERBANDINGAN 3 RUN — LK-06 (3 Metrik Inti)")
    logger.info("=" * 70)
    df = pd.DataFrame(results)
    cols = [c for c in ["run_name","pr_auc","f1_score","recall"] if c in df.columns]
    print("\n" + df[cols].to_string(index=False))

    best_idx = df["pr_auc"].idxmax()
    best = df.iloc[best_idx]
    logger.info("\n>>> Best Run (by PR-AUC): %s", best["run_name"])
    logger.info("    pr_auc=%.4f | f1=%.4f | recall=%.4f",
                best.get("pr_auc",0), best.get("f1_score",0), best.get("recall",0))


# Main
def main():
    parser = argparse.ArgumentParser(description="GempaWas LK-06 Training (Revised)")
    parser.add_argument("--variations",     action="store_true")
    parser.add_argument("--run-name",       default="manual_run")
    parser.add_argument("--n-estimators",   type=int,   default=200)
    parser.add_argument("--max-depth",      type=int,   default=4)
    parser.add_argument("--learning-rate",  type=float, default=0.05)
    parser.add_argument("--threshold",      type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--no-smotetomek",  action="store_true")
    parser.add_argument("--register-model", action="store_true")
    parser.add_argument("--test-size",      type=float, default=0.25)
    parser.add_argument("--features-path",  type=Path, default=FEATURES_PATH)
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger.info("Metrik inti     : pr_auc, f1_score, recall")
    logger.info("Resampling      : %s", "none" if args.no_smotetomek else "SMOTETomek")
    logger.info("Threshold       : %.2f", args.threshold)

    df = load_features(args.features_path)
    X_train, X_test, y_train, y_test = prepare_train_test(df, test_size=args.test_size)

    if args.variations:
        results = run_variations(X_train, X_test, y_train, y_test)
        print_comparison(results)
    else:
        train_one_run(
            X_train, X_test, y_train, y_test,
            n_estimators    = args.n_estimators,
            max_depth       = args.max_depth,
            learning_rate   = args.learning_rate,
            threshold       = args.threshold,
            use_smotetomek  = not args.no_smotetomek,
            run_name        = args.run_name,
            register_model  = args.register_model,
        )

    logger.info("\n✓ Selesai. Buka MLflow UI: mlflow ui --host 0.0.0.0 --port 5000")


if __name__ == "__main__":
    main()
