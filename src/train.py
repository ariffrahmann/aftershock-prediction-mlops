"""
GempaWas — Model Training Module

Latih dua model:
1. Random Forest sebagai baseline (mudah dijelaskan)
2. XGBoost sebagai model utama (performa lebih baik untuk tabular)

Semua eksperimen di-track ke MLflow.
"""

import argparse
import logging
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

logger = logging.getLogger(__name__)

FEATURES = [
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
LABEL = "label_susulan_besar_24jam"


def time_based_split(df: pd.DataFrame, test_year: int = 2024):
    """
    Split time-based: training pakai data sebelum test_year,
    test pakai data dari test_year onwards.

    Time-based split penting karena random split bisa bocorkan info masa depan
    ke training set (data leakage).
    """
    # Asumsi: mainshock_id berisi prefix tahun atau kita sort by snapshot
    # Untuk implementasi nyata, gunakan kolom time_utc dari mainshock
    # Untuk skeleton ini, kita pakai split sederhana 80/20 berdasarkan urutan
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")
    return train_df, test_df


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Hitung metrics dan log ke MLflow."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "f1": f1_score(y_test, y_pred),
        "pr_auc": average_precision_score(y_test, y_proba),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    logger.info(f"\n=== {model_name} ===")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")
        mlflow.log_metric(k, v)

    logger.info(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

    return metrics


def train_random_forest(X_train, y_train, X_test, y_test):
    """Baseline Random Forest dengan class balancing."""
    with mlflow.start_run(run_name="random_forest_baseline", nested=True):
        params = {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 5,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        }
        mlflow.log_params(params)

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test, "Random Forest")
        mlflow.sklearn.log_model(model, "model")

        # Feature importance
        importance = pd.DataFrame({
            "feature": FEATURES,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        logger.info(f"\nFeature importance:\n{importance}")

        return model, metrics


def train_xgboost(X_train, y_train, X_test, y_test, tune: bool = True):
    """XGBoost dengan optional hyperparameter tuning."""
    with mlflow.start_run(run_name="xgboost_main", nested=True):
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        logger.info(f"scale_pos_weight = {scale_pos_weight:.2f}")

        if tune:
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.05, 0.1],
            }
            base_model = xgb.XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                eval_metric="aucpr",
                random_state=42,
                n_jobs=-1,
            )
            grid = GridSearchCV(
                base_model, param_grid, cv=3, scoring="average_precision",
                n_jobs=-1, verbose=1,
            )
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
            logger.info(f"Best params: {grid.best_params_}")
            mlflow.log_params(grid.best_params_)
        else:
            params = {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                "scale_pos_weight": scale_pos_weight,
                "eval_metric": "aucpr",
                "random_state": 42,
                "n_jobs": -1,
            }
            mlflow.log_params(params)
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test, "XGBoost")
        mlflow.xgboost.log_model(model, "model")

        return model, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", type=Path,
                        default=Path("data/features.parquet"))
    parser.add_argument("--no-tune", action="store_true",
                        help="Skip hyperparameter tuning (faster for dev)")
    parser.add_argument("--register-model", action="store_true",
                        help="Register best model to MLflow Model Registry")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # Load data
    df = pd.read_parquet(args.features_path)
    logger.info(f"Loaded {len(df)} rows from {args.features_path}")

    # Split
    train_df, test_df = time_based_split(df)
    X_train, y_train = train_df[FEATURES], train_df[LABEL]
    X_test, y_test = test_df[FEATURES], test_df[LABEL]

    # MLflow setup
    mlflow.set_experiment("gempawas-training")

    with mlflow.start_run(run_name="training_session"):
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))
        mlflow.log_param("class_ratio_train",
                         float(y_train.mean()))

        # Train models
        rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
        xgb_model, xgb_metrics = train_xgboost(
            X_train, y_train, X_test, y_test, tune=not args.no_tune
        )

        # Pilih model terbaik berdasarkan PR-AUC
        if xgb_metrics["pr_auc"] > rf_metrics["pr_auc"]:
            best_model_name = "XGBoost"
            best_metrics = xgb_metrics
        else:
            best_model_name = "Random Forest"
            best_metrics = rf_metrics

        mlflow.set_tag("best_model", best_model_name)
        logger.info(f"\nBest model: {best_model_name} (PR-AUC = {best_metrics['pr_auc']:.4f})")

        if args.register_model:
            mlflow.register_model(
                model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
                name="gempawas-production",
            )
            logger.info("Model registered to MLflow Model Registry")


if __name__ == "__main__":
    main()
