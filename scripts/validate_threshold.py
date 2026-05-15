import argparse
import os
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "gempawas-aftershock-prediction"
TRACKING_URI = "file:./mlruns"


def find_best_run(client: MlflowClient, experiment_name: str):
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        print(f"❌ Experiment '{experiment_name}' tidak ditemukan di {TRACKING_URI}")
        sys.exit(1)

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.pr_auc DESC"],
        max_results=10,
    )

    if not runs:
        print(f"❌ Tidak ada run di experiment '{experiment_name}'")
        sys.exit(1)

    print(f"\n📋 Top {min(len(runs), 5)} runs (sorted by PR-AUC):\n")
    print(f"  {'Run Name':<35} {'PR-AUC':<10} {'F1':<10} {'Recall':<10}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10}")
    for r in runs[:5]:
        name = r.data.tags.get("mlflow.runName", r.info.run_id[:8])[:33]
        pr_auc = r.data.metrics.get("pr_auc", 0)
        f1 = r.data.metrics.get("f1_score", 0)
        recall = r.data.metrics.get("recall", 0)
        print(f"  {name:<35} {pr_auc:<10.4f} {f1:<10.4f} {recall:<10.4f}")

    return runs[0]


def main():
    parser = argparse.ArgumentParser(description="Validate best model vs LK-01 threshold")
    parser.add_argument("--pr-auc-min", type=float, default=0.45)
    parser.add_argument("--f1-min", type=float, default=0.35)
    parser.add_argument("--recall-min", type=float, default=0.40)
    args = parser.parse_args()

    print("=" * 60)
    print("  STAGE 3 — MODEL EVALUATION vs LK-01 THRESHOLD")
    print("=" * 60)
    print(f"\nThreshold dari LK-01:")
    print(f"  PR-AUC ≥ {args.pr_auc_min}")
    print(f"  F1     ≥ {args.f1_min}")
    print(f"  Recall ≥ {args.recall_min}")

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    best = find_best_run(client, EXPERIMENT_NAME)
    metrics = best.data.metrics
    run_id = best.info.run_id
    run_name = best.data.tags.get("mlflow.runName", run_id[:8])

    pr_auc = metrics.get("pr_auc", 0)
    f1 = metrics.get("f1_score", 0)
    recall = metrics.get("recall", 0)

    print(f"\n🏆 Best Run: {run_name}")
    print(f"   Run ID  : {run_id}")
    print(f"   PR-AUC  : {pr_auc:.4f}   (threshold: {args.pr_auc_min})")
    print(f"   F1      : {f1:.4f}   (threshold: {args.f1_min})")
    print(f"   Recall  : {recall:.4f}   (threshold: {args.recall_min})")

    pr_auc_pass = pr_auc >= args.pr_auc_min
    f1_pass = f1 >= args.f1_min
    recall_pass = recall >= args.recall_min

    print("\n📊 Hasil validasi per metric:")
    print(f"   PR-AUC : {'✅ PASS' if pr_auc_pass else '❌ FAIL'}")
    print(f"   F1     : {'✅ PASS' if f1_pass else '❌ FAIL'}")
    print(f"   Recall : {'✅ PASS' if recall_pass else '❌ FAIL'}")

    n_pass = sum([pr_auc_pass, f1_pass, recall_pass])
    overall_pass = pr_auc_pass and n_pass >= 2

    print(f"\n{'='*60}")
    if overall_pass:
        print("✅ VALIDATION PASSED — model layak di-promote ke Staging")
    else:
        print("❌ VALIDATION FAILED — model belum memenuhi standar LK-01")
    print(f"{'='*60}\n")

    # Tulis output untuk GitHub Actions (env file format)
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"passed={'true' if overall_pass else 'false'}\n")
            f.write(f"best_run_id={run_id}\n")
            f.write(f"pr_auc={pr_auc:.4f}\n")
            f.write(f"f1={f1:.4f}\n")
            f.write(f"recall={recall:.4f}\n")
            f.write(f"run_name={run_name}\n")
        print(f"📤 Output disimpan ke GITHUB_OUTPUT")
    else:
        # Local run (debug)
        print(f"passed={overall_pass}")
        print(f"best_run_id={run_id}")
        print(f"pr_auc={pr_auc:.4f}")

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()