from __future__ import annotations

import sys
import time
from pathlib import Path

import mlflow
import yaml
from mlflow import MlflowClient

# Konfigurasi
MLFLOW_TRACKING_URI    = "file:./mlruns"
MLFLOW_EXPERIMENT_NAME = "gempawas-aftershock-prediction"
PARAMS_PATH            = Path("config/params.yaml")
REPORT_PATH            = Path("logs/ci_evaluation_report.md")


def load_thresholds() -> dict:
    with open(PARAMS_PATH) as f:
        params = yaml.safe_load(f)
    thresholds = params.get("ci_thresholds", {})
    return {
        "pr_auc_min" : thresholds.get("pr_auc_min",  0.35),
        "f1_min"     : thresholds.get("f1_min",       0.30),
        "recall_min" : thresholds.get("recall_min",   0.40),
        "model_name" : thresholds.get("model_name",   "gempawas-aftershock-classifier"),
        "stage"      : thresholds.get("auto_register_stage", "Staging"),
    }


def get_best_run(client: MlflowClient) -> tuple[str, dict]:
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(
            f"Experiment '{MLFLOW_EXPERIMENT_NAME}' tidak ditemukan. "
            "Pastikan train.py sudah dijalankan terlebih dahulu."
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["metrics.pr_auc DESC"],    # urutkan dari PR-AUC tertinggi
        max_results=10,
    )

    if not runs:
        raise RuntimeError("Tidak ada run yang selesai ditemukan di MLflow.")

    # Ambil run dengan pr_auc tertinggi
    best_run = runs[0]
    metrics  = {
        "pr_auc"  : best_run.data.metrics.get("pr_auc",   0.0),
        "f1_score": best_run.data.metrics.get("f1_score", 0.0),
        "recall"  : best_run.data.metrics.get("recall",   0.0),
    }
    return best_run.info.run_id, metrics, best_run.data.params


def evaluate_against_thresholds(metrics: dict, thresholds: dict) -> tuple[bool, list]:
    checks = [
        ("pr_auc",   metrics["pr_auc"],   thresholds["pr_auc_min"],  ">="),
        ("f1_score", metrics["f1_score"], thresholds["f1_min"],      ">="),
        ("recall",   metrics["recall"],   thresholds["recall_min"],  ">="),
    ]
    details = []
    all_passed = True
    for name, actual, threshold, op in checks:
        passed = actual >= threshold
        status = "✅ LOLOS" if passed else "❌ GAGAL"
        details.append(
            f"  {status} | {name:<10} = {actual:.4f}  (threshold {op} {threshold})"
        )
        if not passed:
            all_passed = False

    return all_passed, details


def register_model_to_staging(client: MlflowClient, run_id: str,
                                model_name: str, stage: str) -> str:

    model_uri = f"runs:/{run_id}/model"
    print(f"  Mendaftarkan model dari run {run_id[:8]}...")

    # Register model (buat versi baru jika sudah ada)
    model_details = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )
    version = model_details.version

    # Tunggu hingga model siap (state = READY)
    for _ in range(10):
        mv = client.get_model_version(model_name, version)
        if mv.status == "READY":
            break
        time.sleep(1)

    # Transisi ke Staging
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=True,   # arsipkan versi Staging sebelumnya
    )
    return version


def write_report(passed: bool, metrics: dict, thresholds: dict,
                  details: list, run_id: str, version: str | None):
    """
    Tulis laporan evaluasi sebagai Markdown.
    """
    REPORT_PATH.parent.mkdir(exist_ok=True)
    status_badge = "✅ LOLOS — Model Registered to Staging" if passed else "❌ GAGAL — Model NOT Registered"

    lines = [
        f"# Laporan Evaluasi Model CI/CD — LK-08",
        f"",
        f"**Status:** {status_badge}",
        f"",
        f"## Metrik Model (Run ID: `{run_id[:12]}...`)",
        f"",
        f"| Metrik | Nilai | Threshold | Status |",
        f"|--------|-------|-----------|--------|",
        f"| PR-AUC (primary) | {metrics['pr_auc']:.4f} | >= {thresholds['pr_auc_min']} | {'✅' if metrics['pr_auc'] >= thresholds['pr_auc_min'] else '❌'} |",
        f"| F1-Score | {metrics['f1_score']:.4f} | >= {thresholds['f1_min']} | {'✅' if metrics['f1_score'] >= thresholds['f1_min'] else '❌'} |",
        f"| Recall (safety) | {metrics['recall']:.4f} | >= {thresholds['recall_min']} | {'✅' if metrics['recall'] >= thresholds['recall_min'] else '❌'} |",
        f"",
        f"## Detail Validasi",
        f"",
    ]
    lines.extend(f"{d}" for d in details)
    lines += [""]

    if passed and version:
        lines += [
            f"## Auto-Registry",
            f"",
            f"Model berhasil didaftarkan ke MLflow Model Registry:",
            f"- **Nama Model:** `{thresholds['model_name']}`",
            f"- **Versi:** `v{version}`",
            f"- **Stage:** `{thresholds['stage']}`",
            f"",
        ]
    elif not passed:
        lines += [
            f"## Kenapa Gagal?",
            f"",
            f"Model baru tidak memenuhi ambang batas minimum yang ditetapkan di",
            f"`config/params.yaml`. Model TIDAK didaftarkan ke registry.",
            f"",
            f"**Langkah selanjutnya:** Perbaiki model (hyperparameter, fitur, data)",
            f"hingga semua metrik melampaui threshold.",
            f"",
        ]

    lines += [
        f"## Tentang Threshold",
        f"",
        f"Threshold didefinisikan di `config/params.yaml` bagian `ci_thresholds`.",
        f"Ubah nilai di file tersebut untuk menyesuaikan standar kualitas.",
    ]

    REPORT_PATH.write_text("\n".join(lines))
    print(f"  Laporan disimpan: {REPORT_PATH}")


def main():
    print("=" * 60)
    print("STAGE 3 — Evaluasi Model Otomatis")
    print("=" * 60)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # 1. Baca threshold
    thresholds = load_thresholds()
    print(f"\n  Threshold dari params.yaml:")
    print(f"    pr_auc  >= {thresholds['pr_auc_min']}")
    print(f"    f1      >= {thresholds['f1_min']}")
    print(f"    recall  >= {thresholds['recall_min']}")

    # 2. Cari run terbaik
    print(f"\n  Mencari run terbaik di experiment '{MLFLOW_EXPERIMENT_NAME}'...")
    run_id, metrics, params = get_best_run(client)
    print(f"  Run ID    : {run_id[:12]}...")
    print(f"  pr_auc    : {metrics['pr_auc']:.4f}")
    print(f"  f1_score  : {metrics['f1_score']:.4f}")
    print(f"  recall    : {metrics['recall']:.4f}")

    # 3. Evaluasi
    print(f"\n{'='*60}")
    print(f"STAGE 3 — Hasil Validasi")
    print(f"{'='*60}")
    passed, details = evaluate_against_thresholds(metrics, thresholds)
    for d in details:
        print(d)

    version = None
    if passed:
        print(f"\n{'='*60}")
        print(f"STAGE 4 — Auto-Registry ke MLflow Staging")
        print(f"{'='*60}")
        version = register_model_to_staging(
            client, run_id,
            thresholds["model_name"],
            thresholds["stage"],
        )
        print(f"  ✅ Model v{version} berhasil di-register ke '{thresholds['stage']}'!")
    else:
        print(f"\n  ❌ Model TIDAK di-register karena tidak memenuhi threshold.")

    # 4. Tulis laporan
    write_report(passed, metrics, thresholds, details, run_id, version)

    print(f"\n{'='*60}")
    if passed:
        print(f"✅ PIPELINE SUKSES — Model v{version} siap di Staging")
        sys.exit(0)   # GitHub Actions: job sukses
    else:
        print(f"❌ PIPELINE GAGAL — Perbaiki model sebelum merge ke main")
        sys.exit(1)   # GitHub Actions: job gagal, pipeline berhenti


if __name__ == "__main__":
    main()
