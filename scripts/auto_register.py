import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = "file:./mlruns"
METADATA_PATH = Path("models/model_registry_metadata.json")


def get_git_info():
    """Ambil commit SHA + branch sebagai metadata."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        sha = os.environ.get("GITHUB_SHA", "unknown")
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        branch = os.environ.get("GITHUB_REF_NAME", "unknown")
    return sha[:12], branch


def register_model(client: MlflowClient, run_id: str, model_name: str) -> int:
    """Register model dari run ke Model Registry, return version number."""
    model_uri = f"runs:/{run_id}/model"

    # Buat registered model
    try:
        client.create_registered_model(model_name)
        print(f"✓ Created registered model: {model_name}")
    except Exception as exc:
        if "already exists" in str(exc).lower():
            print(f"ℹ Registered model '{model_name}' sudah ada")
        else:
            print(f"⚠ create_registered_model warning: {exc}")

    # Register version baru
    mv = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=run_id,
    )

    # Tunggu state aktif
    for _ in range(10):
        latest = client.get_model_version(model_name, mv.version)
        if latest.status == "READY":
            break
        time.sleep(1)

    print(f"✓ Model version {mv.version} berhasil di-register")
    return int(mv.version)


def transition_to_stage(client: MlflowClient, model_name: str, version: int, stage: str):
    """Transisi stage. Pakai alias untuk MLflow versi baru, fallback ke stage."""
    try:
        client.set_registered_model_alias(model_name, alias=stage.lower(), version=version)
        print(f"✓ Alias '{stage.lower()}' di-set ke version {version}")
    except Exception as exc:
        print(f"⚠ set alias gagal ({exc}), fallback ke transition_model_version_stage")

    try:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
        )
        print(f"✓ Stage transition: version {version} → '{stage}'")
    except Exception as exc:
        print(f"ℹ transition_model_version_stage tidak available (MLflow 3.x): {exc}")


def write_metadata(model_name: str, version: int, run_id: str, stage: str,
                    pr_auc: float):
    """Tulis metadata registry ke JSON untuk lineage tracking."""
    METADATA_PATH.parent.mkdir(exist_ok=True, parents=True)

    git_sha, git_branch = get_git_info()

    # Load metadata existing kalau ada
    history = []
    if METADATA_PATH.exists():
        try:
            existing = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            history = existing.get("history", [])
        except Exception:
            pass

    entry = {
        "model_name": model_name,
        "version": version,
        "stage": stage,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "mlflow_run_id": run_id,
        "git_commit": git_sha,
        "git_branch": git_branch,
        "metrics": {"pr_auc": pr_auc},
        "promoted_by": os.environ.get("GITHUB_ACTOR", "manual"),
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID", "local"),
    }

    history.append(entry)
    # Keep last 20 entries
    history = history[-20:]

    metadata = {
        "current_staging": entry,
        "history": history,
    }

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"✓ Metadata registry disimpan ke {METADATA_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, help="MLflow run ID")
    parser.add_argument("--model-name", required=True, help="Nama registered model")
    parser.add_argument("--stage", default="Staging", choices=["Staging", "Production"])
    parser.add_argument("--pr-auc", type=float, default=0.0)
    args = parser.parse_args()

    print("=" * 60)
    print("  STAGE 4 — AUTO-REGISTER ke MLFLOW REGISTRY")
    print("=" * 60)
    print(f"\nModel name : {args.model_name}")
    print(f"Run ID     : {args.run_id}")
    print(f"Stage      : {args.stage}")
    print(f"PR-AUC     : {args.pr_auc}")

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    #  Register model
    try:
        version = register_model(client, args.run_id, args.model_name)
    except Exception as exc:
        print(f"❌ Register gagal: {exc}")
        sys.exit(1)

    #  Transition ke Staging
    transition_to_stage(client, args.model_name, version, args.stage)

    # Tulis metadata untuk lineage
    write_metadata(args.model_name, version, args.run_id, args.stage, args.pr_auc)

    print("\n" + "=" * 60)
    print(f"✅ {args.model_name} v{version} berhasil di-promote ke {args.stage}")
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()