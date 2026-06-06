"""
Aturan promosi (domain gempa):
  1. GUARDRAIL KESELAMATAN — challenger tidak boleh menurunkan recall lebih
     dari `recall_tolerance` dibanding champion. Recall = kemampuan menangkap
     gempa susulan yang benar-benar terjadi.
  2. KRITERIA UTAMA — challenger.pr_auc >= champion.pr_auc + `min_improvement`.
     PR-AUC adalah metrik primary karena jujur untuk data imbalanced.
  3. PEMULIHAN KESELAMATAN (override) — jika champion sudah "membusuk"
     (recall < `recall_floor`) dan challenger punya recall lebih tinggi,
     challenger tetap dipromosikan meski PR-AUC tidak naik signifikan,
     karena model Production saat ini sudah tidak aman.

Jika tidak ada model lama (cold start), model baru otomatis dipromosikan.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import yaml
from mlflow.tracking import MlflowClient

PARAMS_PATH = Path("config/params.yaml")
DECISION_PATH = Path("logs/ct_promotion_decision.md")
HISTORY_PATH = Path("models/ct_promotion_history.json")


def load_promotion_params() -> dict:
    defaults = {
        "primary_metric": "pr_auc",
        "min_improvement": 0.02,
        "recall_tolerance": 0.02,
        "recall_floor": 0.40,
        "pr_auc_floor": 0.45,
    }
    if not PARAMS_PATH.exists():
        return defaults
    with open(PARAMS_PATH) as f:
        params = yaml.safe_load(f) or {}
    ct = params.get("continuous_training", {})
    promo = ct.get("promotion", {})
    perf = ct.get("performance", {})
    legacy = params.get("promotion", {})
    return {
        "primary_metric": promo.get("primary_metric", legacy.get("champion_metric", defaults["primary_metric"])),
        "min_improvement": promo.get("min_improvement", legacy.get("min_improvement", defaults["min_improvement"])),
        "recall_tolerance": promo.get("recall_tolerance", defaults["recall_tolerance"]),
        "recall_floor": promo.get("recall_floor", defaults["recall_floor"]),
        "pr_auc_floor": perf.get("pr_auc_min", defaults["pr_auc_floor"]),
    }


def get_run_metrics(client: MlflowClient, run_id: str) -> dict:
    m = client.get_run(run_id).data.metrics
    return {
        "pr_auc": m.get("pr_auc", 0.0),
        "f1_score": m.get("f1_score", 0.0),
        "recall": m.get("recall", 0.0),
    }


def find_champion(client: MlflowClient, model_name: str, fallback_file: Path | None = None):
    """Cari model Production saat ini (champion). Return (version, metrics) atau (None, None).

    Urutan pencarian:
      1. alias 'champion' di registry (MLflow 3.x)
      2. stage Production (MLflow 2.x)
      3. fallback: metrics dari models/model_registry_metadata.json (untuk CI tanpa
         MLflow server live — champion dipersistenkan via git).
    """
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except Exception:
        versions = []

    if versions:
        # 1) alias 'champion'
        try:
            mv = client.get_model_version_by_alias(model_name, "champion")
            return mv, get_run_metrics(client, mv.run_id)
        except Exception:
            pass
        # 2) stage Production
        prod = [v for v in versions if getattr(v, "current_stage", "") == "Production"]
        if prod:
            mv = sorted(prod, key=lambda v: int(v.version))[-1]
            return mv, get_run_metrics(client, mv.run_id)

    # 3) fallback file
    if fallback_file and fallback_file.exists():
        try:
            meta = json.loads(fallback_file.read_text(encoding="utf-8"))
            pm = meta.get("production_model") or meta.get("current_staging") or {}
            m = pm.get("metrics") or {}
            if m:
                class _Stub:
                    version = pm.get("version", "?")
                return _Stub(), {
                    "pr_auc": m.get("pr_auc", 0.0),
                    "f1_score": m.get("f1_score", 0.0),
                    "recall": m.get("recall", 0.0),
                }
        except Exception:
            pass

    return None, None


def decide(champion_m: dict | None, challenger_m: dict, p: dict) -> tuple[bool, list[str]]:
    reasons = []
    if champion_m is None:
        return True, ["Tidak ada champion (cold start) → challenger langsung dipromosikan."]

    d_prauc = challenger_m["pr_auc"] - champion_m["pr_auc"]
    d_recall = challenger_m["recall"] - champion_m["recall"]

    reasons.append(
        f"Δ PR-AUC = {d_prauc:+.4f} (challenger {challenger_m['pr_auc']:.4f} "
        f"vs champion {champion_m['pr_auc']:.4f}); butuh >= +{p['min_improvement']}"
    )
    reasons.append(
        f"Δ Recall = {d_recall:+.4f} (challenger {challenger_m['recall']:.4f} "
        f"vs champion {champion_m['recall']:.4f}); toleransi -{p['recall_tolerance']}"
    )

    # 3) Override pemulihan keselamatan
    if champion_m["recall"] < p["recall_floor"] and challenger_m["recall"] > champion_m["recall"]:
        reasons.append(
            f"⚠ OVERRIDE SAFETY: champion recall {champion_m['recall']:.4f} < floor "
            f"{p['recall_floor']} (model membusuk) & challenger recall lebih tinggi → PROMOTE."
        )
        return True, reasons

    # 3b) Champion degenerate — PR-AUC sudah jatuh di bawah lantai Production.
    if (champion_m["pr_auc"] < p["pr_auc_floor"]
            and d_prauc >= p["min_improvement"]
            and challenger_m["recall"] >= p["recall_floor"]):
        reasons.append(
            f"⚠ OVERRIDE DECAY: champion PR-AUC {champion_m['pr_auc']:.4f} < lantai "
            f"{p['pr_auc_floor']} (champion telah decay & recall-nya menyesatkan). "
            f"Challenger PR-AUC jauh lebih baik & recall {challenger_m['recall']:.4f} "
            f">= floor {p['recall_floor']} → PROMOTE."
        )
        return True, reasons

    # 1) Guardrail keselamatan
    if d_recall < -p["recall_tolerance"]:
        reasons.append("❌ Challenger menurunkan recall melebihi toleransi → TOLAK (tidak aman).")
        return False, reasons

    # 2) Kriteria utama
    if d_prauc >= p["min_improvement"]:
        reasons.append("✅ Peningkatan PR-AUC memenuhi ambang & recall aman → PROMOTE.")
        return True, reasons

    reasons.append("❌ Peningkatan PR-AUC belum cukup → TOLAK (pertahankan champion).")
    return False, reasons


def promote(client: MlflowClient, model_name: str, version: str):
    """Promosikan version menjadi champion/Production, arsipkan champion lama."""
    # Alias modern
    try:
        client.set_registered_model_alias(model_name, "champion", version)
        print(f"✓ alias 'champion' → v{version}")
    except Exception as exc:
        print(f"⚠ set alias champion gagal: {exc}")
    # Stage transition (MLflow 2.x); diabaikan jika tidak didukung
    try:
        client.transition_model_version_stage(
            name=model_name, version=version, stage="Production",
            archive_existing_versions=True,
        )
        print(f"✓ stage v{version} → Production (versi lama diarsipkan)")
    except Exception as exc:
        print(f"ℹ transition_model_version_stage tidak tersedia (MLflow 3.x): {exc}")


def update_production_metadata(meta_file: Path, model_name: str, version,
                               run_id: str, metrics: dict, trigger: str):
    """Perbarui models/model_registry_metadata.json: production_model = challenger baru."""
    meta_file.parent.mkdir(exist_ok=True, parents=True)
    base = {}
    if meta_file.exists():
        try:
            base = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            base = {}
    base["production_model"] = {
        "model_name": model_name,
        "version": version,
        "stage": "Production",
        "run_id": run_id,
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "promoted_by_trigger": trigger,
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID", "local"),
    }
    base.setdefault("model_name", model_name)
    base["primary_metric"] = "pr_auc"
    meta_file.write_text(json.dumps(base, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✓ models/model_registry_metadata.json diperbarui → champion v{version}")


def write_history(entry: dict):
    HISTORY_PATH.parent.mkdir(exist_ok=True, parents=True)
    history = []
    if HISTORY_PATH.exists():
        try:
            history = json.loads(HISTORY_PATH.read_text(encoding="utf-8")).get("history", [])
        except Exception:
            pass
    history.append(entry)
    HISTORY_PATH.write_text(
        json.dumps({"latest": entry, "history": history[-50:]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_decision_md(promoted: bool, champion_m, challenger_m, reasons, p,
                      challenger_version):
    DECISION_PATH.parent.mkdir(exist_ok=True, parents=True)
    badge = "🟢 CHALLENGER DIPROMOSIKAN KE PRODUCTION" if promoted else "🔴 CHALLENGER DITOLAK — CHAMPION DIPERTAHANKAN"
    lines = [
        "# Keputusan Evaluasi Komparatif Champion vs Challenger (LK-12)",
        "",
        f"**Hasil:** {badge}",
        f"_Waktu: {datetime.now(timezone.utc).isoformat()}_",
        "",
        "## Perbandingan Metrik",
        "",
        "| Metrik | Champion (lama) | Challenger (baru) | Δ |",
        "|--------|----------------|-------------------|---|",
    ]
    ch = champion_m or {"pr_auc": None, "f1_score": None, "recall": None}
    for k in ("pr_auc", "f1_score", "recall"):
        cv = ch[k]
        nv = challenger_m[k]
        if cv is None:
            lines.append(f"| {k} | — (cold start) | {nv:.4f} | — |")
        else:
            lines.append(f"| {k} | {cv:.4f} | {nv:.4f} | {nv-cv:+.4f} |")
    lines += [
        "",
        f"Challenger version: **v{challenger_version}**",
        "",
        "## Aturan Promosi",
        f"- Kriteria utama : Δ PR-AUC >= +{p['min_improvement']}",
        f"- Guardrail      : recall tidak turun > {p['recall_tolerance']}",
        f"- Override safety : champion recall < {p['recall_floor']} & challenger recall lebih tinggi",
        "",
        "## Penalaran Keputusan",
        "",
    ]
    lines += [f"- {r}" for r in reasons]
    DECISION_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_github_output(promoted: bool, version: str, reasons: list[str]):
    gh = os.environ.get("GITHUB_OUTPUT")
    if not gh:
        return
    with open(gh, "a") as f:
        f.write(f"promoted={'true' if promoted else 'false'}\n")
        f.write(f"challenger_version={version}\n")
        f.write(f"reason={reasons[-1] if reasons else ''}\n")


def main():
    ap = argparse.ArgumentParser(description="Champion vs Challenger evaluator")
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--challenger-run-id", required=True)
    ap.add_argument("--tracking-uri", default="file:./mlruns")
    ap.add_argument("--champion-metrics-file", type=Path,
                    default=Path("models/model_registry_metadata.json"),
                    help="Fallback metrics champion bila registry tidak punya champion live")
    ap.add_argument("--register", action="store_true",
                    help="Register challenger run sebagai versi baru sebelum membandingkan")
    args = ap.parse_args()

    p = load_promotion_params()
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient(tracking_uri=args.tracking_uri)

    print("=" * 64)
    print("  EVALUASI KOMPARATIF — CHAMPION vs CHALLENGER")
    print("=" * 64)
    print(f"  Model      : {args.model_name}")
    print(f"  Primary    : {p['primary_metric']}  (min_improvement=+{p['min_improvement']})")
    print(f"  Recall floor: {p['recall_floor']}  tolerance={p['recall_tolerance']}\n")

    # Register challenger jika diminta
    challenger_version = None
    if args.register:
        mv = client.create_model_version(
            name=args.model_name,
            source=f"runs:/{args.challenger_run_id}/model",
            run_id=args.challenger_run_id,
        )
        challenger_version = mv.version
        print(f"✓ Challenger di-register sebagai v{challenger_version}")
    else:
        # cari versi yang run_id-nya cocok
        for v in client.search_model_versions(f"name='{args.model_name}'"):
            if v.run_id == args.challenger_run_id:
                challenger_version = v.version
                break

    challenger_m = get_run_metrics(client, args.challenger_run_id)
    champion_mv, champion_m = find_champion(client, args.model_name, args.champion_metrics_file)

    print("  Champion  :", "TIDAK ADA" if champion_m is None
          else f"v{champion_mv.version}  pr_auc={champion_m['pr_auc']:.4f} recall={champion_m['recall']:.4f}")
    print(f"  Challenger:  v{challenger_version}  pr_auc={challenger_m['pr_auc']:.4f} recall={challenger_m['recall']:.4f}\n")

    promoted, reasons = decide(champion_m, challenger_m, p)
    for r in reasons:
        print("   ", r)

    print("\n" + "=" * 64)
    if promoted and challenger_version is not None:
        promote(client, args.model_name, challenger_version)
        update_production_metadata(
            args.champion_metrics_file, args.model_name, challenger_version,
            args.challenger_run_id, challenger_m,
            os.environ.get("CT_TRIGGER", "manual"),
        )
        print(f"  ✅ v{challenger_version} dipromosikan ke Production")
    else:
        print("  🛑 Challenger tidak dipromosikan — champion dipertahankan")
    print("=" * 64)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "promoted": promoted,
        "challenger_version": challenger_version,
        "challenger_run_id": args.challenger_run_id,
        "champion_version": None if champion_mv is None else champion_mv.version,
        "champion_metrics": champion_m,
        "challenger_metrics": challenger_m,
        "trigger": os.environ.get("CT_TRIGGER", "manual"),
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID", "local"),
    }
    write_history(entry)
    write_decision_md(promoted, champion_m, challenger_m, reasons, p, challenger_version)
    write_github_output(promoted, str(challenger_version), reasons)

    sys.exit(0)


if __name__ == "__main__":
    main()
