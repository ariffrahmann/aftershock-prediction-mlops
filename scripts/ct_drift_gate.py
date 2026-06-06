from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Fitur yang dipantau drift-nya
DEFAULT_MONITORED = [
    "mainshock_magnitude",
    "mainshock_depth",
    "jam_sejak_mainshock",
    "count_susulan_6jam",
    "count_susulan_24jam",
    "max_mag_susulan_24jam",
    "omori_rate_est",
]
PARAMS_PATH = Path("config/params.yaml")
REPORT_PATH = Path("logs/ct_drift_gate_report.md")


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    PSI < 0.10            : stabil, tidak ada drift signifikan
    0.10 <= PSI < 0.25    : drift moderate, pantau
    PSI >= 0.25           : drift signifikan, retraining direkomendasikan
    """
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    if len(reference) < 10 or len(current) < 5:
        return 0.0

    # Bin berdasarkan kuantil distribusi reference
    breakpoints = np.quantile(reference, np.linspace(0, 1, bins + 1))
    breakpoints = np.unique(breakpoints)          # buang duplikat (fitur low-variance)
    if len(breakpoints) < 3:
        return 0.0
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_hist, _ = np.histogram(reference, bins=breakpoints)
    cur_hist, _ = np.histogram(current, bins=breakpoints)

    ref_pct = ref_hist / max(ref_hist.sum(), 1)
    cur_pct = cur_hist / max(cur_hist.sum(), 1)

    # Smoothing agar tidak log(0)
    ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)
    cur_pct = np.where(cur_pct == 0, 1e-6, cur_pct)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def severity(psi_value: float, warn: float, critical: float) -> str:
    if psi_value >= critical:
        return "CRITICAL"
    if psi_value >= warn:
        return "WARNING"
    return "STABLE"


def load_ct_params() -> dict:
    if not PARAMS_PATH.exists():
        return {}
    with open(PARAMS_PATH) as f:
        params = yaml.safe_load(f) or {}
    return params.get("continuous_training", {}).get("data_drift", {})


def write_github_output(should_retrain: bool, max_psi: float,
                        mean_psi: float, n_critical: int):
    gh = os.environ.get("GITHUB_OUTPUT")
    if not gh:
        return
    with open(gh, "a") as f:
        f.write(f"should_retrain={'true' if should_retrain else 'false'}\n")
        f.write(f"max_psi={max_psi:.4f}\n")
        f.write(f"mean_psi={mean_psi:.4f}\n")
        f.write(f"n_critical={n_critical}\n")


def write_report(rows: list[dict], should_retrain: bool, warn: float,
                 critical: float, max_psi: float, mean_psi: float):
    REPORT_PATH.parent.mkdir(exist_ok=True, parents=True)
    decision = "RETRAINING DIPICU" if should_retrain else "TIDAK ADA DRIFT — SKIP"
    lines = [
        "# Laporan Drift Gate — Continuous Training (LK-12)",
        "",
        f"**Keputusan:** {decision}",
        "",
        f"- Threshold WARNING  : PSI >= {warn}",
        f"- Threshold CRITICAL : PSI >= {critical}",
        f"- Max PSI            : {max_psi:.4f}",
        f"- Mean PSI           : {mean_psi:.4f}",
        "",
        "| Fitur | PSI | Severity |",
        "|-------|-----|----------|",
    ]
    for r in rows:
        lines.append(f"| {r['feature']} | {r['psi']:.4f} | {r['severity']} |")
    lines += [
        "",
        "## Interpretasi",
        "",
        "PSI mengukur seberapa jauh distribusi fitur data terbaru bergeser "
        "dari distribusi data acuan (data saat model Production dilatih). "
        "Pergeseran besar menandakan model berisiko mengalami *decay* karena "
        "melihat pola input yang berbeda dari saat pelatihan.",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="CT Drift Gate (feature-level PSI)")
    p.add_argument("--reference", type=Path, required=True,
                   help="Parquet/CSV data acuan (saat model Production dilatih)")
    p.add_argument("--current", type=Path, required=True,
                   help="Parquet/CSV data terbaru yang akan dicek drift-nya")
    p.add_argument("--psi-warn", type=float, default=None)
    p.add_argument("--psi-critical", type=float, default=None)
    p.add_argument("--features", nargs="*", default=None)
    args = p.parse_args()

    ct = load_ct_params()
    warn = args.psi_warn if args.psi_warn is not None else ct.get("psi_warn", 0.10)
    critical = args.psi_critical if args.psi_critical is not None else ct.get("psi_critical", 0.25)
    monitored = args.features or ct.get("monitored_features") or DEFAULT_MONITORED

    def _read(path: Path) -> pd.DataFrame:
        return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)

    ref_df = _read(args.reference)
    cur_df = _read(args.current)

    print("=" * 64)
    print("  CT DRIFT GATE — Population Stability Index per Fitur")
    print("=" * 64)
    print(f"  Reference : {args.reference}  ({len(ref_df)} baris)")
    print(f"  Current   : {args.current}  ({len(cur_df)} baris)")
    print(f"  Threshold : WARN>={warn}  CRITICAL>={critical}\n")
    print(f"  {'Fitur':<24} {'PSI':<10} {'Severity':<10}")
    print(f"  {'-'*24} {'-'*10} {'-'*10}")

    rows, psi_values, n_critical = [], [], 0
    for feat in monitored:
        if feat not in ref_df.columns or feat not in cur_df.columns:
            continue
        val = psi(ref_df[feat].to_numpy(dtype=float),
                  cur_df[feat].to_numpy(dtype=float))
        sev = severity(val, warn, critical)
        if sev == "CRITICAL":
            n_critical += 1
        flag = "🔴" if sev == "CRITICAL" else ("🟡" if sev == "WARNING" else "🟢")
        print(f"  {feat:<24} {val:<10.4f} {flag} {sev}")
        rows.append({"feature": feat, "psi": val, "severity": sev})
        psi_values.append(val)

    max_psi = max(psi_values) if psi_values else 0.0
    mean_psi = float(np.mean(psi_values)) if psi_values else 0.0

    # Aturan pemicu: minimal satu fitur CRITICAL, ATAU rata-rata PSI >= warn
    should_retrain = (n_critical >= 1) or (mean_psi >= warn)

    print(f"\n  Max PSI  : {max_psi:.4f}")
    print(f"  Mean PSI : {mean_psi:.4f}")
    print(f"  Fitur CRITICAL: {n_critical}")
    print("=" * 64)
    if should_retrain:
        print("  🔴 DRIFT SIGNIFIKAN — Continuous Training DIPICU")
    else:
        print("  🟢 Distribusi stabil — retraining tidak diperlukan")
    print("=" * 64)

    write_report(rows, should_retrain, warn, critical, max_psi, mean_psi)
    write_github_output(should_retrain, max_psi, mean_psi, n_critical)
    print(json.dumps({
        "should_retrain": should_retrain,
        "max_psi": round(max_psi, 4),
        "mean_psi": round(mean_psi, 4),
        "n_critical": n_critical,
    }))

    # exit 0 selalu (gate adalah informasi, bukan kegagalan build)
    sys.exit(0)


if __name__ == "__main__":
    main()
