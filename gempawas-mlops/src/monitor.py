"""
GempaWas — Drift Monitoring Module

Deteksi data drift menggunakan PSI (Population Stability Index)
dan Evidently AI. Memicu CT (Continuous Training) jika drift terdeteksi.
"""

import argparse
import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    Hitung Population Stability Index antara reference dan current distribution.

    Interpretasi:
        PSI < 0.1  : tidak ada perubahan signifikan
        0.1 ≤ PSI < 0.25 : perubahan moderate, monitor
        PSI ≥ 0.25 : perubahan signifikan, retraining direkomendasikan
    """
    breakpoints = np.quantile(reference, np.linspace(0, 1, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_hist, _ = np.histogram(reference, bins=breakpoints)
    cur_hist, _ = np.histogram(current, bins=breakpoints)

    ref_pct = ref_hist / max(ref_hist.sum(), 1)
    cur_pct = cur_hist / max(cur_hist.sum(), 1)

    # Smoothing untuk hindari log(0)
    ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)
    cur_pct = np.where(cur_pct == 0, 1e-6, cur_pct)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def quick_check(db_path: Path = Path("data/gempa.db")) -> dict:
    """
    Quick drift check — bandingkan distribusi magnitude
    7 hari terakhir vs 30 hari sebelumnya.
    """
    conn = sqlite3.connect(db_path)
    events = pd.read_sql("SELECT magnitude, time_utc FROM events", conn)
    conn.close()

    events["time_utc"] = pd.to_datetime(events["time_utc"])
    now = events["time_utc"].max()

    recent = events[events["time_utc"] >= now - pd.Timedelta(days=7)]["magnitude"].values
    reference = events[
        (events["time_utc"] >= now - pd.Timedelta(days=37)) &
        (events["time_utc"] < now - pd.Timedelta(days=7))
    ]["magnitude"].values

    if len(reference) < 50 or len(recent) < 10:
        logger.warning("Insufficient data for drift check")
        return {"status": "insufficient_data"}

    psi_score = psi(reference, recent)
    logger.info(f"PSI magnitude (last 7d vs prior 30d): {psi_score:.4f}")

    return {
        "status": "ok",
        "psi_magnitude": psi_score,
        "should_retrain": psi_score >= 0.1,
    }


def full_report(db_path: Path = Path("data/gempa.db"),
                output_path: Path = Path("drift_report.html")):
    """
    Generate full Evidently AI drift report sebagai HTML.

    Catatan: pakai Evidently >=0.4 API.
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
    except ImportError:
        logger.error("Evidently AI tidak terinstall. pip install evidently")
        return

    conn = sqlite3.connect(db_path)
    events = pd.read_sql("SELECT * FROM events", conn)
    conn.close()

    events["time_utc"] = pd.to_datetime(events["time_utc"])
    cutoff = events["time_utc"].max() - pd.Timedelta(days=7)

    reference = events[events["time_utc"] < cutoff]
    current = events[events["time_utc"] >= cutoff]

    if reference.empty or current.empty:
        logger.error("Tidak cukup data untuk reference/current split")
        return

    feature_cols = ["magnitude", "depth_km", "latitude", "longitude"]
    reference_df = reference[feature_cols].dropna()
    current_df = current[feature_cols].dropna()

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)
    report.save_html(str(output_path))
    logger.info(f"Drift report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick-check", action="store_true")
    parser.add_argument("--full-report", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("drift_report.html"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    if args.quick_check:
        result = quick_check()
        # GitHub Actions output
        if result.get("should_retrain"):
            print("::set-output name=should_retrain::true")
        else:
            print("::set-output name=should_retrain::false")
        return

    if args.full_report:
        full_report(output_path=args.output)


if __name__ == "__main__":
    main()
