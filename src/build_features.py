"""
Feature Engineering

Membangun dataset training untuk binary classification:

    Target: label_susulan_besar_24jam

Pipeline:
  1. Load semua processed events (dari preprocess.py --all)
  2. Identifikasi mainshocks: M >= 5.0 yang tidak didahului M >= 5.0
     dalam radius 100 km / 7 hari sebelumnya
  3. Untuk tiap mainshock, generate snapshot pada t = 1, 3, 6, 12,
     24, 48, 72 jam setelah mainshock
  4. Untuk tiap snapshot, hitung:
     - Fitur rolling: count_susulan_Xjam, max_mag_susulan
     - Fitur statis: mainshock_magnitude, depth, zona_sesar
  5. Label: 1 jika ada susulan M >= 4.0 dalam jendela
     [snapshot_time, snapshot_time + 24h], radius 100 km
  6. Output: data/processed/features.parquet (training-ready)

Cara menjalankan:
  python src/build_features.py
  python src/build_features.py --processed-dir data/processed
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Setup logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "build_features.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Konstanta domain
MAINSHOCK_THRESHOLD = 5.0       # M >= 5.0 dianggap mainshock
AFTERSHOCK_THRESHOLD = 4.0      # M >= 4.0 dianggap susulan signifikan
SPATIAL_RADIUS_KM = 100         # radius pencarian susulan dari mainshock
PREDICTION_WINDOW_HOURS = 24    # prediksi untuk 24 jam ke depan
MAINSHOCK_ISOLATION_DAYS = 7    # mainshock harus tidak didahului M>=5 dalam 7 hari

SNAPSHOT_HOURS = [1, 3, 6, 12, 24, 48, 72]

OMORI_K = 10.0
OMORI_C = 0.1
OMORI_P = 1.0

PROCESSED_DIR = Path("data/processed")
DEFAULT_OUTPUT = PROCESSED_DIR / "features.parquet"


# Helpers
def haversine_km(lat1, lon1, lat2, lon2):
    """Jarak haversine dalam km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def classify_zone(latitude: float, longitude: float) -> int:
    """Klasifikasi zona sesar dari koordinat (0..4)."""
    if -6 < latitude < 6 and 95 < longitude < 103:
        return 2  # Sesar Sumatra
    if -11 < latitude < -6 and 95 < longitude < 120:
        return 0  # Subduksi Jawa-Sumatra-Bali
    if -4 < latitude < 2 and 119 < longitude < 124:
        return 3  # Sesar Palu-Koro
    if -8 < latitude < 2 and 124 < longitude < 135:
        return 1  # Subduksi Maluku-Banda
    return 4  # Papua


def omori_rate(t_hours: float) -> float:
    """Estimasi laju susulan dari Omori's Law (per hari)."""
    t_days = max(t_hours / 24.0, 0.001)
    return OMORI_K / ((t_days + OMORI_C) ** OMORI_P)


# Step 1: Load all processed events
def load_all_processed_events(processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Gabungkan semua processed CSV/Parquet, dedup, return clean DataFrame."""
    csv_files = sorted(processed_dir.glob("gempa_*_processed.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"Tidak ada file processed di {processed_dir}. "
            "Jalankan ingest_data.py + preprocess.py --all dulu."
        )

    logger.info("Loading %d processed files", len(csv_files))
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        dfs.append(df)

    events = pd.concat(dfs, ignore_index=True)
    events = events.drop_duplicates(subset=["event_id"], keep="first")
    events["time_utc"] = pd.to_datetime(events["time_utc"], errors="coerce", utc=True)
    events = events.dropna(subset=["time_utc"]).reset_index(drop=True)
    events = events.sort_values("time_utc").reset_index(drop=True)

    logger.info("Total unique events: %d", len(events))
    logger.info("Rentang waktu: %s s.d. %s",
                events["time_utc"].min(), events["time_utc"].max())
    logger.info("Distribusi magnitude:\n%s",
                events["magnitude"].describe().to_string())
    return events


# Step 2: Identifikasi mainshocks
def find_mainshocks(events: pd.DataFrame) -> pd.DataFrame:
    """
    Mainshock = M >= 5.0 yang tidak didahului M >= 5.0 dalam
    radius 100 km / 7 hari sebelumnya.
    """
    candidates = events[events["magnitude"] >= MAINSHOCK_THRESHOLD].copy()
    logger.info("Candidates M>=%.1f: %d events", MAINSHOCK_THRESHOLD, len(candidates))

    is_mainshock = []
    for idx, row in candidates.iterrows():
        window_start = row["time_utc"] - timedelta(days=MAINSHOCK_ISOLATION_DAYS)
        prior = events[
            (events["time_utc"] >= window_start)
            & (events["time_utc"] < row["time_utc"])
            & (events["magnitude"] >= MAINSHOCK_THRESHOLD)
        ]
        if prior.empty:
            is_mainshock.append(True)
            continue

        distances = haversine_km(
            row["latitude"], row["longitude"],
            prior["latitude"].values, prior["longitude"].values,
        )
        is_mainshock.append(not (distances < SPATIAL_RADIUS_KM).any())

    candidates["is_mainshock"] = is_mainshock
    mainshocks = candidates[candidates["is_mainshock"]].drop(columns=["is_mainshock"])
    logger.info("Mainshocks teridentifikasi: %d", len(mainshocks))
    return mainshocks


# Step 3: Build training rows per (mainshock, snapshot_time)
def build_training_row(mainshock: pd.Series, events: pd.DataFrame,
                        snapshot_hours: float) -> dict:
    """Bangun 1 row training data untuk kombinasi (mainshock, snapshot_time)."""
    snapshot_time = mainshock["time_utc"] + timedelta(hours=snapshot_hours)

    # Cari gempa di radius spasial mainshock
    distances = haversine_km(
        mainshock["latitude"], mainshock["longitude"],
        events["latitude"].values, events["longitude"].values,
    )
    nearby = events[distances < SPATIAL_RADIUS_KM].copy()

    # Past aftershocks: setelah mainshock, sebelum snapshot
    past_aftershocks = nearby[
        (nearby["time_utc"] > mainshock["time_utc"])
        & (nearby["time_utc"] <= snapshot_time)
        & (nearby["event_id"] != mainshock["event_id"])
    ]

    # Future aftershocks (untuk label)
    future_window_end = snapshot_time + timedelta(hours=PREDICTION_WINDOW_HOURS)
    future_aftershocks = nearby[
        (nearby["time_utc"] > snapshot_time)
        & (nearby["time_utc"] <= future_window_end)
    ]
    label = int((future_aftershocks["magnitude"] >= AFTERSHOCK_THRESHOLD).any())

    def count_in_last_hours(hours: float) -> int:
        cutoff = snapshot_time - timedelta(hours=hours)
        return int(((past_aftershocks["time_utc"] >= cutoff)).sum())

    def max_mag_in_last_hours(hours: float) -> float:
        cutoff = snapshot_time - timedelta(hours=hours)
        subset = past_aftershocks[past_aftershocks["time_utc"] >= cutoff]
        return float(subset["magnitude"].max()) if not subset.empty else 0.0

    return {
        "mainshock_id": mainshock["event_id"],
        "mainshock_time": mainshock["time_utc"],
        "snapshot_hours": snapshot_hours,
        # Fitur statis mainshock
        "mainshock_magnitude": float(mainshock["magnitude"]),
        "mainshock_depth": float(mainshock["depth_km"]) if pd.notna(mainshock["depth_km"]) else 10.0,
        "zona_sesar": classify_zone(mainshock["latitude"], mainshock["longitude"]),
        # Fitur temporal
        "jam_sejak_mainshock": float(snapshot_hours),
        # Fitur rolling
        "count_susulan_1jam": count_in_last_hours(1),
        "count_susulan_6jam": count_in_last_hours(6),
        "count_susulan_24jam": count_in_last_hours(24),
        "max_mag_susulan_6jam": max_mag_in_last_hours(6),
        "max_mag_susulan_24jam": max_mag_in_last_hours(24),
        # Fitur fisika
        "omori_rate_est": omori_rate(snapshot_hours),
        # LABEL (target binary classification)
        "label_susulan_besar_24jam": label,
    }


def build_dataset(events: pd.DataFrame) -> pd.DataFrame:
    """Bangun training dataset lengkap."""
    mainshocks = find_mainshocks(events)
    if mainshocks.empty:
        logger.warning("Tidak ada mainshock — dataset training kosong")
        return pd.DataFrame()

    rows = []
    for _, ms in mainshocks.iterrows():
        for t in SNAPSHOT_HOURS:
            rows.append(build_training_row(ms, events, t))

    df = pd.DataFrame(rows)
    logger.info("Training dataset: %d rows (= %d mainshock × %d snapshot_times)",
                len(df), len(mainshocks), len(SNAPSHOT_HOURS))
    logger.info("Distribusi label:\n%s",
                df["label_susulan_besar_24jam"].value_counts())

    pos = df["label_susulan_besar_24jam"].sum()
    total = len(df)
    logger.info("Positive ratio: %.2f%% (%d / %d)", 100 * pos / total, pos, total)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Build training features binary classification",
    )
    parser.add_argument(
        "--processed-dir", type=Path, default=PROCESSED_DIR,
        help="Folder berisi processed CSV/Parquet",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Path output Parquet untuk features",
    )
    args = parser.parse_args()

    events = load_all_processed_events(args.processed_dir)
    df = build_dataset(events)

    if df.empty:
        logger.error("Dataset training kosong. Backfill lebih banyak data.")
        return 1

    args.output.parent.mkdir(exist_ok=True, parents=True)
    df.to_parquet(args.output, index=False)
    df.to_csv(args.output.with_suffix(".csv"), index=False)

    logger.info("=" * 60)
    logger.info("FEATURES BERHASIL DIBANGUN")
    logger.info("=" * 60)
    logger.info("Output Parquet : %s (%d rows, %.1f KB)",
                args.output, len(df), args.output.stat().st_size / 1024)
    logger.info("Output CSV     : %s", args.output.with_suffix(".csv"))
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())