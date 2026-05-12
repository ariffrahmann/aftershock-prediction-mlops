"""
GempaWas — Feature Engineering Module

Mengkonversi raw earthquake events menjadi training dataset
dengan fitur per "kandidat susulan" (window 24 jam pasca mainshock).

Setiap baris dataset = 1 mainshock + snapshot kondisi sekuens pada waktu T.
Label = apakah dalam 24 jam ke depan akan ada susulan ≥ M4.0?
"""

import logging
import sqlite3
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Threshold-threshold yang menentukan logika
MAINSHOCK_THRESHOLD = 5.0      # M ≥ 5.0 dianggap mainshock
AFTERSHOCK_THRESHOLD = 4.0     # M ≥ 4.0 dianggap susulan signifikan
SPATIAL_RADIUS_KM = 100        # radius pencarian susulan dari mainshock
PREDICTION_WINDOW_HOURS = 24   # prediksi untuk 24 jam ke depan

# Omori's Law constants (untuk fitur fisika)
OMORI_K = 10.0
OMORI_C = 0.1
OMORI_P = 1.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Jarak haversine antara dua koordinat dalam km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def classify_zone(latitude: float, longitude: float) -> int:
    """
    Klasifikasi zona sesar berdasarkan koordinat.

    Mapping kasar (untuk versi 1 — bisa di-refine dengan data PVMBG):
        0 = Subduksi Jawa-Sumatra-Lesser Sunda (segmen barat & selatan)
        1 = Subduksi Maluku-Banda (timur)
        2 = Sesar aktif Sumatra (Great Sumatran Fault)
        3 = Sesar Palu-Koro (Sulawesi)
        4 = Lainnya / Papua
    """
    # Sesar Sumatra (Great Sumatran Fault)
    if -6 < latitude < 6 and 95 < longitude < 103:
        return 2
    # Subduksi Jawa-Sumatra-Bali
    if -11 < latitude < -6 and 95 < longitude < 120:
        return 0
    # Sesar Palu-Koro (Sulawesi)
    if -4 < latitude < 2 and 119 < longitude < 124:
        return 3
    # Subduksi Maluku-Banda
    if -8 < latitude < 2 and 124 < longitude < 135:
        return 1
    # Papua dan sisanya
    return 4


def omori_rate(t_hours: float,
               K: float = OMORI_K,
               c: float = OMORI_C,
               p: float = OMORI_P) -> float:
    """Estimasi laju susulan per hari dari Omori's Law."""
    t_days = max(t_hours / 24.0, 0.001)
    return K / ((t_days + c) ** p)


def find_mainshocks(events: pd.DataFrame) -> pd.DataFrame:
    """
    Identifikasi gempa-gempa yang dianggap mainshock.

    Mainshock = M ≥ 5.0 yang tidak didahului gempa lain ≥ M5.0
    dalam radius 100km dan 7 hari sebelumnya.
    """
    events = events.sort_values("time_utc").reset_index(drop=True)
    events["time_utc"] = pd.to_datetime(events["time_utc"])

    candidates = events[events["magnitude"] >= MAINSHOCK_THRESHOLD].copy()
    is_mainshock = []

    for idx, row in candidates.iterrows():
        window_start = row["time_utc"] - timedelta(days=7)
        prior = events[
            (events["time_utc"] >= window_start) &
            (events["time_utc"] < row["time_utc"]) &
            (events["magnitude"] >= MAINSHOCK_THRESHOLD)
        ]
        if prior.empty:
            is_mainshock.append(True)
            continue

        prior_distances = haversine_km(
            row["latitude"], row["longitude"],
            prior["latitude"].values, prior["longitude"].values,
        )
        is_mainshock.append(not (prior_distances < SPATIAL_RADIUS_KM).any())

    candidates["is_mainshock"] = is_mainshock
    return candidates[candidates["is_mainshock"]].drop(columns=["is_mainshock"])


def build_training_row(mainshock: pd.Series,
                       events: pd.DataFrame,
                       snapshot_hours: float) -> dict:
    """
    Bangun satu baris training data untuk satu kombinasi
    (mainshock, snapshot time t jam setelah mainshock).
    """
    snapshot_time = mainshock["time_utc"] + timedelta(hours=snapshot_hours)

    # Cari gempa di radius spasial mainshock
    distances = haversine_km(
        mainshock["latitude"], mainshock["longitude"],
        events["latitude"].values, events["longitude"].values,
    )
    nearby = events[distances < SPATIAL_RADIUS_KM].copy()

    # Susulan = gempa setelah mainshock dan sebelum snapshot
    past_aftershocks = nearby[
        (nearby["time_utc"] > mainshock["time_utc"]) &
        (nearby["time_utc"] <= snapshot_time) &
        (nearby["event_id"] != mainshock["event_id"])
    ]

    # Future aftershocks = label
    future_window_end = snapshot_time + timedelta(hours=PREDICTION_WINDOW_HOURS)
    future_aftershocks = nearby[
        (nearby["time_utc"] > snapshot_time) &
        (nearby["time_utc"] <= future_window_end)
    ]
    label = int((future_aftershocks["magnitude"] >= AFTERSHOCK_THRESHOLD).any())

    # Rolling counts dalam jendela waktu sebelum snapshot
    def count_in_last_hours(hours: float) -> int:
        cutoff = snapshot_time - timedelta(hours=hours)
        return ((past_aftershocks["time_utc"] >= cutoff)).sum()

    def max_mag_in_last_hours(hours: float) -> float:
        cutoff = snapshot_time - timedelta(hours=hours)
        subset = past_aftershocks[past_aftershocks["time_utc"] >= cutoff]
        return float(subset["magnitude"].max()) if not subset.empty else 0.0

    return {
        "mainshock_id": mainshock["event_id"],
        "snapshot_hours": snapshot_hours,

        # --- Fitur ---
        "mainshock_magnitude": float(mainshock["magnitude"]),
        "mainshock_depth": float(mainshock["depth_km"] or 10.0),
        "jam_sejak_mainshock": snapshot_hours,
        "count_susulan_1jam": count_in_last_hours(1),
        "count_susulan_6jam": count_in_last_hours(6),
        "count_susulan_24jam": count_in_last_hours(24),
        "max_mag_susulan_6jam": max_mag_in_last_hours(6),
        "max_mag_susulan_24jam": max_mag_in_last_hours(24),
        "omori_rate_est": omori_rate(snapshot_hours),
        "zona_sesar": classify_zone(
            mainshock["latitude"], mainshock["longitude"]
        ),

        # --- Label ---
        "label_susulan_besar_24jam": label,
    }


def build_dataset(db_path: Path = Path("data/gempa.db")) -> pd.DataFrame:
    """
    Bangun training dataset lengkap dari database.

    Output: DataFrame dengan satu baris per (mainshock × snapshot_time).
    Untuk setiap mainshock, snapshot diambil di t = 1, 3, 6, 12, 24, 48, 72 jam.
    """
    conn = sqlite3.connect(db_path)
    events = pd.read_sql("SELECT * FROM events", conn)
    conn.close()

    if events.empty:
        logger.warning("No events in database — run ingestion.py first")
        return pd.DataFrame()

    events["time_utc"] = pd.to_datetime(events["time_utc"])
    mainshocks = find_mainshocks(events)
    logger.info(f"Found {len(mainshocks)} mainshocks")

    snapshot_times = [1, 3, 6, 12, 24, 48, 72]  # jam setelah mainshock

    rows = []
    for _, ms in mainshocks.iterrows():
        for t in snapshot_times:
            rows.append(build_training_row(ms, events, t))

    df = pd.DataFrame(rows)
    logger.info(f"Built training dataset with {len(df)} rows")
    logger.info(f"Class distribution:\n{df['label_susulan_besar_24jam'].value_counts()}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = build_dataset()
    output_path = Path("data/features.parquet")
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved features to {output_path}")
