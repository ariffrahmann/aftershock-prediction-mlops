"""
GempaWas — Data Ingestion Module

Mengambil data gempa dari:
1. USGS Earthquake API (sumber utama, lebih lengkap untuk Indonesia)
2. BMKG Open API (fallback dan untuk gempa lokal Indonesia yang mungkin
   tidak tertangkap USGS karena magnitudo kecil)

Disimpan ke SQLite database: data/gempa.db
"""

import argparse
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

# Logging setup
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/ingestion.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Konstanta wilayah Indonesia (bounding box)
INDONESIA_BBOX = {
    "min_lat": -11.0,
    "max_lat": 6.0,
    "min_lon": 95.0,
    "max_lon": 141.0,
}

USGS_ENDPOINT = "https://earthquake.usgs.gov/fdsnws/event/1/query"
BMKG_GEMPA_TERKINI = "https://data.bmkg.go.id/DataMKG/TEWS/gempaterkini.json"
BMKG_GEMPA_DIRASAKAN = "https://data.bmkg.go.id/DataMKG/TEWS/gempadirasakan.json"

DB_PATH = Path("data/gempa.db")


def init_database():
    """Inisialisasi skema database SQLite."""
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            time_utc TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            depth_km REAL,
            magnitude REAL NOT NULL,
            magnitude_type TEXT,
            place TEXT,
            felt_intensity TEXT,
            raw_json TEXT,
            ingested_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_time
        ON events(time_utc)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_magnitude
        ON events(magnitude)
    """)
    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {DB_PATH}")


def fetch_usgs(hours_back: int = 2) -> pd.DataFrame:
    """Fetch earthquake events from USGS for Indonesia in the last N hours."""
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=hours_back)

    params = {
        "format": "geojson",
        "starttime": start_time.isoformat(),
        "endtime": end_time.isoformat(),
        "minlatitude": INDONESIA_BBOX["min_lat"],
        "maxlatitude": INDONESIA_BBOX["max_lat"],
        "minlongitude": INDONESIA_BBOX["min_lon"],
        "maxlongitude": INDONESIA_BBOX["max_lon"],
        "minmagnitude": 2.5,
        "orderby": "time-asc",
    }

    logger.info(f"Fetching USGS data from {start_time} to {end_time}")
    try:
        response = requests.get(USGS_ENDPOINT, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        logger.error(f"USGS fetch failed: {e}")
        return pd.DataFrame()

    rows = []
    for feature in data.get("features", []):
        props = feature["properties"]
        coords = feature["geometry"]["coordinates"]
        rows.append({
            "event_id": feature["id"],
            "source": "USGS",
            "time_utc": datetime.fromtimestamp(
                props["time"] / 1000, tz=timezone.utc
            ).isoformat(),
            "latitude": coords[1],
            "longitude": coords[0],
            "depth_km": coords[2] if len(coords) > 2 else None,
            "magnitude": props["mag"],
            "magnitude_type": props.get("magType"),
            "place": props.get("place"),
            "felt_intensity": str(props.get("mmi")) if props.get("mmi") else None,
        })

    logger.info(f"USGS returned {len(rows)} events")
    return pd.DataFrame(rows)


def fetch_bmkg() -> pd.DataFrame:
    """Fetch latest earthquake events from BMKG."""
    logger.info("Fetching BMKG gempaterkini")
    try:
        response = requests.get(BMKG_GEMPA_TERKINI, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        logger.error(f"BMKG fetch failed: {e}")
        return pd.DataFrame()

    rows = []
    gempa_list = data.get("Infogempa", {}).get("gempa", [])
    if isinstance(gempa_list, dict):
        gempa_list = [gempa_list]

    for g in gempa_list:
        try:
            coords = g["point"]["coordinates"].split(",")
            rows.append({
                "event_id": f"BMKG-{g['DateTime']}-{g['Magnitude']}",
                "source": "BMKG",
                "time_utc": g["DateTime"],
                "latitude": float(coords[1]),
                "longitude": float(coords[0]),
                "depth_km": float(g["Kedalaman"].replace(" km", "")),
                "magnitude": float(g["Magnitude"]),
                "magnitude_type": "M",
                "place": g.get("Wilayah"),
                "felt_intensity": g.get("Dirasakan"),
            })
        except (KeyError, ValueError) as e:
            logger.warning(f"Skipping BMKG row, parse error: {e}")

    logger.info(f"BMKG returned {len(rows)} events")
    return pd.DataFrame(rows)


def upsert_events(df: pd.DataFrame):
    """Insert atau update events ke database (idempotent)."""
    if df.empty:
        logger.info("No events to insert")
        return 0

    df["ingested_at"] = datetime.now(timezone.utc).isoformat()
    df["raw_json"] = None  # bisa diisi raw JSON kalau perlu

    conn = sqlite3.connect(DB_PATH)
    inserted = 0
    for _, row in df.iterrows():
        try:
            conn.execute("""
                INSERT OR IGNORE INTO events
                (event_id, source, time_utc, latitude, longitude,
                 depth_km, magnitude, magnitude_type, place,
                 felt_intensity, raw_json, ingested_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row["event_id"], row["source"], row["time_utc"],
                row["latitude"], row["longitude"], row["depth_km"],
                row["magnitude"], row["magnitude_type"], row["place"],
                row["felt_intensity"], row["raw_json"], row["ingested_at"],
            ))
            if conn.total_changes:
                inserted += 1
        except sqlite3.Error as e:
            logger.error(f"Insert error for {row['event_id']}: {e}")
    conn.commit()
    conn.close()
    logger.info(f"Inserted {inserted} new events into database")
    return inserted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=2,
                        help="Berapa jam terakhir yang di-fetch")
    parser.add_argument("--once", action="store_true",
                        help="Sekali jalan saja, untuk testing")
    args = parser.parse_args()

    init_database()

    usgs_df = fetch_usgs(hours_back=args.hours)
    bmkg_df = fetch_bmkg()

    combined = pd.concat([usgs_df, bmkg_df], ignore_index=True)
    if not combined.empty:
        combined = combined.drop_duplicates(subset=["event_id"])

    upsert_events(combined)

    logger.info("Ingestion complete")


if __name__ == "__main__":
    main()
