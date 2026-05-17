import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

# Setup logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "ingest_data.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Konstanta
# Bounding box wilayah Indonesia (sesuai params.yaml)
INDONESIA_BBOX = {
    "min_lat": -11.0,
    "max_lat": 6.0,
    "min_lon": 95.0,
    "max_lon": 141.0,
}

USGS_ENDPOINT = "https://earthquake.usgs.gov/fdsnws/event/1/query"
BMKG_TERKINI_URL = "https://data.bmkg.go.id/DataMKG/TEWS/gempaterkini.json"
BMKG_DIRASAKAN_URL = "https://data.bmkg.go.id/DataMKG/TEWS/gempadirasakan.json"

RAW_DATA_DIR = Path("data/raw")
REQUEST_TIMEOUT = 30  # detik


# Fungsi: Fetch dari USGS
def fetch_usgs(hours_back: int = 2) -> pd.DataFrame:
    """
    Mengambil data gempa dari USGS Earthquake Catalog API menggunakan
    library `requests`. Data difilter untuk wilayah Indonesia saja.

    Args:
        hours_back: Berapa jam ke belakang yang ingin diambil.

    Returns:
        DataFrame berisi event gempa dari USGS, atau DataFrame kosong jika gagal.
    """
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=hours_back)

    params = {
        "format": "geojson",
        "starttime": start_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": end_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "minlatitude": INDONESIA_BBOX["min_lat"],
        "maxlatitude": INDONESIA_BBOX["max_lat"],
        "minlongitude": INDONESIA_BBOX["min_lon"],
        "maxlongitude": INDONESIA_BBOX["max_lon"],
        "minmagnitude": 2.5,
        "orderby": "time-asc",
    }

    logger.info(f"[USGS] Fetching {hours_back} jam terakhir ({start_time} → {end_time})")
    try:
        response = requests.get(USGS_ENDPOINT, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        geojson = response.json()
    except requests.exceptions.ConnectionError:
        logger.error("[USGS] Koneksi gagal — periksa jaringan Anda.")
        return pd.DataFrame()
    except requests.exceptions.Timeout:
        logger.error("[USGS] Request timeout setelah %d detik.", REQUEST_TIMEOUT)
        return pd.DataFrame()
    except requests.exceptions.HTTPError as e:
        logger.error("[USGS] HTTP error: %s", e)
        return pd.DataFrame()

    rows = []
    for feature in geojson.get("features", []):
        props = feature["properties"]
        coords = feature["geometry"]["coordinates"]  # [lon, lat, depth]
        rows.append({
            "event_id": feature["id"],
            "source": "USGS",
            "time_utc": datetime.fromtimestamp(
                props["time"] / 1000, tz=timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            "latitude": coords[1],
            "longitude": coords[0],
            "depth_km": coords[2] if len(coords) > 2 else None,
            "magnitude": props.get("mag"),
            "magnitude_type": props.get("magType"),
            "place": props.get("place"),
            "felt_intensity": props.get("mmi"),
            "status": props.get("status"),
            "ingested_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        })

    logger.info("[USGS] Berhasil mengambil %d event.", len(rows))
    return pd.DataFrame(rows)


# Fungsi: Fetch dari BMKG
def fetch_bmkg() -> pd.DataFrame:
    """
    Mengambil data gempa dari BMKG Open API menggunakan library `requests`.
    Menggabungkan endpoint gempaterkini dan gempadirasakan.

    Returns:
        DataFrame berisi event gempa dari BMKG, atau DataFrame kosong jika gagal.
    """
    rows = []

    for url, label in [
        (BMKG_TERKINI_URL, "terkini"),
        (BMKG_DIRASAKAN_URL, "dirasakan"),
    ]:
        logger.info("[BMKG] Fetching endpoint: %s", label)
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.ConnectionError:
            logger.warning("[BMKG] Koneksi gagal untuk endpoint %s.", label)
            continue
        except requests.exceptions.Timeout:
            logger.warning("[BMKG] Timeout pada endpoint %s.", label)
            continue
        except requests.exceptions.HTTPError as e:
            logger.warning("[BMKG] HTTP error %s: %s", label, e)
            continue

        gempa_list = data.get("Infogempa", {}).get("gempa", [])
        if isinstance(gempa_list, dict):
            gempa_list = [gempa_list]

        for g in gempa_list:
            try:
                # Format Lintang: "-8.43 LS" atau "2.14 LU"
                # Format Bujur : "115.23 BT"
                def parse_coord(raw: str) -> float:
                    raw = str(raw).strip()
                    parts = raw.split()
                    if not parts:
                        return 0.0
                    val = float(parts[0])
                    suffix = parts[1].upper() if len(parts) > 1 else ""
                    if suffix in ("LS", "S", "W", "BB"):
                        val = -val
                    return val

                lat = parse_coord(g.get("Lintang", "0 LS"))
                lon = parse_coord(g.get("Bujur", "0 BT"))

                # Depth dalam format "10 km"
                depth_raw = g.get("Kedalaman", "0 km")
                depth_km = float(depth_raw.replace(" km", "").strip())

                rows.append({
                    "event_id": f"BMKG-{g['DateTime']}-{g['Magnitude']}".replace(" ", "_"),
                    "source": f"BMKG-{label}",
                    "time_utc": g.get("DateTime", ""),
                    "latitude": lat,
                    "longitude": lon,
                    "depth_km": depth_km,
                    "magnitude": float(g.get("Magnitude", 0)),
                    "magnitude_type": "M",
                    "place": g.get("Wilayah", g.get("Keterangan", "")),
                    "felt_intensity": g.get("Dirasakan", None),
                    "status": "reviewed",
                    "ingested_at": datetime.now(timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%S+00:00"
                    ),
                })
            except (KeyError, ValueError, IndexError) as e:
                logger.warning("[BMKG] Baris dilewati karena parse error: %s", e)

        logger.info("[BMKG] Endpoint %s: %d event diproses.", label, len(rows))

    return pd.DataFrame(rows)


# Fungsi: Simpan ke CSV bertimestamp (non-destruktif)
def save_raw_csv(df: pd.DataFrame, dry_run: bool = False) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RAW_DATA_DIR / f"gempa_{timestamp}.csv"

    if dry_run:
        logger.info("[DRY-RUN] Tidak ada file yang disimpan. Preview 5 baris teratas:")
        print(df.head().to_string(index=False))
        return Path()

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info("Data mentah disimpan ke: %s (%d baris)", output_path, len(df))
    return output_path


# Fungsi: Simpan manifest run
def update_ingestion_manifest(output_path: Path, row_count: int):
    manifest_path = RAW_DATA_DIR / "ingestion_manifest.csv"
    entry = pd.DataFrame([{
        "run_timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "output_file": output_path.name,
        "row_count": row_count,
        "status": "success" if row_count > 0 else "empty",
    }])

    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        manifest = pd.concat([manifest, entry], ignore_index=True)
    else:
        manifest = entry

    manifest.to_csv(manifest_path, index=False)
    logger.info("Manifest diperbarui: %s", manifest_path)


# Main
def main():
    parser = argparse.ArgumentParser(
        description="Data Ingestion Script"
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=2,
        help="Berapa jam terakhir yang diambil dari USGS (default: 2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Jalankan tanpa menyimpan file (untuk testing)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("GempaWas — Data Ingestion dimulai")
    logger.info("Mode: %s | Jendela USGS: %d jam", 
                "DRY-RUN" if args.dry_run else "NORMAL", args.hours)
    logger.info("=" * 60)

    # 1. Ambil data dari USGS
    usgs_df = fetch_usgs(hours_back=args.hours)

    # 2. Ambil data dari BMKG
    bmkg_df = fetch_bmkg()

    # 3. Gabungkan kedua sumber
    combined = pd.concat([usgs_df, bmkg_df], ignore_index=True)

    if combined.empty:
        logger.warning("Tidak ada data yang berhasil diambil dari kedua sumber.")
        return

    # 4. Deduplikasi berdasarkan event_id
    before_dedup = len(combined)
    combined = combined.drop_duplicates(subset=["event_id"])
    logger.info(
        "Deduplikasi: %d → %d baris (duplikat dihapus: %d)",
        before_dedup, len(combined), before_dedup - len(combined),
    )

    # 5. Urutkan berdasarkan waktu
    combined = combined.sort_values("time_utc").reset_index(drop=True)

    # 6. Simpan ke file CSV bertimestamp
    output_path = save_raw_csv(combined, dry_run=args.dry_run)

    # 7. Update manifest (hanya jika bukan dry-run)
    if not args.dry_run and output_path.exists():
        update_ingestion_manifest(output_path, len(combined))

    logger.info("=" * 60)
    logger.info("Ingestion selesai. Total event: %d", len(combined))
    logger.info("  - USGS : %d event", len(usgs_df))
    logger.info("  - BMKG : %d event", len(bmkg_df))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
