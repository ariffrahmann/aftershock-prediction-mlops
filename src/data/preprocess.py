"""
Tugas preprocessing:
  1. Memuat file CSV mentah dari data/raw/
  2. Membersihkan missing values (imputation + drop)
  3. Validasi range koordinat & magnitude (filter anomali)
  4. Normalisasi format kolom (tipe data, satuan)
  5. Menambah fitur turunan dasar (waktu, zona)
  6. Menyimpan hasil ke data/processed/ dalam format CSV & Parquet

Output siap digunakan oleh src/features.py untuk ekstraksi fitur.

Cara menjalankan:
  python src/data/preprocess.py                         # proses file terbaru
  python src/data/preprocess.py --input data/raw/gempa_20250512_100000.csv
  python src/data/preprocess.py --all                   # proses semua file raw
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np

# Setup logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "preprocess.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Konstanta
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

# Batas valid wilayah Indonesia
INDONESIA_BBOX = {
    "min_lat": -11.0,
    "max_lat": 6.0,
    "min_lon": 95.0,
    "max_lon": 141.0,
}

# Batas valid magnitude & depth untuk filtering anomali
MAGNITUDE_MIN = 0.0
MAGNITUDE_MAX = 10.0
DEPTH_MIN_KM = 0.0
DEPTH_MAX_KM = 700.0      # gempa dalam maksimum ~700 km

# Nilai default untuk imputation
DEFAULT_DEPTH_KM = 10.0   # shallow earthquake default (USGS convention)

# Step 1: Load data mentah
def load_raw(input_path: Path) -> pd.DataFrame:
    """
    Memuat file CSV mentah hasil ingest_data.py.
    Args:
        input_path: Path ke file CSV.
    Returns:
        DataFrame mentah.
    """
    logger.info("Memuat data mentah dari: %s", input_path)
    try:
        df = pd.read_csv(input_path, encoding="utf-8")
    except FileNotFoundError:
        logger.error("File tidak ditemukan: %s", input_path)
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        logger.warning("File kosong: %s", input_path)
        return pd.DataFrame()

    logger.info("Loaded: %d baris, %d kolom", len(df), len(df.columns))
    logger.info("Kolom: %s", list(df.columns))
    return df

# Step 2: Validasi skema 
REQUIRED_COLUMNS = [
    "event_id", "source", "time_utc",
    "latitude", "longitude", "depth_km",
    "magnitude", "magnitude_type", "place",
]

def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cek semua kolom. Kolom yang hilang diisi dengan NaN.
    """
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        logger.warning("Kolom berikut tidak ditemukan, diisi NaN: %s", missing_cols)
        for col in missing_cols:
            df[col] = np.nan

    return df

# Step 3: Normalisasi tipe data
def normalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mengonversi tipe data ke format yang konsisten:
    - time_utc, ingested_at → datetime (UTC)
    - latitude, longitude, depth_km, magnitude → float
    - event_id, source, place, magnitude_type → string
    """
    logger.info("Normalisasi tipe data...")

    # Datetime
    for col in ["time_utc", "ingested_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    # Float numerik
    for col in ["latitude", "longitude", "depth_km", "magnitude", "felt_intensity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # String
    for col in ["event_id", "source", "place", "magnitude_type", "status"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace("nan", np.nan)

    return df

# Step 4: Pembersihan missing values
def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menangani missing values:
    - event_id kosong   → drop baris (tidak bisa diidentifikasi)
    - time_utc kosong   → drop baris (tidak ada waktu = tidak bisa digunakan)
    - latitude/longitude → drop baris (lokasi wajib ada)
    - magnitude kosong  → drop baris (variabel target utama)
    - depth_km kosong   → impute dengan DEFAULT_DEPTH_KM (10 km, konvensi USGS)
    - magnitude_type    → impute dengan "M" (generic)
    - place             → impute dengan "Unknown"
    """
    before = len(df)

    # Drop kolom kritis yang kosong
    df = df.dropna(subset=["event_id"])
    df = df[df["event_id"].str.strip() != ""]
    df = df.dropna(subset=["time_utc"])
    df = df.dropna(subset=["latitude", "longitude"])
    df = df.dropna(subset=["magnitude"])

    dropped = before - len(df)
    logger.info(
        "Missing values — drop baris kritis: %d baris dihapus (sisa: %d)",
        dropped, len(df)
    )

    # Imputation
    if df["depth_km"].isna().any():
        n_imputed = df["depth_km"].isna().sum()
        df["depth_km"] = df["depth_km"].fillna(DEFAULT_DEPTH_KM)
        logger.info("Impute depth_km dengan %.1f km: %d baris", DEFAULT_DEPTH_KM, n_imputed)

    if "magnitude_type" in df.columns:
        df["magnitude_type"] = df["magnitude_type"].fillna("M")

    if "place" in df.columns:
        df["place"] = df["place"].fillna("Unknown")

    if "felt_intensity" in df.columns:
        df["felt_intensity"] = df["felt_intensity"].fillna(0.0)

    return df

# Step 5: Filter anomali & validasi range
def filter_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membuang baris dengan nilai di luar batas yang masuk akal secara fisika:
    - Koordinat di luar bounding box Indonesia
    - Magnitude negatif atau > 10
    - Depth negatif atau > 700 km
    """
    before = len(df)

    # Filter wilayah Indonesia
    df = df[
        (df["latitude"] >= INDONESIA_BBOX["min_lat"]) &
        (df["latitude"] <= INDONESIA_BBOX["max_lat"]) &
        (df["longitude"] >= INDONESIA_BBOX["min_lon"]) &
        (df["longitude"] <= INDONESIA_BBOX["max_lon"])
    ]

    # Filter magnitude
    df = df[
        (df["magnitude"] >= MAGNITUDE_MIN) &
        (df["magnitude"] <= MAGNITUDE_MAX)
    ]

    # Filter depth
    df = df[
        (df["depth_km"] >= DEPTH_MIN_KM) &
        (df["depth_km"] <= DEPTH_MAX_KM)
    ]

    filtered = before - len(df)
    logger.info(
        "Filter anomali: %d baris dihapus (di luar batas valid), sisa: %d",
        filtered, len(df)
    )
    return df

# Step 6: Deduplikasi
def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghapus baris duplikat berdasarkan event_id.
    Jika ada event_id yang sama, prioritaskan baris dengan source USGS.
    """
    before = len(df)

    # Prioritaskan USGS: sort source sehingga USGS muncul duluan
    df["_source_rank"] = df["source"].apply(
        lambda s: 0 if str(s).startswith("USGS") else 1
    )
    df = df.sort_values("_source_rank").drop(columns=["_source_rank"])
    df = df.drop_duplicates(subset=["event_id"], keep="first")

    logger.info("Deduplikasi: %d → %d baris", before, len(df))
    return df.reset_index(drop=True)


# Step 7: Tambah fitur turunan dasar
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menambah kolom turunan yang dibutuhkan oleh src/features.py:
    - year, month, hour     → dari time_utc
    - depth_category        → shallow/intermediate/deep
    - magnitude_category    → micro/minor/moderate/strong/major
    - zona_sesar            → klasifikasi zona berdasarkan koordinat
    - is_felt               → apakah gempa dirasakan (felt_intensity > 0)
    """
    logger.info("Menambah fitur turunan dasar...")

    # Komponen waktu
    df["year"] = df["time_utc"].dt.year
    df["month"] = df["time_utc"].dt.month
    df["hour_utc"] = df["time_utc"].dt.hour

    # Kategori kedalaman (standar USGS)
    df["depth_category"] = pd.cut(
        df["depth_km"],
        bins=[0, 70, 300, 700],
        labels=["shallow", "intermediate", "deep"],
        right=True,
    ).astype(str)

    # Kategori magnitude (skala Richter informal)
    df["magnitude_category"] = pd.cut(
        df["magnitude"],
        bins=[0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0],
        labels=["micro", "minor", "light", "moderate", "strong", "major"],
        right=True,
    ).astype(str)

    # Zona sesar (disederhanakan dari src/features.py)
    def classify_zone(lat, lon):
        if -6 < lat < 6 and 95 < lon < 103:
            return "Sesar_Sumatra"
        elif -11 < lat < -6 and 95 < lon < 120:
            return "Subduksi_Jawa"
        elif -4 < lat < 2 and 119 < lon < 124:
            return "Sesar_Palu_Koro"
        elif -8 < lat < 2 and 124 < lon < 135:
            return "Subduksi_Maluku"
        else:
            return "Lainnya_Papua"

    df["zona_sesar"] = df.apply(
        lambda r: classify_zone(r["latitude"], r["longitude"]), axis=1
    )

    # Flag: apakah gempa dirasakan
    if "felt_intensity" in df.columns:
        df["is_felt"] = df["felt_intensity"].fillna(0) > 0

    return df

# Step 8: Simpan hasil
def save_processed(df: pd.DataFrame, input_path: Path) -> dict:
    """
    Menyimpan data yang sudah dibersihkan ke:
    - data/processed/<nama_file>_processed.csv    (untuk inspeksi manual)
    - data/processed/<nama_file>_processed.parquet (untuk pipeline ML)

    Returns:
        Dict berisi path output.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem  # e.g. "gempa_20250512_100000"
    csv_path = PROCESSED_DATA_DIR / f"{stem}_processed.csv"
    parquet_path = PROCESSED_DATA_DIR / f"{stem}_processed.parquet"

    # Simpan waktu sebagai string untuk CSV agar mudah dibaca
    df_csv = df.copy()
    for col in ["time_utc", "ingested_at"]:
        if col in df_csv.columns:
            df_csv[col] = df_csv[col].astype(str)

    df_csv.to_csv(csv_path, index=False, encoding="utf-8")
    df.to_parquet(parquet_path, index=False)

    logger.info("Data processed disimpan:")
    logger.info("  CSV     : %s", csv_path)
    logger.info("  Parquet : %s", parquet_path)

    return {"csv": csv_path, "parquet": parquet_path}

# Fungsi utama: pipeline preprocessing satu file
def preprocess_file(input_path: Path) -> pd.DataFrame:
    """
    Menjalankan seluruh pipeline preprocessing untuk satu file CSV mentah.

    Pipeline:
      load → validate_schema → normalize_dtypes → clean_missing
           → filter_anomalies → deduplicate → add_derived_features → save

    Returns:
        DataFrame yang sudah bersih, siap untuk feature engineering.
    """
    logger.info("-" * 50)
    logger.info("Memproses file: %s", input_path.name)

    df = load_raw(input_path)
    if df.empty:
        logger.warning("File kosong atau gagal dimuat, dilewati.")
        return pd.DataFrame()

    df = validate_schema(df)
    df = normalize_dtypes(df)
    df = clean_missing_values(df)

    if df.empty:
        logger.warning("Tidak ada data valid setelah cleaning.")
        return pd.DataFrame()

    df = filter_anomalies(df)
    df = deduplicate(df)
    df = add_derived_features(df)

    # Simpan hasil
    save_processed(df, input_path)

    # Ringkasan statistik
    logger.info("=" * 50)
    logger.info("RINGKASAN PREPROCESSING: %s", input_path.name)
    logger.info("  Total baris bersih : %d", len(df))
    logger.info("  Rentang magnitude  : %.1f – %.1f", df["magnitude"].min(), df["magnitude"].max())
    logger.info("  Rentang kedalaman  : %.1f – %.1f km", df["depth_km"].min(), df["depth_km"].max())
    logger.info("  Sumber data        : %s", df["source"].unique().tolist())
    logger.info("  Zona sesar         : %s", df["zona_sesar"].value_counts().to_dict())
    logger.info("=" * 50)

    return df


# Main
def get_latest_raw_file() -> Path:
    """Mendapatkan file raw terbaru berdasarkan nama (timestamp di nama file)."""
    raw_files = sorted(RAW_DATA_DIR.glob("gempa_*.csv"))
    if not raw_files:
        return None
    return raw_files[-1]


def main():
    parser = argparse.ArgumentParser(
        description="GempaWas — Preprocessing Script (LK-04)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path ke file CSV mentah spesifik. Jika tidak diisi, proses file terbaru.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Proses SEMUA file CSV di data/raw/ (untuk backfill).",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("GempaWas — Preprocessing dimulai")
    logger.info("=" * 60)

    if args.all:
        # Mode: proses semua file
        raw_files = sorted(RAW_DATA_DIR.glob("gempa_*.csv"))
        if not raw_files:
            logger.error("Tidak ada file gempa_*.csv di %s", RAW_DATA_DIR)
            sys.exit(1)
        logger.info("Mode --all: memproses %d file.", len(raw_files))
        for f in raw_files:
            preprocess_file(f)

    elif args.input:
        # file spesifik dari argumen
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error("File tidak ditemukan: %s", input_path)
            sys.exit(1)
        preprocess_file(input_path)

    else:
        # default: file terbaru
        latest = get_latest_raw_file()
        if latest is None:
            logger.error(
                "Tidak ada file gempa_*.csv di '%s'. "
                "Jalankan ingest_data.py terlebih dahulu.", RAW_DATA_DIR
            )
            sys.exit(1)
        logger.info("Memproses file terbaru: %s", latest)
        preprocess_file(latest)

    logger.info("Preprocessing selesai.")


if __name__ == "__main__":
    main()
