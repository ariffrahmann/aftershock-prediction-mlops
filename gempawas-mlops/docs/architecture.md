# Arsitektur Sistem GempaWas

## Alur Data End-to-End

```
┌──────────────┐     ┌──────────────┐
│  BMKG API    │     │  USGS API    │
│ (Indonesia)  │     │  (Global)    │
└──────┬───────┘     └──────┬───────┘
       │                    │
       ▼                    ▼
   ┌─────────────────────────────┐
   │   GitHub Actions Cron       │
   │   scheduled_ingestion.yml   │
   │   (tiap jam, menit ke-5)    │
   └──────────┬──────────────────┘
              │
              ▼
   ┌─────────────────────────────┐
   │   src/ingestion.py          │
   │   - Fetch & dedup            │
   │   - Upsert ke SQLite         │
   └──────────┬──────────────────┘
              │
              ▼
   ┌─────────────────────────────┐
   │   data/gempa.db (SQLite)    │
   │   Tabel: events              │
   └──────────┬──────────────────┘
              │
              ├──────────────────┐
              ▼                  ▼
   ┌──────────────────┐  ┌──────────────────┐
   │  Drift Monitor   │  │ Feature Engineer │
   │  (jam-jaman)     │  │  (weekly cron)   │
   │  PSI per fitur   │  │  → features.parquet
   └────────┬─────────┘  └────────┬─────────┘
            │                     │
            │  drift > 0.1?       ▼
            └─────────────┐  ┌──────────────────┐
                          │  │  src/train.py    │
                          ├──┤  RF + XGBoost    │
                          │  │  → MLflow Registry
                          │  └─────┬────────────┘
                          ▼        │
                ┌────────────────────┐
                │  GitHub Actions    │
                │  weekly_retrain.yml │
                └────────────────────┘
                          │
                          ▼
                ┌────────────────────┐
                │  Model Registry    │
                │  (MLflow)          │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │  src/inference.py  │
                │  FastAPI on :8000  │
                │  POST /predict     │
                └────────────────────┘
                          │
                          ▼
                ┌────────────────────┐
                │  End Users         │
                │  - BPBD dashboard  │
                │  - Mobile app      │
                │  - Media APIs      │
                └────────────────────┘
```

## Komponen Utama

### 1. Ingestion Layer (`src/ingestion.py`)
**Tugas:** Mengambil data gempa baru dari API publik dan menyimpannya ke SQLite.

**Sumber:**
- USGS Earthquake API (sumber utama, lebih lengkap historis)
- BMKG Open API (data Indonesia + felt intensity per kota)

**Triggered by:** GitHub Actions cron `scheduled_ingestion.yml` (tiap jam)

**Idempotent:** YA — pakai `INSERT OR IGNORE` berdasarkan event_id

### 2. Storage Layer (`data/gempa.db`)
**Database:** SQLite untuk simplicity (bisa upgrade ke DuckDB untuk analytics)

**Schema:**
```sql
CREATE TABLE events (
    event_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,        -- 'USGS' atau 'BMKG'
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
);
```

### 3. Feature Engineering (`src/features.py`)
**Tugas:** Konversi raw events menjadi training rows per `(mainshock, snapshot_time)`.

**Logic:**
1. Identifikasi mainshocks: M ≥ 5.0 tanpa preceding event dalam radius 100km / 7 hari
2. Untuk setiap mainshock, buat snapshot di t = 1, 3, 6, 12, 24, 48, 72 jam
3. Untuk setiap snapshot, hitung rolling features dari past events
4. Label = ada susulan ≥ M4.0 dalam 24 jam ke depan?

### 4. Training (`src/train.py`)
**Models:** Random Forest (baseline) + XGBoost (primary)

**Tracking:** MLflow — log params, metrics, model artifacts, feature importance

**Evaluation:** F1-score dan PR-AUC (relevan untuk class imbalance)

**Split:** Time-based (training: data lama, test: data terbaru)

### 5. Serving (`src/inference.py`)
**Framework:** FastAPI

**Endpoint:**
- `GET /health` — health check
- `POST /predict` — prediksi probabilitas susulan
- `GET /docs` — Swagger UI

**Latency target:** < 200ms per request

### 6. Monitoring (`src/monitor.py`)
**Drift Detection:** PSI (Population Stability Index) + Evidently AI

**Threshold:** PSI > 0.1 memicu retraining

**Output:** HTML report sebagai GitHub Actions artifact

### 7. Continuous Training (`weekly_retrain.yml`)
**Schedule:** Setiap Senin pagi (cron `0 3 * * 1` UTC)

**Logic:**
1. Pull data terbaru
2. Cek drift — kalau di atas threshold, retrain
3. Bandingkan dengan model production saat ini
4. Promote kalau metrics ≥ 2% lebih baik
5. Rollback kalau lebih buruk

## Decision: Mengapa SQLite, bukan PostgreSQL?

Untuk MVP project mahasiswa: SQLite cukup karena (1) zero setup, (2) data volume kecil (~6000 events/tahun), (3) bisa di-checkpoint via GitHub Actions cache. Untuk production scale-up, migrasi ke PostgreSQL atau TimescaleDB direkomendasikan.

## Decision: Mengapa MLflow, bukan W&B atau Neptune?

MLflow open-source dan bisa di-self-host di Codespaces. Cocok dengan budget gratis mahasiswa. W&B free tier juga bisa, tapi MLflow lebih mudah diintegrasikan dengan registry workflow.
