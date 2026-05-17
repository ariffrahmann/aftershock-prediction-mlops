# Sistem Prediksi Gempa Susulan di Indonesia

## Deskripsi Proyek

Proyek ini merupakan sistem Machine Learning untuk memonitor dan memprediksi terjadinya gempa susulan
(*aftershock*) signifikan di wilayah Indonesia. Sistem ini dirancang untuk memanfaatkan
data seismik yang terus diperbarui secara berkala dan menghasilkan prediksi probabilitas
kemunculan aftershock bermagnitudo ≥ M4.0 dalam **24 jam ke depan** setelah terjadi
gempa utama (*mainshock*) bermagnitudo ≥ M5.0.

Sumber data diperoleh dari dua API:

1. **USGS Earthquake API** : `https://earthquake.usgs.gov/fdsnws/event/1/query`
2. **BMKG Open Data API** : `https://data.bmkg.go.id/DataMKG/TEWS/gempaterkini.json`

---

## Struktur Direktori Proyek

```
aftershock-prediction-mlops/
├── .devcontainer/                  <- Codespaces config (Python 3.11 + extensions)
│   └── devcontainer.json
├── .dvc/                           <- Konfigurasi DVC
│   ├── .gitignore
│   └── config
├── .github/workflows/              <- GitHub Actions CI/CD
│   └── mlops-automation.yaml       <- "Code as a Trigger" pipeline (LK-08)
├── config/
│   └── params.yaml                 <- Parameter terpusat
├── data/
│   ├── raw/                        <- Raw JSON/CSV snapshot dari API (DVC tracked)
│   ├── interim/                    <- Features siap training (features.parquet)
│   ├── processed/                  <- Data setelah cleaning (events_clean.parquet)
├── mlruns/                         <- MLflow tracking lokal (gitignored)
├── models/                         <- Model artifacts (DVC tracked)
│   └── model_registry_metadata.json
├── scripts/
│   ├── validate_threshold.py       <- cek metric vs threshold
│   ├── auto_register.py            <- register model ke Staging
├── src/                            <- Source code utama
│   ├── data/
│   │   ├── ingest_data.py          <- Ingestion USGS + BMKG dengan timestamp
│   │   └── preprocess.py           <- Cleaning + dedup events
│   ├── build_features.py           <- Feature engineering
│   ├── train.py                    <- Training XGBoost + MLflow logging
│   ├── inference.py                <- Streamlit inference
│   └── model_registry.py           <- Helper untuk MLflow Registry
├── tests/                          <- Unit tests (pytest)
│   ├── test_ingest_data.py
│   ├── test_preprocess.py
│   └── test_features.py
├── .dockerignore
├── .dvcignore
├── .editorconfig
├── .env.example
├── .gitignore
├── LICENSE                         <- MIT
├── Makefile
├── README.md
├── docker-compose.yaml             <- Orkestrasi multi-container (LK-09)
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

## Cara Menjalankan Proyek Menggunakan GitHub Codespaces

### 1. Membuka Codespaces

- Buka repositori GitHub: `https://github.com/<username>/aftershock-prediction-mlops`
- Klik tombol **Code** → tab **Codespaces** → **Create codespace on main**
- GitHub akan otomatis membangun lingkungan pengembangan berbasis cloud dengan Python 3.11 dan Docker yang sudah terkonfigurasi melalui `.devcontainer/devcontainer.json`

### 2. Konfigurasi Environment

Salin file `.env.example` menjadi `.env` dan sesuaikan nilainya jika diperlukan:

```bash
cp .env.example .env
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Arsitektur (Alur MLOps)

### 1. Data Ingestion & Preprocessing

Tahap ini menangani pengambilan dan pembersihan data seismik secara otomatis.

* **Pengambilan Data:** `src/data/ingest_data.py` menarik event seismik dari **USGS API** dan **BMKG API** setiap jam melalui GitHub Actions cron job (`scheduled_ingestion.yml`). Setiap output disimpan dalam format CSV/JSON bertimestamp di `data/raw/`.
* **Pembersihan Data:** `src/data/preprocess.py` menjalankan serangkaian proses pembersihan: parsing datetime ke format UTC, validasi koordinat dalam bounding box wilayah Indonesia, imputasi nilai depth yang hilang, penghapusan outlier, dan deduplication event. Hasil bersih disimpan sebagai `data/processed/.parquet`.

### 2. Manajemen Versi Data (DVC)

DVC (*Data Version Control*) digunakan untuk melacak perubahan dataset gempa secara
terstruktur tanpa membebani repositori Git. Alur standar dalam proses continual learning adalah:

1. Tarik data terbaru dari API: `python src/data/ingest_data.py --hours 24`
2. Lacak file baru via DVC: `dvc add data/raw/gempa_<timestamp>.csv`
3. Commit pointer DVC ke Git: `git add data/raw/*.dvc && git commit -m "data(v1.x): tambah snapshot terbaru"`
4. Tag versi dataset: `git tag data-v1.x`
5. Audit perbedaan antar versi: `dvc diff data-v1.0 data-v1.1`

Pendekatan ini memastikan setiap versi dataset dapat direproduksi dan di-*rollback* kapan pun dibutuhkan.

### 3. Feature Engineering

`src/build_features.py` membangun training samples dari data event yang sudah bersih melalui beberapa tahap:

* **Identifikasi mainshock:** Event bermagnitudo ≥ M5.0 yang terisolasi (tidak ada event ≥ M5.0 lain dalam radius 100 km selama 7 hari sebelumnya)
* **Snapshot temporal:** Untuk tiap mainshock, di-generate 7 snapshot pada t = 1, 3, 6, 12, 24, 48, dan 72 jam pasca-kejadian
* **Labeling biner:** Label = 1 jika terdapat gempa susulan ≥ M4.0 dalam 24 jam ke depan dalam radius 100 km; label = 0 jika tidak ada

### 4. Pelatihan Model & Registrasi

Model **XGBoost binary classifier** dilatih dengan 3 variasi hyperparameter (konfigurasi *shallow*, *medium*, dan *deep*). Seluruh proses eksperimen dilacak via **MLflow**:

* `mlflow.log_param()` — mencatat `n_estimators`, `max_depth`, `learning_rate`, `scale_pos_weight`
* `mlflow.log_metric()` — mencatat `accuracy`, `f1_score`, `roc_auc` sebagai 3 metrik inti
* `mlflow.log_model()` — menyimpan artifact model XGBoost ke MLflow Model Registry

Run terbaik berdasarkan F1-score kemudian di-*promote* melalui MLflow Model Registry dengan alur transisi stage: **None → Staging → Production**.

### 5. CI/CD Automation — "Code as a Trigger"

Pipeline GitHub Actions di `.github/workflows/mlops-automation.yaml` dipicu secara otomatis oleh setiap *push* atau *pull request* ke branch `main`. Pipeline ini menjalankan 4 stage secara berantai:

* **Stage 1 — Automated Testing:** Menjalankan seluruh unit test menggunakan `pytest`
* **Stage 2 — Automated Training:** Melatih ulang model menggunakan data terbaru dari DVC
* **Stage 3 — Model Evaluation:** Memvalidasi metrik model terhadap threshold yang ditetapkan di LK-01 (PR-AUC ≥ 0.45, F1 ≥ 0.35, Recall ≥ 0.40) menggunakan `scripts/validate_threshold.py`
* **Stage 4 — Auto-Registry Update:** Mempromosikan model terbaik ke stage *Staging* secara otomatis jika lolos validasi, menggunakan `scripts/auto_register.py`

### 6. Orkestrasi Layanan Terintegrasi

Proyek menggunakan **Docker Compose** untuk mengorkestrasi seluruh layanan infrastruktur dalam satu *custom bridge network* bernama `gempawas-network`. Terdapat 2 layanan utama yang berjalan:

* **`db`** — PostgreSQL sebagai backend penyimpanan metadata MLflow (port internal 5432, dengan volume persisten `pg-data`)
* **`mlflow-server`** — MLflow Tracking Server & Model Registry yang dapat diakses di `http://localhost:5000` (dengan volume persisten `mlflow-data`)

Untuk menjalankan seluruh sistem:

```bash
docker compose up -d
```

Untuk menghentikan layanan tanpa menghapus data:

```bash
docker compose stop
```


**Nama:** Arif Rahman  
**NIM:** 235150201111012  
**Kelas:** MLOps-B
