# LK-03 — Desain Pipeline Data ETL untuk GempaWas

**Mata Kuliah:** MLOps
**Topik:** Pipeline data dinamis dengan strategi versioning
**Output dokumen ini:** Identifikasi sumber data + desain ETL + diagram arsitektur + rencana versioning (persiapan DVC di LK-05)

---

## 1. Identifikasi Sumber Data Dinamis

Tugas ML GempaWas membutuhkan dataset yang **terus bertambah** (streaming time-series). Berikut dua sumber yang dipilih, beserta justifikasinya:

### 1.1. USGS Earthquake API (Sumber Utama)

| Atribut | Detail |
|---|---|
| Endpoint | `https://earthquake.usgs.gov/fdsnws/event/1/query` |
| Format response | GeoJSON |
| Lisensi | Public domain (US Government work) |
| Otentikasi | Tidak diperlukan |
| Rate limit | Ada batas wajar (~20.000 events / request, ~60 request/menit), namun untuk Indonesia per jam jauh di bawah threshold |
| Frekuensi update | Near real-time, biasanya event terdeteksi terpublish 5–15 menit setelah kejadian |
| Coverage | Global; sangat baik untuk gempa M ≥ 4.0 di Indonesia |
| Field yang dipakai | `time`, `mag`, `magType`, `latitude`, `longitude`, `depth`, `place`, `mmi` |

**Cara pengambilan berkala:**
```
GET /fdsnws/event/1/query
    ?format=geojson
    &starttime=<iso_2_jam_lalu>
    &endtime=<iso_now>
    &minlatitude=-11&maxlatitude=6
    &minlongitude=95&maxlongitude=141
    &minmagnitude=2.5
```

### 1.2. BMKG Open Data (Sumber Pelengkap)

| Atribut | Detail |
|---|---|
| Endpoint utama | `https://data.bmkg.go.id/DataMKG/TEWS/gempaterkini.json` |
| Endpoint tambahan | `https://data.bmkg.go.id/DataMKG/TEWS/gempadirasakan.json` |
| Format | JSON (struktur custom BMKG) |
| Lisensi | Open data BMKG |
| Otentikasi | Tidak diperlukan |
| Karakteristik | Update sangat cepat untuk gempa lokal Indonesia, plus field `Dirasakan` (intensitas MMI per kota) yang tidak ada di USGS |
| Limitasi | Hanya 15 gempa terbaru pada endpoint terkini → wajib polling tiap jam supaya tidak ketinggalan |

### 1.3. Mengapa Dua Sumber?

| Skenario | Mitigasi |
|---|---|
| BMKG endpoint down/maintenance | USGS jadi backup, tetap dapat data |
| USGS lambat publish gempa M kecil di Indonesia | BMKG sering publish lebih dulu |
| Verifikasi konsistensi data | Cross-check magnitude dari dua source untuk event yang sama |

Pendekatan **dual-source ingestion dengan dedup berdasarkan timestamp + jarak spasial** adalah praktik standar di sistem monitoring seismik (mirip pendekatan EMSC di Eropa).

---

## 2. Karakteristik Data Dinamis (Sesuai LK-01)

| Aspek | Detail |
|---|---|
| **Volume** | ~6.000 events/tahun untuk Indonesia ≈ 16/hari ≈ 1 event tiap 1.5 jam (rata-rata) |
| **Velocity** | Bervariasi: bisa 0 event/jam di kondisi tenang, atau 50+ event/jam selama sequence aftershock besar |
| **Variety** | Single modality (numerik + metadata teks), tabular ringan |
| **Veracity** | Tinggi — data sensor seismograf yang sudah diverifikasi |
| **Pergeseran distribusi (drift)** | Distribusi magnitude bisa shift saat ada sequence besar (mainshock + ratusan susulan). Distribusi zona spasial shift saat aktivitas kerak bergeser. |

---

## 3. Desain Pipeline ETL

Pipeline GempaWas mengikuti pola **ELT-then-FE** modern: load mentah ke storage dulu, baru transform untuk modeling. Ini memudahkan re-processing kalau definisi fitur berubah.

### 3.1. Extract — Ingestion Otomatis

**File:** `src/ingestion.py` (sudah ada skeleton, akan disempurnakan di LK-04)

```
┌──────────────────────────┐
│  GitHub Actions cron     │
│  (tiap jam, menit ke-5)  │
└────────────┬─────────────┘
             │
             ▼
   ┌─────────────────────┐     ┌─────────────────────┐
   │  Fetch USGS         │     │  Fetch BMKG         │
   │  (2 jam terakhir)   │     │  (gempa terkini)    │
   └──────────┬──────────┘     └──────────┬──────────┘
              │                            │
              └────────────┬───────────────┘
                           ▼
                ┌─────────────────────┐
                │  Normalize schema   │
                │  (event_id, time,   │
                │   lat, lon, depth,  │
                │   mag, source)      │
                └──────────┬──────────┘
                           ▼
                ┌─────────────────────┐
                │  Dedup by event_id  │
                │  + spatiotemporal   │
                │  proximity match    │
                └──────────┬──────────┘
                           ▼
                ┌─────────────────────┐
                │  INSERT OR IGNORE   │
                │  ke data/gempa.db   │
                │  (SQLite, idempotent)│
                └─────────────────────┘
```

**Aspek penting:**
- **Idempotent**: re-run di jam yang sama tidak menggandakan data, karena `event_id` adalah PK
- **Stateless workflow**: GitHub Actions menggunakan `actions/cache` untuk merestore `data/gempa.db` antar run
- **Resilient**: try/except pada tiap source — kegagalan satu source tidak menggagalkan pipeline

### 3.2. Pola Penamaan Snapshot Mentah

Selain SQLite (untuk query cepat), tiap ingestion juga **menulis raw JSON snapshot** ke `data/raw/` dengan nama berbasis timestamp — supaya kita bisa **re-process** kapan saja kalau ada bug di parser:

```
data/raw/usgs_2026-05-12T08-05-00Z.json
data/raw/bmkg_2026-05-12T08-05-00Z.json
```

Pola `YYYY-MM-DDTHH-MM-SSZ` sengaja menggunakan `-` (bukan `:`) supaya valid di Windows filesystem.

### 3.3. Transform — Cleaning Layer

**File rencana:** `src/preprocess.py` (akan dibuat di LK-04, sesuai instruksi LK-04 #4)

Tahapan cleaning:

| Step | Operasi | Alasan |
|---|---|---|
| 1 | Parse `time_utc` ke datetime UTC | Sumber kasih ISO string atau epoch ms — perlu normalisasi |
| 2 | Validasi koordinat dalam bounding box Indonesia | Buang sampah / event di luar area |
| 3 | Fill missing `depth_km` dengan median 10 km | Rata-rata kedalaman kerak Indonesia |
| 4 | Konversi magnitude type ke skala konsisten | Mw, Ml, mb tidak identik — prioritaskan Mw, drop yang tidak ada |
| 5 | Dedup spatio-temporal | Event yang sama mungkin dilaporkan dua sumber dengan ID berbeda |
| 6 | Outlier removal | Magnitude > 9.5 atau < 0 = anomali sensor |
| 7 | Tulis hasil ke `data/interim/events_clean.parquet` | Parquet lebih cepat di-load oleh pandas |

### 3.4. Transform — Feature Engineering

**File:** `src/features.py`

Pipeline FE mengubah event-level data menjadi sample-level training data. Setiap baris training = `(mainshock, snapshot_time)` dengan snapshot diambil pada t = 1, 3, 6, 12, 24, 48, 72 jam setelah mainshock.

```
┌────────────────────────────────────┐
│  data/interim/events_clean.parquet │
└──────────────────┬─────────────────┘
                   ▼
   ┌────────────────────────────────────┐
   │  Identifikasi mainshocks           │
   │  (M ≥ 5.0, tidak didahului M ≥ 5  │
   │   dalam radius 100 km / 7 hari)    │
   └──────────────────┬─────────────────┘
                      ▼
   ┌────────────────────────────────────┐
   │  Untuk tiap mainshock × snapshot:  │
   │  - Hitung rolling count susulan    │
   │    di window 1, 6, 24 jam sebelum │
   │  - Hitung max_mag_susulan          │
   │  - Hitung omori_rate_est           │
   │  - Encode zona_sesar dari koord    │
   │  - Compute label (event di window  │
   │    24 jam ke depan)                │
   └──────────────────┬─────────────────┘
                      ▼
   ┌────────────────────────────────────┐
   │  data/processed/features.parquet   │
   │  (10 fitur + 1 label)              │
   └────────────────────────────────────┘
```

**Fitur final** (10 buah, ringkas):
1. `mainshock_magnitude` (float)
2. `mainshock_depth` (float)
3. `jam_sejak_mainshock` (float, snapshot time)
4. `count_susulan_1jam` (int)
5. `count_susulan_6jam` (int)
6. `count_susulan_24jam` (int)
7. `max_mag_susulan_6jam` (float)
8. `max_mag_susulan_24jam` (float)
9. `omori_rate_est` (float, dari rumus Omori)
10. `zona_sesar` (int categorical 0..4)

**Label:** `label_susulan_besar_24jam` (binary 0/1)

Detail rumus, threshold, dan justifikasi fisika ada di [`omori_law.md`](omori_law.md) dan [`data_schema.md`](data_schema.md).

### 3.5. Load

| Layer | Lokasi | Format | Kapan diisi |
|---|---|---|---|
| **Raw snapshot** | `data/raw/` | JSON per timestamp | Tiap ingestion run |
| **Operational store** | `data/gempa.db` | SQLite (tabel `events`) | Tiap ingestion run, upsert |
| **Interim cleaned** | `data/interim/events_clean.parquet` | Parquet | On-demand sebelum FE |
| **Processed features** | `data/processed/features.parquet` | Parquet | Tiap retraining (mingguan / drift trigger) |

Pipeline ini **append-only di layer raw** (snapshot lama tidak pernah ditimpa) dan **idempotent di layer operational** — kombinasi yang aman untuk reproducibility.

---

## 4. Visualisasi Arsitektur

Diagram lengkap ada di `docs/figures/lk03_arsitektur_etl.svg` (juga di-embed di PDF LK-01). Sumber teks mermaid ada di `docs/figures/lk03_arsitektur_etl.mmd` untuk editing.

Ringkasan grafis (text-form):

```
   ╔════════════════════════════════════════╗
   ║  EXTERNAL DATA SOURCES                 ║
   ║  ┌──────────┐  ┌──────────┐            ║
   ║  │ USGS API │  │ BMKG API │            ║
   ║  └────┬─────┘  └────┬─────┘            ║
   ╚═══════│═════════════│═══════════════════╝
           │             │
           ▼             ▼
   ╔════════════════════════════════════════╗
   ║  INGESTION (GitHub Actions, hourly)    ║
   ║  src/ingestion.py                      ║
   ╚════════════════════┬═══════════════════╝
                        │
              ┌─────────┼─────────┐
              ▼         ▼         ▼
        data/raw/   data/gempa.db   logs/
        (JSON       (SQLite,        ingestion.log
         snapshot)   operational)
                        │
                        ▼
   ╔════════════════════════════════════════╗
   ║  PREPROCESS — src/preprocess.py        ║
   ║  cleaning, dedup, outlier removal       ║
   ╚════════════════════┬═══════════════════╝
                        │
                        ▼
              data/interim/events_clean.parquet
                        │
                        ▼
   ╔════════════════════════════════════════╗
   ║  FEATURE ENGINEERING — src/features.py ║
   ║  rolling counts, omori_rate, zone enc   ║
   ╚════════════════════┬═══════════════════╝
                        │
                        ▼
              data/processed/features.parquet
                        │       │
                        │       └─────────► DVC tracking
                        │                   (versi v1, v2, ...)
                        ▼
   ╔════════════════════════════════════════╗
   ║  TRAINING — src/train.py (XGBoost)     ║
   ║  + MLflow logging                       ║
   ╚════════════════════┬═══════════════════╝
                        │
                        ▼
   ╔════════════════════════════════════════╗
   ║  MLflow Model Registry                  ║
   ║  None → Staging → Production            ║
   ╚════════════════════┬═══════════════════╝
                        │
              ┌─────────┴──────────┐
              ▼                     ▼
   ┌─────────────────────┐  ┌─────────────────────┐
   │ FastAPI serving     │  │ Streamlit dashboard │
   │ src/inference.py    │  │ (untuk BPBD)        │
   │ POST /predict       │  │                     │
   └─────────────────────┘  └─────────────────────┘
              │
              ▼
   End users: BPBD, BNPB, media, apps
```

Gambar SVG resmi ada di `docs/figures/lk03_arsitektur_etl.svg`.

---

## 5. Rencana Versioning Data (Persiapan DVC — LK-05)

Karena data terus bertambah, versioning **wajib** supaya:
1. Model bisa di-reproduce ke versi data tertentu
2. Eksperimen historis bisa dibandingkan secara apples-to-apples
3. Audit trail "model XYZ dilatih dengan data tanggal A sampai B" jelas

### 5.1. Apa yang Di-version

| Path | Tracker | Alasan |
|---|---|---|
| `data/raw/*.json` | DVC (cache lokal + remote opsional) | Raw bisa besar, tapi reproducibility butuh exact bytes |
| `data/gempa.db` | DVC | SQLite biner — tidak cocok di Git |
| `data/interim/events_clean.parquet` | DVC | Bisa regen, tapi pin version untuk reproducibility |
| `data/processed/features.parquet` | DVC | Dataset training utama — wajib versioned |
| `models/*.joblib` | DVC | Model artifact terkait versi data tertentu |
| `*.dvc` files | **Git** | Pointer kecil yang berisi hash — boleh masuk Git |

### 5.2. Konvensi Tag Versi Data

Pola: `data-v<major>.<minor>` dipasang sebagai Git tag setiap kali snapshot data utama berubah dan akan dipakai untuk training mendatang.

| Tag | Maknanya |
|---|---|
| `data-v0.1` | Snapshot awal, hasil backfill 10 tahun dari USGS |
| `data-v0.2` | + 1 minggu data baru via hourly ingestion |
| `data-v1.0` | Snapshot stabil untuk model production pertama |

Bersamaan dengan tag, Git commit message harus menyebutkan:
- Periode data (`2015-01-01 .. 2026-05-12`)
- Jumlah events
- Sumber (`USGS + BMKG`)

### 5.3. Alur Operasional Continual Learning + DVC

```
1. Hourly ingestion          → push raw JSON ke data/raw/
                                (DVC track tiap minggu via batch)

2. Mingguan (Senin pagi)     → preprocess + feature engineering
                                → dvc add data/processed/features.parquet
                                → git add data/processed/features.parquet.dvc
                                → git commit -m "data: weekly snapshot"
                                → git tag data-vX.Y

3. Drift detection           → kalau PSI > 0.1 ATAU F1 turun
                                → trigger retraining
                                → DVC pin model baru ke versi data terbaru

4. Audit                     → dvc diff data-v0.5 data-v0.6
                                → lihat perubahan ukuran & row count
```

### 5.4. Storage Backend untuk DVC (Opsional di LK-05 #6)

| Pilihan | Pro | Kontra |
|---|---|---|
| **Local only** | Setup nol, gratis | Tidak share-able antar mesin |
| **GitHub LFS** | Mudah, terintegrasi | Quota terbatas (1 GB free) |
| **AWS S3** | Industri standar | Butuh AWS account, kena biaya |
| **GCP Cloud Storage** | GCP free tier | Sama, butuh GCP account |
| **MinIO (self-hosted)** | Open source, kontrol penuh | Butuh server sendiri |

Untuk MVP mahasiswa: **local DVC cache** di Codespaces sudah cukup. Kalau mau ekstra, **GitHub Actions artifacts** (gratis) bisa dipakai sebagai mini remote untuk artifact retention 90 hari.

---

## 6. Hubungan ke LK Sebelumnya & Selanjutnya

| LK | Kontribusi ke LK-03 / dari LK-03 |
|---|---|
| **LK-01** | Memberi problem statement & sumber data utama (LK-03 elaborasi teknisnya) |
| **LK-02** | Memberi struktur folder `data/raw/`, `src/`, `docs/` (LK-03 mengisi desain) |
| **LK-04** | Akan implementasi `src/ingestion.py` dan `src/preprocess.py` berdasarkan desain di sini |
| **LK-05** | Akan menerapkan DVC sesuai rencana versioning di Section 5 |
| **LK-08** | Akan mengotomasi seluruh pipeline ETL via GitHub Actions trigger |

---

## 7. Ringkasan Eksekutif

GempaWas menggunakan **dual-source ingestion** dari USGS + BMKG, di-orkestrasi hourly via GitHub Actions, distorage di SQLite (operational) + Parquet (analytical), dengan **DVC sebagai versioning layer**. Pipeline ETL bersifat **idempotent & resumable** sehingga aman di-trigger ulang kapan saja. Feature engineering menghasilkan dataset tabular 10 fitur untuk training XGBoost binary classifier.

Pipeline ini adalah fondasi untuk Continual Training (LK-08) — setiap dataset versi baru otomatis memicu retraining via GitHub Actions kalau drift terdeteksi, dengan jaminan reproducibility lewat DVC pointer + MLflow Model Registry.
