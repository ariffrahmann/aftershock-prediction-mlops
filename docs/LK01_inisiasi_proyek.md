---
title: "LK-01 — Inisiasi Proyek Pengembangan Sistem AI Production"
subtitle: "MLOps-GempaWas: Sistem Prediksi Probabilitas Gempa Susulan di Indonesia"
author: "Niki"
date: "12 Mei 2026"
---

# LK-01 — Inisiasi Proyek

**Nama Proyek:** MLOps-GempaWas
**Mata Kuliah:** MLOps
**Tipe Tugas:** Individual
**Tanggal:** 12 Mei 2026

---

## 1. Identifikasi Masalah & Domain

### 1.1. Latar Belakang

Indonesia berada di Pacific Ring of Fire dan mengalami rata-rata sekitar 6.000 gempa bumi per tahun. Setelah gempa besar (M ≥ 5.0), biasanya terjadi serangkaian gempa susulan (*aftershocks*) yang dapat berlangsung berhari-hari hingga berminggu-minggu. Banyak korban jiwa dan kerusakan tambahan justru disebabkan oleh gempa susulan yang merobohkan bangunan yang sudah retak — pola ini terdokumentasi di kasus gempa Lombok 2018, Palu 2018, dan Cianjur 2022.

Saat ini, BPBD dan tim SAR mengandalkan intuisi senior atau aturan empiris (Omori's Law klasik) yang **belum disesuaikan untuk karakteristik geologi Indonesia per zona seismik**. Tidak ada API publik yang menyediakan probabilitas gempa susulan berbasis kombinasi sekuens seismik dan karakteristik zona.

### 1.2. Domain

| Aspek | Detail |
|---|---|
| **Domain utama** | Mitigasi bencana geofisika (*disaster response & earth sciences*) |
| **Sub-domain** | Operasional darurat BPBD, perencanaan SAR, asuransi |
| **Stakeholder** | BPBD provinsi/kabupaten, BNPB, media massa, aplikasi kesiapsiagaan warga, perusahaan asuransi |

### 1.3. Machine Learning Task

| Item | Detail |
|---|---|
| **Tipe task** | **Binary Classification** (klasifikasi biner) |
| **Pertanyaan utama** | *"Setelah gempa M ≥ 5.0, apakah dalam 24 jam ke depan akan ada susulan M ≥ 4.0 dalam radius 100 km?"* |
| **Output** | Probabilitas kelas positif (0.0 – 1.0) yang di-update tiap jam selama 7 hari pasca mainshock |
| **Model terpilih** | **XGBoost** (single framework — sesuai instruksi "pilih satu saja") |
| **Baseline pembanding** | Omori's Law klasik ($n(t) = K/(t+c)^p$) |

Mengapa XGBoost? Karena data bersifat tabular dengan ~10 fitur numerik + 1 categorical, class imbalance moderat (~25% positif), dan XGBoost terbukti dominan di Kaggle untuk skenario ini. Dibandingkan random forest, XGBoost punya `scale_pos_weight` built-in dan early stopping yang lebih elegant.

---

## 2. Analisis Karakteristik Data Bergerak

### 2.1. Sumber Data (Streaming, Public APIs)

Data berasal dari **dua sumber publik gratis tanpa API key**:

| Sumber | Endpoint | Karakteristik |
|---|---|---|
| **USGS Earthquake API** | `https://earthquake.usgs.gov/fdsnws/event/1/query` | Global, GeoJSON, near-real-time (lag 5–15 menit), public domain |
| **BMKG Open Data** | `https://data.bmkg.go.id/DataMKG/TEWS/gempaterkini.json` + `gempadirasakan.json` | Indonesia-spesifik, JSON, sangat cepat untuk gempa lokal, plus intensitas MMI per kota |

Pemakaian dua sumber memberikan **resiliensi** (jika satu down, satu lagi tetap aktif) dan **cross-verification** (gempa yang sama dilaporkan dua sumber dapat dibandingkan magnitudonya).

### 2.2. Modalitas & Tipe Data

Data adalah **tabular numerik + metadata teks**, time-stamped per event. Bukan gambar, bukan teks panjang, bukan audio. Volume per event sangat kecil (~1 KB JSON), namun frekuensinya **terus mengalir** (streaming, event-driven).

### 2.3. Cara Data Berubah / Bertambah

| Dimensi | Bagaimana berubah |
|---|---|
| **Volume** | ~16 event baru per hari rata-rata (~6.000/tahun) |
| **Velocity** | Bervariasi: 0 event/jam di kondisi tenang, sampai 50+ event/jam selama aftershock sequence besar |
| **Distribusi magnitude** | Shift saat ada sequence besar (banyak event kecil-menengah dalam waktu singkat) |
| **Distribusi spasial** | Shift saat aktivitas tektonik bergeser antar zona (Jawa vs Sumatra vs Sulawesi) |
| **Coverage temporal** | Dataset historis ~10 tahun (2015–2025) untuk training awal, terus bertambah ke depan |

### 2.4. Tipe Drift yang Diantisipasi

| Tipe drift | Contoh skenario di GempaWas |
|---|---|
| **Covariate drift** — P(X) berubah | Sequence aftershock besar menggeser distribusi magnitude & rate event |
| **Prior probability drift** — P(y) berubah | Periode tenang vs periode aktif → proporsi label positif berubah |
| **Concept drift** — P(y &#124; X) berubah | Karakteristik fisika zona kerak berubah perlahan (jangka tahunan); juga peningkatan kualitas sensor BMKG bisa mengubah completeness magnitude |

Karena P(y&nbsp;|&nbsp;X) **secara fisika relatif stabil** (Omori's Law berlaku universal), drift utama yang harus ditangani adalah **covariate drift** akibat sequence events. Inilah yang akan dimonitor menggunakan Population Stability Index (PSI).

---

## 3. Strategi Continual Learning (CL) / Continuous Training (CT)

Sistem GempaWas mengadopsi pendekatan **Continuous Training berbasis hybrid trigger**: kombinasi *scheduled retraining* dan *drift-triggered retraining*.

### 3.1. Mekanisme Pengambilan Data Berkala

| Aktivitas | Frekuensi | Implementasi |
|---|---|---|
| **Hourly ingestion** | Setiap jam (menit ke-5) | GitHub Actions cron `5 * * * *` menjalankan `src/ingestion.py` |
| **Backfill historis** | Sekali di awal, on-demand | Manual run dengan `--hours 720` untuk 30 hari |
| **Weekly snapshot DVC** | Setiap Senin pagi | `dvc add data/processed/features.parquet` lalu commit & tag `data-vX.Y` |

### 3.2. Pemicu (Trigger) Retraining Otomatis

Sistem menggunakan **3 jenis trigger** yang bersifat komplementer:

| Trigger | Kondisi | Aksi |
|---|---|---|
| **Scheduled (mingguan)** | Tiap Senin 03:00 UTC | Retrain tanpa kondisi, untuk safety net |
| **Drift-based** | PSI feature `magnitude` ≥ 0.1 ATAU PSI `depth` ≥ 0.1 | Trigger workflow `weekly_retrain.yml` dengan `force_retrain=true` |
| **Performance-based** | F1-score 30-hari sliding window turun ≥ 10% dari champion | Sama: trigger retrain + alert ke Slack/email (kalau di-setup) |
| **Code-based (LK-08)** | Push ke `main` yang mengubah `src/train.py` atau `src/features.py` | "Code as Trigger" — full automation chain |

### 3.3. Alur CT End-to-End

```
   [hourly cron] ─► ingestion ─► SQLite ─► quick drift check
                                                  │
                                                  ▼ PSI ≥ 0.1?
                                              ─── ya ──► trigger weekly_retrain
                                              ─── tidak ► continue

   [weekly cron] ─► features.py ─► train.py ─► MLflow run
                                                  │
                                                  ▼ PR-AUC ≥ champion + 2%?
                                              ─── ya ──► register sebagai Production
                                              ─── tidak ► tetap pakai champion lama
                                                          (rollback policy)
```

### 3.4. Reproducibility & Lineage

Setiap model production memiliki **lineage trail** yang dapat di-trace:

1. **Code version**: Git commit SHA
2. **Data version**: DVC pointer hash (file `*.dvc`)
3. **Experiment params**: MLflow run ID
4. **Model artifact**: MLflow Model Registry version number
5. **Tag rilis**: Git tag `v0.X.Y` di `main`

Audit "model XYZ dilatih kapan dengan data apa" cukup query MLflow Registry → ikuti tag → cocokkan dengan DVC pointer → reproduce dengan `dvc checkout` + `git checkout`.

---

## 4. Penetapan Kriteria Keberhasilan

### 4.1. Metrik Teknis (ML)

| Metrik | Target | Justifikasi |
|---|---|---|
| **Primary: F1-score (kelas positif)** | ≥ **0.65** di test set | Class imbalance → F1 lebih informatif dari accuracy |
| **Secondary: PR-AUC** | ≥ **0.72** | Robust untuk imbalanced binary classification |
| **Recall@Precision=0.7** | ≥ 0.60 | BPBD lebih takut miss (false negative) daripada false alarm — tapi tetap precision dijaga supaya tidak cry-wolf |
| **Beating baseline** | XGBoost F1 > Omori-baseline F1 + 0.10 | Wajib mengalahkan baseline fisika murni |

### 4.2. Metrik Operasional (MLOps)

| Metrik | Target | Tools |
|---|---|---|
| **Pipeline uptime ingestion** | ≥ 99% (per bulan) | GitHub Actions success rate |
| **Latency inference** | < 200 ms p95 | FastAPI + lazy model loading |
| **Drift detection latency** | < 1 jam setelah event | Quick check setelah tiap ingestion |
| **Retraining cycle time** | < 30 menit | GitHub Actions runner |
| **Model rollback time** | < 5 menit | MLflow Registry stage transition |
| **Coverage tests** | ≥ 80% (`src/`) | pytest --cov |

### 4.3. Metrik Bisnis / Produk

| Metrik | Target | Mengapa penting |
|---|---|---|
| **Coverage geografis** | 5 zona seismik utama (Jawa, Sumatra, Sulawesi, Maluku, Papua) | Indonesia luas, model harus generalize |
| **Update frequency API** | Update probabilitas tiap jam | Lebih cepat dari laporan manual BMKG |
| **Adoption** | Dipakai minimal 1 BPBD pilot dalam 6 bulan | Bukti relevansi nyata |
| **Time-to-warning** | < 1 jam dari kejadian mainshock | Window kritis untuk evakuasi bangunan retak |
| **False alarm rate** | < 30% di precision 0.7 | Cry-wolf effect menurunkan kepercayaan stakeholder |
| **Efisiensi operasional** | Mengurangi waktu intuisi manual BPBD dari ~2 jam → < 5 menit per assessment | ROI utama dari segi tenaga kerja |

### 4.4. Definition of Done untuk Production

Sebuah model dianggap **siap production** jika dan hanya jika:

1. F1 ≥ 0.65 di test set time-based (bukan random split)
2. PR-AUC ≥ 0.72
3. Mengalahkan Omori baseline minimal +0.10 F1
4. Latency p95 < 200 ms saat di-load dari MLflow Registry
5. Semua unit test pass (`pytest`)
6. Tidak ada data leakage (validated via feature audit)
7. Lineage lengkap: Git SHA + DVC pointer + MLflow run ID + tag rilis

---

## 5. Perancangan Diagram Arsitektur Pipeline

Diagram lengkap (7 layer: Source → Ingestion → Storage → Feature Engineering → Training → Serving → Users) ada di file terlampir:

- **PNG:** `docs/figures/lk03_arsitektur_etl.png` (gambar di halaman berikutnya)
- **SVG:** `docs/figures/lk03_arsitektur_etl.svg` (sumber editable)
- **Mermaid:** `docs/figures/lk03_arsitektur_etl.mmd` (sumber text)

Cuplikan struktur layer:

| Layer | Komponen | Tool/Tech |
|---|---|---|
| **1. Sources** | USGS API, BMKG API | HTTP / public APIs |
| **2. Ingestion** | scheduled cron, ingestion.py, preprocess.py, quick drift | GitHub Actions, Python `requests` |
| **3. Storage** | raw JSON, SQLite operational, interim parquet, processed features | SQLite, Parquet, **DVC versioning** |
| **4. Feature Engineering** | identifikasi mainshock, rolling features, label compute | pandas, numpy |
| **5. Training** | weekly cron, train.py, MLflow tracking, Model Registry | **XGBoost**, **MLflow** |
| **6. Serving** | FastAPI inference, Streamlit dashboard, Docker container | **FastAPI**, **Streamlit**, **Docker** |
| **7. Users** | BPBD, BNPB, media, app warga, asuransi | REST API + UI |

**Continual Learning feedback loop** (panah merah putus-putus di diagram) menghubungkan layer 2 (Quick Drift Check) langsung ke layer 5 (Training) — saat PSI melewati threshold, retraining ter-trigger tanpa intervensi manual.

---

## 6. Asumsi, Batasan, dan Risiko

### 6.1. Asumsi

1. Data BMKG & USGS akan terus tersedia gratis tanpa API key sepanjang proyek
2. Definisi "gempa besar" mengikuti standar BMKG (M ≥ 5.0)
3. Definisi "susulan signifikan" adalah M ≥ 4.0 dalam radius 100 km dan 24 jam setelah mainshock
4. GitHub Actions free tier mencukupi untuk hourly cron (2.000 menit/bulan untuk repo public, free)

### 6.2. Batasan

1. Tidak ada akses ke data sensor seismograf mentah (waveform), hanya katalog gempa
2. Tidak ada budget untuk hosting eksternal — semua di Codespaces + GitHub Actions
3. Dataset historis terbatas ~10 tahun data berkualitas dari BMKG
4. Single training framework (XGBoost) sesuai instruksi tugas

### 6.3. Risiko & Mitigasi

| Risiko | Probabilitas | Dampak | Mitigasi |
|---|---|---|---|
| BMKG API down/berubah format | Sedang | Sedang | USGS sebagai fallback otomatis di `ingestion.py` |
| Class imbalance ekstrem | Tinggi | Sedang | `scale_pos_weight` XGBoost + fokus PR-AUC, bukan accuracy |
| Sekuens jarang di test set | Sedang | Tinggi | Time-based train/test split, bukan random |
| Model overfit ke zona Jawa (data terbanyak) | Sedang | Sedang | Stratified sampling per zona + separate evaluation per zona |
| Tidak ada gempa besar saat demo | Tinggi | Rendah | Replay demo pakai data historis Cianjur 2022 |
| GitHub Actions quota habis | Rendah | Tinggi | Optimasi run time, cache pip, batas timeout 10 menit |

---

## 7. Roadmap Eksekusi (Mapping ke LK)

| LK | Fokus | Estimasi |
|---|---|---|
| **LK-01** (selesai) | Inisiasi, problem framing, arch design | 1–2 hari |
| **LK-02** | GitHub setup + Codespaces + cookiecutter | 1–2 hari |
| **LK-03** | Desain ETL detail + diagram | 2–3 hari |
| LK-04 | Implementasi `ingest_data.py` + `preprocess.py` | 3–5 hari |
| LK-05 | DVC tracking + simulasi continual learning | 2–3 hari |
| LK-06 | `train.py` + MLflow tracking + 3+ runs | 3–4 hari |
| LK-07 | MLflow Model Registry + stage transition | 2–3 hari |
| LK-08 | GitHub Actions automation chain | 3–4 hari |
| **TOTAL** | | **~3–4 minggu kerja** |

---

## 8. Kesimpulan Inisiasi

GempaWas adalah proyek MLOps yang **memenuhi semua syarat data dinamis**: ingestion otomatis dari API publik yang terus mengalirkan data baru tiap jam, dengan dua trigger continual training (scheduled + drift-based) untuk memastikan model tetap relevan terhadap perubahan karakteristik seismik Indonesia.

Pemilihan domain mitigasi gempa bukan sekadar untuk memenuhi tugas akademis — tapi punya **pain point nyata** (banyak korban Cianjur 2022 jatuh karena susulan), **label paling bersih** (gempa terjadi atau tidak, diverifikasi sensor), dan **baseline fisika defensible** (Omori's Law) sebagai pembanding ML.

Tools MLOps yang akan dipakai (cookiecutter, GitHub Codespaces, GitHub Actions, DVC, MLflow, Streamlit, Docker, XGBoost) sudah dipetakan secara eksplisit ke tiap LK berikutnya. Sistem ini dirancang dari awal untuk **production-grade**, bukan eksperimen one-shot.

---

**Lampiran:** Diagram arsitektur (`lk03_arsitektur_etl.png`)
