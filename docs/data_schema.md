# Skema Data & Fitur GempaWas

## Sumber Data

### 1. USGS Earthquake API (Sumber Utama)

**Endpoint:** `https://earthquake.usgs.gov/fdsnws/event/1/query`

**Akses:** Gratis, tanpa API key, tanpa rate limit ketat

**Bounding box Indonesia:**
```
minlatitude:  -11.0
maxlatitude:   6.0
minlongitude: 95.0
maxlongitude: 141.0
```

**Format response:** GeoJSON

**Contoh request:**
```
https://earthquake.usgs.gov/fdsnws/event/1/query?
  format=geojson
  &starttime=2025-01-01
  &endtime=2025-01-02
  &minlatitude=-11&maxlatitude=6
  &minlongitude=95&maxlongitude=141
  &minmagnitude=2.5
```

**Field penting dari response:**

| Field | Tipe | Deskripsi |
|---|---|---|
| `features[].id` | string | Unique event ID |
| `features[].properties.time` | int (ms epoch) | Waktu kejadian UTC |
| `features[].properties.mag` | float | Magnitude |
| `features[].properties.magType` | string | Tipe pengukuran magnitude |
| `features[].properties.place` | string | Deskripsi lokasi |
| `features[].properties.mmi` | float | Modified Mercalli Intensity |
| `features[].geometry.coordinates` | [lon, lat, depth] | Posisi episenter |

### 2. BMKG Open API (Sumber Pelengkap)

**Endpoints:**
- `https://data.bmkg.go.id/DataMKG/TEWS/gempaterkini.json` — 15 gempa terbaru
- `https://data.bmkg.go.id/DataMKG/TEWS/gempadirasakan.json` — gempa yang dirasakan warga

**Akses:** Gratis, tanpa API key

**Keunggulan vs USGS:**
- Coverage gempa kecil di Indonesia lebih baik
- Field "Dirasakan" memberikan intensitas MMI per kota
- Update lebih cepat untuk gempa lokal

**Field penting:**

| Field | Deskripsi |
|---|---|
| `DateTime` | Waktu UTC ISO format |
| `Coordinates` | "lat,lon" string |
| `Magnitude` | Magnitude (float as string) |
| `Kedalaman` | Kedalaman + " km" |
| `Wilayah` | Deskripsi area Indonesia |
| `Dirasakan` | Skala MMI per kota (text) |

## Schema Tabel Database

### Tabel `events` (raw data)

| Kolom | Tipe | Deskripsi |
|---|---|---|
| `event_id` | TEXT PK | Unique ID dari source |
| `source` | TEXT | 'USGS' atau 'BMKG' |
| `time_utc` | TEXT (ISO) | Waktu kejadian UTC |
| `latitude` | REAL | Lintang episenter |
| `longitude` | REAL | Bujur episenter |
| `depth_km` | REAL | Kedalaman fokus |
| `magnitude` | REAL | Magnitude |
| `magnitude_type` | TEXT | Mw, Ml, Mb, dll |
| `place` | TEXT | Deskripsi lokasi |
| `felt_intensity` | TEXT | MMI atau Dirasakan text |
| `raw_json` | TEXT | Raw response untuk debugging |
| `ingested_at` | TEXT (ISO) | Kapan data masuk database |

## Fitur untuk Model

Setiap baris training data merepresentasikan satu kombinasi `(mainshock, snapshot_time)`. Snapshot diambil pada `t = 1, 3, 6, 12, 24, 48, 72` jam setelah mainshock.

### Fitur Statis Mainshock

| Fitur | Tipe | Rentang | Deskripsi |
|---|---|---|---|
| `mainshock_magnitude` | float | 5.0 – 9.5 | Magnitudo gempa utama |
| `mainshock_depth` | float | 0 – 700 km | Kedalaman fokus mainshock |
| `zona_sesar` | int | 0 – 4 | Encoded geological zone |

**Encoding `zona_sesar`:**
- `0` = Subduksi Jawa-Sumatra-Lesser Sunda
- `1` = Subduksi Maluku-Banda
- `2` = Sesar aktif Sumatra (Great Sumatran Fault)
- `3` = Sesar Palu-Koro (Sulawesi)
- `4` = Lainnya / Papua

### Fitur Temporal

| Fitur | Tipe | Rentang | Deskripsi |
|---|---|---|---|
| `jam_sejak_mainshock` | float | 0 – 168 | Berapa jam telah berlalu (= variabel t di Omori's Law) |

### Fitur Sekuens (Rolling Counts)

| Fitur | Tipe | Deskripsi |
|---|---|---|
| `count_susulan_1jam` | int | Jumlah susulan dalam 1 jam terakhir |
| `count_susulan_6jam` | int | Jumlah susulan dalam 6 jam terakhir |
| `count_susulan_24jam` | int | Jumlah susulan dalam 24 jam terakhir |
| `max_mag_susulan_6jam` | float | Magnitudo terbesar susulan 6 jam terakhir |
| `max_mag_susulan_24jam` | float | Magnitudo terbesar susulan 24 jam terakhir |

### Fitur Fisika

| Fitur | Tipe | Deskripsi |
|---|---|---|
| `omori_rate_est` | float | Estimasi laju susulan dari Omori's Law: K / (t+c)^p |

## Target Label

| Label | Tipe | Definisi |
|---|---|---|
| `label_susulan_besar_24jam` | int (0 atau 1) | Apakah ada susulan ≥ M4.0 dalam radius 100 km dan 24 jam ke depan? |

### Catatan Penting tentang Label

- **Self-labeling:** Label di-generate otomatis dari data yang sama (event mendatang yang sudah terjadi), tidak perlu anotasi manual
- **Class imbalance:** Ekspektasi sekitar 20–40% sampel positif (tergantung definisi mainshock dan window)
- **No data leakage:** Snapshot pada waktu T hanya pakai data dari sebelum T; label dihitung dari window setelah T

## Data Quality Considerations

1. **Duplikasi USGS vs BMKG:** Banyak gempa Indonesia tercatat di kedua sumber. Dedup berdasarkan timestamp & koordinat dalam toleransi.

2. **Missing depth:** Kadang `depth_km` null untuk gempa kecil. Default ke 10.0 km (rata-rata kedalaman kerak Indonesia).

3. **Magnitude type heterogeneity:** Mw, Ml, mb tidak persis sama. Kalau ada dua nilai untuk event sama, prioritaskan Mw.

4. **Time zone:** Semua disimpan dalam UTC. Konversi ke WIB/WITA/WIT hanya saat display.
