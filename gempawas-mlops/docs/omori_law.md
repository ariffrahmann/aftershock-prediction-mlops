# Omori's Law — Fondasi Fisika untuk GempaWas

## Rumus

$$n(t) = \frac{K}{(t + c)^p}$$

Atau dalam notasi kode:
```python
n(t) = K / (t + c) ** p
```

## Definisi Variabel

| Simbol | Nama | Tipikal nilai |
|---|---|---|
| `n(t)` | Jumlah susulan per hari | output |
| `t` | Hari sejak gempa utama | input (0 hingga 30+) |
| `K` | Konstanta skala | bergantung magnitudo mainshock; ~5–50 |
| `c` | Konstanta peluruhan dini | 0.01 – 0.1 (mencegah pembagian dengan nol) |
| `p` | Eksponen decay rate | 0.8 – 1.2 (biasanya ~1.0) |

## Intuisi

Setelah gempa besar, kerak bumi mengalami stress release secara bertahap. Pelepasan ini menghasilkan banyak gempa kecil di hari-hari pertama, lalu makin jarang seiring waktu. Omori (1894) mengamati pola ini di gempa Mino-Owari Jepang 1891 dan memformalkannya menjadi rumus di atas.

**Versi sederhana:** "Susulan banyak di awal, makin jarang ke depan."

## Mengapa Penting untuk GempaWas

### 1. Justifikasi Fitur

Rumus Omori adalah alasan mengapa fitur-fitur berikut masuk akal secara fisika:

- `jam_sejak_mainshock` — variabel `t` langsung
- `count_susulan_1jam`, `count_susulan_6jam` — empirical observation dari `n(t)`
- `omori_rate_est` — implementasi langsung dari rumus

Tanpa pemahaman ini, fitur-fitur tersebut hanya angka random. Dengan pemahaman ini, dosen akan mengerti kenapa kamu memilihnya.

### 2. Baseline yang Defensible

Sebelum kamu menyentuh ML, kamu sudah punya baseline yang lebih baik dari random:

```python
def omori_baseline_predict(jam_sejak_mainshock):
    rate = OMORI_K / ((jam_sejak_mainshock/24 + OMORI_C) ** OMORI_P)
    # Probabilitas paling tidak satu susulan ≥ M4.0 dalam 24 jam
    # = 1 - exp(-rate * 24jam * fraction_above_M4)
    return 1 - np.exp(-rate * 0.15)  # 15% asumsi proporsi susulan ≥ M4
```

Lalu Random Forest dan XGBoost kamu harus mengalahkan ini.

### 3. Interpretability untuk Presentasi

Ketika dosen tanya "Kenapa model kamu memprediksi probabilitas 0.7?", kamu bisa jawab:
- "Karena Omori's Law memang memprediksi rate tinggi di t=2 jam"
- "Tapi model juga mempertimbangkan zona sesar dan magnitudo mainshock"
- "Lihat SHAP values di slide ini..."

Itu adalah jawaban yang membedakan kamu dari mahasiswa lain yang sekadar "saya pakai XGBoost".

## Variasi Modifikasi (Untuk Eksplorasi Lanjutan)

Omori-Utsu Law (1961) — versi paling sering dipakai sekarang:
$$n(t) = \frac{K}{(t + c)^p}$$
Dengan p tidak harus 1.0, biasanya 0.8–1.2 tergantung tipe sesar.

Reasenberg-Jones (1989) — versi yang menggabungkan magnitudo:
$$\lambda(t, M) = 10^{a + b(M_m - M)} \cdot (t + c)^{-p}$$
Di mana `M_m` adalah magnitudo mainshock dan `M` adalah threshold magnitudo susulan yang dilihat. Ini lebih canggih, tapi untuk MVP kita tetap pakai Omori klasik.

## Implementasi di GempaWas

Lihat `src/features.py` fungsi `omori_rate()`:

```python
def omori_rate(t_hours, K=10.0, c=0.1, p=1.0):
    t_days = max(t_hours / 24.0, 0.001)
    return K / ((t_days + c) ** p)
```

Nilai default `K=10, c=0.1, p=1.0` adalah pilihan generik. Untuk versi lebih akurat, parameter ini bisa di-fit per zona sesar Indonesia menggunakan maximum likelihood estimation pada data historis BMKG. Ini bisa jadi extension menarik untuk laporan akhir.

## Untuk Slide Presentasi

Satu kalimat yang harus muncul di slide:

> *"Model GempaWas tidak menggantikan Omori's Law — model ini memperkaya prediksi Omori dengan fitur sekuens dan karakteristik geologi spesifik Indonesia yang tidak ditangkap oleh rumus original."*

Itu framing yang membuat proyekmu terdengar matang secara saintifik, bukan hanya "saya pakai ML untuk semuanya".
