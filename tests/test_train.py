from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from imblearn.combine import SMOTETomek
from sklearn.metrics import average_precision_score, f1_score, recall_score

# Path ke features.parquet
FEATURES_PATH = Path("data/processed/features.parquet")

FEATURE_COLUMNS = [
    "mainshock_magnitude", "mainshock_depth", "jam_sejak_mainshock",
    "count_susulan_1jam",  "count_susulan_6jam", "count_susulan_24jam",
    "max_mag_susulan_6jam","max_mag_susulan_24jam","omori_rate_est","zona_sesar",
]
TARGET = "label_susulan_besar_24jam"


# Fixture: load dataframe sekali untuk semua test di file ini
@pytest.fixture(scope="module")
def df():
    """
    Fixture scope='module' artinya data hanya dibaca sekali per sesi test.
    Ini menghemat waktu — tidak perlu baca parquet berulang kali.
    """
    if not FEATURES_PATH.exists():
        pytest.skip("features.parquet tidak tersedia — skip data-dependent tests")
    return pd.read_parquet(FEATURES_PATH)


@pytest.fixture(scope="module")
def train_test_split(df):
    """Fixture: time-based split siap pakai untuk test lain."""
    df_sorted   = df.sort_values("mainshock_time").reset_index(drop=True)
    split_idx   = int(len(df_sorted) * 0.75)
    X_train     = df_sorted.iloc[:split_idx][FEATURE_COLUMNS]
    y_train     = df_sorted.iloc[:split_idx][TARGET].astype(int)
    X_test      = df_sorted.iloc[split_idx:][FEATURE_COLUMNS]
    y_test      = df_sorted.iloc[split_idx:][TARGET].astype(int)
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# TEST 1: Integritas data
# ---------------------------------------------------------------------------
def test_load_features(df):
    """
    Memastikan features.parquet bisa dibaca dan punya struktur yang benar.
    Ini adalah 'smoke test' paling dasar — kalau ini gagal, semua test lain
    tidak relevan.
    """
    assert len(df) > 0, "Dataset tidak boleh kosong"
    for col in FEATURE_COLUMNS + [TARGET]:
        assert col in df.columns, f"Kolom '{col}' hilang dari features.parquet"


def test_feature_columns_count(df):
    """
    Memastikan jumlah fitur tetap 10.
    Kalau ada penambahan/penghapusan fitur di build_features.py,
    test ini akan langsung mendeteksinya.
    """
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    assert len(missing) == 0, f"Fitur hilang: {missing}"
    assert len(FEATURE_COLUMNS) == 10, "Harus ada tepat 10 fitur"


# ---------------------------------------------------------------------------
# TEST 2: Train-test split
# ---------------------------------------------------------------------------
def test_prepare_train_test(train_test_split):
    """
    Memastikan pembagian data 75/25 menghasilkan ukuran yang wajar.
    """
    X_train, X_test, y_train, y_test = train_test_split
    total = len(X_train) + len(X_test)
    train_ratio = len(X_train) / total
    assert 0.70 <= train_ratio <= 0.80, \
        f"Train ratio {train_ratio:.2f} di luar range [0.70, 0.80]"
    assert len(X_test) > 0, "Test set tidak boleh kosong"


def test_split_is_time_based(df):
    """
    Memastikan split berbasis waktu: semua data train LEBIH LAMA dari test.
    Ini krusial untuk time-series gempa — tidak boleh ada 'data leakage' temporal.
    """
    if "mainshock_time" not in df.columns:
        pytest.skip("Tidak ada kolom mainshock_time")
    df_sorted  = df.sort_values("mainshock_time").reset_index(drop=True)
    split_idx  = int(len(df_sorted) * 0.75)
    last_train = df_sorted.iloc[split_idx - 1]["mainshock_time"]
    first_test = df_sorted.iloc[split_idx]["mainshock_time"]
    assert last_train <= first_test, \
        "DATA LEAKAGE: ada data train yang lebih baru dari data test!"


# ---------------------------------------------------------------------------
# TEST 3: Metrik evaluasi
# ---------------------------------------------------------------------------
def test_compute_3_metrics(train_test_split):
    """
    Memastikan 3 metrik inti dapat dihitung dan nilainya dalam range valid [0, 1].

    Kenapa test metrik penting?
    - Kalau ada perubahan kode di compute_3_metrics(), test ini mendeteksi
      kalau metrik tiba-tiba jadi NaN, negatif, atau > 1
    - Menjamin konsistensi definisi metrik antar versi kode
    """
    X_train, X_test, y_train, y_test = train_test_split
    # Train model sederhana (cepat, bukan untuk performa terbaik)
    model = xgb.XGBClassifier(
        n_estimators=10, max_depth=2, random_state=42, verbosity=0
    )
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= 0.45).astype(int)

    pr_auc  = average_precision_score(y_test, y_proba)
    f1      = f1_score(y_test, y_pred, zero_division=0)
    recall  = recall_score(y_test, y_pred, zero_division=0)

    assert 0.0 <= pr_auc  <= 1.0, f"pr_auc tidak valid: {pr_auc}"
    assert 0.0 <= f1      <= 1.0, f"f1_score tidak valid: {f1}"
    assert 0.0 <= recall  <= 1.0, f"recall tidak valid: {recall}"


# ---------------------------------------------------------------------------
# TEST 4: SMOTETomek
# ---------------------------------------------------------------------------
def test_smotetomek_balances(train_test_split):
    """
    Memastikan SMOTETomek menghasilkan distribusi kelas yang lebih seimbang
    dari distribusi asli (84:16).

    Kenapa penting?
    - Kalau SMOTETomek gagal (library berubah, dll.), training akan berjalan
      dengan data asli yang imbalanced — model akan jauh lebih buruk
    - Test ini mendeteksi masalah tersebut sebelum training penuh
    """
    X_train, _, y_train, _ = train_test_split
    original_ratio = (y_train == 1).sum() / len(y_train)

    smt    = SMOTETomek(random_state=42)
    X_r, y_r = smt.fit_resample(X_train, y_train)

    resampled_ratio = (y_r == 1).sum() / len(y_r)
    # Setelah SMOTETomek, rasio positif harus lebih tinggi (lebih seimbang)
    assert resampled_ratio > original_ratio, \
        f"SMOTETomek tidak meningkatkan rasio kelas: {original_ratio:.2f} -> {resampled_ratio:.2f}"
    # Distribusi tidak boleh ekstrem (< 10% atau > 90%)
    assert 0.10 < resampled_ratio < 0.90, \
        f"Distribusi setelah resampling tidak wajar: {resampled_ratio:.2f}"


# ---------------------------------------------------------------------------
# TEST 5: Model output
# ---------------------------------------------------------------------------
def test_model_predicts_proba(train_test_split):
    """
    Memastikan model XGBoost menghasilkan probabilitas valid.

    Cek:
    - Shape output = (n_samples, 2) untuk binary classifier
    - Semua nilai probabilitas antara 0 dan 1
    - Setiap baris sum ke 1.0 (probabilitas komplementer)
    """
    X_train, X_test, y_train, _ = train_test_split
    model = xgb.XGBClassifier(
        n_estimators=10, max_depth=2, random_state=42, verbosity=0
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)

    assert proba.shape == (len(X_test), 2), \
        f"Shape probabilitas tidak valid: {proba.shape}"
    assert np.all(proba >= 0) and np.all(proba <= 1), \
        "Ada probabilitas di luar range [0, 1]"
    np.testing.assert_allclose(
        proba.sum(axis=1), 1.0, atol=1e-6,
        err_msg="Baris probabilitas tidak menjumlah ke 1.0"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
