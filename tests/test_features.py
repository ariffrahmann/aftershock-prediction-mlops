"""Unit tests untuk feature engineering."""

import pandas as pd
import pytest

from src.features import (
    classify_zone,
    haversine_km,
    omori_rate,
)


def test_omori_rate_decay():
    """Susulan rate di hari 1 harus lebih besar dari hari 7."""
    rate_day1 = omori_rate(t_hours=24)
    rate_day7 = omori_rate(t_hours=168)
    assert rate_day1 > rate_day7, \
        "Omori's Law harus monotonically decreasing"


def test_omori_rate_handles_zero():
    """t=0 tidak boleh menyebabkan division by zero."""
    rate = omori_rate(t_hours=0)
    assert rate > 0 and rate != float("inf")


def test_haversine_known_distance():
    """Jakarta ke Bandung sekitar 117 km."""
    jakarta = (-6.2088, 106.8456)
    bandung = (-6.9175, 107.6191)
    dist = haversine_km(*jakarta, *bandung)
    assert 100 < dist < 140, f"Expected ~117 km, got {dist:.1f}"


def test_classify_zone_jawa():
    """Jakarta seharusnya di zona Jawa subduksi (0)."""
    # Tapi koordinat Jakarta di darat, jadi mungkin tidak terklasifikasi
    # sebagai subduksi. Test dengan koordinat selatan Jawa.
    zona = classify_zone(-9.0, 110.0)  # Selatan Jogja
    assert zona == 0


def test_classify_zone_sumatra_fault():
    """Padang ada di Sesar Sumatra."""
    zona = classify_zone(-0.95, 100.35)
    assert zona == 2


def test_classify_zone_palu():
    """Palu seharusnya di zona Palu-Koro."""
    zona = classify_zone(-0.9, 119.9)
    assert zona == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
