from __future__ import annotations

import numpy as np

# Konstanta Omori's Law (sama dengan build_features.py)
OMORI_K = 10.0
OMORI_C = 0.1
OMORI_P = 1.0


def haversine_km(lat1: float, lon1: float,
                  lat2: float, lon2: float) -> float:
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def classify_zone(latitude: float, longitude: float) -> int:
    if -6 < latitude < 6 and 95 < longitude < 103:
        return 2   # Sesar Sumatra
    if -11 < latitude < -6 and 95 < longitude < 120:
        return 0   # Subduksi Jawa-Sumatra-Bali
    if -4 < latitude < 2 and 119 < longitude < 124:
        return 3   # Sesar Palu-Koro
    if -8 < latitude < 2 and 124 < longitude < 135:
        return 1   # Subduksi Maluku-Banda
    return 4       # Lainnya / Papua


def omori_rate(t_hours: float,
               K: float = OMORI_K,
               c: float = OMORI_C,
               p: float = OMORI_P) -> float:
    t_days = max(t_hours / 24.0, 0.001)   # cegah division by zero
    return K / ((t_days + c) ** p)
