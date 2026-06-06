# Laporan Drift Gate — Continuous Training (LK-12)

**Keputusan:** RETRAINING DIPICU

- Threshold WARNING  : PSI >= 0.1
- Threshold CRITICAL : PSI >= 0.25
- Max PSI            : 9.6291
- Mean PSI           : 2.9132

| Fitur | PSI | Severity |
|-------|-----|----------|
| mainshock_magnitude | 2.4239 | CRITICAL |
| mainshock_depth | 3.0497 | CRITICAL |
| jam_sejak_mainshock | 1.1263 | CRITICAL |
| count_susulan_6jam | 0.0000 | STABLE |
| count_susulan_24jam | 4.0419 | CRITICAL |
| max_mag_susulan_24jam | 0.1216 | WARNING |
| omori_rate_est | 9.6291 | CRITICAL |

## Interpretasi

PSI mengukur seberapa jauh distribusi fitur data terbaru bergeser dari distribusi data acuan (data saat model Production dilatih). Pergeseran besar menandakan model berisiko mengalami *decay* karena melihat pola input yang berbeda dari saat pelatihan.