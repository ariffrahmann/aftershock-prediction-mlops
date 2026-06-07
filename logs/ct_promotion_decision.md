# Keputusan Evaluasi Komparatif Champion vs Challenger (LK-12)

**Hasil:** 🟢 CHALLENGER DIPROMOSIKAN KE PRODUCTION
_Waktu: 2026-06-06T23:00:21.886816+00:00_

## Perbandingan Metrik

| Metrik | Champion (lama) | Challenger (baru) | Δ |
|--------|----------------|-------------------|---|
| pr_auc | 0.3061 | 0.5829 | +0.2767 |
| f1_score | 0.5428 | 0.5736 | +0.0309 |
| recall | 1.0000 | 0.8043 | -0.1957 |

Challenger version: **v2**

## Aturan Promosi
- Kriteria utama : Δ PR-AUC >= +0.02
- Guardrail      : recall tidak turun > 0.02
- Override safety : champion recall < 0.4 & challenger recall lebih tinggi

## Penalaran Keputusan

- Δ PR-AUC = +0.2767 (challenger 0.5829 vs champion 0.3061); butuh >= +0.02
- Δ Recall = -0.1957 (challenger 0.8043 vs champion 1.0000); toleransi -0.02
- ⚠ OVERRIDE DECAY: champion PR-AUC 0.3061 < lantai 0.45 (champion telah decay & recall-nya menyesatkan). Challenger PR-AUC jauh lebih baik & recall 0.8043 >= floor 0.4 → PROMOTE.