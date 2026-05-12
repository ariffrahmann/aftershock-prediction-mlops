# Branching Strategy — GitHub Flow untuk GempaWas

Proyek ini mengadopsi **GitHub Flow**, model branching yang sederhana namun cocok untuk proyek MLOps individu/kecil. Berbeda dengan GitFlow yang punya `develop`, `release`, `hotfix` (terlalu kompleks untuk skala mahasiswa), GitHub Flow cuma butuh dua hal: `main` dan **feature branch**.

---

## 1. Prinsip Dasar

1. **`main` selalu deployable.** Apapun yang ada di `main` harus bisa dijalankan dan modelnya bisa di-load. Tidak ada commit langsung ke `main`.
2. **Setiap perubahan lahir dari branch baru.** Branch dibuat dari `main`, dikerjakan, lalu di-merge balik via Pull Request.
3. **PR adalah tempat review.** Bahkan kalau kerjanya sendirian — review diri sendiri (cold review setelah jeda 30 menit) atau minta teman/dosen review.
4. **CI hijau dulu, baru merge.** Pipeline GitHub Actions (`pytest`, `lint`) harus lulus sebelum tombol merge bisa diklik.
5. **Hapus branch setelah merge.** Untuk menjaga repo bersih.

---

## 2. Konvensi Nama Branch

Format: `<type>/<deskripsi-singkat-pakai-dash>`

| Prefix | Untuk apa | Contoh |
|---|---|---|
| `feat/` | Fitur baru | `feat/initial-eda`, `feat/xgboost-tuning` |
| `fix/` | Bug fix | `fix/ingestion-bmkg-timeout` |
| `docs/` | Dokumentasi saja | `docs/lk01-pdf`, `docs/api-examples` |
| `chore/` | Maintenance | `chore/bump-mlflow-2.10` |
| `experiment/` | Eksperimen ML | `experiment/smote-balancing` |
| `refactor/` | Restructure tanpa ubah behavior | `refactor/features-modular` |
| `ci/` | GitHub Actions / CI | `ci/add-coverage-job` |

**Aturan tambahan:**
- Lowercase, gunakan `-` sebagai separator, jangan pakai underscore
- Maksimum 50 karakter total
- Hindari nama generic seperti `feat/update`, `fix/bug` — selalu spesifik

---

## 3. Alur Kerja Standar

```bash
# 1. Pastikan main up-to-date
git checkout main
git pull origin main

# 2. Buat branch baru
git checkout -b feat/initial-eda

# 3. Kerja, commit kecil-kecil dengan pesan jelas
git add notebooks/01_eda.ipynb
git commit -m "feat(eda): tambah analisis distribusi magnitude per zona"

# (commit lain di branch ini...)

# 4. Push ke remote
git push -u origin feat/initial-eda

# 5. Buka Pull Request di GitHub UI
#    - Title: "feat(eda): Initial EDA notebook untuk LK-02"
#    - Body: jelaskan apa, kenapa, hasil utama
#    - Link ke issue/task kalau ada

# 6. Tunggu CI hijau (pytest, lint, dll)

# 7. Self-review atau minta review
#    - Diff dibaca ulang line by line
#    - Cek apakah ada print/debug yang ketinggalan
#    - Cek apakah notebook output sudah di-strip (nbstripout)

# 8. Squash & merge ke main (default di repo ini)

# 9. Hapus branch
git checkout main
git pull origin main
git branch -d feat/initial-eda
git push origin --delete feat/initial-eda
```

---

## 4. Conventional Commits

Pesan commit mengikuti **Conventional Commits** supaya history terbaca dan changelog bisa di-generate otomatis:

```
<type>(<scope>): <subject>

[body opsional]
[footer opsional]
```

| Type | Kapan dipakai |
|---|---|
| `feat` | Fitur baru |
| `fix` | Bug fix |
| `docs` | Dokumentasi saja |
| `style` | Formatting (tidak ubah logic) |
| `refactor` | Restructure code |
| `test` | Tambah/ubah test |
| `chore` | Tooling, bump version, dll |
| `ci` | GitHub Actions config |
| `perf` | Performance improvement |

**Contoh baik:**
```
feat(ingestion): fallback ke USGS kalau BMKG timeout >30s

BMKG endpoint sering 504 di jam padat. USGS lebih reliable
untuk Indonesia jadi cocok sebagai fallback. Re-test dengan
3 jam data backlog menunjukkan 0 missing events.

Refs LK-04
```

**Contoh buruk:**
```
update stuff
fix bug
asdf
```

---

## 5. Pull Request Template (Wajib LK-02)

Setiap PR ke `main` minimal harus menjawab:

```markdown
## What
Singkat: apa yang diubah di PR ini?

## Why
Konteks: kenapa perlu diubah? (link ke LK / issue / drift report)

## How
Penjelasan teknis singkat tentang pendekatan yang dipilih.

## Test Plan
- [ ] pytest pass: `make test`
- [ ] lint pass: `make lint`
- [ ] Manual smoke test: `make ingest && make features`
- [ ] (Untuk ML PR) MLflow run logged dengan metric ≥ baseline

## Screenshot / Bukti
(Kalau ada UI / plot / log relevan)

## Checklist
- [ ] Branch nama mengikuti konvensi `<type>/<deskripsi>`
- [ ] Commit messages conventional
- [ ] No data file >2MB di-commit ke Git (pakai DVC)
- [ ] No secret/API key di-commit
- [ ] README/docs diperbarui kalau ada perubahan public interface
```

---

## 6. Aturan Khusus untuk Branch ML / Eksperimen

Karena ini proyek MLOps, eksperimen yang menghasilkan model HARUS punya bukti reproducibility:

1. **MLflow run ID di-commit message:** `feat(train): xgboost depth=8 — mlflow_run=a1b2c3d4`
2. **DVC pointer di-update:** kalau dataset berubah, jalankan `dvc add data/processed/features.parquet` dan commit `.dvc` filenya
3. **Metric dilaporkan di PR body:** F1, PR-AUC, ROC-AUC + perbandingan dengan baseline
4. **Tidak boleh merge ke main kalau metric < baseline production** (kecuali untuk eksplorasi terbatas yang jelas-jelas ditandai `experiment/`)

---

## 7. Yang Harus Dilakukan untuk LK-02 (Demo Branching)

Demo wajib untuk LK-02:

```bash
# 1. Pastikan di branch main yang clean
git checkout main && git pull

# 2. Buat branch eksperimen awal — INI YANG DIWAJIBKAN OLEH LK-02
git checkout -b feat/initial-eda

# 3. Tambahkan notebook EDA pertama
mkdir -p notebooks
cat > notebooks/01_eda.ipynb << 'NB'
# (buat di Jupyter, isi minimal: load data, plot distribusi magnitude, save plot ke reports/figures)
NB

# 4. Commit
git add notebooks/01_eda.ipynb
git commit -m "feat(eda): notebook 01 — distribusi magnitude & lokasi mainshock"

# 5. Push & buka PR
git push -u origin feat/initial-eda
gh pr create --base main --title "feat(eda): Initial EDA" \
  --body "Notebook EDA awal — sesuai requirement LK-02 #4"

# 6. Tunggu CI, lalu merge
gh pr merge --squash --delete-branch
```

Screenshot dari step #5 dan #6 (atau output `gh pr view`) bisa dipakai sebagai bukti pengerjaan LK-02 nomor 4.

---

## 8. FAQ

**Q: Boleh commit langsung ke `main`?**
A: Tidak. Bahkan untuk typo. Setiap perubahan via PR.

**Q: Berapa lama branch boleh hidup?**
A: Targetnya < 1 minggu. Branch yang hidup lama akan susah di-merge dan rentan konflik.

**Q: Kalau eksperimen gagal, branch-nya diapain?**
A: Tutup PR tanpa merge, hapus branch. Tapi sebelum hapus, **tag commit terakhir** kalau eksperimen itu insightful: `git tag experiment/smote-failed-2026-05-12` lalu `git push --tags`.

**Q: Bagaimana kalau dua eksperimen jalan paralel?**
A: Tidak masalah. Dua branch dari `main`, dikerjakan paralel. Yang lebih dulu siap merge duluan, yang kedua rebase dari `main` baru lanjut.

**Q: Tag rilis?**
A: Setiap kali model production di-promote (LK-07), tag rilis `v0.1.0`, `v0.2.0`, dst. di `main`.
