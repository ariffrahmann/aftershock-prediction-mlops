# =============================================================================
# GempaWas — Makefile
# =============================================================================
# Common tasks untuk development & operasi MLOps.
# Jalankan `make help` untuk daftar perintah.
# =============================================================================

.DEFAULT_GOAL := help
PYTHON ?= python
PIP ?= pip
VENV ?= .venv
PORT ?= 8000

# Warna untuk terminal output
BLUE := \033[36m
NC := \033[0m

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

.PHONY: help
help:  ## Tampilkan daftar perintah
	@echo "GempaWas Makefile — perintah yang tersedia:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-20s$(NC) %s\n", $$1, $$2}'

.PHONY: venv
venv:  ## Buat virtual environment Python
	$(PYTHON) -m venv $(VENV)
	@echo "Aktifkan dengan: source $(VENV)/bin/activate  (Linux/Mac)"
	@echo "                 $(VENV)\\Scripts\\activate    (Windows)"

.PHONY: install
install:  ## Install dependencies utama
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

.PHONY: install-dev
install-dev: install  ## Install dependencies + dev tools
	$(PIP) install -r requirements-dev.txt
	pre-commit install

# -----------------------------------------------------------------------------
# Data pipeline
# -----------------------------------------------------------------------------

.PHONY: ingest
ingest:  ## Jalankan ingestion sekali (2 jam terakhir)
	$(PYTHON) src/ingestion.py --once --hours 2

.PHONY: backfill
backfill:  ## Backfill 30 hari data terakhir
	$(PYTHON) src/ingestion.py --hours 720

.PHONY: features
features:  ## Bangun training dataset dari raw events
	$(PYTHON) src/features.py

.PHONY: data
data: ingest features  ## Ingestion + feature engineering sekaligus

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

.PHONY: train
train:  ## Latih model (dengan hyperparam tuning)
	$(PYTHON) src/train.py

.PHONY: train-fast
train-fast:  ## Latih model tanpa tuning (cepat, untuk dev)
	$(PYTHON) src/train.py --no-tune

.PHONY: register
register:  ## Latih dan register model ke MLflow Registry
	$(PYTHON) src/train.py --register-model

# -----------------------------------------------------------------------------
# Serving
# -----------------------------------------------------------------------------

.PHONY: serve
serve:  ## Jalankan FastAPI inference server (port 8000)
	uvicorn src.inference:app --reload --port $(PORT) --host 0.0.0.0

.PHONY: mlflow-ui
mlflow-ui:  ## Buka MLflow tracking UI (port 5000)
	mlflow ui --port 5000

# -----------------------------------------------------------------------------
# Monitoring
# -----------------------------------------------------------------------------

.PHONY: drift-check
drift-check:  ## Quick PSI drift check
	$(PYTHON) src/monitor.py --quick-check

.PHONY: drift-report
drift-report:  ## Generate full Evidently HTML drift report
	$(PYTHON) src/monitor.py --full-report --output reports/drift_report.html
	@echo "Report disimpan di reports/drift_report.html"

# -----------------------------------------------------------------------------
# Quality
# -----------------------------------------------------------------------------

.PHONY: test
test:  ## Jalankan unit tests
	pytest -v

.PHONY: test-cov
test-cov:  ## Jalankan tests + coverage report
	pytest --cov=src --cov-report=term-missing --cov-report=html:reports/coverage

.PHONY: lint
lint:  ## Cek kode dengan ruff + black --check
	ruff check src tests
	black --check src tests

.PHONY: format
format:  ## Auto-format kode dengan black + ruff fix
	black src tests
	ruff check --fix src tests

.PHONY: typecheck
typecheck:  ## Type checking dengan mypy (kalau di-install)
	mypy src || echo "mypy belum di-install — skip"

.PHONY: check
check: lint test  ## Jalankan lint + test (CI-style)

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------

.PHONY: clean
clean:  ## Hapus file cache & temporary
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf reports/coverage htmlcov .coverage build dist

.PHONY: clean-data
clean-data:  ## Hapus data cache (HATI-HATI: hapus DB lokal)
	@echo "Akan menghapus data/gempa.db. Tekan Ctrl+C untuk batal."
	@sleep 3
	rm -f data/gempa.db data/features.parquet
	rm -rf data/raw/* data/interim/* data/processed/*

.PHONY: clean-models
clean-models:  ## Hapus model artifacts lokal
	rm -rf mlruns models/*.pkl models/*.joblib
