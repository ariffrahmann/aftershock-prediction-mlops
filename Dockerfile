FROM python:3.12-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi>=0.110.0 \
    "uvicorn[standard]>=0.27.0" \
    mlflow>=2.10.0 \
    xgboost>=2.0.0 \
    scikit-learn>=1.4.0 \
    pandas>=2.1.0 \
    numpy>=1.26.0 \
    "prometheus-client>=0.20.0" \
    pydantic>=2.6.0

COPY src/__init__.py ./src/__init__.py
COPY src/inference.py ./src/inference.py

EXPOSE 8080

# Jalankan inference server
CMD ["uvicorn", "src.inference:app", "--host", "0.0.0.0", "--port", "8080"]
