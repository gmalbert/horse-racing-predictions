# Equine Edge API - FastAPI on Oracle Cloud (or any container host)
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Data directory is mounted at runtime (synced from the repo / pipeline):
#   docker run -v /srv/horse-racing/data:/data ...
ENV HRP_DATA_DIR=/data/processed \
    HRP_RAW_DIR=/data/raw \
    HRP_STATE_DIR=/state \
    HRP_CORS_ORIGINS="*"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
