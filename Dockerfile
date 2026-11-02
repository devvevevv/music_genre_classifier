# syntax=docker/dockerfile:1

FROM python:3.12-slim

# System libraries required to decode audio (soundfile + ffmpeg for MP3, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first so Docker can cache this layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code and the trained model registry.
COPY src ./src
COPY eval ./eval
COPY app ./app
COPY models ./models

ENV MODEL_PATH=/app/models/model.pkl
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

# Bind to $PORT so PaaS platforms (Render, etc.) can inject the port;
# fall back to 8000 for local `docker run`.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
