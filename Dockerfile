FROM python:3.11-slim

# Arabic + fallback fonts so matplotlib renders figures correctly when generating
# at runtime (the baked figures already render fine without this).
RUN apt-get update && apt-get install -y --no-install-recommends \
        fonts-noto-core fonts-noto-naskh-arabic fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run sets $PORT (8080). gunicorn: 1 worker (in-process generation state +
# the shared API semaphore are per-process), many threads for serving, long
# timeout so a manual /api/generate run isn't killed mid-flight.
ENV PORT=8080
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 exam_app:app
