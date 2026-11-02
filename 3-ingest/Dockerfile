# Build from main-pipeline/:  docker build -f 3-ingest/Dockerfile -t r2ai-ingest:v1 .
# CPU image for `rag chunk` / `rag index` (and `rag embed --server`). GCS auth: VM service account (ADC).
FROM python:3.11.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 INGEST_WORKDIR=/work
WORKDIR /app
COPY 3-ingest/requirements.txt ./
RUN pip install -r requirements.txt pytest==8.3.5
COPY 2-extract/extract ./extract
COPY 3-ingest/ingest ./ingest
RUN useradd --uid 1000 --create-home app && mkdir -p /work && chown app:app /work
USER app
VOLUME ["/work"]
ENTRYPOINT ["python", "-m", "ingest"]
CMD ["--help"]
