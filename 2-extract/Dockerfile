# Clean-markdown extraction for the R2AI2026 crawl warehouse. Runs on a GCP VM; auth comes from the VM's
# service account via the metadata server (Application Default Credentials). No credentials in the image.
FROM python:3.11.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 EXTRACT_WORKDIR=/work
WORKDIR /app
COPY requirements-dev.txt requirements.txt constraints.txt ./
RUN pip install -r requirements-dev.txt -c constraints.txt

COPY extract ./extract
COPY tests ./tests
RUN useradd --uid 1000 --create-home app && mkdir -p /work && chown app:app /work
USER app

# /work: local disk for downloaded dbs (largest is 6.6 GB) and shards; mount a big disk here.
VOLUME ["/work"]
ENTRYPOINT ["python", "-m", "extract"]
CMD ["--help"]
