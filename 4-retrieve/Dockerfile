# Build from main-pipeline/:  docker build -f 4-retrieve/Dockerfile -t r2ai-retrieve:v1 .
# Retrieval on a CPU VM: queries are encoded with bge-m3 on CPU (or via --server). GCS auth: ADC.
# The LLM key comes only from env LLM_API_KEY at run time, never from the image.
FROM python:3.11.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/work/hf
WORKDIR /app
# FlagEmbedding pulls a recent `datasets` (pyarrow>=21, newer fsspec), so 3-ingest's pyarrow/gcsfs pins are left
# to the resolver here; parquet files stay compatible either way.
RUN pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install FlagEmbedding==1.4.2 transformers==4.57.1 pandas==2.2.3 "numpy<2.1" "scipy>=1.11,<1.15" gcsfs \
    google-auth requests==2.32.3 rapidfuzz==3.9.7 scikit-learn==1.5.2 pytest==8.3.5
COPY 2-extract/extract ./extract
COPY 3-ingest/ingest ./ingest
COPY 4-retrieve/retrieve ./retrieve
COPY 4-retrieve/models ./models
COPY 4-retrieve/configs ./configs
COPY 4-retrieve/bench ./bench
COPY 4-retrieve/tests ./tests
RUN useradd --uid 1000 --create-home app && mkdir -p /work && chown app:app /work
USER app
VOLUME ["/work"]
ENTRYPOINT ["python", "-m", "retrieve"]
CMD ["--help"]
