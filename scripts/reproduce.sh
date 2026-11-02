#!/bin/bash
# Reproduce the reported submission (variant i_dprf15, public FINAL 0.3577) on a Linux VM with Docker.
# Run from the repository root:  scripts/reproduce.sh <step> [<step> ...]
#
#   fast      released lists + chunk texts -> submission (minutes, no GPU, no LLM)
#   full      every step below in order (the embed step needs a GPU; see README §4)
#
#   images    build the Docker images (r2ai-extract, r2ai-ingest, r2ai-retrieve)
#   extract   crawl warehouse (SQLite per host) -> clean parquet store            (CPU, ~1 h on 16 vCPU)
#   chunk     clean store -> verbatim chunks + doc cards                          (CPU)
#   embed     bge-m3 dense + sparse for every chunk                               (GPU, ~10 h on one T4)
#   index     merge the sparse shards into one index per language                 (CPU)
#   sync      copy <ingest>/rag to local disk (indexes are memory-mapped)
#   llm       start Qwen3.5-9B (GGUF Q4_K_M) on llama.cpp at localhost:8080
#   retrieve  plans (LLM) + bge-m3 dense/sparse lists for 1,200 queries -> lists.pkl
#   prf       dense Rocchio PRF lists                                    -> lists_prf.pkl
#   doc       doc-level dense lists (first chunk, mean chunk vector)     -> lists_doc.pkl
#   docprf    doc-level dense lists of the PRF query vectors             -> lists_docprf.pkl
#   build     weighted RRF fusion + selection, variant i_dprf15          -> submission.zip
#   verify    compare the built submission.json with the released one
#
# Environment (defaults match the team VM):
#   BUCKET  gs://aiguru_crawler   source data and released artifacts (read access needed; write for extract..index)
#   WORK    /mnt/full             work dir: runs, caches, outputs
#   INGEST  /mnt/ingest           local copy of <bucket>/ingest/v1/rag (must be a local disk)
#   DATA    $HOME/data            organiser files: query.parquet
#   HF      /mnt/hf               Hugging Face cache (BAAI/bge-m3, ~2.3 GB)
#   LLMDIR  /mnt/llm              GGUF model dir
#   PLANS   (empty)               reuse a plans.json instead of calling the LLM (the released one gives exact lists)
set -euo pipefail
BUCKET=${BUCKET:-gs://aiguru_crawler}
WORK=${WORK:-/mnt/full}
INGEST=${INGEST:-/mnt/ingest}
DATA=${DATA:-$HOME/data}
HF=${HF:-/mnt/hf}
LLMDIR=${LLMDIR:-/mnt/llm}
PLANS=${PLANS:-}
REL=$BUCKET/release/v1
D=/full/subs/deep_bge_vizh_k100  # run dir as seen inside the retrieve container
V=i_dprf15
ROOT=$(pwd)

# retrieve container: r <env flags...> -- <python args...>
r() {
  local env=()
  while [ "$1" != "--" ]; do env+=("$1"); shift; done; shift
  sudo docker run --rm --network host "${env[@]}" -v "$WORK":/full -v "$INGEST":/full/ingest:ro -v "$HF":/work/hf \
    -v "$DATA":/data:ro -v "$ROOT/4-retrieve/configs":/cfg:ro --entrypoint python r2ai-retrieve:v1 -u "$@"
}
# ingest / extract containers write to GCS with the VM's Application Default Credentials
g() { sudo docker run --rm -v /mnt/work:/work "$@"; }

step_images() {
  sudo docker build -t r2ai-extract:v1 2-extract
  sudo docker build -f 3-ingest/Dockerfile -t r2ai-ingest:v1 .
  sudo docker build -f 4-retrieve/Dockerfile -t r2ai-retrieve:v1 .
}
step_extract() {
  sudo mkdir -p /mnt/work && sudo chown 1000:1000 /mnt/work
  g r2ai-extract:v1 run --out "$BUCKET/clean/v1" --workdir /work
  g r2ai-extract:v1 validate --out "$BUCKET/clean/v1"
}
step_chunk() {
  g -e INGEST_WORKDIR=/work r2ai-ingest:v1 rag chunk --clean "$BUCKET/clean/v1" --out "$BUCKET/ingest/v1" --workers 16 --workdir /work
}
step_embed() {
  if command -v nvidia-smi > /dev/null; then
    (cd 3-ingest && pip install -r requirements-gpu.txt && python -m ingest rag embed --out "$BUCKET/ingest/v1")
  else
    echo "embed needs a GPU: run 3-ingest/colab/embed_colab.ipynb (T4) or this step on a GPU machine, then continue" >&2
    exit 1
  fi
}
step_index() { sudo docker run --rm r2ai-ingest:v1 rag index --out "$BUCKET/ingest/v1"; }
step_sync() {
  sudo mkdir -p "$INGEST/rag" && sudo chown "$(id -u)" "$INGEST/rag"
  for p in chunks dense sparse_index; do gcloud storage rsync -r "$BUCKET/ingest/v1/rag/$p" "$INGEST/rag/$p"; done
}
step_llm() {
  sudo mkdir -p "$LLMDIR" && sudo chown "$(id -u)" "$LLMDIR"
  [ -s "$LLMDIR/qwen3.5-9b-q4km.gguf" ] || curl -L -o "$LLMDIR/qwen3.5-9b-q4km.gguf" \
    https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf
  sudo docker run -d --name qwen --network host -v "$LLMDIR":/models ghcr.io/ggml-org/llama.cpp:server \
    -m /models/qwen3.5-9b-q4km.gguf -c 16384 -np 4 -t 16 --jinja --reasoning-budget 0 --host 0.0.0.0 --port 8080
  until curl -sf localhost:8080/health > /dev/null; do sleep 5; done
}
step_retrieve() {
  mkdir -p "$WORK/subs" "$WORK/bge_cache"
  local plan=(--llm-url http://localhost:8080/v1 --llm-model qwen3.5-9b-q4km --llm-workers 4)
  if [ -n "$PLANS" ]; then cp "$PLANS" "$WORK/plans_in.json"; plan=(--plans /full/plans_in.json); fi
  r -- -m retrieve run --modes deep --queries /data/query.parquet --ingest /full/ingest --server cpu \
    --config /cfg/deep_bge_vizh_k100.json "${plan[@]}" --run-dir $D --cache-dir /full/bge_cache
}
step_prf() { r -- bench/prf.py $D /full/ingest /data/query.parquet $D/plans.json /full/bge_cache $D/lists_prf.pkl 10 0.5; }
step_doc() {
  r -e LISTS=lists_prf.pkl -- bench/doc_dense.py $D /full/ingest /data/query.parquet $D/plans.json /full/bge_cache $D/lists_doc.pkl
}
step_docprf() {
  r -e LISTS=lists_doc.pkl -- bench/doc_dense_prf.py $D /full/ingest /data/query.parquet $D/plans.json /full/bge_cache \
    $D/lists_docprf.pkl
}
step_build() {
  mkdir -p "$WORK/variants"
  r -e LISTS=lists_docprf.pkl -- bench/variants.py $D /full/ingest /data/query.parquet /full/variants $V
  ls -la "$WORK/variants/$V/submission.zip"
}
step_verify() {
  mkdir -p "$WORK/release"
  gcloud storage cp "$REL/submissions/$V.zip" "$WORK/release/$V.zip"
  r -- -c "
import json, zipfile
a = json.loads(zipfile.ZipFile('/full/variants/$V/submission.zip').read('submission.json'))
b = json.loads(zipfile.ZipFile('/full/release/$V.zip').read('submission.json'))
print('IDENTICAL to the released $V' if a == b else 'DIFFERENT from the released $V')"
}
step_fast() {  # released lists (built by retrieve..docprf) + chunk texts from the bucket -> submission
  mkdir -p "$WORK/subs/deep_bge_vizh_k100"
  gcloud storage cp "$REL/run/plans.json" "$REL/run/lists_docprf.pkl" "$WORK/subs/deep_bge_vizh_k100/"
  sudo mkdir -p "$INGEST/rag" && sudo chown "$(id -u)" "$INGEST/rag"
  gcloud storage rsync -r "$BUCKET/ingest/v1/rag/chunks" "$INGEST/rag/chunks"
  step_build
  step_verify
}
step_full() {
  for s in images extract chunk embed index sync llm retrieve prf doc docprf build verify; do "step_$s"; done
}

[ $# -gt 0 ] || { sed -n '2,32p' "$0"; exit 1; }
for s in "$@"; do echo "== $s"; "step_$s"; done
