#!/bin/bash
# Final 1,200-query submissions with no GPU and no paid API: BM25 + CPU dense rerank (rerank_docs 30) +
# Qwen3.5-9B Q4_K_M on llama.cpp at localhost:8080 (plans; small judge prompts). Re-runnable: plans, judge replies,
# query and chunk vectors are cached under /full/bm25/cache. Log: ~/logs/full.log ([final] lines).
R="sudo docker run --rm --network host -v /mnt/full:/full -v /mnt/ingest:/full/bm25/ingest -v /mnt/hf:/work/hf
   -v $HOME/data:/data:ro -e RETRIEVE_MODELS=/app/models/bm25_rerank r2ai-retrieve:v1"
COMMON="--queries /data/query.parquet --ingest /full/bm25/ingest --server bm25+cpu --config /app/configs/bm25_rerank_cpu.json
        --llm-url http://localhost:8080/v1 --llm-model qwen3.5-9b-q4km --llm-workers 4 --cache-dir /full/bm25/cache"
B="sudo docker run --rm --network host -v $HOME/bench2/_bench:/bench -v /mnt/hf:/work/hf -e LLM_WORKERS=4
   -e RETRIEVE_MODELS=/app/models/bm25_rerank --entrypoint python r2ai-retrieve:v1"
log() { echo "$(date -u +%FT%TZ) [final] $*" >> ~/logs/full.log; }
log START
$R run --modes v0,v1 --run-dir /full/bm25/runs/final $COMMON >> ~/logs/final_v1.log 2>&1 && log "v0,v1 done" \
  || { log "v0,v1 failed"; exit 1; }
# check the local LLM and the small CPU judge on the silver bench before spending hours on v2
$B -e RUN=b3 bench/evaluate.py /bench bm25+cpu http://localhost:8080/v1 qwen3.5-9b-q4km v1,v2 /app/configs/bm25_rerank_cpu.json \
  >> ~/logs/bench_b3.log 2>&1 && log "bench b3 done" || log "bench b3 failed"
$R run --modes v2 --run-dir /full/bm25/runs/final_v2 $COMMON >> ~/logs/final_v2.log 2>&1 && log "v2 done" \
  || { log "v2 failed"; exit 1; }
log "ALL DONE"
