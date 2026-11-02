#!/bin/bash
# Every submission for the 1,200 queries, one plan set for all (plans.json of the local-LLM run), judge on
# OpenRouter Qwen3.5-9B. Outputs: /mnt/full/subs/<name>/<mode>/submission.zip. Log: ~/logs/full.log ([subs] lines).
#   bm25         v1, v2   BM25 only
#   bm25_rerank  v0, v1   BM25 + CPU dense rerank (from final_cpu.sh, re-run here if that run is gone), v2 here
#   bge          v0-v2    full bge-m3 index, once embed_colab.ipynb has embedded every part
# Usage: LLM_API_KEY=... setsid nohup ./subs.sh > ~/logs/subs.out 2>&1 &
A=(-v /etc/r2ai:/creds:ro -e GOOGLE_APPLICATION_CREDENTIALS=/creds/adc.json -e GOOGLE_CLOUD_PROJECT=aiguru26)
P=/full/bm25/runs/final/plans.json
LLM=(--llm-url https://openrouter.ai/api/v1 --llm-model qwen/qwen3.5-9b --llm-workers 8
     --llm-extra '{"reasoning": {"enabled": false}}')
log() { echo "$(date -u +%FT%TZ) [subs] $*" >> ~/logs/full.log; }
run() {  # run <models dir> <name> <args...>
  local models=$1 name=$2; shift 2
  sudo docker run --rm --network host -v /mnt/full:/full -v /mnt/ingest:/full/bm25/ingest -v /mnt/hf:/work/hf \
    -v "$HOME/data:/data:ro" -e LLM_API_KEY="$LLM_API_KEY" -e RETRIEVE_MODELS="$models" r2ai-retrieve:v1 run \
    --queries /data/query.parquet --ingest /full/bm25/ingest --plans $P "${LLM[@]}" --cache-dir /full/bm25/cache \
    --run-dir /full/subs/$name "$@" >> ~/logs/subs_$name.log 2>&1 && log "$name $* done" || { log "$name failed"; return 1; }
}
sudo mkdir -p /mnt/full/subs && sudo chown 1000:1000 /mnt/full/subs
log START

run /app/models/bm25 bm25 --modes v1,v2 --server bm25 --config /app/configs/bm25.json

# BM25 + rerank: v0/v1 come from final_cpu.sh's still-running container; wait for it, re-run here if it died
F=/mnt/full/bm25/runs/final
while [ ! -f $F/v1/submission.zip ] && sudo docker ps --no-trunc --format '{{.Command}}' | grep -q 'runs/final '; do
  sleep 300
done
if [ -f $F/v1/submission.zip ]; then
  sudo mkdir -p /mnt/full/subs/bm25_rerank && sudo cp -r $F/v0 $F/v1 /mnt/full/subs/bm25_rerank/ && log "bm25_rerank v0,v1 copied"
else
  run /app/models/bm25_rerank bm25_rerank --modes v0,v1 --server bm25+cpu --config /app/configs/bm25_rerank.json
fi
run /app/models/bm25_rerank bm25_rerank_v2 --modes v2 --server bm25+cpu --config /app/configs/bm25_rerank.json

# full bge-m3 path once the notebook is done
count() {
  sudo docker run --rm "${A[@]}" --entrypoint python r2ai-ingest:v1 -c "
import gcsfs
fs = gcsfs.GCSFileSystem(token='google_default')
r = 'aiguru_crawler/ingest/v1/rag/'
n = lambda p: len(fs.glob(r + p))
print(n('chunks/*/*.parquet'), min(n('dense/bge-m3/*/*.ids.parquet'), n('sparse/bge-m3/*/*.ids.parquet')))" 2>/dev/null
}
while true; do
  read parts done_ <<< "$(count)"
  [ -n "$parts" ] && [ "$parts" -gt 0 ] && [ "$done_" -ge "$parts" ] && break
  [ "$done_" != "$last" ] && log "embedded $done_ / $parts parts" && last=$done_
  sleep 600
done
log "embeddings complete"
[ -f ~/logs/.index_bge ] || { sudo docker run --rm "${A[@]}" -v /mnt/work:/work r2ai-ingest:v1 rag index \
  --out gs://aiguru_crawler/ingest/v1 --workdir /work/index >> ~/logs/index_bge.log 2>&1 && touch ~/logs/.index_bge \
  || { log "bge index failed"; exit 1; }; }
[ -f ~/logs/.fetch_bge ] || { sudo docker run --rm "${A[@]}" -v /mnt/ingest:/ingest --entrypoint python r2ai-ingest:v1 -c "
import gcsfs
fs = gcsfs.GCSFileSystem(token='google_default')
for d in ('dense/bge-m3', 'sparse_index/bge-m3'):
    fs.get('aiguru_crawler/ingest/v1/rag/' + d + '/', '/ingest/rag/' + d + '/', recursive=True)
" >> ~/logs/fetch_bge.log 2>&1 && touch ~/logs/.fetch_bge || { log "bge fetch failed"; exit 1; }; }
run /app/models bge --modes v0,v1,v2 --server cpu
log "ALL DONE"
