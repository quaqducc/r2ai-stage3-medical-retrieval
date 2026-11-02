#!/bin/bash
# Waits until every chunk part has its bge-m3 dense + sparse shard on GCS (written by 3-ingest/colab/embed_colab.ipynb),
# then: sparse index -> local copy -> v0/v1/v2 for the 1,200 queries with the full bge-m3 path and the local
# Qwen3.5-9B (llama.cpp, localhost:8080). Reuses the BM25 run's cache (plans, judge replies). Log: ~/logs/full.log.
A="-v /etc/r2ai:/creds:ro -e GOOGLE_APPLICATION_CREDENTIALS=/creds/adc.json -e GOOGLE_CLOUD_PROJECT=aiguru26"
OUT=gs://aiguru_crawler/ingest/v1
log() { echo "$(date -u +%FT%TZ) [bge] $*" >> ~/logs/full.log; }
count() {
  sudo docker run --rm $A --entrypoint python r2ai-ingest:v1 -c "
import gcsfs
fs = gcsfs.GCSFileSystem(token='google_default')
r = 'aiguru_crawler/ingest/v1/rag/'
n = lambda p: len(fs.glob(r + p))
print(n('chunks/*/*.parquet'), min(n('dense/bge-m3/*/*.ids.parquet'), n('sparse/bge-m3/*/*.ids.parquet')))" 2>/dev/null
}
log "waiting for embeddings"
while true; do
  read parts done_ <<< "$(count)"
  [ -n "$parts" ] && [ "$parts" -gt 0 ] && [ "$done_" -ge "$parts" ] && break
  [ "$done_" != "$last" ] && log "embedded $done_ / $parts parts" && last=$done_
  sleep 600
done
log "embeddings complete: $parts parts"
[ -f ~/logs/.index_bge ] || { sudo docker run --rm $A -v /mnt/work:/work r2ai-ingest:v1 rag index --out $OUT \
  --workdir /work/index >> ~/logs/index_bge.log 2>&1 && touch ~/logs/.index_bge || { log "index failed"; exit 1; }; }
[ -f ~/logs/.fetch_bge ] || { sudo docker run --rm $A -v /mnt/ingest:/ingest --entrypoint python r2ai-ingest:v1 -c "
import gcsfs
fs = gcsfs.GCSFileSystem(token='google_default')
for d in ('dense/bge-m3', 'sparse_index/bge-m3'):
    fs.get('aiguru_crawler/ingest/v1/rag/' + d + '/', '/ingest/rag/' + d + '/', recursive=True)
" >> ~/logs/fetch_bge.log 2>&1 && touch ~/logs/.fetch_bge || { log "fetch failed"; exit 1; }; }
log "index ready: $(du -sh /mnt/ingest | cut -f1)"
sudo docker run --rm --network host -v /mnt/full:/full -v /mnt/ingest:/full/bm25/ingest -v /mnt/hf:/work/hf \
  -v $HOME/data:/data:ro r2ai-retrieve:v1 run --modes v0,v1,v2 --queries /data/query.parquet \
  --ingest /full/bm25/ingest --run-dir /full/bge/runs/final --server cpu --config /app/configs/bge_cpu.json \
  --llm-url http://localhost:8080/v1 --llm-model qwen3.5-9b-q4km --llm-workers 4 --cache-dir /full/bm25/cache \
  >> ~/logs/final_bge.log 2>&1 && log "ALL DONE" || log "final run failed"
