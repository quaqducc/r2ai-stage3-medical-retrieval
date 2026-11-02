#!/bin/bash
# Hybrid (bge-m3 dense + sparse) deep submissions with the listwise LLM judge, once embed_colab.ipynb has embedded
# every vi part (the notebook goes en -> vi -> zh): wait -> sparse index -> local copy -> runs below. Plans reused from plans.json (one plan set for all);
# judge = Qwen3.5-9B on OpenRouter. Outputs /mnt/full/subs/<config>/deep/. Log: ~/logs/full.log ([hybrid] lines).
# Usage: LLM_API_KEY=... setsid nohup ./hybrid.sh > ~/logs/hybrid.out 2>&1 &
A=(-v /etc/r2ai:/creds:ro -e GOOGLE_APPLICATION_CREDENTIALS=/creds/adc.json -e GOOGLE_CLOUD_PROJECT=aiguru26)
log() { echo "$(date -u +%FT%TZ) [hybrid] $*" >> ~/logs/full.log; }
count() {
  sudo docker run --rm "${A[@]}" --entrypoint python r2ai-ingest:v1 -c "
import gcsfs
fs = gcsfs.GCSFileSystem(token='google_default')
r = 'aiguru_crawler/ingest/v1/rag/'
n = lambda p: len(fs.glob(r + p))
print(n('chunks/vi/*.parquet'), min(n('dense/bge-m3/vi/*.ids.parquet'), n('sparse/bge-m3/vi/*.ids.parquet')))" 2>/dev/null
}
log START
while true; do
  read parts done_ <<< "$(count)"
  [ -n "$parts" ] && [ "$parts" -gt 0 ] && [ "$done_" -ge "$parts" ] && break
  [ "$done_" != "$last" ] && log "embedded $done_ / $parts parts" && last=$done_
  sleep 600
done
log "vi embeddings complete (configs use vi only; zh joins in a later run)"
[ -f ~/logs/.index_bge ] || { sudo docker run --rm "${A[@]}" -v /mnt/work:/work r2ai-ingest:v1 rag index \
  --out gs://aiguru_crawler/ingest/v1 --workdir /work/index >> ~/logs/index_bge.log 2>&1 && touch ~/logs/.index_bge \
  || { log "index failed"; exit 1; }; }
[ -f ~/logs/.fetch_bge ] || { sudo docker run --rm "${A[@]}" -v /mnt/ingest:/ingest --entrypoint python r2ai-ingest:v1 -c "
import gcsfs
fs = gcsfs.GCSFileSystem(token='google_default')
for d in ('dense/bge-m3', 'sparse_index/bge-m3'):
    fs.get('aiguru_crawler/ingest/v1/rag/' + d + '/', '/ingest/rag/' + d + '/', recursive=True)
" >> ~/logs/fetch_bge.log 2>&1 && touch ~/logs/.fetch_bge || { log "fetch failed"; exit 1; }; }
log "index ready"
for C in deep_bge_k100 deep_bge_judge_k100 deep_bge_judge_k200; do
  sudo docker run --rm -v /mnt/full:/full -v /mnt/ingest:/full/ingest -v /mnt/hf:/work/hf -v "$HOME/data:/data:ro" \
    -e LLM_API_KEY="$LLM_API_KEY" r2ai-retrieve:v1 run --modes deep --queries /data/query.parquet --ingest /full/ingest \
    --server cpu --config /app/configs/$C.json --plans /full/bm25/runs/final/plans.json --run-dir /full/subs/$C \
    --llm-url https://openrouter.ai/api/v1 --llm-model qwen/qwen3.5-9b --llm-workers 16 \
    --llm-extra '{"reasoning": {"enabled": false}}' --cache-dir /full/bge_cache >> ~/logs/$C.log 2>&1 \
    && log "$C done" || log "$C failed"
done
log "ALL DONE"
