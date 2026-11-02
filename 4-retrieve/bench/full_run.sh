#!/bin/bash
# Full corpus on the CPU VM, every step resumable (done markers in ~/logs/.<step>_<mode>):
#   index:  bge-m3 = embed via the HTTP server (2 clients, retried) + sparse index;  bm25 = BM25 on CPU (no GPU)
#   -> local copy of the index -> silver dev on the full corpus -> eval with the shipped rankers
#   -> retrain rankers on the full corpus -> eval -> v0/v1/v2 submissions for the 1,200 queries with both ranker sets.
# Usage: LLM_API_KEY=... setsid nohup ./full_run.sh bm25|<bge-m3 server url> > ~/logs/full.out 2>&1 &
# Needs: images r2ai-ingest:v1 / r2ai-retrieve:v1, ADC at /etc/r2ai/adc.json, queries at ~/data/query.parquet.
if [ "$1" = bm25 ]; then
  M=bm25; ENC=bm25; CFG=/app/configs/bm25.json; SHIPPED=/app/models/bm25; PARTS="chunks sparse_index/bm25"
else
  M=bge; URL=$1; ENC=cpu; CFG=""; SHIPPED=/app/models; PARTS="chunks dense sparse_index/bge-m3"
fi
OUT=gs://aiguru_crawler/ingest/v1
A="-v /etc/r2ai:/creds:ro -e GOOGLE_APPLICATION_CREDENTIALS=/creds/adc.json -e GOOGLE_CLOUD_PROJECT=aiguru26"
LLM="--llm-url https://openrouter.ai/api/v1 --llm-model qwen/qwen3.5-9b"
EXTRA='{"reasoning":{"enabled":false}}'
F=/full/$M  # this mode's dev set, caches, runs, models
mkdir -p ~/logs && sudo mkdir -p /mnt/work /mnt/ingest /mnt/full/$M/cache /mnt/hf \
  && sudo chown -R 1000:1000 /mnt/work /mnt/ingest /mnt/full /mnt/hf
log() { echo "$(date -u +%FT%TZ) [$M] $*" >> ~/logs/full.log; }
done_() { [ -f ~/logs/.$1_$M ]; }
mark() { touch ~/logs/.$1_$M; }
R="sudo docker run --rm $A -v /mnt/full:/full -v /mnt/ingest:$F/ingest -v /mnt/hf:/work/hf -v $HOME/data:/data:ro
   -e LLM_API_KEY=$LLM_API_KEY -e LLM_EXTRA=$EXTRA -e CLEAN_DOCS=gs://aiguru_crawler/clean/v1/docs/"
PY="--entrypoint python r2ai-retrieve:v1"

embed() {
  local n=0
  until sudo docker run --rm $A -v /mnt/work:/work r2ai-ingest:v1 rag embed --out $OUT --server "$URL" \
        --batch-size 64 --shard "$1/2" --workdir "/work/embed$1" >> ~/logs/embed_$1.log 2>&1; do
    n=$((n + 1)); log "embed $1/2 failed (try $n), retry in 120 s"; sleep 120
  done
}

log "START"
if [ $M = bm25 ]; then
  done_ index || { $R -v /mnt/work:/work $PY -m ingest rag bm25 --out $OUT --workdir /work/bm25 >> ~/logs/bm25.log 2>&1 \
    && mark index || { log "bm25 index failed"; exit 1; }; }
else
  done_ embed || { embed 0 & embed 1 & wait; mark embed; }
  done_ index || { sudo docker run --rm $A -v /mnt/work:/work r2ai-ingest:v1 rag index --out $OUT --workdir /work/index \
    >> ~/logs/index.log 2>&1 && mark index || { log "index failed"; exit 1; }; }
fi
log "index done"
done_ fetch || { sudo docker run --rm $A -v /mnt/ingest:/ingest --entrypoint python r2ai-ingest:v1 -c "
import gcsfs, sys
fs = gcsfs.GCSFileSystem(token='google_default')
for d in sys.argv[1:]:
    fs.get('aiguru_crawler/ingest/v1/rag/' + d + '/', '/ingest/rag/' + d + '/', recursive=True)
    print(d, 'ok', flush=True)
" $PARTS >> ~/logs/fetch.log 2>&1 && mark fetch || { log "fetch failed"; exit 1; }; }
log "fetch done: $(du -sh /mnt/ingest | cut -f1)"

[ -f /mnt/full/$M/dev.parquet ] || $R $PY bench/make_dev.py $F https://openrouter.ai/api/v1 qwen/qwen3.5-9b 300 \
  >> ~/logs/make_dev_$M.log 2>&1 || { log "make_dev failed"; exit 1; }
log "dev ready"
Q="$F $ENC https://openrouter.ai/api/v1 qwen/qwen3.5-9b v0,v1,v2 $CFG"
$R -e RUN=f1 -e RETRIEVE_MODELS=$SHIPPED $PY bench/evaluate.py $Q >> ~/logs/eval_f1_$M.log 2>&1 || { log "eval f1 failed"; exit 1; }
$R -e MODELS_DIR=$F/models_f1 $PY bench/train_ranker.py $F $ENC $F/runs/f1 >> ~/logs/train_f1_$M.log 2>&1 \
  || { log "train f1 failed"; exit 1; }
$R -e RUN=f1b -e RETRIEVE_MODELS=$F/models_f1 $PY bench/evaluate.py $Q >> ~/logs/eval_f1b_$M.log 2>&1 \
  || { log "eval f1b failed"; exit 1; }
log "silver eval done"

for S in shipped f1; do
  MD=$SHIPPED; [ $S = f1 ] && MD=$F/models_f1
  $R -e RETRIEVE_MODELS=$MD r2ai-retrieve:v1 run --modes v0,v1,v2 --queries /data/query.parquet --ingest $F/ingest \
    --run-dir $F/runs/test_$S --server $ENC ${CFG:+--config $CFG} $LLM --llm-extra "$EXTRA" --cache-dir $F/cache \
    >> ~/logs/test_${S}_$M.log 2>&1 || { log "test run $S failed"; exit 1; }
  log "test run $S done"
done
log "ALL DONE"
