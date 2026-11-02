"""CLI: python -m ingest rag {chunk,embed,index} --help   (an `okf` command group is reserved for flow B)."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(prog="ingest")
    groups = p.add_subparsers(dest="group", required=True)
    rag = groups.add_parser("rag", help="flow A: chunks, bge-m3 dense + sparse").add_subparsers(dest="cmd", required=True)
    workdir = os.environ.get("INGEST_WORKDIR", "/tmp/ingest")

    c = rag.add_parser("chunk", help="clean store -> chunks + doc cards")
    c.add_argument("--clean", required=True, help="2-extract output, e.g. gs://aiguru_crawler/clean/v1")
    c.add_argument("--out", required=True, help="ingest root, e.g. gs://<bucket>/ingest/v1")
    c.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    c.add_argument("--hosts")
    c.add_argument("--limit", type=int, help="docs per part (smoke tests)")
    c.add_argument("--workdir", default=workdir)
    c.add_argument("--force", action="store_true")

    e = rag.add_parser("embed", help="bge-m3 dense + sparse for every chunk part not embedded yet (GPU)")
    e.add_argument("--out", required=True)
    e.add_argument("--server", help="bge-m3 HTTP server base URL (POST /encode_batch); else load the model locally (GPU)")
    e.add_argument("--model", default="BAAI/bge-m3")
    e.add_argument("--tag", default="bge-m3")
    e.add_argument("--devices", help="e.g. cuda:0,cuda:1")
    e.add_argument("--batch-size", type=int, default=32)
    e.add_argument("--max-length", type=int, default=2048, help="tokens; longer chunks are truncated for the encoder only")
    e.add_argument("--langs")
    e.add_argument("--max-parts", type=int, help="stop after N parts (time-boxed sessions, e.g. Kaggle)")
    e.add_argument("--shard", default="0/1", help="i/n: this session's share of the parts (n parallel sessions)")
    e.add_argument("--workdir", default=workdir)

    i = rag.add_parser("index", help="stack sparse shards into one index per language")
    i.add_argument("--out", required=True)
    i.add_argument("--tag", default="bge-m3")
    i.add_argument("--workdir", default=workdir)

    b = rag.add_parser("bm25", help="CPU fallback: BM25 over XLM-R tokens as sparse_index/bm25/<lang> (no GPU)")
    b.add_argument("--out", required=True)
    b.add_argument("--langs")
    b.add_argument("--workdir", default=workdir)

    a = p.parse_args(argv)
    from . import rag as R
    if a.cmd == "chunk":
        res = R.run_chunk(a.clean, a.out, a.workers, a.hosts.split(",") if a.hosts else None, a.workdir, a.limit, a.force)
    elif a.cmd == "embed":
        enc = (R.HttpEncoder(a.server, a.batch_size) if a.server else
               R.BgeM3Encoder(a.model, a.devices.split(",") if a.devices else None, a.batch_size, a.max_length))
        i, n = (int(x) for x in a.shard.split("/"))
        if not 0 <= i < n:
            raise SystemExit(f"--shard {a.shard}: need 0 <= i < n")
        res = R.run_embed(a.out, enc, a.tag, a.langs.split(",") if a.langs else None, a.workdir, a.max_parts, (i, n))
    elif a.cmd == "bm25":
        res = R.run_bm25(a.out, a.langs.split(",") if a.langs else None, a.workdir)
    else:
        res = R.run_index(a.out, a.tag, a.workdir)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
