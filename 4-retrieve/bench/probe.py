"""Leaderboard probes from an existing run's lists.pkl (no re-retrieval): each probe fuses only some lists and submits
the top-k docs (best fused chunk each), so its leaderboard precision/recall tells e.g. which language the gold docs
are in. Usage: python bench/probe.py <run_dir with lists.pkl> <ingest root> <queries.parquet> <out_dir>"""
import os
import pickle
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from retrieve import search, select, submit  # noqa: E402
from retrieve.config import Config  # noqa: E402

VI = (lambda r: r["channel"] == "sparse_vi")
PROBES = {  # name -> (keep(list record), docs per query, docs per query that also get their best chunk)
    # round 1 (21/10): which language holds the gold -> vi P .23-.28, zh .08, en exactly 0; |G| ~ 37 docs/query
    # "probe_vi_orig_k5": (lambda r: r["channel"] == "sparse_vi" and r["variant"] == "orig", 5, 5),
    # "probe_vi_all_k3": (VI, 3, 3), "probe_zh_k3": (lambda r: r["channel"] == "sparse_zh", 3, 3),
    # "probe_en_k3": (lambda r: r["channel"] == "sparse_en", 3, 3),
    # round 2: precision vs depth
    "probe_vi_all_k30": (VI, 30, 10),
    "probe_vi_all_k100": (VI, 100, 10),
}


def main(run_dir, ingest, queries_path, out_dir, *_):
    with open(os.path.join(run_dir, "lists.pkl"), "rb") as f:
        lists = pickle.load(f)
    queries = pd.read_parquet(queries_path, columns=["id", "query"])
    names = sys.argv[5].split(",") if len(sys.argv) > 5 else list(PROBES)
    for name in names:
        keep, k, kc = PROBES[name]
        cfg = Config(select="topk", k_docs=k, k_chunks=1, cand_docs=k, cand_chunks=1, allow_degraded=True)
        cand = search.fuse(lists, cfg.rrf_k, keep=keep, cfg=cfg)
        sel = select.select(cand, cfg)
        for s in sel.values():  # chunks only for the first kc docs (keeps the file and the scorer's LCS work bounded)
            s["chunks"] = {d: s["chunks"][d] for d in s["docs"][:kc]}
        store = select.load_chunks(ingest, {c for s in sel.values() for cs in s["chunks"].values() for c in cs})
        texts = select.chunk_texts(sel, store, False)
        entries = {q: {"docs": s["docs"], "chunk_texts": texts.get(q, [])} for q, s in sel.items()}
        zp = submit.write(queries, entries, os.path.join(out_dir, name), 1.0)  # a probe may leave queries empty
        n = sum(1 for s in sel.values() if s["docs"])
        print(name, "queries with docs:", n, "->", zp, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
