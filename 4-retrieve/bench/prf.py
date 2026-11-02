"""Dense pseudo-relevance feedback (Rocchio) on top of an existing run: for each dense list of the given variants,
q' = normalise(q + alpha * mean(top-n chunk vectors)), searched again over the same dense shards; the new lists are
added as variants "<v>_prf". Usage:
python bench/prf.py <run_dir with lists.pkl> <ingest> <queries.parquet> <plans.json> <cache_dir> <out lists.pkl> [n alpha]"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ingest import dense  # noqa: E402
from retrieve import search  # noqa: E402
from retrieve.__main__ import query_encoder  # noqa: E402
from retrieve.config import Config  # noqa: E402
from retrieve.encode_cache import CachedEncoder  # noqa: E402

VARIANTS = ("orig", "zh_q")


def chunk_vectors(ingest, langs, need: set) -> dict:
    out = {}
    for lang in langs:
        for s in search.dense_shards(ingest, "bge-m3", lang):
            ids = pd.read_parquet(s + ".ids.parquet").id.to_numpy()
            rows = [i for i, c in enumerate(ids) if c in need]
            if rows:
                v = np.load(s + ".npy", mmap_mode="r")
                out.update({ids[i]: np.asarray(v[i], np.float32) for i in rows})
    return out


def main(run_dir, ingest, queries_path, plans_path, cache_dir, out_path, n=10, alpha=0.5):
    n, alpha = int(n), float(alpha)
    lists = pickle.load(open(os.path.join(run_dir, "lists.pkl"), "rb"))
    cfg = Config(dense_langs=["vi", "zh"], k_dense=200)
    queries = pd.read_parquet(queries_path, columns=["id", "query"])
    plans = {int(k): v for k, v in json.load(open(plans_path, encoding="utf-8")).items()}
    text = {}
    for qid, q in zip(queries["id"].astype(int), queries["query"]):
        text[(qid, "orig")] = q
        p = plans.get(qid) or {}
        if p.get("ok") and p.get("zh_q"):
            text[(qid, "zh_q")] = p["zh_q"]
    recs = [r for r in lists if r["channel"] == "dense" and r["variant"] in VARIANTS and r["ids"]
            and (r["query_id"], r["variant"]) in text]
    vec = chunk_vectors(ingest, cfg.dense_langs, {c for r in recs for c in r["ids"][:n]})
    enc = CachedEncoder(query_encoder("cpu"), os.path.join(cache_dir, "query_enc.sqlite"))
    qv, _ = enc.encode_queries([text[(r["query_id"], r["variant"])] for r in recs])
    new = []
    for u, r in zip(qv, recs):
        fb = [vec[c] for c in r["ids"][:n] if c in vec]
        x = np.asarray(u, np.float32) + (alpha * np.mean(fb, axis=0) if fb else 0)
        new.append(x / np.linalg.norm(x))
    q2 = np.stack(new)
    per_lang = [(lang, *dense.search(q2, search.dense_shards(ingest, "bge-m3", lang), k=cfg.k_dense))
                for lang in cfg.dense_langs]
    added = []
    for j, r in enumerate(recs):
        merged = sorted(((float(s), c, lang) for lang, ids, sc in per_lang for c, s in zip(ids[j], sc[j])),
                        reverse=True)[:cfg.k_dense]
        added.append({"query_id": r["query_id"], "variant": r["variant"] + "_prf", "channel": "dense",
                      "ids": [c for _, c, _ in merged], "langs": [l for _, _, l in merged],
                      "scores": [s for s, _, _ in merged]})
    with open(out_path, "wb") as f:
        pickle.dump(lists + added, f)
    print("prf lists added:", len(added), "->", out_path, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
