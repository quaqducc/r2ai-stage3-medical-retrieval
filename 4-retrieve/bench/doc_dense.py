"""Doc-level bge-m3 lists, to mimic a gold built by embedding whole documents: for the orig and zh_q queries, dense
search over (a) each doc's first chunk only (variants <v>_head) and (b) each doc's mean chunk vector (<v>_mean).
Ids are "<doc>:0" so the lists rank docs; use them for doc ranking only (chunk_keep without them). Usage:
python bench/doc_dense.py <run_dir with LISTS> <ingest> <queries.parquet> <plans.json> <cache_dir> <out lists.pkl>"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from retrieve import search  # noqa: E402
from retrieve.__main__ import query_encoder  # noqa: E402
from retrieve.encode_cache import CachedEncoder  # noqa: E402

K = 200


def doc_vectors(ingest, lang):
    """(doc ids, head vectors, mean vectors) over every dense shard of a language (a doc's chunks share one part)."""
    ids_all, head, mean = [], [], []
    for s in search.dense_shards(ingest, "bge-m3", lang):
        ids = pd.read_parquet(s + ".ids.parquet").id.astype(str)
        v = np.load(s + ".npy", mmap_mode="r")
        doc = ids.str.split(":").str[0].to_numpy()
        idx = ids.str.split(":").str[1].astype(int).to_numpy()
        order = np.lexsort((idx, doc))
        doc, idx = doc[order], idx[order]
        start = np.r_[0, np.flatnonzero(doc[1:] != doc[:-1]) + 1]
        vv = np.asarray(v, np.float32)[order]
        head.append(vv[start])
        sums = np.add.reduceat(vv, start, axis=0)
        mean.append(sums / np.linalg.norm(sums, axis=1, keepdims=True))
        ids_all.append(doc[start])
    return np.concatenate(ids_all), np.concatenate(head), np.concatenate(mean)


def topk(q, mat, k=K, block=256):
    out_i, out_s = [], []
    for a in range(0, len(q), block):
        sc = q[a:a + block] @ mat.T
        j = np.argpartition(-sc, k - 1, axis=1)[:, :k]
        s = np.take_along_axis(sc, j, axis=1)
        o = np.argsort(-s, axis=1)
        out_i.append(np.take_along_axis(j, o, axis=1))
        out_s.append(np.take_along_axis(s, o, axis=1))
    return np.concatenate(out_i), np.concatenate(out_s)


def main(run_dir, ingest, queries_path, plans_path, cache_dir, out_path):
    lists = pickle.load(open(os.path.join(run_dir, os.environ.get("LISTS", "lists.pkl")), "rb"))
    queries = pd.read_parquet(queries_path, columns=["id", "query"])
    plans = {int(k): v for k, v in json.load(open(plans_path, encoding="utf-8")).items()}
    reqs = []
    for qid, q in zip(queries["id"].astype(int), queries["query"]):
        reqs.append((qid, "orig", q))
        p = plans.get(qid) or {}
        if p.get("ok") and p.get("zh_q"):
            reqs.append((qid, "zh_q", p["zh_q"]))
    enc = CachedEncoder(query_encoder("cpu"), os.path.join(cache_dir, "query_enc.sqlite"))  # cached: no re-encoding
    qv = np.asarray(enc.encode_queries([t for _, _, t in reqs])[0], np.float32)
    added = []
    for kind in ("head", "mean"):
        per = []
        for lang in ("vi", "zh"):
            ids, head, mean = doc_vectors(ingest, lang)
            j, s = topk(qv, head if kind == "head" else mean)
            per.append((lang, ids, j, s))
            print(kind, lang, len(ids), "docs", flush=True)
        for r, (qid, v, _) in enumerate(reqs):
            merged = sorted(((float(s[r, c]), ids[j[r, c]], lang) for lang, ids, j, s in per for c in range(j.shape[1])),
                            reverse=True)[:K]
            added.append({"query_id": qid, "variant": f"{v}_{kind}", "channel": "dense",
                          "ids": [f"{d}:0" for _, d, _ in merged], "langs": [l for _, _, l in merged],
                          "scores": [x for x, _, _ in merged]})
    with open(out_path, "wb") as f:
        pickle.dump(lists + added, f)
    print("doc-level lists added:", len(added), "->", out_path, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
