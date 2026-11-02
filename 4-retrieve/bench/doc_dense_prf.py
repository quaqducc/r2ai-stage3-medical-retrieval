"""Doc-level bge-m3 lists for the PRF query vectors: q' = normalise(q + alpha * mean(top-n chunk vectors of the q dense
list)), searched over each doc's first chunk (<v>_prf_head) and mean chunk vector (<v>_prf_mean). Combines the two
signals that each helped (dense PRF, doc-level lists). Usage:
python bench/doc_dense_prf.py <run_dir with LISTS> <ingest> <queries.parquet> <plans.json> <cache_dir> <out lists.pkl>"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from doc_dense import K, doc_vectors, topk  # noqa: E402
from prf import chunk_vectors  # noqa: E402
from retrieve.__main__ import query_encoder  # noqa: E402
from retrieve.encode_cache import CachedEncoder  # noqa: E402

N, ALPHA = 10, 0.5  # same PRF settings as lists_prf.pkl


def main(run_dir, ingest, queries_path, plans_path, cache_dir, out_path):
    lists = pickle.load(open(os.path.join(run_dir, os.environ.get("LISTS", "lists.pkl")), "rb"))
    queries = pd.read_parquet(queries_path, columns=["id", "query"])
    plans = {int(k): v for k, v in json.load(open(plans_path, encoding="utf-8")).items()}
    text = {}
    for qid, q in zip(queries["id"].astype(int), queries["query"]):
        text[(qid, "orig")] = q
        p = plans.get(qid) or {}
        if p.get("ok") and p.get("zh_q"):
            text[(qid, "zh_q")] = p["zh_q"]
    recs = [r for r in lists if r["channel"] == "dense" and r["variant"] in ("orig", "zh_q") and r["ids"]
            and (r["query_id"], r["variant"]) in text]
    vec = chunk_vectors(ingest, ("vi", "zh"), {c for r in recs for c in r["ids"][:N]})
    enc = CachedEncoder(query_encoder("cpu"), os.path.join(cache_dir, "query_enc.sqlite"))
    qv, _ = enc.encode_queries([text[(r["query_id"], r["variant"])] for r in recs])
    q2 = []
    for u, r in zip(qv, recs):
        fb = [vec[c] for c in r["ids"][:N] if c in vec]
        x = np.asarray(u, np.float32) + (ALPHA * np.mean(fb, axis=0) if fb else 0)
        q2.append(x / np.linalg.norm(x))
    q2 = np.stack(q2).astype(np.float32)
    added = []
    for kind in ("head", "mean"):
        per = []
        for lang in ("vi", "zh"):
            ids, head, mean = doc_vectors(ingest, lang)
            j, s = topk(q2, head if kind == "head" else mean)
            per.append((lang, ids, j, s))
        for r_i, r in enumerate(recs):
            merged = sorted(((float(s[r_i, c]), ids[j[r_i, c]], lang) for lang, ids, j, s in per for c in range(j.shape[1])),
                            reverse=True)[:K]
            added.append({"query_id": r["query_id"], "variant": f"{r['variant']}_prf_{kind}", "channel": "dense",
                          "ids": [f"{d}:0" for _, d, _ in merged], "langs": [l for _, _, l in merged],
                          "scores": [x for x, _, _ in merged]})
        print(kind, "done", flush=True)
    with open(out_path, "wb") as f:
        pickle.dump(lists + added, f)
    print("doc-level PRF lists added:", len(added), "->", out_path, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
