"""Second dense retriever: multilingual-e5-large lists for the orig and zh_q queries, added to an existing lists file
(variants orig_e5 / zh_q_e5, channel dense). Needs the e5 shards in <ingest>/rag/dense/e5-large/<lang>/ (embed_e5_colab
notebook, then fetched locally). Usage:
python bench/add_e5.py <run_dir with LISTS> <ingest> <queries.parquet> <plans.json> <out lists.pkl>"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ingest import dense  # noqa: E402
from ingest.rag import E5Encoder  # noqa: E402
from retrieve import search  # noqa: E402


def main(run_dir, ingest, queries_path, plans_path, out_path):
    lists = pickle.load(open(os.path.join(run_dir, os.environ.get("LISTS", "lists.pkl")), "rb"))
    queries = pd.read_parquet(queries_path, columns=["id", "query"])
    plans = {int(k): v for k, v in json.load(open(plans_path, encoding="utf-8")).items()}
    reqs = []
    for qid, q in zip(queries["id"].astype(int), queries["query"]):
        reqs.append((qid, "orig_e5", q))
        p = plans.get(qid) or {}
        if p.get("ok") and p.get("zh_q"):
            reqs.append((qid, "zh_q_e5", p["zh_q"]))
    enc = E5Encoder(device="cpu", batch_size=32, fp16=False)
    qv, _ = enc.encode_queries([t for _, _, t in reqs])
    per_lang = []
    for lang in ("vi", "zh"):
        shards = search.dense_shards(ingest, "e5-large", lang)
        if not shards:
            raise SystemExit(f"no e5 shards for {lang} under {ingest}")
        per_lang.append((lang, *dense.search(np.asarray(qv, np.float32), shards, k=200)))
    added = []
    for j, (qid, v, _) in enumerate(reqs):
        merged = sorted(((float(s), c, lang) for lang, ids, sc in per_lang for c, s in zip(ids[j], sc[j])), reverse=True)[:200]
        added.append({"query_id": qid, "variant": v, "channel": "dense", "ids": [c for _, c, _ in merged],
                      "langs": [l for _, _, l in merged], "scores": [s for s, _, _ in merged]})
    with open(out_path, "wb") as f:
        pickle.dump(lists + added, f)
    print("e5 lists added:", len(added), "->", out_path, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
