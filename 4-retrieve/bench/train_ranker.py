"""Train the learned doc scorer on the silver dev set with query-grouped 5-fold CV, compare with RRF, save the model.
Usage: LLM_API_KEY=... python bench/train_ranker.py <bench_dir> <bge-m3 url> <run_dir with plans.json>"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from retrieve import learn, search, select  # noqa: E402
from retrieve.__main__ import KEEP, query_encoder  # noqa: E402
from retrieve.config import Config  # noqa: E402
from retrieve.encode_cache import CachedEncoder  # noqa: E402
from retrieve.metric import evaluate  # noqa: E402

MODELS = os.environ.get("MODELS_DIR", os.path.join(os.path.dirname(HERE), "models"))


def submit_from_p(pt, cand, store, gold_ids, missing, max_docs):
    """Docs by p with the expected-F2 cut, best fused chunk per doc -> predictions for the metric."""
    pred = []
    by_q = {q: g for q, g in cand.groupby("query_id")}
    for q in gold_ids:
        g = pt[pt.query_id == q].sort_values("p", ascending=False)
        k = select.f2_cut(g.p.tolist(), missing, max_docs)
        docs = g.doc_id.head(k).tolist()
        cs = by_q.get(q)
        chunks = []
        for d in docs:
            if cs is not None:
                c = cs[cs.doc_id == d].sort_values("score", ascending=False).chunk_id.head(1).tolist()
                if c and c[0] in store:
                    chunks.append({"doc_id": int(d), "chunk_text": store[c[0]]["text"]})
        pred.append({"id": int(q), "relevant_docs": [int(d) for d in docs], "relevant_chunks": chunks})
    return pred


def main(bench, server, run_dir):
    dev = pd.read_parquet(os.path.join(bench, "dev.parquet"))
    gold = [{"id": int(r.id), **json.loads(r.gold)} for r in dev.itertuples()]
    gdocs = {g["id"]: set(g["relevant_docs"]) for g in gold}
    plans = {int(k): v for k, v in json.load(open(os.path.join(run_dir, "plans.json"), encoding="utf-8")).items()}
    cached = next((p for p in (os.path.join(run_dir, f) for f in ("lists_scored.pkl", "lists.pkl"))
                   if os.path.exists(p) and all("scores" in r for r in pickle.load(open(p, "rb")))), None)
    if cached:
        with open(cached, "rb") as f:
            lists = pickle.load(f)
    else:
        enc = CachedEncoder(query_encoder(server), os.path.join(bench, "cache", "query_enc.sqlite"))
        lists = search.collect(dev[["id", "query"]], plans, os.path.join(bench, "ingest"), enc, Config())
        with open(os.path.join(run_dir, "lists_scored.pkl"), "wb") as f:
            pickle.dump(lists, f)
    ing = os.path.join(bench, "ingest")
    report = {}
    for mode in ("v1", "v2"):
        df = learn.features(lists, KEEP[mode])
        y = np.array([d in gdocs[q] for q, d in zip(df.query_id, df.doc_id)], dtype=int)
        qs = np.array(sorted(df.query_id.unique()))
        rng = np.random.default_rng(0)
        folds = np.array_split(rng.permutation(qs), 5)
        oof = np.zeros(len(df))
        for f in folds:
            te = df.query_id.isin(f).to_numpy()
            oof[te] = learn.Ranker.fit(df[~te], y[~te]).p(df[te])
        pt = df.assign(p=oof)
        cand = search.fuse(lists, 60, keep=KEEP[mode], cfg=Config(cand_docs=200, cand_chunks=3))
        store = select.load_chunks(ing, set(cand.chunk_id))
        ids = [g["id"] for g in gold]
        # ranking quality: learned p vs RRF
        def r_at(table, col, k):
            top = table.sort_values(col, ascending=False).groupby("query_id").head(k).groupby("query_id").doc_id.apply(set)
            return round(float(np.mean([bool(top.get(q, set()) & gdocs[q]) for q in ids])), 3)
        rep = {"R@1 rrf": r_at(pt, "rrf", 1), "R@1 learned": r_at(pt, "p", 1),
               "R@5 rrf": r_at(pt, "rrf", 5), "R@5 learned": r_at(pt, "p", 5)}
        for m in (0.1, 0.3, 0.5):
            r = evaluate(submit_from_p(pt, cand, store, ids, m, 5), gold)
            rep[f"final m={m}"] = round(r["final"], 4)
            rep[f"docs/q m={m}"] = round(float(np.mean([len(p["relevant_docs"]) for p in submit_from_p(pt, cand, store, ids, m, 5)])), 2)
        by = {}
        for s in sorted(set(f"{r.lang}/{r.doc_type}" for r in dev.itertuples())):
            sid = [int(r.id) for r in dev.itertuples() if f"{r.lang}/{r.doc_type}" == s]
            by[s] = round(evaluate(submit_from_p(pt, cand, store, sid, 0.3, 5), [g for g in gold if g["id"] in sid])["final"], 3)
        rep["by_stratum m=0.3"] = by
        report[mode] = rep
        os.makedirs(MODELS, exist_ok=True)
        final = learn.Ranker.fit(df, y)  # one model per mode: its lists (so n_lists, d_subq) differ
        final.save(os.path.join(MODELS, f"ranker_{mode}.json"))
        rep["coef"] = {k: round(float(v), 3) for k, v in sorted(zip(learn.FEATURES, final.coef), key=lambda kv: -abs(kv[1]))}
    print(json.dumps(report, ensure_ascii=False, indent=1))
    with open(os.path.join(run_dir, "ranker_cv.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
