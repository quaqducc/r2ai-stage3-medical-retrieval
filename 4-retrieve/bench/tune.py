"""Grid search over fusion / selection knobs on saved ranked lists (no re-encoding, no LLM): reports doc recall@k of
the fused ranking and the metric of the selected submission. Usage: python bench/tune.py <bench_dir> <run_dir>
Silver questions have one source doc: trust it for ranking/chunk knobs; tune how many docs to submit on the leaderboard."""
import itertools
import json
import os
import pickle
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from retrieve import search, select  # noqa: E402
from retrieve.__main__ import KEEP  # noqa: E402
from retrieve.config import for_mode  # noqa: E402
from retrieve.metric import evaluate  # noqa: E402


def recall_at(cand: pd.DataFrame, gold: dict, ks=(1, 5, 20, 100)) -> dict:
    d = select.doc_table(cand)
    out = {}
    for k in ks:
        top = d[d["rank"] <= k].groupby("query_id").doc_id.apply(set).to_dict()
        out[f"R@{k}"] = sum(bool(top.get(q, set()) & set(g)) for q, g in gold.items()) / len(gold)
    return out


def main(bench, run_dir):
    dev = pd.read_parquet(os.path.join(bench, "dev.parquet"))
    gold = [{"id": int(r.id), **json.loads(r.gold)} for r in dev.itertuples()]
    gdocs = {g["id"]: g["relevant_docs"] for g in gold}
    queries = dev[["id", "query"]]
    lists = pickle.load(open(os.path.join(run_dir, "lists.pkl"), "rb"))
    ing = os.path.join(bench, "ingest")
    all_c = {c for rec in lists for c in rec["ids"][:50]}
    store = select.load_chunks(ing, all_c)
    rows = []
    grid = itertools.product(["v0", "v1", "v2"], [20, 60], [0.5, 1.0, 2.0], [0.5, 1.0, 2.0])
    for mode, rrf_k, w_sparse, w_trans in grid:
        if mode == "v0" and w_trans != 1.0:
            continue
        cfg = for_mode(mode)
        cfg.rrf_k, cfg.w_sparse, cfg.w_trans = rrf_k, w_sparse, w_trans
        cand = search.fuse(lists, rrf_k, keep=KEEP[mode], cfg=cfg)
        base = {"mode": mode, "rrf_k": rrf_k, "w_sparse": w_sparse, "w_trans": w_trans, **recall_at(cand, gdocs)}
        for k_docs, tau, kmax in itertools.product([1, 2], [0.0, 0.8, 1.0], [1, 3]):
            cfg.select, cfg.k_docs, cfg.chunk_tau, cfg.k_chunks_max, cfg.k_chunks = "topk", k_docs, tau, kmax, kmax
            sel = select.select(cand, cfg)
            if tau:  # topk with a score floor relative to the doc's best chunk
                for q, s in sel.items():
                    for d in s["docs"]:
                        cs = cand[(cand.query_id == q) & (cand.doc_id == d)].sort_values("score", ascending=False)
                        s["chunks"][d] = cs[cs.score >= tau * cs.score.iloc[0]].chunk_id.head(kmax).tolist()
            texts = select.chunk_texts(sel, store, True)
            pred = [{"id": q, "relevant_docs": s["docs"],
                     "relevant_chunks": [{"doc_id": d, "chunk_text": t} for d, t in texts.get(q, [])]} for q, s in sel.items()]
            r = evaluate(pred, gold)
            rows.append({**base, "k_docs": k_docs, "chunk_tau": tau, "k_chunks": kmax,
                         **{k: round(r[k], 4) for k in ("final", "doc_f2", "chunk_f2", "chunk_p", "chunk_r")}})
    df = pd.DataFrame(rows).sort_values("final", ascending=False)
    df.to_csv(os.path.join(run_dir, "tune.csv"), index=False)
    pd.set_option("display.width", 220)
    print(df.head(15).to_string(index=False))
    print("\nbest per mode:\n", df.groupby("mode").head(1).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
