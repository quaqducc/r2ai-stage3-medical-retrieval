"""S5 select: fixed top-k (v0) or expected-F2 cut on calibrated p (v1+), optional judge (v2); verbatim chunk assembly."""
from __future__ import annotations

import glob
import logging
import os

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ingest.chunk import split_spans

log = logging.getLogger("retrieve")


def doc_table(cand: pd.DataFrame) -> pd.DataFrame:
    """Per (query, doc): best chunk score, rank within the query, share of lists that hit the doc."""
    if cand.empty:
        return pd.DataFrame(columns=["query_id", "doc_id", "score", "rank", "lists_frac"])
    if "n_lists" not in cand:
        cand = cand.assign(n_lists=1, lists_total=1)
    d = cand.groupby(["query_id", "doc_id"]).agg(score=("score", "max"), n_lists=("n_lists", "max"),
                                                 lists_total=("lists_total", "max")).reset_index()
    d["lists_frac"] = d.n_lists / d.lists_total.clip(lower=1)
    d = d.sort_values(["query_id", "score", "doc_id"], ascending=[True, False, True])
    d["rank"] = d.groupby("query_id").cumcount() + 1
    return d


def calibrate(d: pd.DataFrame, cfg) -> pd.Series:
    """p(doc relevant) without labels: monotone in rank and agreement across lists; parameters tuned on the leaderboard."""
    z = cfg.p_bias - cfg.p_rank * (d["rank"].astype(float) - 1) + cfg.p_lists * d["lists_frac"].astype(float)
    return pd.Series(1 / (1 + np.exp(-np.clip(z.to_numpy(), -50, 50))), index=d.index)


def doc_probs(cand: pd.DataFrame, cfg, learned: pd.DataFrame | None = None) -> pd.DataFrame:
    """doc_table + p: the learned ranker's p (rank re-sorted by p) when given, else the label-free calibration."""
    d = doc_table(cand)
    if d.empty:
        return d.assign(p=[])
    if learned is None:
        d["p"] = calibrate(d, cfg)
        return d
    d = d.drop(columns="rank").merge(learned[["query_id", "doc_id", "p"]], how="left").fillna({"p": 0.0})
    d = d.sort_values(["query_id", "p", "doc_id"], ascending=[True, False, True])
    d["rank"] = d.groupby("query_id").cumcount() + 1
    return d


def f2_cut(p: list[float], missing: float, max_docs: int) -> int:
    """How many of the p-sorted docs to submit: add the next doc while p > E[F2]/5 = S/(k + 4E|G|). At least 1."""
    if not p:
        return 0
    eg = sum(p) + missing
    s, k = p[0], 1
    for pi in p[1:max_docs]:
        if pi > s / (k + 4 * eg):
            s, k = s + pi, k + 1
        else:
            break
    return k


def select(cand: pd.DataFrame, cfg, p_override: dict | None = None, evidence: dict | None = None,
           learned: pd.DataFrame | None = None) -> dict:
    """{query_id: {"docs": [...], "chunks": {doc: [chunk_id...]}, "p": {doc: p}}}.
    p_override / evidence: {(qid, doc): p} and {(qid, doc): [chunk_id...]} from the judge.
    learned: (query_id, doc_id, p) from learn.doc_p when cfg.select == "learned"."""
    out = {}
    cut = cfg.select != "topk"  # "f2" (calibrated p) and "learned" (ranker p) share the expected-F2 cut
    d = doc_probs(cand, cfg, learned) if cut else doc_table(cand)
    if cut and not d.empty:
        if p_override:
            d["p"] = [p_override.get((q, doc), p) for q, doc, p in zip(d.query_id, d.doc_id, d.p)]
    by_chunk = {q: g.sort_values("score", ascending=False, kind="stable") for q, g in cand.groupby("query_id")}
    for qid, g in d.groupby("query_id"):
        if cut:
            g = g.sort_values(["p", "rank"], ascending=[False, True], kind="stable")
            docs = g.doc_id.head(f2_cut(g.p.tolist(), cfg.f2_missing, cfg.max_docs)).tolist()
            pmap = dict(zip(g.doc_id, g.p))
        else:
            docs = g.doc_id.head(cfg.k_docs).tolist()
            pmap = {}
        cs = by_chunk[qid]
        chunks = {}
        for doc in docs:
            ev = (evidence or {}).get((int(qid), int(doc)))
            dc = cs[cs.doc_id == doc]
            if ev:
                chunks[int(doc)] = ev
            elif cut:
                best = dc.score.iloc[0]
                chunks[int(doc)] = dc[dc.score >= cfg.chunk_tau * best].chunk_id.head(cfg.k_chunks_max).tolist()
            else:
                chunks[int(doc)] = dc.chunk_id.head(cfg.k_chunks).tolist()
        if getattr(cfg, "chunk_docs", 0):  # chunks for the first N docs only (file size, scorer time)
            chunks = {int(d): chunks[int(d)] for d in docs[:cfg.chunk_docs] if int(d) in chunks}
        out[int(qid)] = {"docs": [int(x) for x in docs], "chunks": chunks, "p": {int(k): float(v) for k, v in pmap.items()}}
    return out


def span_texts(chunk: dict) -> dict:
    """{(start, end): text} of a chunk, recovered from its verbatim text (3-ingest join_spans invariant)."""
    return split_spans(chunk["text"], chunk["spans"])


def merge_chunks(chunks: list[dict]) -> str:
    """Chunks of one doc -> one verbatim text: union of their spans in document order. Touching spans (consecutive
    article chunks, a qa question and its first answer) concatenate as is, so the result is an exact substring of the
    doc; a shared qa question appears once."""
    pieces = {}
    for c in chunks:
        pieces.update(span_texts(c))
    out, prev = [], None
    for (s, e) in sorted(pieces):
        if out and prev != s:
            out.append("\n\n")
        out.append(pieces[(s, e)])
        prev = e
    return "".join(out)


def groups(chunks: list[dict]) -> list[list[dict]]:
    """Consecutive chunk idx -> one group (merged into one span); others stay separate."""
    out = []
    for c in sorted(chunks, key=lambda c: c["idx"]):
        if out and c["idx"] == out[-1][-1]["idx"] + 1:
            out[-1].append(c)
        else:
            out.append([c])
    return out


def load_chunks(ingest: str, chunk_ids: set) -> dict:
    """chunk_id -> {idx, spans, text, title} for the requested ids. Reads the small chunk_id column of every part, and
    the text only of the rows asked for."""
    out = {}
    if not chunk_ids:
        return out
    import pyarrow as pa
    pa_set = pa.array(sorted(chunk_ids))
    for p in glob.glob(os.path.join(ingest, "rag", "chunks", "*", "*.parquet")):
        pf = pq.ParquetFile(p)
        mask = pc.is_in(pf.read(columns=["chunk_id"]).column("chunk_id"), value_set=pa_set)
        if not pc.any(mask).as_py():
            continue
        t = pf.read(columns=["chunk_id", "idx", "spans", "text", "title"]).filter(mask)
        for r in t.to_pylist():
            out[r["chunk_id"]] = {"idx": int(r["idx"]), "spans": [tuple(int(x) for x in s) for s in r["spans"]],
                                  "text": r["text"], "title": r["title"]}
    missing = len(chunk_ids) - len(out)
    if missing:
        log.warning("load_chunks: %d of %d requested chunks not found in %s", missing, len(chunk_ids), ingest)
    return out


def chunk_texts(sel: dict, store: dict, merge_adjacent: bool = True) -> dict:
    """{query_id: [(doc_id, chunk_text), ...]} in doc order of `sel`."""
    out = {}
    for qid, s in sel.items():
        items = []
        for d in s["docs"]:
            chs = [store[c] for c in s["chunks"].get(d, []) if c in store]
            for grp in (groups(chs) if merge_adjacent else [[c] for c in chs]):
                items.append((d, merge_chunks(grp)))
        out[qid] = items
    return out
