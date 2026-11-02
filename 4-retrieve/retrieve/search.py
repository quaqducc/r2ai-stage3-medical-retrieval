"""S2 retrieve. `collect` encodes every query variant once and runs every channel once, returning ranked lists;
`fuse` turns the subset of lists a mode wants into candidates (weighted RRF at chunk level). Modes share the cost.
Dense is ONE list per variant across languages (merged by cosine), so a language with no relevant chunk does not get
free rank-1 credit. The ingest root must be a local directory (sync it from GCS first): shards/indexes are mmapped."""
from __future__ import annotations

import glob
import logging
import os

import numpy as np
import pandas as pd

from ingest import dense, sparse

log = logging.getLogger("retrieve")


class DegradedRun(RuntimeError):
    """Raised when inputs are missing or stale enough that a submission would be silently bad."""


def rrf(lists: list[list], k: int = 60, weights: list[float] | None = None) -> dict:
    """Reciprocal rank fusion: score(x) = sum over lists of w / (k + rank), rank from 1. Sorted, best first."""
    s = {}
    for li, lst in enumerate(lists):
        w = weights[li] if weights else 1.0
        for r, x in enumerate(lst, 1):
            s[x] = s.get(x, 0.0) + w / (k + r)
    return dict(sorted(s.items(), key=lambda kv: -kv[1]))


def list_weight(rec: dict, cfg) -> float:
    if cfg is None:
        return 1.0
    w = cfg.w_dense if rec["channel"].startswith("dense") else cfg.w_sparse
    v = rec["variant"]
    vw = getattr(cfg, "variant_weight", None) or {}
    if v in vw:  # explicit per-variant weight (e.g. LLM-generated hyde_* lists) replaces the variant-kind factor
        return w * vw[v]
    if v.startswith("subq"):
        w *= cfg.w_subq
    elif v != "orig":
        w *= cfg.w_trans
    return w


def dense_shards(ingest: str, tag: str, lang: str) -> list[str]:
    return sorted(p[:-len(".ids.parquet")] for p in glob.glob(os.path.join(ingest, "rag", "dense", tag, lang, "*.ids.parquet")))


def check_ingest(ingest: str, cfg) -> None:
    """Every chunk part must have its dense shard, and the sparse index must cover every chunk (stale index guard)."""
    problems = []
    for lang in sorted(set(cfg.dense_langs) | set(cfg.sparse_langs)):
        parts = glob.glob(os.path.join(ingest, "rag", "chunks", lang, "*.parquet"))
        if not parts:
            problems.append(f"{lang}: no chunk parts")
            continue
        if lang in cfg.dense_langs and len(dense_shards(ingest, cfg.tag, lang)) != len(parts):
            problems.append(f"{lang}: {len(dense_shards(ingest, cfg.tag, lang))} dense shards for {len(parts)} chunk parts")
        pj = os.path.join(ingest, "rag", "sparse_index", cfg.tag, lang, "params.json")
        if lang in cfg.sparse_langs:
            if not os.path.exists(pj):
                problems.append(f"{lang}: no sparse index")
            else:
                import json
                n_idx = json.load(open(pj, encoding="utf-8"))["n_docs"]
                n_ch = sum(pd.read_parquet(p, columns=["chunk_id"]).shape[0] for p in parts)
                if n_idx != n_ch:
                    problems.append(f"{lang}: sparse index has {n_idx} chunks, chunk parts have {n_ch}")
    if problems:
        msg = "ingest incomplete or stale: " + "; ".join(problems)
        if not cfg.allow_degraded:
            raise DegradedRun(msg)
        log.warning(msg)


def variants(query: str, plan: dict | None) -> list[tuple]:
    """(variant name, text, channels) for one query. Channels: dense (all languages) and sparse_<lang>."""
    out = [("orig", query, ["dense", "sparse_vi"])]
    if plan and plan.get("ok"):
        if plan["vi_kw"]:
            out.append(("vi_kw", plan["vi_kw"], ["sparse_vi"]))
        if plan["zh_q"]:
            out.append(("zh_q", plan["zh_q"], ["sparse_zh", "dense"]))
        if plan["zh_kw"]:
            out.append(("zh_kw", plan["zh_kw"], ["sparse_zh"]))
        if plan["en_kw"]:
            out.append(("en_kw", plan["en_kw"], ["sparse_en"]))
    for i, s in enumerate((plan or {}).get("subqs") or []):
        out.append((f"subq{i}", s, ["dense", "sparse_vi"]))
    return out


def collect(queries: pd.DataFrame, plans: dict | None, ingest: str, encoder, cfg) -> list[dict]:
    """Ranked chunk lists: [{query_id, variant, channel, ids, langs}]. Every distinct text is encoded once."""
    check_ingest(ingest, cfg)
    reqs = []  # (qid, variant, text, channels)
    for qid, q in zip(queries["id"].tolist(), queries["query"].tolist()):
        for v, text, chans in variants(q, (plans or {}).get(int(qid))):
            reqs.append((int(qid), v, text, chans))
    texts = list(dict.fromkeys(t for _, _, t, _ in reqs))
    pos = {t: i for i, t in enumerate(texts)}
    vecs, lex = encoder.encode_queries(texts)
    out = []
    dense_reqs = [r for r in reqs if "dense" in r[3]]
    if dense_reqs:
        qv = np.stack([vecs[pos[t]] for _, _, t, _ in dense_reqs])
        per_lang = []
        for lang in cfg.dense_langs:
            shards = dense_shards(ingest, cfg.tag, lang)
            if shards:
                ids, sc = dense.search(qv, shards, k=cfg.k_dense)
                per_lang.append((lang, ids, sc))
        for j, (q, v, _, _) in enumerate(dense_reqs):  # one list per variant: languages merged by cosine
            merged = sorted(((float(s), c, lang) for lang, ids, sc in per_lang for c, s in zip(ids[j], sc[j])), reverse=True)
            merged = merged[:cfg.k_dense]
            out.append({"query_id": q, "variant": v, "channel": "dense", "ids": [c for _, c, _ in merged],
                        "langs": [l for _, _, l in merged], "scores": [s_ for s_, _, _ in merged]})
    for lang in cfg.sparse_langs:
        rs = [r for r in reqs if f"sparse_{lang}" in r[3]]
        path = os.path.join(ingest, "rag", "sparse_index", cfg.tag, lang)
        if not rs or not os.path.exists(os.path.join(path, "params.json")):
            continue
        ids, scores = sparse.load(path).search([lex[pos[t]] for _, _, t, _ in rs], k=cfg.k_sparse)
        for (q, v, _, _), i, s in zip(rs, ids, scores):
            keep = list(i[s > 0])  # zero score = no shared token
            out.append({"query_id": q, "variant": v, "channel": f"sparse_{lang}", "ids": keep, "langs": [lang] * len(keep),
                        "scores": [float(x) for x in s[s > 0]]})
    if not any(rec["ids"] for rec in out):
        raise DegradedRun("every retrieval list is empty: wrong --ingest path or empty index")
    return out


def fuse(lists: list[dict], rrf_k: int = 60, keep=lambda rec: True, cfg=None) -> pd.DataFrame:
    """Candidates per (query, chunk) from the lists `keep` accepts: weighted RRF score, lists hit, lang.
    Truncated to the top cfg.cand_docs docs per query and cfg.cand_chunks chunks per doc."""
    cand_docs = getattr(cfg, "cand_docs", 50)
    cand_chunks = getattr(cfg, "cand_chunks", 3)
    by_q = {}
    for rec in lists:
        if keep(rec):
            by_q.setdefault(rec["query_id"], []).append(rec)
    rows = []
    for qid, recs in by_q.items():
        lang_of, hits = {}, {}
        for rec in recs:
            for c, l in zip(rec["ids"], rec.get("langs") or [rec.get("lang")] * len(rec["ids"])):
                lang_of.setdefault(c, l)
                hits[c] = hits.get(c, 0) + 1
        docs, per_doc = [], {}
        fused = rrf([r["ids"] for r in recs], rrf_k, [list_weight(r, cfg) for r in recs])
        lw = getattr(cfg, "lang_weight", None)
        if lw:  # e.g. the leaderboard gold has no PubMed: {"en": 0} drops it, {"zh": 0.5} demotes zh
            fused = {c: s * lw.get(lang_of[c], 1.0) for c, s in fused.items() if lw.get(lang_of[c], 1.0) > 0}
            fused = dict(sorted(fused.items(), key=lambda kv: -kv[1]))
        n_agg = getattr(cfg, "doc_agg", 1)
        if n_agg > 1:  # doc score = sum of its n best chunk scores (several matching passages = stronger doc)
            per = {}
            for c, s in fused.items():
                per.setdefault(int(str(c).split(":")[0]), []).append(s)
            dsc = {d: sum(v[:n_agg]) for d, v in per.items()}  # fused is sorted, so v is too
            fused = {c: dsc[int(str(c).split(":")[0])] + 1e-3 * s for c, s in fused.items()}  # tie-break: chunk order
            fused = dict(sorted(fused.items(), key=lambda kv: -kv[1]))
        for c, sc in fused.items():
            d = int(str(c).split(":")[0])
            if d not in per_doc:
                if len(docs) >= cand_docs:
                    continue
                docs.append(d)
                per_doc[d] = 0
            if per_doc[d] >= cand_chunks:
                continue
            per_doc[d] += 1
            rows.append({"query_id": int(qid), "chunk_id": c, "doc_id": d, "lang": lang_of[c], "score": sc,
                         "n_lists": hits[c], "lists_total": len(recs)})
    return pd.DataFrame(rows, columns=["query_id", "chunk_id", "doc_id", "lang", "score", "n_lists", "lists_total"])


def retrieve(queries: pd.DataFrame, ingest: str, encoder, cfg, plans: dict | None = None) -> pd.DataFrame:
    """v0 convenience: original-query lists only."""
    return fuse(collect(queries, plans, ingest, encoder, cfg), cfg.rrf_k, keep=lambda r: r["variant"] == "orig", cfg=cfg)
