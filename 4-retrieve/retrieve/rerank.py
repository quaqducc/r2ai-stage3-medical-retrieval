"""Dense rerank on CPU for the BM25-only index: embed only the fused top candidates of each query with bge-m3
(tens of thousands of chunks instead of the 1.57M corpus) and turn their cosine with each dense-capable query variant
into ordinary "dense" lists, so the learned ranker gets its dense features back. Chunk vectors are cached by chunk id."""
from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import replace

import numpy as np

from . import search, select

log = logging.getLogger("retrieve")
DENSE_VARIANTS = ("orig", "zh_q")  # + subq*: the variants `search.variants` sends to the dense channel


class ChunkVectors:
    def __init__(self, encoder, path: str, max_chars: int = 2000):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.enc, self.max_chars = encoder, max_chars
        self.con = sqlite3.connect(path)
        self.con.execute("create table if not exists v (id text primary key, dense blob)")

    def get(self, ids: list[str], texts: dict) -> dict:
        out = {}
        for i in range(0, len(ids), 500):
            part = ids[i:i + 500]
            q = f"select id, dense from v where id in ({','.join('?' * len(part))})"
            out.update({k: np.frombuffer(d, np.float32) for k, d in self.con.execute(q, part)})
        todo = [c for c in ids if c not in out and c in texts]
        for i in range(0, len(todo), 64):  # commit per batch: an interrupted run keeps what it encoded
            part = todo[i:i + 64]
            vecs, _ = self.enc.encode([texts[c][:self.max_chars] for c in part])
            for c, v in zip(part, vecs):
                v = np.asarray(v, np.float32)
                self.con.execute("insert or replace into v values (?,?)", (c, v.tobytes()))
                out[c] = v
            self.con.commit()
            if i % 6400 == 0:
                log.info("rerank: encoded %d / %d candidate chunks", i + len(part), len(todo))
        return out


def dense_lists(lists: list[dict], queries, plans, ingest: str, query_enc, chunk_vecs: ChunkVectors, cfg) -> list[dict]:
    """Dense records for every dense-capable variant, over the fused top cfg.rerank_docs docs (best chunk each)."""
    cand = search.fuse(lists, cfg.rrf_k, cfg=replace(cfg, cand_docs=cfg.rerank_docs, cand_chunks=1))
    store = select.load_chunks(ingest, set(cand.chunk_id))
    texts = {c: "\n".join(x for x in (s["title"], s["text"]) if x) for c, s in store.items()}
    vec = chunk_vecs.get(sorted(texts), texts)
    by_q = {int(q): g for q, g in cand.groupby("query_id")}
    reqs = []
    for qid, q in zip(queries["id"].tolist(), queries["query"].tolist()):
        for v, text, chans in search.variants(q, (plans or {}).get(int(qid))):
            if "dense" in chans and int(qid) in by_q:
                reqs.append((int(qid), v, text))
    qv, _ = query_enc.encode_queries([t for _, _, t in reqs])
    out = []
    for (qid, v, _), u in zip(reqs, qv):
        g = by_q[qid]
        ids = [c for c in g.chunk_id if c in vec]
        if not ids:
            continue
        s = np.stack([vec[c] for c in ids]) @ np.asarray(u, np.float32)
        order = np.argsort(-s, kind="stable")
        lang = dict(zip(g.chunk_id, g.lang))
        out.append({"query_id": qid, "variant": v, "channel": "dense", "ids": [ids[i] for i in order],
                    "langs": [lang[ids[i]] for i in order], "scores": [float(s[i]) for i in order]})
    log.info("rerank: %d dense lists over %d candidate chunks", len(out), len(vec))
    return out


class HybridQueryEncoder:
    """BM25 lexical weights + bge-m3 dense vector for each query (the BM25 index has no dense side)."""
    name = "bm25+bge-m3"

    def __init__(self, lexical, dense):
        self.lexical, self.dense = lexical, dense

    def encode_queries(self, texts: list[str]):
        return self.dense.encode_queries(texts)[0], self.lexical.encode_queries(texts)[1]

    encode = encode_queries
