"""Dense vectors stored as float16 shards (one per chunk part). Exact search for a batch of queries: stream the
shards, matmul, keep a running top-k. No ANN index: 1,200 fixed queries make brute force cheap and exact."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd


def save_shard(path: str, ids: list, vecs: np.ndarray) -> str:
    """Writes <path>.npy (float16, L2-normalised by the encoder) + <path>.ids.parquet; returns path."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    assert vecs.shape[0] == len(ids)
    np.save(path + ".npy", vecs.astype(np.float16))
    pd.DataFrame({"id": list(ids)}).to_parquet(path + ".ids.parquet")
    return path


def search(queries: np.ndarray, shards: list[str], k: int = 100, block: int = 512) -> tuple[list[np.ndarray], np.ndarray]:
    """queries: (n_q, dim) normalised. Top-k ids and cosine scores per query across all shards (exact).
    Memory stays bounded: queries go in blocks, each shard is reduced to its own top-k before merging, and rows are
    tracked as int64 global positions (ids are looked up once at the end)."""
    q = np.ascontiguousarray(queries, dtype=np.float32)
    nq = len(q)
    best_s = np.full((nq, 0), -np.inf, np.float32)
    best_r = np.zeros((nq, 0), np.int64)
    all_ids, offset = [], 0
    for s in shards:
        v = np.load(s + ".npy", mmap_mode="r")
        n = len(v)
        if not n:
            continue
        all_ids.append(pd.read_parquet(s + ".ids.parquet").id.to_numpy())
        vt = np.asarray(v, dtype=np.float32).T
        kk = min(k, n)
        sh_s = np.empty((nq, kk), np.float32)
        sh_r = np.empty((nq, kk), np.int64)
        for i in range(0, nq, block):
            sc = q[i:i + block] @ vt
            top = np.argpartition(-sc, kk - 1, axis=1)[:, :kk] if kk < n else np.broadcast_to(np.arange(n), sc.shape)
            sh_s[i:i + block] = np.take_along_axis(sc, top, axis=1)
            sh_r[i:i + block] = top + offset
        offset += n
        cat_s = np.concatenate([best_s, sh_s], axis=1)
        cat_r = np.concatenate([best_r, sh_r], axis=1)
        kk = min(k, cat_s.shape[1])
        top = np.argpartition(-cat_s, kk - 1, axis=1)[:, :kk]
        best_s, best_r = np.take_along_axis(cat_s, top, axis=1), np.take_along_axis(cat_r, top, axis=1)
    order = np.argsort(-best_s, axis=1, kind="stable")
    best_s, best_r = np.take_along_axis(best_s, order, axis=1), np.take_along_axis(best_r, order, axis=1)
    ids = np.concatenate(all_ids) if all_ids else np.zeros(0, dtype=object)
    return [ids[r] for r in best_r], best_s
