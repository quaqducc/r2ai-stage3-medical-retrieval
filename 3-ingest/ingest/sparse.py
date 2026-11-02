"""Lexical index over bge-m3 sparse (lexical) weights. `rag embed` writes one CSR shard per chunk part
(rows = chunks, columns = XLM-R token ids); `build` stacks them into one CSC index that loads with mmap.
score(query, chunk) = sum over shared tokens of q_weight * chunk_weight (bge-m3's lexical matching score)."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp

VOCAB = 250002  # bge-m3 / XLM-R tokenizer size


def lexical_to_csr(lexical: list[dict], n_cols: int = VOCAB) -> sp.csr_matrix:
    """bge-m3 `lexical_weights` ([{token_id_str: weight}, ...]) -> CSR rows."""
    indptr, indices, data = [0], [], []
    for d in lexical:
        cols = np.fromiter((int(k) for k in d), dtype=np.int32, count=len(d))
        vals = np.fromiter((float(v) for v in d.values()), dtype=np.float32, count=len(d))
        order = np.argsort(cols)
        indices.append(cols[order])
        data.append(vals[order])
        indptr.append(indptr[-1] + len(d))
    cat = (lambda xs, dt: np.concatenate(xs) if xs else np.zeros(0, dt))
    return sp.csr_matrix((cat(data, np.float32), cat(indices, np.int32), np.asarray(indptr, np.int64)),
                         shape=(len(lexical), n_cols))


def save_shard(path: str, ids: list, m: sp.csr_matrix) -> str:
    """Writes <path>.npz + <path>.ids.parquet; returns path (the shard key)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    assert m.shape[0] == len(ids)
    sp.save_npz(path + ".npz", m.tocsr(), compressed=False)
    pd.DataFrame({"id": list(ids)}).to_parquet(path + ".ids.parquet")
    return path


def build(shards: list[str], out: str, n_cols: int = VOCAB) -> dict:
    mats, ids = [], []
    for s in shards:
        mats.append(sp.load_npz(s + ".npz").astype(np.float32, copy=False))
        ids += pd.read_parquet(s + ".ids.parquet").id.tolist()
    W = sp.vstack(mats, format="csr") if mats else sp.csr_matrix((0, n_cols), dtype=np.float32)
    return write_index(W, ids, out, {"shards": len(shards)})


def write_index(W: sp.spmatrix, ids: list, out: str, extra: dict | None = None) -> dict:
    """Rows = chunks (ids), columns = token ids -> CSC index files in `out` (Index loads them with mmap)."""
    W = W.tocsc()
    n_cols = W.shape[1]
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "data.npy"), W.data.astype(np.float32, copy=False))
    np.save(os.path.join(out, "indices.npy"), W.indices.astype(np.int32, copy=False))
    np.save(os.path.join(out, "indptr.npy"), W.indptr.astype(np.int64, copy=False))
    pd.DataFrame({"id": ids}).to_parquet(os.path.join(out, "ids.parquet"))
    params = {"n_docs": len(ids), "nnz": int(W.nnz), "n_cols": n_cols, **(extra or {})}
    with open(os.path.join(out, "params.json"), "w", encoding="utf-8") as f:
        json.dump(params, f)
    return params


class Index:
    def __init__(self, path: str, mmap: bool = True):
        mode = "r" if mmap else None
        self.data = np.load(os.path.join(path, "data.npy"), mmap_mode=mode)
        self.indices = np.load(os.path.join(path, "indices.npy"), mmap_mode=mode)
        self.indptr = np.load(os.path.join(path, "indptr.npy"), mmap_mode=mode)
        self.ids = pd.read_parquet(os.path.join(path, "ids.parquet")).id.to_numpy()
        with open(os.path.join(path, "params.json"), encoding="utf-8") as f:
            self.params = json.load(f)

    def scores(self, query: dict) -> np.ndarray:
        s = np.zeros(len(self.ids), np.float32)
        n_cols = len(self.indptr) - 1
        for c, w in query.items():
            c = int(c)
            if not 0 <= c < n_cols:  # token id outside the index vocabulary
                continue
            a, b = self.indptr[c], self.indptr[c + 1]
            if b > a:  # row indices are unique within a CSC column: plain fancy-index add is exact (and fast)
                s[self.indices[a:b]] += np.float32(w) * self.data[a:b]
        return s

    def search(self, queries: list[dict], k: int = 100) -> tuple[list[np.ndarray], np.ndarray]:
        """queries: bge-m3 lexical weights per query. Top-k ids and scores, descending."""
        k = min(k, len(self.ids))
        out_ids, out_scores = [], np.zeros((len(queries), k), np.float32)
        for qi, q in enumerate(queries):
            s = self.scores(q)
            top = np.argpartition(-s, k - 1)[:k] if k < len(s) else np.arange(len(s))
            top = top[np.argsort(-s[top], kind="stable")]
            out_ids.append(self.ids[top])
            out_scores[qi] = s[top]
        return out_ids, out_scores


def load(path: str, mmap: bool = True) -> Index:
    return Index(path, mmap)
