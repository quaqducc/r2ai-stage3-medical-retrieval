"""CPU-only lexical index: Okapi BM25 over bge-m3's XLM-R tokens (tokenizer only, no model forward pass).
Same on-disk layout as the bge-m3 sparse index (rag/sparse_index/<tag>/<lang>/), so retrieval treats it as one more
sparse channel: chunk-side weights = idf * saturated tf, query-side weight = query term count.
Used when no GPU is available for `rag embed`."""
from __future__ import annotations

import logging
import os

import numpy as np
import scipy.sparse as sp

from . import sparse

log = logging.getLogger("ingest")
K1, B = 1.2, 0.75
TAG = "bm25"


class Tokenizer:
    def __init__(self, model: str = "BAAI/bge-m3"):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model)

    def ids(self, texts: list[str]) -> list[list[int]]:
        """Lowercased XLM-R token ids, no special tokens (case would split vocabulary for a lexical match)."""
        return self.tok([t.lower() for t in texts], add_special_tokens=False)["input_ids"]

    def tf(self, texts: list[str]) -> sp.csr_matrix:
        rows = self.ids(texts)
        indptr, indices, data = [0], [], []
        for r in rows:
            cols, cnt = np.unique(np.asarray(r, np.int32), return_counts=True)
            indices.append(cols)
            data.append(cnt.astype(np.float32))
            indptr.append(indptr[-1] + len(cols))
        cat = (lambda xs, dt: np.concatenate(xs) if xs else np.zeros(0, dt))
        return sp.csr_matrix((cat(data, np.float32), cat(indices, np.int32), np.asarray(indptr, np.int64)),
                             shape=(len(rows), sparse.VOCAB))


def weigh(tf: sp.csr_matrix) -> sp.csr_matrix:
    """tf (rows = chunks) -> BM25 chunk-side weights idf(t) * tf*(k1+1) / (tf + k1*(1 - b + b*dl/avgdl))."""
    tf = tf.tocsr().astype(np.float32)
    n = tf.shape[0]
    dl = np.asarray(tf.sum(1)).ravel()
    avgdl = float(dl.mean()) if n else 1.0
    df = np.bincount(tf.indices, minlength=tf.shape[1])
    idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
    row_dl = np.repeat(dl, np.diff(tf.indptr))
    w = tf.copy()
    w.data = idf[tf.indices] * tf.data * (K1 + 1) / (tf.data + K1 * (1 - B + B * row_dl / max(avgdl, 1e-9)))
    return w


class QueryEncoder:
    """retrieve's encoder interface: (dense placeholder, [{token_id: count}]). Dense is a 1-dim zero: no dense here."""
    name = TAG

    def __init__(self, model: str = "BAAI/bge-m3"):
        self.t = Tokenizer(model)

    def encode_queries(self, texts: list[str]):
        lex = []
        for r in self.t.ids(texts):
            cols, cnt = np.unique(np.asarray(r, np.int64), return_counts=True)
            lex.append({str(int(c)): float(k) for c, k in zip(cols, cnt)})
        return np.zeros((len(texts), 1), np.float32), lex

    encode = encode_queries


def build_lang(texts_by_part: list[tuple[list, list[str]]], out: str, tok: Tokenizer) -> dict:
    """[(chunk ids, model inputs)] of one language -> BM25 index in `out`."""
    ids, mats = [], []
    for part_ids, texts in texts_by_part:
        ids += list(part_ids)
        mats.append(tok.tf(texts))
    tf = sp.vstack(mats, format="csr") if mats else sp.csr_matrix((0, sparse.VOCAB), dtype=np.float32)
    return sparse.write_index(weigh(tf), ids, out, {"kind": "bm25", "k1": K1, "b": B})
