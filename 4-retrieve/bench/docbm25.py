"""Doc-level BM25 over plain word tokens (Vietnamese syllables), closer to a textbook BM25 than the chunk-level XLM-R
index: probes whether the leaderboard gold follows such a BM25 (organizer BM25 baseline: docs P .88 / R .51).
Usage: python bench/docbm25.py build <ingest root> <index dir> [langs=vi]
       python bench/docbm25.py probe <ingest root> <index dir> <queries.parquet> <out dir> <k docs> <k chunk docs>"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ingest.bm25 import weigh  # noqa: E402
from retrieve import submit  # noqa: E402

HV = HashingVectorizer(n_features=2 ** 22, alternate_sign=False, norm=None, lowercase=True,
                       token_pattern=r"(?u)\b\w+\b", dtype=np.float32)


def parts(ingest, langs):
    return sorted(p for l in langs for p in glob.glob(os.path.join(ingest, "rag", "chunks", l, "*.parquet")))


def build(ingest, out, langs="vi"):
    ids, mats = [], []
    for p in parts(ingest, langs.split(",")):
        df = pq.read_table(p, columns=["doc_id", "idx", "title", "text"]).to_pandas().sort_values(["doc_id", "idx"])
        docs = df.groupby("doc_id", sort=True).agg(title=("title", "first"), text=("text", "\n".join))
        ids += docs.index.tolist()
        mats.append(HV.transform((docs.title.fillna("") + "\n" + docs.text).tolist()))
        print(p, len(docs), flush=True)
    tf = sp.vstack(mats, format="csr")
    n = tf.shape[0]
    df_ = np.bincount(tf.indices, minlength=tf.shape[1])
    idf = np.log1p((n - df_ + 0.5) / (df_ + 0.5)).astype(np.float32)
    os.makedirs(out, exist_ok=True)
    sp.save_npz(os.path.join(out, "wT.npz"), weigh(tf).T.tocsr(), compressed=False)  # terms x docs, for Q @ wT
    np.save(os.path.join(out, "ids.npy"), np.asarray(ids, np.int64))
    np.save(os.path.join(out, "idf.npy"), idf)
    print("docs", n, "nnz", tf.nnz, flush=True)


def best_chunks(ingest, langs, need: dict, qtf: dict, idf) -> dict:
    """need: {doc_id: [query ids]}, qtf: {qid: sparse 1 x F} -> {(qid, doc_id): chunk text} = the doc's chunk with the most idf-weighted
    query-term overlap."""
    out, ids = {}, pa.array(sorted(need), pa.int64())
    for p in parts(ingest, langs):
        t = pq.read_table(p, columns=["doc_id", "text"])
        t = t.filter(pc.is_in(t.column("doc_id"), value_set=ids))
        if not t.num_rows:
            continue
        df = t.to_pandas()
        X = (HV.transform(df.text.tolist()) > 0).astype(np.float32).multiply(idf).tocsr()
        for doc, g in df.groupby("doc_id").groups.items():
            rows = X[list(g)]
            for q in need[doc]:
                s = (rows @ qtf[q].T).toarray().ravel()
                out[(q, int(doc))] = df.text.iloc[list(g)[int(np.argmax(s))]]
    return out


def probe(ingest, index, queries_path, out_dir, k, kc, langs="vi"):
    k, kc = int(k), int(kc)
    wT = sp.load_npz(os.path.join(index, "wT.npz")).tocsr()
    ids, idf = np.load(os.path.join(index, "ids.npy")), np.load(os.path.join(index, "idf.npy"))
    q = pd.read_parquet(queries_path, columns=["id", "query"])
    Q = HV.transform(q["query"].tolist()).tocsr()
    top = {}
    for a in range(0, Q.shape[0], 64):
        S = (Q[a:a + 64] @ wT).toarray()
        for i, row in enumerate(S):
            j = np.argpartition(-row, k)[:k]
            j = j[np.argsort(-row[j], kind="stable")]
            top[int(q.id.iloc[a + i])] = [int(ids[x]) for x in j if row[x] > 0]
    need = {}
    for qid, docs in top.items():
        for d in docs[:kc]:
            need.setdefault(d, []).append(qid)
    qtf = {int(qid): Q[i] for i, qid in enumerate(q.id)}  # sparse rows (dense would be 4M floats per query)
    texts = best_chunks(ingest, langs.split(","), need, qtf, idf)
    entries = {qid: {"docs": docs, "chunk_texts": [(d, texts[(qid, d)]) for d in docs[:kc] if (qid, d) in texts]}
               for qid, docs in top.items()}
    print(submit.write(q, entries, out_dir, 1.0), flush=True)


if __name__ == "__main__":
    {"build": build, "probe": probe}[sys.argv[1]](*sys.argv[2:])
