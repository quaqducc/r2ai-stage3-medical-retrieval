"""End to end on real fixtures: 2-extract clean store -> rag chunk -> rag embed (fake encoder) -> rag index -> search."""
import glob
import os
import sqlite3
import zlib

import numpy as np
import pandas as pd
import pytest

from conftest import FIXTURES
from ingest import dense, rag, sparse

HOSTS = ["www.120ask.com", "ask.39.net", "hellobacsi.com", "pubmed.ncbi.nlm.nih.gov", "vnexpress.net"]


class FakeEncoder:
    """Deterministic stand-in for bge-m3: lexical weight 1 per distinct word (hashed into the vocab), dense = hash vec."""

    def _lex(self, t):
        return {str(zlib.crc32(w.encode()) % sparse.VOCAB): 1.0 for w in t.lower().split()}

    def _vec(self, t):
        v = np.random.default_rng(zlib.crc32(t.encode())).normal(size=16).astype(np.float32)
        return v / np.linalg.norm(v)

    def encode(self, texts):
        return np.stack([self._vec(t) for t in texts]), [self._lex(t) for t in texts]


@pytest.fixture(scope="module")
def clean_store(tmp_path_factory):
    from extract.__main__ import main as extract_main
    root = tmp_path_factory.mktemp("store")
    rows = []
    for h in HOSTS:
        con = sqlite3.connect(os.path.join(FIXTURES, f"{h}.sqlite"))
        lang = "en" if h.startswith("pubmed") else ("zh" if h in ("www.120ask.com", "ask.39.net") else "vi")
        rows += [(i, u, h, lang, "lane") for i, u in con.execute("select id, url from pages")]
    man = str(root / "manifest.parquet")
    pd.DataFrame(rows, columns=["id", "url", "host", "lang", "lane"]).to_parquet(man)
    extract_main(["run", "--db-root", FIXTURES, "--manifest", man, "--out", str(root / "clean"), "--workers", "1",
                  "--workdir", str(root / "w"), "--hosts", ",".join(HOSTS)])
    return str(root / "clean")


def test_chunk_embed_index_search(clean_store, tmp_path):
    out, wd = str(tmp_path / "ingest"), str(tmp_path / "w")
    rep = rag.run_chunk(clean_store, out, workers=1, workdir=wd)
    assert rep["chunks"] > 0 and rep["docs_chunked"] > 0
    assert {"zh", "vi", "en"} <= {os.path.basename(d) for d in glob.glob(os.path.join(out, "rag", "chunks", "*"))}
    chunks = pd.concat(pd.read_parquet(f) for f in glob.glob(os.path.join(out, "rag", "chunks", "*", "*.parquet")))
    cards = pd.concat(pd.read_parquet(f) for f in glob.glob(os.path.join(out, "rag", "docs", "*", "*.parquet")))
    assert chunks.chunk_id.is_unique and set(chunks.doc_id) <= set(cards.doc_id)
    assert (cards[cards.host == "pubmed.ncbi.nlm.nih.gov"].meta_text.str.contains("PMID")).all()

    # resume: a second run does nothing
    again = rag.run_chunk(clean_store, out, workers=1, workdir=wd)
    assert again["parts_this_run"] == 0 and again["chunks"] == rep["chunks"]  # report still covers all parts

    enc = FakeEncoder()
    # two parallel sessions split the parts without overlap, then a plain run finds nothing left
    halves = [rag.run_embed(out, enc, workdir=wd, shard=(i, 2))["chunks"] for i in range(2)]
    assert sum(halves) == len(chunks) and min(halves) > 0
    assert rag.run_embed(out, enc, workdir=wd)["parts_embedded"] == 0  # all shards present
    res = rag.run_index(out, workdir=wd)
    assert sum(r["n_docs"] for r in res.values()) == len(chunks)

    # a chunk's own text finds itself first, sparse and dense
    target = chunks[chunks.lang == "vi"].iloc[0]
    q = rag.embed_input(target.to_dict())
    idx = sparse.load(os.path.join(out, "rag", "sparse_index", "bge-m3", "vi"))
    ids, _ = idx.search([enc._lex(q)], k=3)
    assert ids[0][0] == target.chunk_id
    shards = [p[:-len(".ids.parquet")] for p in glob.glob(os.path.join(out, "rag", "dense", "bge-m3", "vi", "*.ids.parquet"))]
    ids, scores = dense.search(enc._vec(q)[None], shards, k=3)
    assert ids[0][0] == target.chunk_id and np.isclose(scores[0][0], 1.0, atol=1e-2)


def test_embed_input_keeps_chunk_text_verbatim_last():
    c = {"title": "T", "heading_path": "A > B", "text": "body"}
    assert rag.embed_input(c) == "T\nA > B\nbody"
    assert rag.embed_input({"title": "", "heading_path": "", "text": "body"}) == "body"


class _Resp:
    def __init__(self, data, status=200):
        self._data, self.status_code = data, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def json(self):
        return {"data": self._data}


def test_http_encoder_batches_and_retries(monkeypatch):
    calls = []

    def post(url, json, timeout):
        calls.append((url, len(json["input"])))
        if len(calls) == 1:
            return _Resp([], 502)  # transient tunnel error -> retried
        return _Resp([{"dense": [1.0] + [0.0] * 1023, "sparse": {"5": 0.3}} for _ in json["input"]])

    monkeypatch.setattr("time.sleep", lambda s: None)
    enc = rag.HttpEncoder("https://x.example/", batch_size=2)
    monkeypatch.setattr(enc.s, "post", post)
    vecs, lex = enc.encode(["a", "b", "c"])
    assert vecs.shape == (3, 1024) and lex == [{"5": 0.3}] * 3
    assert [n for _, n in calls] == [2, 2, 1] and calls[0][0] == "https://x.example/encode_batch"


def test_http_encoder_rejects_short_response(monkeypatch):
    enc = rag.HttpEncoder("https://x.example", batch_size=4, retries=1)
    monkeypatch.setattr(enc.s, "post", lambda url, json, timeout: _Resp([{"dense": [0.0] * 1024, "sparse": {}}]))
    with pytest.raises(ValueError):
        enc.encode(["a", "b"])


class FakeTok:
    """Offline stand-in for the XLM-R tokenizer: one hashed id per lowercased word."""

    def ids(self, texts):
        return [[zlib.crc32(w.encode()) % sparse.VOCAB for w in t.lower().split()] for t in texts]

    from ingest.bm25 import Tokenizer as _Real
    tf = _Real.tf  # the real counting code, fed by the fake ids()


def test_bm25_weights_rare_terms_and_saturates():
    from ingest import bm25
    import scipy.sparse as sp
    tf = sp.csr_matrix(np.array([[1, 1, 0], [1, 0, 0], [1, 0, 0], [5, 0, 1]], np.float32))
    w = bm25.weigh(tf).toarray()
    assert w[0, 1] > w[0, 0]  # term 1 (in 1 of 4 chunks) outweighs term 0 (in all)
    assert w[3, 0] < 5 * w[1, 0]  # tf saturates


def test_bm25_index_finds_chunk_by_its_text(clean_store, tmp_path, monkeypatch):
    from ingest import bm25
    monkeypatch.setattr(bm25, "Tokenizer", FakeTok)
    out, wd = str(tmp_path / "ingest"), str(tmp_path / "w")
    rag.run_chunk(clean_store, out, workers=1, workdir=wd)
    res = rag.run_bm25(out, workdir=wd)
    chunks = pd.concat(pd.read_parquet(f) for f in glob.glob(os.path.join(out, "rag", "chunks", "*", "*.parquet")))
    assert sum(r["n_docs"] for r in res.values()) == len(chunks)
    target = chunks[chunks.lang == "vi"].iloc[0]
    idx = sparse.load(os.path.join(out, "rag", "sparse_index", "bm25", "vi"))
    q = {str(c): 1.0 for c in set(FakeTok().ids([rag.embed_input(target.to_dict())])[0])}
    ids, _ = idx.search([q], k=1)
    assert ids[0][0] == target.chunk_id
