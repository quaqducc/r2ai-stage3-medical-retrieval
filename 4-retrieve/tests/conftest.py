import os
import sqlite3
import sys
import zlib

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MP = os.path.dirname(ROOT)
sys.path[:0] = [ROOT, os.path.join(MP, "3-ingest"), os.path.join(MP, "2-extract")]
FIXTURES = os.path.join(MP, "2-extract", "tests", "fixtures", "db")
HOSTS = ["www.120ask.com", "ask.39.net", "hellobacsi.com", "pubmed.ncbi.nlm.nih.gov", "vnexpress.net", "www.vinmec.com"]


class FakeEncoder:
    """Deterministic bge-m3 stand-in: lexical = hashed words, dense = bag-of-words random projection (similar texts
    -> similar vectors, so a query made of a chunk's words finds that chunk)."""
    DIM = 256

    def _words(self, t):
        import re
        return re.findall(r"[一-鿿]|[^\W_]+", t.lower())  # CJK: one token per character

    def _lex(self, t):
        from ingest import sparse
        return {str(zlib.crc32(w.encode()) % sparse.VOCAB): 1.0 for w in self._words(t)}

    def _vec(self, t):
        v = np.zeros(self.DIM, np.float32)
        for w in self._words(t):
            v += np.random.default_rng(zlib.crc32(w.encode())).normal(size=self.DIM).astype(np.float32)
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode(self, texts):
        return (np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.DIM), np.float32),
                [self._lex(t) for t in texts])

    encode_queries = encode


@pytest.fixture(scope="session")
def ingest_root(tmp_path_factory):
    """2-extract clean store -> 3-ingest chunks + fake embeddings + sparse index, from real fixture pages."""
    from extract.__main__ import main as extract_main
    from ingest import rag
    root = tmp_path_factory.mktemp("store")
    rows = []
    for h in HOSTS:
        lang = "en" if h.startswith("pubmed") else ("zh" if h in ("www.120ask.com", "ask.39.net") else "vi")
        rows += [(i, u, h, lang, "lane") for i, u in sqlite3.connect(os.path.join(FIXTURES, f"{h}.sqlite")).execute("select id, url from pages")]
    man = str(root / "manifest.parquet")
    pd.DataFrame(rows, columns=["id", "url", "host", "lang", "lane"]).to_parquet(man)
    extract_main(["run", "--db-root", FIXTURES, "--manifest", man, "--out", str(root / "clean"), "--workers", "1",
                  "--workdir", str(root / "w"), "--hosts", ",".join(HOSTS)])
    out = str(root / "ingest")
    rag.run_chunk(str(root / "clean"), out, workers=1, workdir=str(root / "w2"))
    rag.run_embed(out, FakeEncoder(), workdir=str(root / "w3"))
    rag.run_index(out, workdir=str(root / "w4"))
    return out
