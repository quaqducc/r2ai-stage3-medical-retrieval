import json
import os
import zipfile

import pandas as pd
import pytest

from conftest import FakeEncoder
from retrieve import search, select, submit
from retrieve.config import Config


def chunks_of(ingest_root):
    import glob
    return pd.concat(pd.read_parquet(f) for f in glob.glob(os.path.join(ingest_root, "rag", "chunks", "*", "*.parquet")))


def test_rrf_fuses_ranked_lists():
    lists = [["a", "b", "c"], ["b", "a"], ["c"]]
    fused = search.rrf(lists, k=60)
    assert list(fused)[:2] == ["a", "b"] or list(fused)[:2] == ["b", "a"]
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62) and fused["c"] == pytest.approx(1 / 63 + 1 / 61)


def test_candidates_find_own_doc(ingest_root):
    ch = chunks_of(ingest_root)
    vi = ch[ch.lang == "vi"]
    a, b = vi.iloc[0], vi[vi.doc_id != vi.iloc[0].doc_id].iloc[0]
    queries = pd.DataFrame({"id": [1, 2], "query": [a.text[:300], b.text[:300]]})
    cand = search.retrieve(queries, ingest_root, FakeEncoder(), Config(k_dense=5, k_sparse=5))
    assert set(cand.columns) >= {"query_id", "chunk_id", "doc_id", "score", "lang"}
    top = cand.sort_values("score", ascending=False).groupby("query_id").head(1).set_index("query_id")
    assert top.loc[1, "doc_id"] == a.doc_id and top.loc[2, "doc_id"] == b.doc_id


def test_select_top_k_docs_and_chunks():
    cand = pd.DataFrame({"query_id": [1, 1, 1, 1, 2], "chunk_id": ["10:0", "10:1", "20:0", "30:0", "40:0"],
                         "doc_id": [10, 10, 20, 30, 40], "score": [0.9, 0.5, 0.8, 0.1, 0.3]})
    sel = select.select(cand, Config(k_docs=2, k_chunks=1))
    assert sel[1]["docs"] == [10, 20] and sel[1]["chunks"] == {10: ["10:0"], 20: ["20:0"]}
    assert sel[2]["docs"] == [40]


def test_merge_adjacent_chunks_keeps_verbatim_spans():
    # doc text "Hỏi?\n\nĐáp án một.\n\nĐáp hai."; qa answer chunks (3-ingest) both carry the question span (0, 4)
    a = {"idx": 0, "spans": [(0, 4), (6, 17)], "text": "Hỏi?\n\nĐáp án một."}
    b = {"idx": 1, "spans": [(0, 4), (19, 27)], "text": "Hỏi?\n\nĐáp hai."}
    assert select.span_texts(a) == {(0, 4): "Hỏi?", (6, 17): "Đáp án một."}
    assert select.merge_chunks([b, a]) == "Hỏi?\n\nĐáp án một.\n\nĐáp hai."
    # adjacent article chunks: one span each
    c = {"idx": 0, "spans": [(0, 5)], "text": "alpha"}
    d = {"idx": 1, "spans": [(7, 11)], "text": "beta"}
    assert select.merge_chunks([c, d]) == "alpha\n\nbeta"


def test_submission_written_and_validated(tmp_path):
    queries = pd.DataFrame({"id": [1, 2, 3], "query": ["a", "b", "c"]})
    entries = {1: {"docs": [10, 20], "chunk_texts": [(10, "x"), (20, "y")]}, 2: {"docs": [], "chunk_texts": []}}
    path = submit.write(queries, entries, str(tmp_path))
    data = json.load(open(os.path.join(tmp_path, "submission.json"), encoding="utf-8"))
    assert [d["id"] for d in data] == [1, 2, 3]  # every query present, even without candidates
    assert data[0] == {"id": 1, "relevant_docs": [10, 20],
                       "relevant_chunks": [{"doc_id": 10, "chunk_text": "x"}, {"doc_id": 20, "chunk_text": "y"}]}
    assert data[2] == {"id": 3, "relevant_docs": [], "relevant_chunks": []}
    with zipfile.ZipFile(path) as z:
        assert z.namelist() == ["submission.json"]


def test_validator_rejects_chunk_outside_docs_and_missing_query(tmp_path):
    queries = pd.DataFrame({"id": [1, 2], "query": ["a", "b"]})
    with pytest.raises(ValueError, match="not in relevant_docs"):
        submit.validate([{"id": 1, "relevant_docs": [10], "relevant_chunks": [{"doc_id": 11, "chunk_text": "x"}]},
                         {"id": 2, "relevant_docs": [], "relevant_chunks": []}], queries)
    with pytest.raises(ValueError, match="missing"):
        submit.validate([{"id": 1, "relevant_docs": [], "relevant_chunks": []}], queries)


def test_run_v0_end_to_end(ingest_root, tmp_path):
    from retrieve.__main__ import run_v0
    ch = chunks_of(ingest_root)
    queries = pd.DataFrame({"id": list(range(1, 6)), "query": [t[:200] for t in ch.text.head(5)]})
    qpath = str(tmp_path / "query.parquet")
    queries.to_parquet(qpath)
    out = run_v0(qpath, ingest_root, str(tmp_path / "run"), FakeEncoder(), Config(k_docs=2, k_chunks=1))
    data = json.load(open(os.path.join(tmp_path, "run", "submission.json"), encoding="utf-8"))
    assert len(data) == 5 and all(len(d["relevant_docs"]) == 2 for d in data)
    texts = set(ch.text)
    assert all(c["chunk_text"] in texts for d in data for c in d["relevant_chunks"])
    assert json.load(open(os.path.join(tmp_path, "run", "config.json")))["k_docs"] == 2
    assert out["queries"] == 5
