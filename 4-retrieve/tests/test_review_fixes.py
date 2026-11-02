"""Regression tests for the code review of 2026-10-20."""
import json
import os

import pandas as pd
import pytest

from conftest import FakeEncoder
from retrieve import search, select, submit
from retrieve.config import Config, for_mode
from retrieve.search import DegradedRun


def test_calibrate_no_overflow_at_deep_ranks():
    d = pd.DataFrame({"rank": [1, 800, 5000], "lists_frac": [1.0, 0.0, 0.0]})
    p = select.calibrate(d, for_mode("v1"))
    assert p.iloc[0] > 0.9 and p.iloc[2] < 1e-6


def test_fuse_truncates_candidates():
    lists = [{"query_id": 1, "variant": "orig", "channel": "dense", "ids": [f"{d}:{i}" for d in range(100) for i in range(5)],
              "langs": ["vi"] * 500}]
    cand = search.fuse(lists, 60, cfg=Config(cand_docs=10, cand_chunks=2))
    assert cand.doc_id.nunique() == 10 and cand.groupby("doc_id").size().max() == 2


def test_dense_is_one_list_per_variant(ingest_root):
    q = pd.DataFrame({"id": [1], "query": ["đau đầu"]})
    lists = search.collect(q, None, ingest_root, FakeEncoder(), Config(k_dense=5, k_sparse=5))
    dense = [r for r in lists if r["channel"] == "dense"]
    assert len(dense) == 1 and len(dense[0]["ids"]) == 5 and len(set(dense[0]["langs"])) >= 1


def test_wrong_ingest_path_raises(tmp_path):
    q = pd.DataFrame({"id": [1], "query": ["x"]})
    with pytest.raises(DegradedRun):
        search.collect(q, None, str(tmp_path / "nope"), FakeEncoder(), Config())


def test_plan_guard_when_llm_is_down(ingest_root, tmp_path):
    from retrieve.__main__ import run_modes

    class DeadLLM:
        def map_json(self, prompts):
            return [None] * len(prompts)
    qp = str(tmp_path / "q.parquet")
    pd.DataFrame({"id": [1, 2], "query": ["a", "b"]}).to_parquet(qp)
    with pytest.raises(DegradedRun):
        run_modes(qp, ingest_root, str(tmp_path / "run"), FakeEncoder(), DeadLLM(), ["v1"])


def test_validate_rejects_mostly_empty_submission():
    q = pd.DataFrame({"id": list(range(100))})
    data = [{"id": i, "relevant_docs": [], "relevant_chunks": []} for i in range(100)]
    with pytest.raises(ValueError, match="no document"):
        submit.validate(data, q, max_empty=0.01)


def test_merge_touching_spans_is_exact_substring():
    doc = "Para one.\n\nPara two.\nPara three."
    a = {"idx": 0, "spans": [(0, 11)], "text": doc[0:11]}
    b = {"idx": 1, "spans": [(11, len(doc))], "text": doc[11:]}
    assert select.merge_chunks([a, b]) == doc
