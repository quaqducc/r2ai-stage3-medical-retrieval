import json
import os

import pandas as pd
import pyarrow.parquet as pq
import pytest

from extract import pipeline
from extract.writer import SCHEMA

GOOD = "<html><body><article><h1>T</h1>" + "<p>" + "Nội dung bài viết về sốt xuất huyết. " * 20 + "</p></article></body></html>"


def manifest(ids, host="h.vn"):
    return pd.DataFrame({"id": ids, "url": [f"https://{host}/{i}" for i in ids], "host": host, "lang": "vi", "lane": "vi-news"})


def read_docs(out, host="h.vn"):
    files = sorted(f for f in os.listdir(os.path.join(out, "docs", host)) if f.endswith(".parquet"))
    return pd.concat([pq.read_table(os.path.join(out, "docs", host, f)).to_pandas() for f in files])


@pytest.fixture
def host_db(db_factory):
    return db_factory("h.vn.sqlite", [
        {"id": 1, "url": "https://h.vn/1", "kind": "ok", "html": GOOD},
        {"id": 2, "url": "https://h.vn/2", "kind": "blocked"},
        {"id": 3, "url": "https://h.vn/3", "kind": "ok", "html": GOOD, "source": "wayback_direct"},
        {"id": 4, "url": "https://h.vn/4", "kind": "ok", "html": "<p>tiny</p>"},
    ])


def test_process_host_one_row_per_manifest_id(host_db, tmp_path):
    out = str(tmp_path / "out")
    stats = pipeline.process_host(host_db, "h.vn", manifest([1, 2, 3, 4, 5]), out, workers=1, workdir=str(tmp_path / "w"))
    df = read_docs(out)
    assert sorted(df.id) == [1, 2, 3, 4, 5]
    assert pq.read_schema(os.path.join(out, "docs", "h.vn", "part-00000.parquet")).names == SCHEMA.names
    by = df.set_index("id")
    assert by.loc[1, "status"] == "ok" and "sốt xuất huyết" in by.loc[1, "text"] and by.loc[1, "title"] == "T"
    assert by.loc[2, "status"] == "blocked" and list(by.loc[2, "flags"]) == ["empty"]
    assert "archive" in list(by.loc[3, "flags"])
    assert "short" in list(by.loc[4, "flags"])
    assert by.loc[5, "status"] == "missing"
    assert by.loc[1, "content_sha1"] and by.loc[1, "n_chars"] == len(by.loc[1, "text"])
    assert stats["rows"] == 5 and stats["statuses"]["missing"] == 1
    assert json.load(open(os.path.join(out, "hosts", "h.vn.json")))["rows"] == 5


def test_parse_error_still_emits_row(host_db, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("bad page")
    monkeypatch.setattr(pipeline, "parse", boom)
    out = str(tmp_path / "out")
    stats = pipeline.process_host(host_db, "h.vn", manifest([1, 2, 3, 4]), out, workers=1, workdir=str(tmp_path / "w"))
    df = read_docs(out).set_index("id")
    assert df.loc[1, "parser"] == "error:RuntimeError" and list(df.loc[1, "flags"]) == ["empty"]
    assert stats["errors"] == 3


def test_completed_host_skipped_and_partial_host_redone(host_db, tmp_path):
    out = str(tmp_path / "out")
    m = manifest([1, 2, 3, 4])
    pipeline.process_host(host_db, "h.vn", m, out, workers=1, workdir=str(tmp_path / "w"))
    assert pipeline.process_host(host_db, "h.vn", m, out, workers=1, workdir=str(tmp_path / "w")).get("skipped")
    # simulate a crash after shards were uploaded but before stats were written
    os.remove(os.path.join(out, "hosts", "h.vn.json"))
    open(os.path.join(out, "docs", "h.vn", "part-00099.parquet"), "w").write("partial garbage")
    pipeline.process_host(host_db, "h.vn", m, out, workers=1, workdir=str(tmp_path / "w"))
    assert sorted(read_docs(out).id) == [1, 2, 3, 4]


def test_limit_emits_no_missing_rows(host_db, tmp_path):
    out = str(tmp_path / "out")
    stats = pipeline.process_host(host_db, "h.vn", manifest([1, 2, 3, 4, 5]), out, workers=1, limit=2, workdir=str(tmp_path / "w"))
    assert sorted(read_docs(out).id) == [1, 2] and stats["limited"]


def test_multiprocess_matches_single(host_db, tmp_path):
    a = pipeline.process_host(host_db, "h.vn", manifest([1, 2, 3, 4]), str(tmp_path / "a"), workers=1, workdir=str(tmp_path / "w"))
    b = pipeline.process_host(host_db, "h.vn", manifest([1, 2, 3, 4]), str(tmp_path / "b"), workers=2, workdir=str(tmp_path / "w"))
    da, db_ = read_docs(str(tmp_path / "a")), read_docs(str(tmp_path / "b"))
    assert da.drop(columns="flags").equals(db_.drop(columns="flags")) and a["statuses"] == b["statuses"]
