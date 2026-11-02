import os
import sqlite3
import sys

import pytest
import zstandard

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
FIXTURES = os.path.join(ROOT, "tests", "fixtures")

SCHEMA = """CREATE TABLE pages(
  id INTEGER PRIMARY KEY, url TEXT NOT NULL, final_url TEXT, status INTEGER, kind TEXT NOT NULL,
  source TEXT, lane TEXT, ctype TEXT, charset TEXT, nbytes INTEGER, sha1 TEXT, fetched_at REAL,
  attempts INTEGER NOT NULL DEFAULT 0, err TEXT, vflag TEXT, headers_json TEXT, html_zstd BLOB);
CREATE INDEX idx_pages_kind ON pages(kind);"""


def make_db(path, rows):
    """rows: dicts with id, url, kind and optional html (str|bytes), charset, source, ctype."""
    if os.path.exists(path):
        os.remove(path)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    cz = zstandard.ZstdCompressor()
    for r in rows:
        html = r.get("html")
        if isinstance(html, str):
            html = html.encode(r.get("encode_as") or r.get("charset") or "utf-8")
        con.execute("insert into pages(id,url,final_url,status,kind,source,ctype,charset,fetched_at,html_zstd) values (?,?,?,?,?,?,?,?,?,?)",
                    (r["id"], r["url"], r.get("final_url"), r.get("status", 200), r["kind"], r.get("source", "direct"),
                     r.get("ctype", "text/html"), r.get("charset"), 1.7e9, cz.compress(html) if html is not None else None))
    con.commit()
    con.close()
    return path


@pytest.fixture
def db_factory(tmp_path):
    return lambda name, rows: make_db(str(tmp_path / name), rows)


def fixture_db(host):
    return os.path.join(FIXTURES, "db", f"{host}.sqlite")


def fixture_rows(host):
    """All ok rows of a committed fixture db as (id, url, html)."""
    from extract.source import iter_rows, decode
    return [(r.id, r.final_url or r.url, decode(r.blob, r.charset, r.ctype))
            for r in iter_rows(fixture_db(host)) if r.kind == "ok" and r.blob]
