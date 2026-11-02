"""Copy a few rows of a host's remote db into a small local SQLite with the same schema (test fixtures)."""
from __future__ import annotations

import os
import random
import sqlite3

from . import storage
from .remote_sqlite import connect

_SCHEMA = """CREATE TABLE IF NOT EXISTS pages(
  id INTEGER PRIMARY KEY, url TEXT NOT NULL, final_url TEXT, status INTEGER, kind TEXT NOT NULL,
  source TEXT, lane TEXT, ctype TEXT, charset TEXT, nbytes INTEGER, sha1 TEXT, fetched_at REAL,
  attempts INTEGER NOT NULL DEFAULT 0, err TEXT, vflag TEXT, headers_json TEXT, html_zstd BLOB);
CREATE INDEX IF NOT EXISTS idx_pages_kind ON pages(kind);"""
_COLS = "id,url,final_url,status,kind,source,lane,ctype,charset,nbytes,sha1,fetched_at,attempts,err,vflag,headers_json,html_zstd"


def ok_ids(con) -> list[int]:
    return [r[0] for r in con.execute("select id from pages indexed by idx_pages_kind where kind='ok'")]


def sample_ids(con, n: int, seed: str) -> list[int]:
    ids = ok_ids(con)
    return random.Random(seed).sample(ids, min(n, len(ids)))


def copy_rows(host: str, db_root: str, dest_dir: str, n: int = 3, ids: list[int] | None = None, with_bad: bool = True) -> str:
    src = connect(storage.join(db_root, f"{host}.sqlite"))
    ids = list(ids or sample_ids(src, n, host))
    if with_bad:
        bad = list(src.execute("select id from pages indexed by idx_pages_kind where kind!='ok' limit 1"))
        ids += [b[0] for b in bad]
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, f"{host}.sqlite")
    if os.path.exists(path):  # a fixture is exactly the requested rows, never an accumulation of past runs
        os.remove(path)
    dst = sqlite3.connect(path)
    dst.executescript(_SCHEMA)
    for i in ids:
        for row in src.execute(f"select {_COLS} from pages where id=?", (i,)):
            dst.execute(f"insert or replace into pages({_COLS}) values ({','.join('?' * 17)})", row)
    dst.commit()
    dst.execute("vacuum")
    dst.close()
    return path
