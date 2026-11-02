"""Rows of one crawl db (table `pages`, id INTEGER PRIMARY KEY so ids are unique per file)."""
from __future__ import annotations

import pathlib
import re
import sqlite3
import threading
from dataclasses import dataclass
from typing import Iterator

import zstandard

COLS = "id, url, final_url, status, kind, source, charset, ctype, fetched_at, html_zstd"
_CS_ALIAS = {"gb2312": "gb18030", "gbk": "gb18030", "x-gbk": "gb18030", "gb_2312-80": "gb18030", "utf8": "utf-8",
             "iso-8859-1": "cp1252", "latin-1": "cp1252", "latin1": "cp1252", "us-ascii": "cp1252", "ascii": "cp1252"}
_META_CS = re.compile(rb"""<meta[^>]+charset\s*=\s*["']?\s*([A-Za-z0-9_\-]+)""", re.I)
_CT_CS = re.compile(r"charset\s*=\s*([A-Za-z0-9_\-]+)", re.I)
_WAYBACK = re.compile(r"<!-- BEGIN WAYBACK TOOLBAR INSERT -->.*?<!-- END WAYBACK TOOLBAR INSERT -->", re.S)
_CJK_CS = {"gb18030", "big5", "big5-hkscs", "shift_jis", "euc-jp", "euc-kr", "cp936", "cp950"}
_TLS = threading.local()  # ZstdDecompressor is not thread-safe (audit uses threads)
MAX_DECOMPRESSED = 64 << 20


@dataclass
class Row:
    id: int
    url: str
    final_url: str | None
    status: int | None
    kind: str
    source: str | None
    charset: str | None
    ctype: str | None
    fetched_at: float | None
    blob: bytes | None


def iter_rows(db: str, limit: int | None = None, remote: bool = False) -> Iterator[Row]:
    if remote:
        from .remote_sqlite import connect
        con = connect(db)
    else:
        con = sqlite3.connect(pathlib.Path(db).resolve().as_uri() + "?mode=ro", uri=True)
    sql = f"select {COLS} from pages order by id" + (f" limit {int(limit)}" if limit is not None else "")
    try:
        for r in con.execute(sql):
            yield Row(*r)
    finally:
        con.close()


def _norm_cs(cs: str | None) -> str | None:
    if not cs:
        return None
    cs = cs.strip().lower()
    return _CS_ALIAS.get(cs, cs)


def decode(blob: bytes, charset: str | None, ctype: str | None) -> str:
    """zstd blob -> str. Labels: db column, Content-Type, <meta>; utf-8 first unless labelled CJK. Never raises."""
    dz = getattr(_TLS, "dz", None) or zstandard.ZstdDecompressor()
    _TLS.dz = dz
    raw = dz.decompress(blob, max_output_size=MAX_DECOMPRESSED)
    m = _CT_CS.search(ctype or "")
    sniff = _META_CS.search(raw[:4096])
    labels = [_norm_cs(charset), _norm_cs(m.group(1) if m else None),
              _norm_cs(sniff.group(1).decode("ascii", "ignore") if sniff else None)]
    first = next((c for c in labels if c), None)
    # A multibyte CJK label is trusted (short GBK text can also be valid UTF-8); any other label (often a wrong
    # default like iso-8859-1 on Vietnamese sites) loses to bytes that are valid UTF-8.
    cands = (labels if first in _CJK_CS else ["utf-8"] + labels) + ["utf-8"]
    text = None
    for cs in dict.fromkeys(c for c in cands if c):
        try:
            text = raw.decode(cs)
            break
        except (LookupError, UnicodeDecodeError):
            continue
    if text is None:
        cs = next((c for c in cands if c and _known(c)), "utf-8")
        text = raw.decode(cs, "replace")
    return _WAYBACK.sub("", text) if "WAYBACK TOOLBAR" in text else text


def _known(cs: str) -> bool:
    import codecs
    try:
        codecs.lookup(cs)
        return True
    except LookupError:
        return False
