"""Extraction audit on random remote pages: how many long text blocks of the HTML survive in the clean text.
Low recall = content lost (e.g. forum answers); inspect the dump and write a host rule."""
from __future__ import annotations

import os
import re
import statistics
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import lxml.html
import pandas as pd

from . import storage
from .fixture import sample_ids
from .pipeline import build_record
from .remote_sqlite import connect
from .source import COLS, Row, decode

_LEAF = {"p", "li", "td", "dd", "dt", "div", "section", "article", "blockquote", "h1", "h2", "h3", "h4", "pre", "span", "font"}
_BLOCKY = {"p", "div", "li", "td", "table", "ul", "ol", "section", "article", "blockquote", "dd", "h1", "h2", "h3", "h4", "pre"}
_JUNK = "//script|//style|//noscript|//nav|//header|//footer|//aside|//form|//select|//button|//iframe|//template"
_CJK = re.compile(r"[一-鿿]")
_WS = re.compile(r"\s+")


def _key(s: str) -> str:
    return _WS.sub("", unicodedata.normalize("NFC", s))


def blocks(html: str) -> list[str]:
    """Visible leaf text blocks long enough to be content (>= 20 CJK chars or >= 80 chars)."""
    try:
        tree = lxml.html.fromstring(re.sub(r"</(?:html|body)\s*>", "", html, flags=re.I))
    except Exception:
        return []
    for n in tree.xpath(_JUNK):
        if n.getparent() is not None:
            n.drop_tree()
    out = []
    for el in tree.iter():
        if not isinstance(el.tag, str) or el.tag not in _LEAF:
            continue
        if any(isinstance(d.tag, str) and d.tag in _BLOCKY for d in el.iterdescendants()):
            continue
        t = " ".join(el.text_content().split())
        if len(_CJK.findall(t)) >= 20 or len(t) >= 80:
            out.append(t)
    return list(dict.fromkeys(out))


def paragraph_recall(html: str, text: str) -> tuple[float, list[str]]:
    bl = blocks(html)
    if not bl:
        return 1.0, []
    tk = _key(text)
    missed = []
    for b in bl:
        k = _key(b)
        mid = len(k) // 2
        if k[max(0, mid - 12): mid + 12] not in tk:
            missed.append(b)
    return 1 - len(missed) / len(bl), missed


def audit_host(host: str, n: int, db_root: str, dump: str | None = None) -> dict:
    con = connect(storage.join(db_root, f"{host}.sqlite"))
    ids = sample_ids(con, n, "audit:" + host)
    recs = []
    for pid in ids:
        for r in con.execute(f"select {COLS} from pages where id=?", (pid,)):
            row = Row(*r)
            html = decode(row.blob, row.charset, row.ctype)
            rec = build_record(row, host, None)
            rc, missed = paragraph_recall(html, rec["text"])
            recs.append((rec, rc, missed))
            if dump:
                os.makedirs(dump, exist_ok=True)
                base = os.path.join(dump, f"{host}_{pid}")
                open(base + ".html", "w", encoding="utf-8").write(html)
                open(base + ".md", "w", encoding="utf-8").write(f"<!-- {rec['url']} parser={rec['parser']} title={rec['title']} -->\n" + rec["markdown"])
                open(base + ".missed.txt", "w", encoding="utf-8").write("\n\n".join(missed))
    if not recs:
        return {"host": host, "n": 0}
    rcs = [r[1] for r in recs]
    worst = min(recs, key=lambda r: r[1])
    return {"host": host, "n": len(recs), "recall_med": round(statistics.median(rcs), 2), "recall_min": round(min(rcs), 2),
            "chars_med": int(statistics.median(r[0]["n_chars"] for r in recs)),
            "parsers": dict(Counter(r[0]["parser"] for r in recs)),
            "fallback": sum("fallback" in r[0]["flags"] for r in recs),
            "worst_id": worst[0]["id"], "worst_missed": (max(worst[2], key=len)[:90] if worst[2] else "")}


def run(hosts: list[str], n: int, db_root: str, dump: str | None = None, workers: int = 4) -> pd.DataFrame:
    def one(h):
        try:
            return audit_host(h, n, db_root, dump)
        except Exception as e:  # audit is diagnostic: report the host as failed, keep going
            return {"host": h, "n": 0, "error": f"{type(e).__name__}: {e}"}
    from . import rules
    rules.get("")  # import the host rule modules once, before threads race on the lazy import
    with ThreadPoolExecutor(workers) as ex:
        return pd.DataFrame(list(ex.map(one, hosts)))
