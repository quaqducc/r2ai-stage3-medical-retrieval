"""One host: crawl db -> clean rows -> parquet shards under <out>/docs/<host>/ + <out>/hosts/<host>.json.
The stats JSON is written last and marks the host complete; a host without it is wiped and redone."""
from __future__ import annotations

import hashlib
import logging
import multiprocessing as mp
import os
import re
import shutil
import signal
import time
from collections import Counter

import pandas as pd
import pyarrow.parquet as pq

from . import VERSION, storage
from .clean import clean_markdown, flags, to_text
from .parsers import PageTimeout, parse
from .source import Row, decode, iter_rows
from .writer import ShardWriter

log = logging.getLogger("extract")
PAGE_TIMEOUT = 120      # seconds of CPU-bound parsing for one page before it is recorded as error:PageTimeout
HOST_STALL = 1800       # seconds without any finished page -> the host fails loudly instead of hanging forever


def _utf8(s: str) -> str:
    """Lone surrogates (e.g. from a truncated JSON emoji escape) cannot be encoded/written; replace them."""
    return s.encode("utf-8", "replace").decode("utf-8")


def build_record(row: Row, host: str, meta: tuple | None) -> dict:
    url, lang, lane = meta or (row.url, "", "")
    rec = dict(id=row.id, url=url, host=host, lang=lang, text_lang="", lane=lane, status=row.kind, source=row.source or "",
               fetched_at=row.fetched_at, title="", markdown="", text="", n_chars=0, parser="none",
               flags=["empty"], content_sha1="", pipeline_version=VERSION)
    if row.kind != "ok" or not row.blob:
        return rec
    try:
        html = decode(row.blob, row.charset, row.ctype)
        doc = parse(host, row.final_url or row.url, html)
        cnt = {}
        raw_len = len(doc.markdown)
        md = _utf8(clean_markdown(doc.markdown, doc.drop_lines, cnt))
        text = to_text(md)
        extra = list(doc.notes)
        if row.status not in (None, 200):
            extra.append(f"http_{row.status}")
        if text and text.count("\ufffd") > 0.005 * len(text):
            extra.append("replacement_chars")  # likely wrong charset
        if raw_len and cnt.get("dropped_chars", 0) > 0.3 * raw_len:
            extra.append("heavy_drop")  # drop_lines removed >30% of the page: check the patterns
        rec.update(title=_utf8(doc.title), markdown=md, text=text, n_chars=len(text), parser=doc.parser,
                   text_lang=detect_lang(text),
                   flags=flags(text, row.source, doc.fallback, extra),
                   content_sha1=hashlib.sha1(text.encode("utf-8")).hexdigest() if text else "")
    except (Exception, PageTimeout) as e:  # one bad page must not kill the host; counted in stats["errors"]
        log.warning("parse failed host=%s id=%s: %s: %s", host, row.id, type(e).__name__, e)
        rec.update(title="", markdown="", text="", n_chars=0, flags=["empty"], content_sha1="",
                   parser=f"error:{type(e).__name__}")
    return rec


_CJK = re.compile(r"[\u3400-\u9fff]")
_VI = re.compile(r"[ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]", re.I)
_LATIN = re.compile(r"[A-Za-z]")


def detect_lang(text: str) -> str:
    """vi / zh / en / mixed from character classes of the first 5000 chars ('' when there is no text)."""
    t = text[:5000]
    cjk, vi, lat = len(_CJK.findall(t)), len(_VI.findall(t)), len(_LATIN.findall(t))
    total = cjk + lat
    if not total:
        return ""
    if cjk / total > 0.6:
        return "zh"
    if cjk / total > 0.15:
        return "mixed"
    return "vi" if vi > 0.02 * lat else "en"


def _on_alarm(signum, frame):
    raise PageTimeout()


def _work(args):
    if hasattr(signal, "setitimer"):  # POSIX only (the VM); dev on Windows runs without the per-page timeout
        signal.signal(signal.SIGALRM, _on_alarm)
        signal.setitimer(signal.ITIMER_REAL, PAGE_TIMEOUT)
    try:
        return build_record(*args)
    finally:
        if hasattr(signal, "setitimer"):
            signal.setitimer(signal.ITIMER_REAL, 0)


def missing_record(pid: int, host: str, meta: tuple) -> dict:
    url, lang, lane = meta
    return dict(id=pid, url=url, host=host, lang=lang, text_lang="", lane=lane, status="missing", source="", fetched_at=None,
                title="", markdown="", text="", n_chars=0, parser="none", flags=["empty"], content_sha1="",
                pipeline_version=VERSION)


SUSPECT = {"error_rate": 0.005, "empty_rate": 0.3, "fallback_rate": 0.2, "missing_rate": 0.0}


def suspect(host: str, st: dict, rows: int) -> dict:
    """Rates that reveal systematic breakage (a host can be 'complete' yet useless); listed in stats + summary."""
    from . import rules
    ok = st["statuses"].get("ok", 0)
    rates = {"error_rate": sum(v for k, v in st["parsers"].items() if k.startswith("error:")) / max(ok, 1),
             "empty_rate": max(0, st["flags"].get("empty", 0) - (rows - ok)) / max(ok, 1),
             "fallback_rate": st["flags"].get("fallback", 0) / max(ok, 1),
             "missing_rate": st["statuses"].get("missing", 0) / max(rows, 1)}
    bad = [k for k, v in rates.items() if v > SUSPECT[k]]
    if rules.get(host) is not None and ok and st["parsers"].get("rule", 0) / ok < 0.5:
        bad.append("rule_share")
    if bad:
        log.warning("suspect host %s: %s", host, {k: round(rates.get(k, 0), 4) for k in bad})
    return {"rates": {k: round(v, 4) for k, v in rates.items()}, "suspect": bad}


def is_complete(stats_path: str) -> bool:
    """A finished full run of this pipeline version (smoke runs with --limit never count)."""
    if not storage.exists(stats_path):
        return False
    try:
        st = storage.read_json(stats_path)
    except ValueError:
        return False
    return not st.get("limited") and st.get("pipeline_version") == VERSION


def process_host(db: str, host: str, manifest: pd.DataFrame, out: str, workers: int = 1, limit: int | None = None,
                 remote: bool = False, workdir: str = "/tmp/extract", force: bool = False) -> dict:
    stats_path = storage.join(out, "hosts", f"{host}.json")
    if not force and is_complete(stats_path):
        return {"host": host, "skipped": True}
    t0 = time.time()
    host_docs = storage.join(out, "docs", host)
    if storage.exists(stats_path):  # a stale/limited marker must not survive a crash of this run
        storage.rm_tree(stats_path)
    storage.rm_tree(host_docs)  # leftovers of an interrupted run
    shard_dir = os.path.join(workdir, "shards", host)
    shutil.rmtree(shard_dir, ignore_errors=True)
    local_db = db
    try:
        local_db = db if remote else storage.download(db, os.path.join(workdir, "db"))
        man = manifest[manifest.host == host]
        meta = {int(i): (u, l, ln) for i, u, l, ln in zip(man.id, man.url, man.lang, man.lane)}
        writer = ShardWriter(shard_dir)
        st = {"statuses": Counter(), "parsers": Counter(), "flags": Counter()}
        seen, errors = set(), 0

        def consume(rec):
            nonlocal errors
            writer.add(rec)
            seen.add(rec["id"])
            st["statuses"][rec["status"]] += 1
            st["parsers"][rec["parser"]] += 1
            st["flags"].update(rec["flags"])
            errors += rec["parser"].startswith("error:")

        jobs = ((r, host, meta.get(r.id)) for r in iter_rows(local_db, limit=limit, remote=remote))
        if workers > 1:
            with mp.get_context("spawn").Pool(workers, maxtasksperchild=2000) as pool:
                it = pool.imap(_work, jobs)  # chunksize 1: only IMapIterator.next() takes a timeout
                while True:
                    try:
                        rec = it.next(timeout=HOST_STALL)
                    except StopIteration:
                        break
                    consume(rec)
        else:
            for job in jobs:
                consume(_work(job))
        extra = len(seen - meta.keys())
        if limit is None:
            for pid in sorted(meta.keys() - seen):
                consume(missing_record(pid, host, meta[pid]))
        files = writer.close()
        written = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
        if written != len(seen):
            raise RuntimeError(f"{host}: wrote {written} rows for {len(seen)} ids")
        for f in files:
            storage.put(f, storage.join(host_docs, os.path.basename(f)))
        stats = {"host": host, "rows": len(seen), "manifest": len(meta), "extra_ids": extra, "errors": errors,
                 **suspect(host, st, len(seen)),
                 "statuses": dict(st["statuses"]), "parsers": dict(st["parsers"]), "flags": dict(st["flags"]),
                 "shards": len(files), "limited": limit is not None, "remote": remote,
                 "seconds": round(time.time() - t0, 1), "pipeline_version": VERSION}
        storage.write_json(stats_path, stats)
        return stats
    finally:
        shutil.rmtree(shard_dir, ignore_errors=True)
        if local_db != db and os.path.exists(local_db):
            os.remove(local_db)
