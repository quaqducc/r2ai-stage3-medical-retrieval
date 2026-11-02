"""CLI: python -m extract {run,validate,audit,fixture} --help"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback

DEFAULT_DB_ROOT = "gs://aiguru_crawler/crawl/db"
DEFAULT_MANIFEST = "gs://aiguru_crawler/crawl/manifest.parquet"


def _hosts_for(manifest, hosts: str | None, shard: str | None) -> list[str]:
    sizes = manifest.host.value_counts()
    names = list(sizes.index)
    if hosts:
        want = [h.strip() for h in hosts.split(",") if h.strip()]
        unknown = set(want) - set(names)
        if unknown:
            sys.exit(f"hosts not in manifest: {sorted(unknown)}")
        names = [h for h in names if h in want]
    if shard:
        try:
            i, n = map(int, shard.split("/"))
            assert 0 <= i < n
        except (ValueError, AssertionError):
            sys.exit(f"--shard must be i/n with 0 <= i < n, got {shard!r}")
        load, part = [0] * n, {}
        for h in names:  # biggest first onto the least loaded shard: balanced, deterministic
            k = load.index(min(load))
            part[h] = k
            load[k] += int(sizes[h])
        names = [h for h in names if part[h] == i]
    return names


def cmd_run(a):
    import pandas as pd
    from . import pipeline, storage
    fs, mpath = storage.filesystem(a.manifest)
    man = pd.read_parquet(mpath, columns=["id", "url", "host", "lang", "lane"], filesystem=fs)
    man[["url", "lang", "lane"]] = man[["url", "lang", "lane"]].fillna("")
    hosts = _hosts_for(man, a.hosts, a.shard)
    logging.info("%d hosts -> %s", len(hosts), a.out)
    failed = []
    for i, h in enumerate(hosts, 1):
        try:
            st = _with_retry(lambda: pipeline.process_host(
                storage.join(a.db_root, f"{h}.sqlite"), h, man, a.out, workers=a.workers,
                limit=a.limit, remote=a.remote, workdir=a.workdir, force=a.force), h)
            logging.info("[%d/%d] %s %s", i, len(hosts), h, json.dumps({k: st.get(k) for k in
                         ("skipped", "rows", "errors", "statuses", "parsers", "seconds")}, ensure_ascii=False))
        except Exception:
            tb = traceback.format_exc()
            logging.error("[%d/%d] %s FAILED\n%s", i, len(hosts), h, tb)
            failed.append(h)
            try:  # durable record for unattended VMs (the container log may be gone)
                storage.write_json(storage.join(a.out, "hosts", f"{h}.failed.json"), {"host": h, "traceback": tb})
            except Exception:
                logging.error("could not write failure marker for %s", h)
    if failed:
        sys.exit(f"failed hosts: {failed}")


def _with_retry(fn, host, attempts=2):
    """Transient GCS/network errors on multi-GB downloads: retry the host once before giving up."""
    for k in range(attempts):
        try:
            return fn()
        except Exception:
            if k == attempts - 1:
                raise
            logging.warning("%s failed, retrying\n%s", host, traceback.format_exc())


def cmd_validate(a):
    from . import validate
    s = validate.run(a.manifest, a.out)
    print(json.dumps(s, ensure_ascii=False, indent=1))
    if not s["complete"]:
        sys.exit(1)


def cmd_audit(a):
    from . import audit
    df = audit.run([h.strip() for h in a.hosts.split(",")], a.n, a.db_root, dump=a.dump, workers=a.workers)
    import pandas as pd
    with pd.option_context("display.width", 250, "display.max_colwidth", 90, "display.max_rows", 500):
        print(df.to_string(index=False))
    if a.csv:
        df.to_csv(a.csv, index=False, encoding="utf-8")


def cmd_fixture(a):
    from .fixture import copy_rows
    ids = [int(x) for x in a.ids.split(",")] if a.ids else None
    print(copy_rows(a.host, a.db_root, a.dest, n=a.n, ids=ids, with_bad=not a.no_bad))


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252; output contains CJK/Vietnamese
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for name in ("trafilatura", "htmldate", "courlan", "charset_normalizer"):
        logging.getLogger(name).setLevel(logging.ERROR)
    p = argparse.ArgumentParser(prog="extract")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="extract hosts into the clean store")
    r.add_argument("--db-root", default=DEFAULT_DB_ROOT)
    r.add_argument("--manifest", default=DEFAULT_MANIFEST)
    r.add_argument("--out", required=True, help="gs://aiguru_crawler/clean/v1 or a local dir")
    r.add_argument("--hosts", help="comma-separated subset")
    r.add_argument("--shard", help="i/n: this machine's share of hosts (balanced by size)")
    r.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    r.add_argument("--workdir", default=os.environ.get("EXTRACT_WORKDIR", "/tmp/extract"),
                   help="local disk for downloaded dbs (largest 6.6 GB) and shards; env EXTRACT_WORKDIR")
    r.add_argument("--limit", type=int, help="rows per host (smoke tests; no 'missing' rows are emitted)")
    r.add_argument("--remote", action="store_true", help="read dbs via HTTP range instead of downloading (dev only)")
    r.add_argument("--force", action="store_true", help="redo hosts that are already complete")
    r.set_defaults(fn=cmd_run)

    v = sub.add_parser("validate", help="prove every manifest id has exactly one row")
    v.add_argument("--manifest", default=DEFAULT_MANIFEST)
    v.add_argument("--out", required=True)
    v.set_defaults(fn=cmd_validate)

    au = sub.add_parser("audit", help="paragraph recall of extraction on random remote pages")
    au.add_argument("--hosts", required=True)
    au.add_argument("--n", type=int, default=6)
    au.add_argument("--db-root", default=DEFAULT_DB_ROOT)
    au.add_argument("--dump", help="dir to write <host>_<id>.html/.md/.missed.txt for inspection")
    au.add_argument("--csv")
    au.add_argument("--workers", type=int, default=4)
    au.set_defaults(fn=cmd_audit)

    fx = sub.add_parser("fixture", help="copy rows of a remote host db into a small local test db")
    fx.add_argument("--host", required=True)
    fx.add_argument("--n", type=int, default=3)
    fx.add_argument("--ids")
    fx.add_argument("--no-bad", action="store_true", help="do not add one non-ok row")
    fx.add_argument("--db-root", default=DEFAULT_DB_ROOT)
    fx.add_argument("--dest", default="tests/fixtures/db")
    fx.set_defaults(fn=cmd_fixture)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
