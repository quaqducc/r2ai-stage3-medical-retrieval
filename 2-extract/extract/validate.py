"""Coverage proof: every manifest id has exactly one row in <out>/docs. Writes reports/coverage.csv + summary.json."""
from __future__ import annotations

import io

import pandas as pd
import pyarrow.dataset as ds

from . import storage


def load_docs(out: str, columns=("id", "host", "status", "flags", "n_chars", "parser")) -> pd.DataFrame:
    fs, path = storage.filesystem(storage.join(out, "docs"))
    files = [f for f in storage.glob(storage.join(out, "docs", "*", "*.parquet"))]
    if not files:
        return pd.DataFrame(columns=list(columns))
    paths = [storage.filesystem(f)[1] for f in files]
    return ds.dataset(paths, format="parquet", filesystem=fs).to_table(columns=list(columns)).to_pandas()


def _hosts_without_stats(out: str, hosts: list) -> list:
    from .pipeline import is_complete
    return [h for h in hosts if not is_complete(storage.join(out, "hosts", f"{h}.json"))]


def _suspect_hosts(out: str) -> dict:
    """host -> suspect reasons from hosts/<host>.json (quality signal; does not affect 'complete')."""
    res = {}
    for p in storage.glob(storage.join(out, "hosts", "*.json")):
        if p.endswith(".failed.json"):
            continue
        st = storage.read_json(p)
        if st.get("suspect"):
            res[st["host"]] = st["suspect"]
    return res


def run(manifest_path: str, out: str) -> dict:
    fs, mpath = storage.filesystem(manifest_path)
    man = pd.read_parquet(mpath, columns=["id", "host"], filesystem=fs)
    man_dups = int(man.id.duplicated().sum())
    docs = load_docs(out)
    dup_ids = docs.id[docs.id.duplicated()].unique()
    missing = man[~man.id.isin(docs.id)]
    extra = docs[~docs.id.isin(man.id)]
    f = docs["flags"].map(lambda x: set(x) if x is not None else set())
    docs = docs.assign(ok=docs.status.eq("ok"), empty=f.map(lambda s: "empty" in s), short=f.map(lambda s: "short" in s),
                       usable=docs.status.eq("ok") & ~f.map(lambda s: bool(s & {"empty", "short"})),
                       error=docs.parser.str.startswith("error:"))
    per = docs.groupby("host").agg(rows=("id", "size"), ok=("ok", "sum"), usable=("usable", "sum"), empty=("empty", "sum"),
                                   short=("short", "sum"), parse_errors=("error", "sum"),
                                   status_missing=("status", lambda s: int((s == "missing").sum())))
    rep = man.groupby("host").size().rename("manifest").to_frame().join(per, how="outer").fillna(0).astype(int)
    rep["not_in_docs"] = missing.groupby("host").size().reindex(rep.index).fillna(0).astype(int)
    rep["duplicate_ids"] = docs[docs.id.isin(dup_ids)].groupby("host").id.nunique().reindex(rep.index).fillna(0).astype(int)
    rep = rep.sort_values("manifest", ascending=False)
    summary = {"manifest_ids": int(len(man)), "doc_rows": int(len(docs)), "not_in_docs": int(len(missing)),
               "duplicate_ids": int(len(dup_ids)), "extra_ids": int(len(extra)), "ok": int(docs.ok.sum()),
               "usable": int(docs.usable.sum()), "parse_errors": int(docs.error.sum()),
               "hosts_with_gaps": rep.index[rep.not_in_docs > 0].tolist(), "manifest_duplicate_ids": man_dups,
               "hosts_without_stats": _hosts_without_stats(out, sorted(docs.host.unique())),
               "suspect_hosts": _suspect_hosts(out)}
    summary["complete"] = (summary["not_in_docs"] == 0 and summary["duplicate_ids"] == 0 and summary["extra_ids"] == 0
                           and man_dups == 0 and not summary["hosts_without_stats"])
    buf = io.StringIO()
    rep.to_csv(buf)
    storage.write_text(storage.join(out, "reports", "coverage.csv"), buf.getvalue())
    storage.write_json(storage.join(out, "reports", "summary.json"), summary)
    return summary
