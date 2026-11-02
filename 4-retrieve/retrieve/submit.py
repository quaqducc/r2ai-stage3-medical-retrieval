"""S6: submission.json in the competition schema + a validator + the zip (exactly one file, no folder)."""
from __future__ import annotations

import json
import logging
import os
import zipfile

import pandas as pd


log = logging.getLogger("retrieve")


def build(queries: pd.DataFrame, entries: dict) -> list[dict]:
    """entries: {query_id: {"docs": [int], "chunk_texts": [(doc_id, text)]}}; every query gets a row (empty lists if none)."""
    data = []
    for qid in queries["id"].tolist():
        e = entries.get(int(qid), {"docs": [], "chunk_texts": []})
        chunks = [{"doc_id": int(d), "chunk_text": t} for d, t in e["chunk_texts"] if t and t.strip()]
        if len(chunks) < len(e["chunk_texts"]):
            log.warning("query %s: dropped %d empty chunk(s)", qid, len(e["chunk_texts"]) - len(chunks))
        data.append({"id": int(qid), "relevant_docs": [int(d) for d in e["docs"]], "relevant_chunks": chunks})
    return data


def validate(data: list[dict], queries: pd.DataFrame, max_empty: float = 1.0) -> None:
    """Raises ValueError on anything the scorer would reject or that silently wastes score."""
    want = [int(q) for q in queries["id"].tolist()]
    got = [d["id"] for d in data]
    missing = sorted(set(want) - set(got))
    if missing:
        raise ValueError(f"missing query ids: {missing[:10]} ({len(missing)})")
    if len(got) != len(set(got)):
        raise ValueError("duplicate query ids")
    for d in data:
        if not isinstance(d["id"], int) or not all(isinstance(x, int) for x in d["relevant_docs"]):
            raise ValueError(f"query {d['id']}: ids must be int")
        if len(d["relevant_docs"]) != len(set(d["relevant_docs"])):
            raise ValueError(f"query {d['id']}: duplicate docs")
        docs = set(d["relevant_docs"])
        for c in d["relevant_chunks"]:
            if c["doc_id"] not in docs:
                raise ValueError(f"query {d['id']}: chunk doc {c['doc_id']} not in relevant_docs")
            if not c["chunk_text"].strip():
                raise ValueError(f"query {d['id']}: empty chunk_text")
    empty = sum(1 for d in data if not d["relevant_docs"])
    if data and empty / len(data) > max_empty:
        raise ValueError(f"{empty}/{len(data)} queries have no document (limit {max_empty:.0%}): retrieval is broken")
    if empty:
        log.warning("%d queries have no document", empty)


def write(queries: pd.DataFrame, entries: dict, out_dir: str, max_empty: float = 1.0) -> str:
    data = build(queries, entries)
    validate(data, queries, max_empty)
    os.makedirs(out_dir, exist_ok=True)
    js = os.path.join(out_dir, "submission.json")
    with open(js, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    zp = os.path.join(out_dir, "submission.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_BZIP2, compresslevel=9) as z:  # ~35% smaller; the scorer reads it (22/10)
        z.write(js, "submission.json")
    return zp
