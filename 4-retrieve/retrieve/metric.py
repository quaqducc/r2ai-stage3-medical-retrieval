"""Re-implementation of the competition metric (for the self-built silver dev set):
docs: P = |D∩G|/|D|, R = |D∩G|/|G|; chunks: a submitted chunk is correct if LCS_tokens(c, g)/|g| >= 0.4 for some gold
chunk g of the same doc; chunk recall = gold chunks found / gold chunks; F2 per query, macro-averaged; final = mean.
Text normalisation as in the rules: NFKC, HTML unescape, lowercase, collapse whitespace. The official tokenizer is not
published: here a token is one CJK character or one word (punctuation dropped) — an approximation."""
from __future__ import annotations

import html
import re
import unicodedata

from rapidfuzz.distance import LCSseq

THRESHOLD = 0.4
_TOK = re.compile(r"[㐀-鿿豈-﫿]|[^\W_㐀-鿿豈-﫿]+", re.UNICODE)


def tokens(s: str) -> list[str]:
    s = " ".join(html.unescape(unicodedata.normalize("NFKC", s or "")).lower().split())
    return _TOK.findall(s)


def overlap(chunk_tokens: list[str], gold_tokens: list[str]) -> float:
    return LCSseq.similarity(chunk_tokens, gold_tokens) / len(gold_tokens) if gold_tokens else 0.0


def f2(p: float, r: float) -> float:
    return 5 * p * r / (4 * p + r) if p + r > 0 else 0.0


def score_query(pred: dict, gold: dict) -> dict:
    """pred/gold: {"relevant_docs": [int], "relevant_chunks": [{"doc_id", "chunk_text"}]}."""
    pd_, gd = set(pred.get("relevant_docs") or []), set(gold.get("relevant_docs") or [])
    hit = len(pd_ & gd)
    dp, dr = (hit / len(pd_) if pd_ else 0.0), (hit / len(gd) if gd else 0.0)
    pc, gc = pred.get("relevant_chunks") or [], gold.get("relevant_chunks") or []
    gtok = [(g["doc_id"], tokens(g["chunk_text"])) for g in gc]
    found, correct = set(), 0
    for c in pc:
        ct = tokens(c["chunk_text"])
        ok = False
        for j, (gdoc, gt) in enumerate(gtok):
            if gdoc == c["doc_id"] and overlap(ct, gt) >= THRESHOLD:
                ok = True
                found.add(j)
        correct += ok
    cp, cr = (correct / len(pc) if pc else 0.0), (len(found) / len(gc) if gc else 0.0)
    return {"doc_p": dp, "doc_r": dr, "doc_f2": f2(dp, dr), "chunk_p": cp, "chunk_r": cr, "chunk_f2": f2(cp, cr)}


def evaluate(pred: list[dict], gold: list[dict]) -> dict:
    """Macro averages over the gold queries (a query missing from pred scores 0)."""
    by_id = {p["id"]: p for p in pred}
    rows = [score_query(by_id.get(g["id"], {}), g) for g in gold]
    n = max(len(rows), 1)
    out = {k: sum(r[k] for r in rows) / n for k in (rows[0] if rows else {})}
    out["final"] = (out.get("doc_f2", 0) + out.get("chunk_f2", 0)) / 2
    out["n"] = len(rows)
    return out
