"""Clean doc -> chunks per CHUNKING.md. Every chunk is verbatim: text == join_spans(doc.text, spans), where touching
spans are concatenated as is and separate spans are joined by a blank line. Spans of article / pubmed / qa_single
chunks are contiguous, so their text is an exact substring of the doc; a qa_answer chunk for answer i > 0 is the
question + that answer (two non-adjacent spans), as decided for CHUNKING.md."""
from __future__ import annotations

import re

from extract.clean import to_text

MIN_WORDS, MAX_WORDS = 500, 1000   # article chunks
QA_SPLIT_WORDS = 600               # qa page above this (with >= 2 answers) -> one chunk per answer
CJK_PER_WORD = 1.6

_CJK = re.compile(r"[㐀-鿿豈-﫿]")
_LINE = re.compile(r"[^\n]+")
_SENT_END = re.compile(r"[.!?]+[\"”’)）]*(?=\s|$)|[。！？]+[”’）」]*")  # '0.5 mg' / URLs are not sentence ends
_WORDS = re.compile(r"[^\W_]+")
_TRA_LOI = re.compile(r"^Trả lời:?$")
_39_DOCTOR = re.compile(r"^\S{1,12} (?:主任医师|副主任医师|主治医师|住院医师|医师|主任药师|副主任药师|主管药师|药师|"
                        r"主管护师|护师|心理咨询师|心理治疗师|营养师)(?: |$)")
_120_STRUCT = re.compile(r"^病情分析[：:]")
_GREETING = re.compile(r"^(?:您好|你好)")
ZH_QA = {"www.120ask.com", "ask.39.net"}
PUBMED = "pubmed.ncbi.nlm.nih.gov"


def count_words(s: str) -> float:
    """vi/en: whitespace tokens; zh: CJK chars / 1.6 (CJK_PER_WORD); mixed text adds both."""
    cjk = len(_CJK.findall(s))
    return len(_WORDS.findall(_CJK.sub(" ", s))) + cjk / CJK_PER_WORD  # punctuation (also full-width) is not a word


def join_spans(text: str, spans) -> str:
    out, prev = [], None
    for s_, e in spans:
        if out and prev != s_:
            out.append("\n\n")
        out.append(text[s_:e])
        prev = e
    return "".join(out)


def split_spans(chunk_text: str, spans) -> dict:
    """Inverse of join_spans: {(start, end): text of that span} recovered from the chunk text alone."""
    out, pos, prev = {}, 0, None
    for s_, e in spans:
        s_, e = int(s_), int(e)
        if prev is not None and prev != s_:
            pos += 2
        out[(s_, e)] = chunk_text[pos:pos + (e - s_)]
        pos += e - s_
        prev = e
    return out


def _lines(text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group()) for m in _LINE.finditer(text) if m.group().strip()]


def doc_type(doc: dict) -> str:
    if doc["host"] == PUBMED:
        return "pubmed"
    if doc["host"] in ZH_QA:
        return "qa"
    lines = _lines(doc["text"])
    marks = [i for i, (_, _, ln) in enumerate(lines) if _TRA_LOI.match(ln)]
    # vinmec "Hỏi bác sĩ" / vnexpress "Trả lời:" pages: a short question near the top and few answers. A long FAQ
    # article that happens to contain a "Trả lời:" line stays an article.
    if marks and len(marks) <= 3 and marks[0] <= 12 and count_words(doc["text"][:lines[marks[0]][0]]) <= 400:
        return "qa"
    return "article"


# ---------------------------------------------------------------- qa
def _answer_starts(host: str, lines: list) -> list[int]:
    """Line indexes where an answer begins (the question is everything before the first one)."""
    if host == "ask.39.net":
        return [i for i, (_, _, ln) in enumerate(lines) if i and _39_DOCTOR.match(ln)]
    if host == "www.120ask.com":
        starts, style = [], None
        for i, (_, _, ln) in enumerate(lines):
            if not i:
                continue
            if _120_STRUCT.match(ln):          # structured answer: 病情分析 / 指导意见 / 生活护理
                starts.append(i)
                style = "struct"
            elif _GREETING.match(ln) and style != "struct":  # free-form answers each open with a greeting
                starts.append(i)
                style = "free"
        return starts
    return [i for i, (_, _, ln) in enumerate(lines) if _TRA_LOI.match(ln)]


def chunk_qa(doc: dict) -> list[dict]:
    text, lines = doc["text"], _lines(doc["text"])
    starts = _answer_starts(doc["host"], lines)
    if not starts:
        return [_mk(doc, [(lines[0][0], lines[-1][1])], "qa_question")]
    # spans tile the page (each ends where the next begins), so adjacent pieces concatenate verbatim
    q = (lines[0][0], lines[starts[0]][0]) if starts[0] > 0 else None
    ends = [lines[b][0] for b in starts[1:]] + [lines[-1][1]]
    answers = [(lines[a][0], e) for a, e in zip(starts, ends)]
    total = count_words(text[lines[0][0]:lines[-1][1]])
    if len(answers) == 1 or total <= QA_SPLIT_WORDS:
        return [_mk(doc, [(lines[0][0], lines[-1][1])], "qa_single")]
    return [_mk(doc, ([q] if q else []) + [a], "qa_answer", answer_idx=i) for i, a in enumerate(answers)]


# ---------------------------------------------------------------- article
def _headings(markdown: str) -> dict[str, int]:
    """Heading text as it appears in `text` -> heading level, from the markdown '#' lines."""
    out = {}
    for ln in markdown.splitlines():
        m = re.match(r"^(#{1,6})\s+", ln)
        if m:
            out.setdefault(to_text(ln), len(m.group(1)))
    return out


def _sentences(start: int, line: str) -> list[tuple[int, int]]:
    spans, pos = [], 0
    for m in _SENT_END.finditer(line):
        seg = line[pos:m.end()]
        if seg.strip():
            lead = len(seg) - len(seg.lstrip())
            spans.append((start + pos + lead, start + m.end()))
        pos = m.end()
    if line[pos:].strip():
        lead = len(line[pos:]) - len(line[pos:].lstrip())
        spans.append((start + pos + lead, start + len(line.rstrip())))
    return spans


def _units(text: str, lines: list, heads: dict[str, int]) -> list[dict]:
    """Sections cut at headings; a section over MAX_WORDS is cut at lines, a line over MAX_WORDS at sentences."""
    sections, cur = [], []
    for ln in lines:
        if ln[2] in heads and cur:
            sections.append(cur)
            cur = []
        cur.append(ln)
    if cur:
        sections.append(cur)

    units = []
    for sec in sections:
        words = sum(count_words(ln[2]) for ln in sec)
        if words <= MAX_WORDS:
            units.append({"start": sec[0][0], "end": sec[-1][1], "words": words})
            continue
        pieces = []
        for s, e, ln in sec:
            w = count_words(ln)
            pieces += [(a, b, count_words(text[a:b])) for a, b in _sentences(s, ln)] if w > MAX_WORDS else [(s, e, w)]
        acc = None
        for a, b, w in pieces:  # pack pieces back up to MAX_WORDS
            if acc and acc["words"] + w > MAX_WORDS:
                units.append(acc)
                acc = None
            acc = {"start": a, "end": b, "words": w} if acc is None else {**acc, "end": b, "words": acc["words"] + w}
        if acc:
            units.append(acc)
    return units


def _tile(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Extend each span to where the next starts: consecutive chunks then concatenate verbatim."""
    return [(s_, spans[i + 1][0] if i + 1 < len(spans) else e) for i, (s_, e) in enumerate(spans)]


def _pack(units: list[dict]) -> list[tuple[int, int]]:
    """Greedy: grow a chunk until >= MIN_WORDS, never past MAX_WORDS; a small tail joins the previous chunk if it fits."""
    chunks, cur = [], None
    for u in units:
        if cur and cur["words"] + u["words"] > MAX_WORDS:
            chunks.append(cur)
            cur = None
        cur = dict(u) if cur is None else {**cur, "end": u["end"], "words": cur["words"] + u["words"]}
        if cur["words"] >= MIN_WORDS:
            chunks.append(cur)
            cur = None
    if cur:
        if chunks and cur["words"] < MIN_WORDS and chunks[-1]["words"] + cur["words"] <= MAX_WORDS:
            chunks[-1] = {**chunks[-1], "end": cur["end"], "words": chunks[-1]["words"] + cur["words"]}
        else:
            chunks.append(cur)
    return [(c["start"], c["end"]) for c in chunks]


def _heading_path(lines: list, heads: dict[str, int], title: str) -> dict[int, str]:
    """line start offset -> 'H2 > H3' path in effect at that line (the doc title is left out)."""
    path, out = [], {}
    for s, _, ln in lines:
        if ln in heads:
            lvl = heads[ln]
            path = [p for p in path if p[0] < lvl] + [(lvl, ln)]
        out[s] = " > ".join(t for _, t in path if t != title)
    return out


def chunk_article(doc: dict) -> list[dict]:
    text, lines = doc["text"], _lines(doc["text"])
    heads = _headings(doc.get("markdown") or "")
    hp = _heading_path(lines, heads, doc.get("title") or "")
    starts = sorted(hp)
    out = []
    for s, e in _tile(_pack(_units(text, lines, heads))):
        line_start = max((x for x in starts if x <= s), default=starts[0])
        out.append(_mk(doc, [(s, e)], "section", heading_path=hp[line_start]))
    return out


# ---------------------------------------------------------------- pubmed
def _pubmed_parts(lines: list):
    """(title line, abstract lines, metadata lines) of a 2-extract PubMed text."""
    meta = [ln for ln in lines if "· PMID" in ln[2] or ln[2].startswith(("MeSH:", "Keywords:"))]
    body = [ln for ln in lines[1:] if ln not in meta and ln[2] != "Abstract"]
    return lines[0], body, meta


def pubmed_meta(doc: dict) -> str:
    """Journal / PMID / MeSH / keyword lines: a separate BM25 field for the doc card, never chunk text."""
    lines = _lines(doc["text"])
    return "\n".join(ln[2] for ln in _pubmed_parts(lines)[2]) if lines else ""


def chunk_pubmed(doc: dict) -> list[dict]:
    text, lines = doc["text"], _lines(doc["text"])
    title, body, _ = _pubmed_parts(lines)
    if not body:  # no abstract: the title line is all the text there is
        return [_mk(doc, [(title[0], title[1])], "pubmed")]
    # abstract only (one contiguous span); the title goes into the embedding input, not into the chunk
    if sum(count_words(ln[2]) for ln in body) <= MAX_WORDS:
        return [_mk(doc, [(body[0][0], body[-1][1])], "pubmed")]
    return [_mk(doc, [span], "pubmed") for span in _tile(_pack(_units(text, body, {})))]


# ---------------------------------------------------------------- entry point
def _mk(doc: dict, spans: list, unit_kind: str, answer_idx: int = -1, heading_path: str = "") -> dict:
    text = join_spans(doc["text"], spans)
    return {"doc_id": int(doc["id"]), "unit_kind": unit_kind, "answer_idx": answer_idx, "spans": [list(x) for x in spans],
            "text": text, "n_words": round(count_words(text), 1), "title": doc.get("title") or "",
            "heading_path": heading_path, "host": doc["host"], "lang": doc.get("lang") or "",
            "text_lang": doc.get("text_lang") or "", "doc_flags": list(doc.get("flags") or [])}


def usable(doc: dict) -> bool:
    flags = set(doc.get("flags") or [])
    return doc.get("status") == "ok" and "empty" not in flags and bool((doc.get("text") or "").strip())


def chunk_doc(doc: dict) -> list[dict]:
    if not usable(doc):
        return []
    kind = doc_type(doc)
    chunks = {"qa": chunk_qa, "pubmed": chunk_pubmed}.get(kind, chunk_article)(doc)
    for i, c in enumerate(chunks):
        c.update(chunk_id=f"{c['doc_id']}:{i}", idx=i, doc_type=kind)
        c["spans"] = [tuple(x) for x in c["spans"]]
    return chunks
