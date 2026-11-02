"""S4 judge (v2): LLM reads grey-zone docs (uncertain p) and grades them 0-3, pointing at the chunks holding the answer."""
from __future__ import annotations

SYSTEM = """Bạn là bác sĩ đánh giá tài liệu y khoa (tiếng Việt, tiếng Trung hoặc tiếng Anh) có trả lời được câu hỏi của người bệnh không.
Chỉ trả về MỘT object JSON, không giải thích."""

USER = """Câu hỏi (tiếng Việt): {q}

Tài liệu: {title}
{chunks}

Chấm mức liên quan:
3 = trả lời trực tiếp câu hỏi; 2 = trả lời một phần quan trọng; 1 = cùng chủ đề nhưng không trả lời; 0 = không liên quan.
Trả về: {{"rel": <0-3>, "evidence": [<số thứ tự các đoạn [c..] chứa câu trả lời>]}}"""

MAX_CHARS = 3000  # chunks run 500-1000 words; the answer is often past the first few hundred chars


def prompt(query: str, title: str, chunks: list[str], max_chars: int = MAX_CHARS) -> list[dict]:
    body = "\n\n".join(f"[c{i}] {t[:max_chars]}" for i, t in enumerate(chunks))
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER.format(q=query, title=title or "", chunks=body)}]


def parse(raw: dict | None, n_chunks: int) -> tuple[int | None, list[int]]:
    if not isinstance(raw, dict):
        return None, []
    try:
        rel = int(raw.get("rel"))
    except (TypeError, ValueError):
        return None, []
    if not 0 <= rel <= 3:
        return None, []
    ev = []
    for x in raw.get("evidence") or []:
        try:
            i = int(str(x).lstrip("c"))
        except ValueError:
            continue
        if 0 <= i < n_chunks and i not in ev:
            ev.append(i)
    return rel, ev


# --- listwise (deep mode): one call grades a batch of candidate docs for one query
LIST_USER = """Câu hỏi (tiếng Việt): {q}

Các tài liệu ứng viên:
{docs}

Chấm TỪNG tài liệu (đủ {n} tài liệu): 3 = trả lời trực tiếp câu hỏi; 2 = trả lời một phần quan trọng; 1 = cùng chủ đề nhưng không trả lời; 0 = không liên quan.
Trả về đúng object JSON này, thay mỗi ? bằng điểm 0-3: {keys}"""


def list_prompt(query: str, docs: list[tuple[str, str]], snip: int = 700) -> list[dict]:
    """docs: [(title, best chunk text)]. Keyed answer ({"d0": .., "d1": ..}): a bare list came back one short."""
    body = "\n\n".join(f"[d{i}] {t or ''}\n{x[:snip]}" for i, (t, x) in enumerate(docs))
    keys = "{" + ", ".join(f'"d{i}": ?' for i in range(len(docs))) + "}"
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": LIST_USER.format(q=query, docs=body, n=len(docs), keys=keys)}]


def list_parse(raw: dict | None, n: int) -> list[int | None] | None:
    """Grade per doc (None where missing or invalid); None if fewer than half the docs got a valid grade."""
    if not isinstance(raw, dict):
        return None
    out = []
    for i in range(n):
        v = raw.get(f"d{i}", raw.get(str(i)))
        try:
            v = int(v)
        except (TypeError, ValueError):
            v = None
        out.append(v if v is not None and 0 <= v <= 3 else None)
    return out if sum(v is not None for v in out) * 2 >= n else None
