"""S1 plan: one LLM call per query -> search variants per language. Any failure -> plan of the original query only."""
from __future__ import annotations

import re

SYSTEM = """Bạn là chuyên gia y khoa song ngữ Việt–Trung–Anh, chuẩn bị truy vấn tìm kiếm cho một kho tài liệu y khoa gồm
bài báo/hỏi đáp tiếng Việt, diễn đàn hỏi đáp và bách khoa tiếng Trung, và tóm tắt PubMed tiếng Anh.
Chỉ trả về MỘT object JSON, không giải thích."""

USER = """Câu hỏi của người bệnh (tiếng Việt):
{q}

Trả về JSON đúng các khoá sau:
{{"type": "<một trong: symptom, disease, drug, pregnancy, pediatric, lab_result, nutrition, mental, procedure, other>",
 "vi_kw": "<từ khoá y khoa tiếng Việt, bỏ phần kể lể, giữ chỉ số/tên thuốc>",
 "zh_q": "<câu hỏi bằng tiếng Trung giản thể, viết như tiêu đề diễn đàn hỏi đáp y khoa (120ask, 39问医生)>",
 "zh_kw": "<thuật ngữ y khoa tiếng Trung, cách nhau bằng dấu cách>",
 "en_kw": "<medical English keywords, space separated, no AND/OR/quotes/brackets>",
 "entities": [{{"vi": "<tên trong câu hỏi>", "en": "<tên chuẩn/hoạt chất>", "zh": "<tên tiếng Trung>"}}],
 "subqs": [<chỉ khi câu hỏi có nhiều ý: mỗi ý là một câu hỏi tiếng Việt ngắn; ngược lại để []>]}}"""

KEYS = ("vi_kw", "zh_q", "zh_kw", "en_kw")
_LONG = 60  # words: below this, sub-questions are ignored even if the model returns some


def prompt(query: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER.format(q=query)}]


def normalize(query: str, raw: dict | None) -> dict:
    """Clean plan; keeps only string variants that are non-empty; `ok` tells whether the LLM plan was usable."""
    p = {"type": "other", "vi_kw": "", "zh_q": "", "zh_kw": "", "en_kw": "", "entities": [], "subqs": [], "ok": False}
    if isinstance(raw, dict):
        for k in KEYS:
            v = raw.get(k)
            p[k] = " ".join(v.split()) if isinstance(v, str) else ""
        p["type"] = raw.get("type") if isinstance(raw.get("type"), str) else "other"
        ents, subqs = raw.get("entities"), raw.get("subqs")
        p["entities"] = [e for e in ents if isinstance(e, dict)] if isinstance(ents, list) else []
        subqs = [s for s in subqs if isinstance(s, str) and s.strip()] if isinstance(subqs, list) else []
        long_q = len(query.split()) >= _LONG or query.count("?") >= 2
        p["subqs"] = subqs[:4] if long_q and len(subqs) > 1 else []
        p["ok"] = any(p[k] for k in KEYS)
    # entity names (en / zh) widen the lexical queries of their language; keyword lists are de-duplicated word-wise
    for key, lang in (("zh_kw", "zh"), ("en_kw", "en")):
        words = _clean_kw(p[key]).split() + [w for e in p["entities"] if isinstance(e.get(lang), str) for w in _clean_kw(e[lang]).split()]
        p[key] = " ".join(dict.fromkeys(words))
    p["vi_kw"] = _clean_kw(p["vi_kw"])
    return p


def _clean_kw(s: str) -> str:
    """Search-engine syntax is noise for bge-m3 sparse: drop boolean operators, quotes, brackets."""
    s = re.sub(r"\b(?:AND|OR|NOT)\b", " ", s or "")
    return " ".join(re.sub(r"[\"()\[\]{}]", " ", s).split())


def expand(query: str, plan: dict) -> list[str]:
    """Hook for extra query variants (e.g. OKF aliases later). No-op for now."""
    return []


def run(queries, llm) -> dict:
    """{query_id: plan}"""
    qs = queries["query"].tolist()
    raws = llm.map_json([prompt(q) for q in qs]) if llm is not None else [None] * len(qs)
    return {int(i): normalize(q, r) for i, q, r in zip(queries["id"].tolist(), qs, raws)}


def has_cjk(s: str) -> bool:
    return bool(re.search(r"[一-鿿]", s or ""))
