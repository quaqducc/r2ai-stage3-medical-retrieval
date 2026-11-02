"""Normalise extracted markdown, derive plain text (for verbatim chunk matching) and quality flags."""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

SHORT = 300
_CJK = re.compile(r"[\u3400-\u9fff]")
ARCHIVE = {"wayback", "wayback_direct", "commoncrawl", "mirror"}
_INVISIBLE = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)
# Whole lines that are site chrome on many hosts; host-specific ones live in Rule.drop_lines.
GLOBAL_DROP = (
    r"^Chọn cỡ chữ\b", r"^Theo dõi .{0,40}(Google News|Zalo|Facebook)", r"^(Chia sẻ|Bình luận|In bài|Lưu bài)$",
    r"^.{0,12}请使用\*?微信\*?扫一扫.{0,12}$", r"^扫描二维码", r"^(分享到|举报|我要咨询|向医生提问|举报相关问题)[:：]?$",
)


@lru_cache(maxsize=256)
def _compile(patterns: tuple) -> tuple:
    """Each pattern compiled on its own: inline flags like (?i) are only legal at a pattern's start (3.11)."""
    return tuple(re.compile(p) for p in patterns)


def clean_markdown(md: str, drop_lines: tuple = (), counter: dict | None = None) -> str:
    """counter (optional) accumulates 'dropped_chars': chars removed by drop_lines patterns."""
    md = unicodedata.normalize("NFC", md).translate(_INVISIBLE).replace("\xa0", " ")
    drop = _compile(GLOBAL_DROP + tuple(drop_lines))
    out, prev, dropped, seen = [], None, 0, set()
    for line in md.splitlines():
        line = " ".join(line.split())
        if line and any(r.search(line) for r in drop):
            dropped += len(line)
            continue
        data_row = line.startswith(("|", "- "))  # repeated table rows / list items are data, kept
        if line and line == prev and not data_row:
            continue
        key = line.lstrip("#").strip()  # '# Title' and a later 'Title' paragraph are the same text
        if not data_row and (len(key) >= 20 or len(_CJK.findall(key)) >= 8):  # paragraph seen earlier (title/lead)
            if key in seen:
                continue
            seen.add(key)
        if not line and (not out or out[-1] == ""):
            continue
        out.append(line)
        prev = line or prev
    while out and out[-1] == "":
        out.pop()
    if counter is not None:
        counter["dropped_chars"] = counter.get("dropped_chars", 0) + dropped
    return "\n".join(out)


_HEAD = re.compile(r"^#{1,6}\s+")
_BULLET = re.compile(r"^(?:[-*+]|>)\s+")
_EMPH = (re.compile(r"\*\*(\S(?:.*?\S)?)\*\*"), re.compile(r"__(\S(?:.*?\S)?)__"),
         re.compile(r"(?<![*\w])\*(\S(?:.*?\S)?)\*(?![*\w])"), re.compile(r"`([^`]+)`"))  # paired markers only: '2*3' stays
_IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_SEP = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?$")


def to_text(md: str) -> str:
    out = []
    for line in md.splitlines():
        if _SEP.match(line):
            continue
        if line.startswith("|") and line.endswith("|"):
            line = " ".join(c.strip() for c in line.strip("|").split("|") if c.strip())
        if line[:1] == "\\" and line[1:2] in "#>+*-":  # escaped source text (htmlmd), not syntax
            out.append(" ".join(line[1:].split()))
            continue
        line = _HEAD.sub("", line)
        line = _BULLET.sub("", line)
        line = _LINK.sub(r"\1", _IMG.sub("", line))
        for r in _EMPH:
            line = r.sub(r"\1", line)
        line = " ".join(line.split())
        out.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def flags(text: str, source: str | None, fallback: bool, extra=()) -> list[str]:
    f = []
    if not text:
        f.append("empty")
    elif len(text) < SHORT:
        f.append("short")
    if fallback:
        f.append("fallback")
    if source in ARCHIVE:
        f.append("archive")
    f.extend(x for x in extra if x not in f)
    return f
