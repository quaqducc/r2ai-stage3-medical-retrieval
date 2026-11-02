"""Per-host extraction rules: XPath selectors for the content of a site template."""
from __future__ import annotations

import json
from dataclasses import dataclass

import lxml.html

from .htmlmd import to_markdown


@dataclass(frozen=True)
class Rule:
    body: tuple[str, ...]            # XPath; matched nodes, in selector order, form the document
    title: str | None = None         # XPath to the title element/text
    drop: tuple[str, ...] = ()       # XPath of nodes removed before conversion (ads, related lists, buttons)
    drop_lines: tuple[str, ...] = ()  # regexes of whole markdown lines to remove after conversion
    min_chars: int = 50              # body shorter than this -> rule failed, fall back to generic
    meta_fallback: bool = False      # rule failed -> title + og/meta description, not generic extraction
    next_data: tuple[str, ...] = ()  # dotted paths into the __NEXT_DATA__ JSON whose HTML strings are appended
                                     # after the body (sections a Next.js page never renders server-side);
                                     # 'a.b[].c' iterates a list; absent paths are skipped


def _json_values(data, path: str) -> list:
    cur = [data]
    for seg in path.split("."):
        each = seg.endswith("[]")
        key = seg[:-2] if each else seg
        nxt = []
        for c in cur:
            v = c.get(key) if isinstance(c, dict) else None
            if v is None:
                continue
            nxt.extend(v if each and isinstance(v, list) else [v])
        cur = nxt
    return [v for v in cur if isinstance(v, str) and v.strip()]


def _next_data_md(tree, paths: tuple) -> str:
    raw = tree.xpath('//script[@id="__NEXT_DATA__"]/text()')
    if not raw:
        return ""
    try:
        data = json.loads(raw[0])
    except ValueError:
        return ""
    parts = []
    for p in paths:
        for html in _json_values(data, p):
            frag = lxml.html.fragment_fromstring(html, create_parent="div")
            parts.append(to_markdown([frag]))
    return "\n\n".join(x for x in parts if x)


def apply(rule: Rule, tree) -> tuple[str, str]:
    """(title, markdown). tree is consumed (drop removes nodes) — pass a copy."""
    title = ""
    extra = _next_data_md(tree, rule.next_data) if rule.next_data else ""
    if rule.title:
        hit = tree.xpath(rule.title)
        if hit:
            h = hit[0]
            title = " ".join((h if isinstance(h, str) else "".join(h.itertext())).split())
    for x in rule.drop:
        for n in tree.xpath(x):
            if n.getparent() is not None:
                n.drop_tree()
    nodes, seen = [], set()
    for x in rule.body:
        for n in tree.xpath(x):
            if isinstance(n, str) or id(n) in seen or any(id(a) in seen for a in n.iterancestors()):
                continue
            seen.add(id(n))
            nodes.append(n)
    md = to_markdown(nodes)
    return title, "\n\n".join(x for x in (md, extra) if x)


def get(host: str) -> Rule | None:
    """Exact host first, then suffix keys written with a leading dot ('.39.net' matches 'heart.39.net')."""
    from .hosts import RULES
    if host in RULES:
        return RULES[host]
    for k, r in RULES.items():
        if k.startswith(".") and host.endswith(k):
            return r
    return None
