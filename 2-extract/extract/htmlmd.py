"""Minimal deterministic HTML (lxml elements) -> markdown: headings, paragraphs, lists, tables, line breaks."""
from __future__ import annotations

import re

SKIP = {"script", "style", "noscript", "iframe", "button", "svg", "select", "input", "textarea",
        "nav", "template", "object", "embed", "canvas", "video", "audio", "map", "head"}
BLOCK = {"p", "div", "section", "article", "main", "header", "footer", "aside", "blockquote", "pre", "ul", "ol",
         "dl", "dt", "dd", "figure", "figcaption", "address", "center", "details", "summary", "body", "html", "hr"}
_LAYOUT = {"p", "div", "ul", "ol", "table", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article"}
_MDSTART = re.compile(r"^(?:#{1,6}|[>+*-])\s")  # source text that would read as markdown syntax: escaped with "\\"
HEAD = {"h1": "# ", "h2": "## ", "h3": "### ", "h4": "#### ", "h5": "##### ", "h6": "###### "}


class _Out:
    def __init__(self):
        self.blocks: list[tuple[str, str]] = []  # (kind, text)
        self.cur: list[str] = []
        self.prefix, self.kind = "", "p"

    def text(self, s):
        if s:
            self.cur.append(s)

    def flush(self):
        t = " ".join("".join(self.cur).split())
        self.cur = []
        if t:
            if _MDSTART.match(t):
                t = "\\" + t
            self.blocks.append((self.kind, self.prefix + t))
        self.prefix, self.kind = "", "p"

    def render(self) -> str:
        out, prev = [], None
        for kind, t in self.blocks:
            if out:
                out.append("\n" if kind == prev and kind in ("li", "tr") else "\n\n")
            out.append(t)
            prev = kind
        return "".join(out)


def _tag(el) -> str:
    t = el.tag
    return t.lower() if isinstance(t, str) else ""


def _text(el) -> str:
    """Text of el without SKIP subtrees (script/style/nav inside a cell must not leak)."""
    out = [el.text or ""]
    for c in el:
        if _tag(c) not in SKIP:
            out.append(_text(c))
        out.append(c.tail or "")
    return "".join(out)


def _cell(el) -> str:
    return " ".join(_text(el).split()).replace("|", "/")


def _is_layout(table) -> bool:
    """Old CN/VN sites lay out whole pages with tables: cells holding blocks are content, not data."""
    return any(_tag(d) in _LAYOUT for c in table.iter("td", "th") for d in c.iterdescendants())


def _walk(el, o: _Out):
    tag = _tag(el)
    if not tag or tag in SKIP:  # comments, processing instructions, junk
        o.text(el.tail)
        return
    if tag == "br":
        o.flush()
    elif tag in HEAD:
        o.flush()
        t = " ".join(_text(el).split())
        if t:
            o.blocks.append(("h", HEAD[tag] + t))
    elif tag == "li":
        o.flush()
        o.prefix, o.kind = "- ", "li"  # used by the first flush inside the item, even if a block child triggers it
        o.text(el.text)
        for c in el:
            _walk(c, o)
        o.flush()
    elif tag == "table" and not _is_layout(el):
        o.flush()
        for cap in el.xpath("./caption"):
            o.blocks.append(("p", _cell(cap)))
        for tr in el.xpath("./tr|./thead/tr|./tbody/tr|./tfoot/tr"):  # direct rows: nested tables are not repeated
            cells = [_cell(c) for c in tr if _tag(c) in ("td", "th")]
            if any(cells):
                o.blocks.append(("tr", "| " + " | ".join(cells) + " |"))
    elif tag in BLOCK or tag in ("table", "thead", "tbody", "tfoot", "tr", "td", "th", "caption"):
        o.flush()
        o.text(el.text)
        for c in el:
            _walk(c, o)
        o.flush()
    else:  # inline
        o.text(el.text)
        for c in el:
            _walk(c, o)
    o.text(el.tail)


def to_markdown(nodes) -> str:
    o = _Out()
    for n in nodes:
        tail, n.tail = n.tail, None  # a selected node's tail text is outside the selection
        try:
            _walk(n, o)
        finally:
            n.tail = tail
        o.flush()
    return o.render()
