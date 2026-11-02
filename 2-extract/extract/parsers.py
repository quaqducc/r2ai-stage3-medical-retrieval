"""Page -> Doc(title, markdown). Order: PubMed XML, host rule, trafilatura (+ Next.js JSON), lxml fallback."""
from __future__ import annotations

import copy
import html as htmllib
import json
import logging
import re
from dataclasses import dataclass

import lxml.etree
import lxml.html
import trafilatura

from . import rules
from .clean import clean_markdown
from .htmlmd import to_markdown

log = logging.getLogger("extract")
SHORT = 300
_NEXT = re.compile(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S)
_CJK = re.compile(r"[一-鿿]")
_TAGS = re.compile(r"<[^>]+>")
_CLOSE = re.compile(r"</(?:html|body)\s*>", re.I)  # optional tags; a stray early </html> makes libxml2 stop reading
_XMLDECL = re.compile(r"^\s*<\?xml[^>]*\?>", re.I)  # lxml refuses str input with an encoding declaration
_FALLBACK_DROP = "//script|//style|//noscript|//nav|//header|//footer|//aside|//iframe"  # not //form: ASP.NET wraps whole pages in one


class PageTimeout(BaseException):
    """Raised by the worker's alarm; BaseException so no `except Exception` on the way swallows it."""


@dataclass
class Doc:
    title: str
    markdown: str
    parser: str
    fallback: bool = False
    drop_lines: tuple = ()
    notes: tuple = ()  # rule_error / tree_none / extractor_error: why a page left the intended path


def parse(host: str, url: str, html: str) -> Doc:
    if not html or not html.strip():
        return Doc("", "", "none")
    head = html.lstrip()[:400]
    if "<PubmedArticle" in head or "<PubmedBookArticle" in head or (host == "pubmed.ncbi.nlm.nih.gov" and head.startswith("<?xml")):
        doc = pubmed(html)
        if doc is not None:
            return doc
    html = _CLOSE.sub("", _XMLDECL.sub("", html))
    tree = _tree(html)
    title = resolve_title(tree) if tree is not None else ""
    fallback, notes = False, []
    rule = rules.get(host)
    if rule is not None:
        fallback = True  # cleared only when the rule produces the document
        if tree is None:
            notes.append("tree_none")
        else:
            try:
                rtitle, md = rules.apply(rule, copy.deepcopy(tree))
            except Exception as e:  # broken selector in a rule: generic parse, flagged, never an empty row
                log.warning("rule failed host=%s url=%s: %s: %s", host, url, type(e).__name__, e)
                rtitle, md = "", ""
                notes.append("rule_error")
            if len(clean_markdown(md, rule.drop_lines)) >= rule.min_chars:  # judged after cleaning: chrome-only matches fail
                return Doc(rtitle or title, md, "rule", False, rule.drop_lines)
            if rule.meta_fallback:
                desc = meta_description(tree)
                if desc:
                    return Doc(title, desc, "meta", True, rule.drop_lines, tuple(notes) + ("meta_only",))
    try:
        md, parser = _trafilatura(html, url), "trafilatura"
    except Exception as e:
        log.warning("trafilatura failed url=%s: %s: %s", url, type(e).__name__, e)
        md, parser = "", "trafilatura"
        notes.append("extractor_error")
    if len(md) < SHORT and "__NEXT_DATA__" in html:
        nd = next_data_text(html)
        if len(nd) > len(md):
            md, parser = nd, "next_data"
    if not md.strip() and tree is not None:
        md, parser = lxml_fallback(tree), "lxml"
    return Doc(title, md, parser, fallback, rule.drop_lines if rule else (), tuple(notes))


def _tree(html: str):
    try:
        return lxml.html.fromstring(html)
    except (lxml.etree.ParserError, ValueError):  # empty document / unsupported declaration
        try:
            return lxml.html.fromstring(html.encode("utf-8", "replace"), parser=lxml.html.HTMLParser(encoding="utf-8"))
        except (lxml.etree.ParserError, ValueError):
            return None


def _trafilatura(html: str, url: str) -> str:
    return trafilatura.extract(html, url=url, output_format="markdown", include_tables=True, include_comments=True,
                               include_links=False, include_images=False, favor_recall=True, deduplicate=False) or ""


def _clean_text(s: str) -> str:
    return " ".join(s.split())


def resolve_title(tree) -> str:
    for xp in ('//meta[@property="og:title"]/@content', "//h1", "//title"):
        hit = tree.xpath(xp)
        if hit:
            h = hit[0]
            t = _clean_text(h if isinstance(h, str) else "".join(h.itertext()))
            if t:
                return t
    return ""


def meta_description(tree) -> str:
    for xp in ('//meta[@property="og:description"]/@content', '//meta[@name="description"]/@content'):
        hit = tree.xpath(xp)
        if hit and _clean_text(hit[0]):
            return _clean_text(hit[0])
    return ""


def lxml_fallback(tree) -> str:
    tree = copy.deepcopy(tree)
    for n in tree.xpath(_FALLBACK_DROP):
        if n.getparent() is not None:
            n.drop_tree()
    main = tree.xpath("//article|//main") or tree.xpath("//body") or [tree]
    return to_markdown(main[:1])


def next_data_text(html: str) -> str:
    """Next.js sites (msdmanuals.cn) ship the article only inside the __NEXT_DATA__ JSON."""
    m = _NEXT.search(html)
    if not m:
        return ""
    try:
        data = json.loads(m.group(1))
    except ValueError:
        return ""
    out, seen = [], set()

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str):
            s = x.lstrip()
            if s[:1] in "[{":
                try:
                    walk(json.loads(s))  # JSON nested in a string
                    return
                except ValueError:
                    pass
            if len(_CJK.findall(x)) >= 15:
                t = _clean_text(htmllib.unescape(_TAGS.sub(" ", x)))
                if t not in seen:
                    seen.add(t)
                    out.append(t)

    walk(data)
    return "\n\n".join(out)


def pubmed(xml: str) -> Doc | None:
    """efetch PubmedArticle XML -> title + journal line + abstract sections + MeSH + keywords.
    None when the document is not a PubMed record (e.g. an HTML error page) so the generic path handles it."""
    xml = _XMLDECL.sub("", xml)
    try:
        root = lxml.etree.fromstring(xml.encode("utf-8"), parser=lxml.etree.XMLParser(recover=True, huge_tree=True))
    except lxml.etree.XMLSyntaxError:
        return None
    if root is None or (root.find(".//PubmedArticle") is None and root.find(".//PubmedBookArticle") is None
                        and root.tag not in ("PubmedArticle", "PubmedBookArticle") and root.find(".//ArticleTitle") is None):
        return None

    def txt(path):
        return " ".join(_clean_text("".join(e.itertext())) for e in root.iterfind(path)).strip()

    title = txt(".//ArticleTitle") or txt(".//BookTitle") or txt(".//VernacularTitle")
    year = txt(".//JournalIssue/PubDate/Year") or txt(".//PubDate/MedlineDate") or txt(".//PubDate/Year")
    parts = [f"*{txt('.//Journal/Title') or txt('.//PublisherName')}* ({year}) · PMID {txt('.//MedlineCitation/PMID') or txt('.//BookDocument/PMID')}"]
    abstract = []
    for ab in root.iterfind(".//Abstract/AbstractText"):
        body, label = _clean_text("".join(ab.itertext())), ab.get("Label")
        if body:
            abstract.append(f"**{label}.** {body}" if label else body)
    if abstract:
        parts += ["## Abstract", *abstract]
    mesh = [_clean_text("".join(m.itertext())) for m in root.iterfind(".//MeshHeading/DescriptorName")]
    if mesh:
        parts.append("**MeSH:** " + "; ".join(mesh))
    kw = [_clean_text("".join(k.itertext())) for k in root.iterfind(".//Keyword")]
    if kw:
        parts.append("**Keywords:** " + "; ".join(kw))
    if title:
        parts.insert(0, f"# {title}")
    return Doc(title, "\n\n".join(parts), "pubmed_xml")
