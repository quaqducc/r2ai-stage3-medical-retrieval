import lxml.html
import pytest

from conftest import fixture_rows
from extract import parsers, rules
from extract.htmlmd import to_markdown
from extract.rules import Rule

PAGE = """<html><head><title>T - Site</title><meta property="og:title" content="OG title"></head><body>
<nav>Home | News</nav><div id="main"><h1>Head</h1><p>First <b>para</b> text.</p><ul><li>one</li><li>two</li></ul>
<table><tr><th>a</th><th>b</th></tr><tr><td>1</td><td>2</td></tr></table>
<div class="ad">BUY NOW</div><p>Line<br>break</p><script>var x=1</script></div>
<div class="related">Related junk</div></body></html>"""


def test_to_markdown_structure():
    tree = lxml.html.fromstring(PAGE)
    md = to_markdown(tree.xpath('//div[@id="main"]'))
    assert md.splitlines()[0] == "# Head"
    assert "First para text." in md
    assert "- one\n- two" in md
    assert "| a | b |\n| 1 | 2 |" in md
    assert "Line\n\nbreak" in md
    assert "var x" not in md and "Home | News" not in md


def test_rule_body_drop_and_title(monkeypatch):
    rule = Rule(body=('//div[@id="main"]',), title="//h1", drop=('//div[@class="ad"]',))
    monkeypatch.setattr(rules, "get", lambda host: rule)
    doc = parsers.parse("x.com", "http://x.com/a", PAGE)
    assert doc.parser == "rule" and not doc.fallback
    assert doc.title == "Head"
    assert "First para text." in doc.markdown and "BUY NOW" not in doc.markdown and "Related junk" not in doc.markdown


def test_rule_not_matching_falls_back(monkeypatch):
    monkeypatch.setattr(rules, "get", lambda host: Rule(body=('//div[@id="gone"]',)))
    doc = parsers.parse("x.com", "http://x.com/a", PAGE)
    assert doc.fallback and doc.parser != "rule"
    assert "First para text." in " ".join(doc.markdown.split())


def test_title_fallback_order():
    assert parsers.parse("x.com", "u", PAGE).title == "OG title"


@pytest.mark.parametrize("host", ["pubmed.ncbi.nlm.nih.gov"])
def test_pubmed_xml(host):
    rows = fixture_rows(host)
    assert rows
    for _, url, xml in rows:
        doc = parsers.parse(host, url, xml)
        assert doc.parser == "pubmed_xml"
        assert doc.title and "PMID" in doc.markdown
        assert "<" not in doc.markdown


def test_msdmanuals_next_data():
    for _, url, html in fixture_rows("www.msdmanuals.cn"):
        doc = parsers.parse("www.msdmanuals.cn", url, html)
        assert len(doc.markdown) > 300, doc.parser
        assert sum("一" <= c <= "鿿" for c in doc.markdown) > 150


def test_broken_html_not_empty():
    # baotayninh: '<html></html><head>...' — trafilatura returns nothing on it
    for _, url, html in fixture_rows("baotayninh.vn"):
        doc = parsers.parse("baotayninh.vn", url, html)
        assert len(doc.markdown) > 300, doc.parser


def test_empty_and_garbage_input():
    assert parsers.parse("x.com", "u", "").markdown == ""
    assert parsers.parse("x.com", "u", "\x00\x01 not html").markdown is not None
