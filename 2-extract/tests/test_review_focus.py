"""Failure modes from PLAN.md 'Review focus' and the core code review."""
import json
import os
import re

import lxml.etree
import pandas as pd
import pytest
import zstandard

from extract import parsers, pipeline, rules
from extract.clean import clean_markdown, to_text
from extract.rules import Rule
from extract.source import Row, decode


def z(b):
    return zstandard.ZstdCompressor().compress(b)


def row(html, **kw):
    return Row(id=kw.get("id", 1), url="https://x.vn/a", final_url=None, status=200, kind="ok", source="direct",
               charset=kw.get("charset"), ctype="text/html", fetched_at=0.0, blob=z(html.encode(kw.get("enc", "utf-8"))))


def test_every_host_rule_compiles():
    """Selectors and drop_lines of all host modules must be valid on this Python (3.11 rejects mid-pattern (?i))."""
    from extract.hosts import RULES
    for host, r in RULES.items():
        for x in r.body + r.drop + ((r.title,) if r.title else ()):
            lxml.etree.XPath(x)
        for p in r.drop_lines:
            re.compile(p)
        assert r.body, host


def test_inline_flag_drop_line_works():
    assert clean_markdown("keep\nQuảng Cáo here\nend", drop_lines=(r"(?i)^quảng cáo",)) == "keep\nend"


def test_utf8_page_with_wrong_latin1_label():
    assert "sốt xuất huyết" in decode(z("<p>sốt xuất huyết</p>".encode()), "iso-8859-1", "text/html; charset=ISO-8859-1")


def test_lone_surrogate_does_not_kill_record(monkeypatch):
    monkeypatch.setattr(pipeline, "parse", lambda *a: parsers.Doc("t\ud83d", "nội dung \ud83d " * 40, "trafilatura"))
    rec = pipeline.build_record(row("<p>x</p>"), "x.vn", None)
    assert not rec["parser"].startswith("error") and rec["n_chars"] > 0
    rec["text"].encode("utf-8")  # must not raise


def test_broken_rule_falls_back_not_empty(monkeypatch):
    monkeypatch.setattr(rules, "get", lambda h: Rule(body=("count(//p)",)))  # XPath returning a number
    html = "<html><body><article>" + "<p>Nội dung đầy đủ của bài viết về bệnh tim mạch.</p>" * 15 + "</article></body></html>"
    doc = parsers.parse("x.vn", "u", html)
    assert doc.fallback and "bệnh tim mạch" in doc.markdown


def test_xml_declaration_page_keeps_title_and_rule(monkeypatch):
    monkeypatch.setattr(rules, "get", lambda h: Rule(body=('//div[@id="c"]',), title="//h1"))
    html = '<?xml version="1.0" encoding="gb2312"?><html><body><h1>标题</h1><div id="c">' + "".join(f"<p>中医治疗失眠的方法很多，第{i}种方法。</p>" for i in range(10)) + "</div></body></html>"
    doc = parsers.parse("x.cn", "u", html)
    assert doc.parser == "rule" and doc.title == "标题"


def test_pubmed_host_html_error_page_goes_generic():
    html = "<html><body><article>" + "<p>Service temporarily unavailable, please retry later.</p>" * 8 + "</article></body></html>"
    doc = parsers.parse("pubmed.ncbi.nlm.nih.gov", "u", html)
    assert doc.parser != "pubmed_xml"


def test_aspnet_form_page_fallback_keeps_content():
    from extract.parsers import lxml_fallback
    import lxml.html
    tree = lxml.html.fromstring("<html><body><form runat='server'><div>" + "Bệnh viện thông báo lịch khám mới. " * 10 + "</div></form></body></html>")
    assert "lịch khám" in lxml_fallback(tree)


def test_to_text_keeps_lone_asterisks():
    assert to_text("liều 2*3 lần, **quan trọng**") == "liều 2*3 lần, quan trọng"


def test_limited_stats_do_not_mark_host_complete(db_factory, tmp_path):
    db = db_factory("h.vn.sqlite", [{"id": i, "url": f"u{i}", "kind": "ok", "html": "<p>" + "nội dung " * 50 + "</p>"} for i in (1, 2, 3)])
    man = pd.DataFrame({"id": [1, 2, 3], "url": ["a", "b", "c"], "host": "h.vn", "lang": "vi", "lane": "x"})
    out = str(tmp_path / "out")
    pipeline.process_host(db, "h.vn", man, out, workdir=str(tmp_path / "w"), limit=1)
    st = pipeline.process_host(db, "h.vn", man, out, workdir=str(tmp_path / "w"))
    assert not st.get("skipped") and st["rows"] == 3


def test_markdown_looking_source_text_survives_in_text():
    import lxml.html
    from extract.htmlmd import to_markdown
    md = to_markdown([lxml.html.fromstring("<div><p>> 140/90 mmHg là tăng huyết áp</p><p>- 5 mg mỗi ngày</p></div>")])
    assert to_text(clean_markdown(md)) == "> 140/90 mmHg là tăng huyết áp\n\n- 5 mg mỗi ngày"


def test_layout_table_is_walked_as_blocks():
    import lxml.html
    from extract.htmlmd import to_markdown
    md = to_markdown([lxml.html.fromstring("<table><tr><td><h2>Triệu chứng</h2><p>Sốt cao.</p><p>Đau đầu.</p></td></tr></table>")])
    assert md == "## Triệu chứng\n\nSốt cao.\n\nĐau đầu."


def test_data_table_skips_script_and_keeps_caption():
    import lxml.html
    from extract.htmlmd import to_markdown
    md = to_markdown([lxml.html.fromstring("<table><caption>Liều</caption><tr><td>A<script>x=1</script></td><td>5mg</td></tr></table>")])
    assert md == "Liều\n\n| A | 5mg |"


def test_http_status_and_timeout_rows(monkeypatch):
    from extract.parsers import PageTimeout
    r = row("<html><body><article>" + "<p>Nội dung trang lỗi máy chủ nhưng vẫn có chữ.</p>" * 12 + "</article></body></html>")
    r.status = 503
    assert "http_503" in pipeline.build_record(r, "x.vn", None)["flags"]

    def slow(*a):
        raise PageTimeout()
    monkeypatch.setattr(pipeline, "parse", slow)
    assert pipeline.build_record(row("<p>x</p>"), "x.vn", None)["parser"] == "error:PageTimeout"


def test_suspect_host_reported(db_factory, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "parse", lambda *a: (_ for _ in ()).throw(ValueError("x")))
    db = db_factory("h.vn.sqlite", [{"id": i, "url": f"u{i}", "kind": "ok", "html": "<p>a</p>"} for i in (1, 2)])
    man = pd.DataFrame({"id": [1, 2], "url": ["a", "b"], "host": "h.vn", "lang": "vi", "lane": "x"})
    st = pipeline.process_host(db, "h.vn", man, str(tmp_path / "o"), workdir=str(tmp_path / "w"))
    assert "error_rate" in st["suspect"]


def test_rule_next_data_sections_longchau_disease_page():
    from conftest import fixture_rows
    page = [p for p in fixture_rows("nhathuoclongchau.com.vn") if p[0] == 282674]
    assert page
    _, url, html = page[0]
    doc = parsers.parse("nhathuoclongchau.com.vn", url, html)
    md = clean_markdown(doc.markdown, doc.drop_lines)
    assert doc.parser == "rule"
    assert "Phương pháp xét nghiệm và chẩn đoán dương vật cong" in md  # only in __NEXT_DATA__
    assert md.count("Những ai có nguy cơ mắc phải dương vật cong") == 1


def test_json_values_paths():
    from extract.rules import _json_values
    data = {"a": {"b": "<p>x</p>", "l": [{"q": "<p>1</p>"}, {"q": ""}, {"z": 1}]}}
    assert _json_values(data, "a.b") == ["<p>x</p>"]
    assert _json_values(data, "a.l[].q") == ["<p>1</p>"]
    assert _json_values(data, "a.missing") == []


def test_non_adjacent_duplicate_paragraph_removed_but_data_rows_kept():
    md = "Tiêu đề bài viết về sốt xuất huyết\n\nĐoạn mở đầu của bài viết dài hơn hai mươi ký tự.\n\nTiêu đề bài viết về sốt xuất huyết\n\n| a | 1 |\n| a | 1 |\n\n- mục lặp lại đủ dài để vượt ngưỡng\n- mục lặp lại đủ dài để vượt ngưỡng"
    out = clean_markdown(md)
    assert out.count("Tiêu đề bài viết") == 1 and out.count("| a | 1 |") == 2 and out.count("- mục lặp") == 2


def test_detect_lang():
    from extract.pipeline import detect_lang
    assert detect_lang("Sốt xuất huyết là bệnh truyền nhiễm cấp tính do virus Dengue gây ra.") == "vi"
    assert detect_lang("骨头囊肿的治疗可以考虑穿刺引流、局部注射。") == "zh"
    assert detect_lang("Malabsorption of carbohydrates such as fructose or lactose.") == "en"
    assert detect_lang("") == ""


def test_pubmed_text_starts_with_title():
    from conftest import fixture_rows
    _, url, xml = fixture_rows("pubmed.ncbi.nlm.nih.gov")[0]
    doc = parsers.parse("pubmed.ncbi.nlm.nih.gov", url, xml)
    assert to_text(doc.markdown).startswith(doc.title)


def test_cjk_title_repeated_as_paragraph_removed():
    out = clean_markdown("# 先天性心脏病轻微的不需要治疗吗\n\n先天性心脏病轻微的不需要治疗吗\n\n补充说明：先天性心脏病轻微的不需要治疗吗")
    assert out == "# 先天性心脏病轻微的不需要治疗吗\n\n补充说明：先天性心脏病轻微的不需要治疗吗"
