from extract.clean import clean_markdown, flags, to_text


def test_clean_markdown_whitespace_dupes_and_blank_lines():
    md = "# Tiêu đề​\n\n\n\nĐoạn   một\nĐoạn   một\n\n\n- mục\n﻿cuối  "
    assert clean_markdown(md) == "# Tiêu đề\n\nĐoạn một\n\n- mục\ncuối"


def test_clean_markdown_nfc():
    decomposed = "Việt"  # NFD 'Việt'
    assert clean_markdown(decomposed) == "Việt"


def test_clean_markdown_global_and_rule_drop_lines():
    md = "Bài viết\nChọn cỡ chữ: A A A\n请使用微信扫一扫\nTheo dõi chúng tôi trên Google News\nnội dung\nQUẢNG CÁO"
    assert clean_markdown(md, drop_lines=(r"^QUẢNG CÁO$",)) == "Bài viết\nnội dung"


def test_to_text_strips_markdown_keeps_words():
    md = "# Head\n\n**Bold** and *it* `code`\n\n- item one\n1. kept number\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n> quote"
    assert to_text(md) == "Head\n\nBold and it code\n\nitem one\n1. kept number\n\na b\n1 2\n\nquote"


def test_to_text_keeps_cjk_numbering():
    assert to_text("1.穿刺引流\n2.局部注射") == "1.穿刺引流\n2.局部注射"


def test_flags():
    assert flags("", "direct", False) == ["empty"]
    assert flags("x" * 299, "wayback_direct", True) == ["short", "fallback", "archive"]
    assert flags("x" * 300, "mirror", False) == ["archive"]
    assert flags("x" * 300, "direct", False) == []
