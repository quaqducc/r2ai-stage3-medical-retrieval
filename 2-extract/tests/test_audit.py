from extract.audit import blocks, paragraph_recall

HTML = """<html><body><nav><p>""" + "Menu item that is long enough to look like content but lives in nav " * 2 + """</p></nav>
<div class="q"><p>Câu hỏi: con tôi bị sốt cao liên tục ba ngày nay, uống hạ sốt không đỡ, có nguy hiểm không ạ?</p></div>
<div class="a"><div><p>Trả lời: sốt cao kéo dài trên ba ngày ở trẻ cần được khám ngay để loại trừ sốt xuất huyết và viêm phổi.</p></div></div>
<div><p>短</p><p>这是一个足够长的中文段落用于测试召回率计算是否正确处理中文字符的情况</p></div></body></html>"""


def test_blocks_skip_nav_and_short():
    bl = blocks(HTML)
    assert len(bl) == 3 and not any("Menu item" in b for b in bl)


def test_recall_counts_missing_answer():
    text = "Câu hỏi: con tôi bị sốt cao liên tục ba ngày nay, uống hạ sốt không đỡ, có nguy hiểm không ạ?\n这是一个足够长的中文段落用于测试召回率计算是否正确处理中文字符的情况"
    rc, missed = paragraph_recall(HTML, text)
    assert round(rc, 2) == 0.67 and len(missed) == 1 and missed[0].startswith("Trả lời")


def test_recall_whitespace_insensitive():
    rc, _ = paragraph_recall(HTML, " ".join(b.replace(" ", "  ") for b in blocks(HTML)))
    assert rc == 1.0
