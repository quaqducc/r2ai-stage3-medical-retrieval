import pytest

from conftest import clean_doc
from ingest.chunk import MAX_WORDS, MIN_WORDS, QA_SPLIT_WORDS, chunk_doc, count_words, doc_type, join_spans, split_spans


def joined(doc, ch):
    return join_spans(doc["text"], ch["spans"])


def check_verbatim(doc, chunks):
    assert chunks
    for i, ch in enumerate(chunks):
        assert ch["text"] == joined(doc, ch)
        assert ch["chunk_id"] == f"{doc['id']}:{i}" and ch["idx"] == i and ch["doc_id"] == doc["id"]
        assert ch["n_words"] == round(count_words(ch["text"]), 1)
        if ch["unit_kind"] != "qa_answer" or ch["answer_idx"] == 0:
            assert ch["text"] in doc["text"]  # exact substring of the source document
        assert "\n\n".join(split_spans(ch["text"], ch["spans"]).values()).replace("\n", "") == \
            "".join(doc["text"][s_:e] for s_, e in ch["spans"]).replace("\n", "")


def test_count_words():
    assert count_words("Sốt xuất huyết là bệnh truyền nhiễm") == 7
    assert count_words("骨头囊肿怎么治疗") == pytest.approx(8 / 1.6)
    assert count_words("服用 HCG 2次") == pytest.approx(3 / 1.6 + 2)
    assert count_words("你好，可以看出来的。") == pytest.approx(8 / 1.6)  # full-width punctuation is not a word


@pytest.mark.parametrize("host,kind", [("www.120ask.com", "qa"), ("ask.39.net", "qa"), ("pubmed.ncbi.nlm.nih.gov", "pubmed"),
                                       ("hellobacsi.com", "article"), ("www.msdmanuals.cn", "article")])
def test_doc_type(host, kind):
    assert doc_type(clean_doc(host)) == kind


def test_qa_single_answer_is_one_chunk_with_question():
    doc = clean_doc("www.120ask.com", 1041244)
    chunks = chunk_doc(doc)
    check_verbatim(doc, chunks)
    assert len(chunks) == 1 and chunks[0]["unit_kind"] == "qa_single"
    assert chunks[0]["text"].startswith("同房21天验孕棒准确吗") and "病情分析" in chunks[0]["text"]


def test_qa_question_without_answer():
    doc = clean_doc("www.120ask.com", 1202820)
    chunks = chunk_doc(doc)
    assert len(chunks) == 1 and chunks[0]["unit_kind"] == "qa_question"


def test_qa_two_answers_short_total_stays_one_chunk():
    doc = clean_doc("ask.39.net", 2231109)  # 2 doctors, well under 600 words in total
    assert count_words(doc["text"]) <= QA_SPLIT_WORDS
    chunks = chunk_doc(doc)
    check_verbatim(doc, chunks)
    assert len(chunks) == 1 and chunks[0]["unit_kind"] == "qa_single"


def test_qa_many_long_answers_split_and_each_starts_with_full_question(monkeypatch):
    import ingest.chunk as C
    monkeypatch.setattr(C, "QA_SPLIT_WORDS", 50)  # force the split on the real 2-answer page
    doc = clean_doc("ask.39.net", 2231109)
    chunks = chunk_doc(doc)
    check_verbatim(doc, chunks)
    assert [c["unit_kind"] for c in chunks] == ["qa_answer", "qa_answer"]
    question = doc["text"][chunks[0]["spans"][0][0]:chunks[0]["spans"][0][1]]
    assert "补充说明" in question and "杨猛" not in question
    assert all(c["text"].startswith(question) for c in chunks)
    assert "杨猛 副主任医师" in chunks[0]["text"] and "陈肖贤 主治医师" in chunks[1]["text"]
    assert "陈肖贤" not in chunks[0]["text"]


def test_vi_qa_tra_loi_marker():
    for host, did in [("www.vinmec.com", 743963), ("vnexpress.net", 702208)]:
        doc = clean_doc(host, did)
        assert doc_type(doc) == "qa"
        chunks = chunk_doc(doc)
        check_verbatim(doc, chunks)
        assert len(chunks) == 1 and chunks[0]["unit_kind"] == "qa_single"


def test_pubmed_title_plus_abstract_without_metadata_lines():
    doc = clean_doc("pubmed.ncbi.nlm.nih.gov")
    chunks = chunk_doc(doc)
    check_verbatim(doc, chunks)
    assert len(chunks) == 1
    t = chunks[0]["text"]
    assert not t.startswith(doc["title"]) and "PMID" not in t and "MeSH:" not in t and t in doc["text"]
    assert chunks[0]["unit_kind"] == "pubmed"


def test_article_sizes_coverage_and_heading_boundaries():
    doc = clean_doc("hellobacsi.com")  # ~10k chars, many headings
    chunks = chunk_doc(doc)
    check_verbatim(doc, chunks)
    assert len(chunks) >= 2
    for c in chunks[:-1]:
        assert MIN_WORDS <= c["n_words"] <= MAX_WORDS, c["n_words"]
    assert chunks[-1]["n_words"] <= MAX_WORDS
    # contiguous and complete: chunks tile the text (only whitespace between them)
    spans = [c["spans"][0] for c in chunks]
    assert spans[0][0] == 0 and spans[-1][1] == len(doc["text"].rstrip())
    for (a, b), (c, d) in zip(spans, spans[1:]):
        assert b <= c and doc["text"][b:c].strip() == ""


def test_article_short_doc_one_chunk():
    doc = clean_doc("www.cnkang.com")  # a few hundred chars
    assert count_words(doc["text"]) < MIN_WORDS
    chunks = chunk_doc(doc)
    check_verbatim(doc, chunks)
    assert len(chunks) == 1 and chunks[0]["spans"] == [(0, len(doc["text"]))]


def test_article_long_section_split_by_paragraph_then_sentence():
    sent = "Đây là một câu dài mô tả triệu chứng của bệnh sốt xuất huyết ở trẻ em. "
    text = "Tiêu đề\n\n" + "\n\n".join(sent * 12 for _ in range(10)) + "\n\n" + sent * 200
    doc = {"id": 1, "host": "x.vn", "lang": "vi", "text_lang": "vi", "title": "Tiêu đề", "url": "u", "status": "ok",
           "flags": [], "text": text, "markdown": "# Tiêu đề\n\n" + text.split("\n\n", 1)[1], "n_chars": len(text)}
    chunks = chunk_doc(doc)
    check_verbatim(doc, chunks)
    assert all(c["n_words"] <= MAX_WORDS for c in chunks)
    assert all(c["text"].rstrip().endswith(".") for c in chunks)  # never cut mid-sentence


def test_unusable_docs_give_no_chunks():
    base = {"id": 1, "host": "x.vn", "lang": "vi", "text_lang": "vi", "title": "", "url": "u", "markdown": ""}
    assert chunk_doc({**base, "status": "blocked", "flags": ["empty"], "text": "", "n_chars": 0}) == []
    assert chunk_doc({**base, "status": "ok", "flags": ["empty"], "text": "", "n_chars": 0}) == []
