import pytest

from retrieve import metric


def test_tokens_normalise_and_split_cjk():
    assert metric.tokens("  Sốt&amp;Ho  CAO ") == ["sốt", "ho", "cao"]
    assert metric.tokens("服用ＨＣＧ２次") == ["服", "用", "hcg2", "次"]  # NFKC folds full-width forms


def test_chunk_rule_threshold_and_same_doc():
    gold = {"id": 1, "relevant_docs": [7], "relevant_chunks": [{"doc_id": 7, "chunk_text": "a b c d e f g h i j"}]}
    long_ok = {"relevant_docs": [7], "relevant_chunks": [{"doc_id": 7, "chunk_text": "x y a b c d z z"}]}  # 4/10 = 0.4
    short = {"relevant_docs": [7], "relevant_chunks": [{"doc_id": 7, "chunk_text": "a b c"}]}             # 3/10
    wrong_doc = {"relevant_docs": [7, 8], "relevant_chunks": [{"doc_id": 8, "chunk_text": "a b c d e f g h i j"}]}
    assert metric.score_query(long_ok, gold)["chunk_f2"] == pytest.approx(1.0)
    assert metric.score_query(short, gold)["chunk_f2"] == 0
    s = metric.score_query(wrong_doc, gold)
    assert s["chunk_f2"] == 0 and s["doc_p"] == 0.5 and s["doc_r"] == 1.0


def test_doc_f2_formula_and_macro():
    gold = [{"id": 1, "relevant_docs": [1, 2], "relevant_chunks": []}, {"id": 2, "relevant_docs": [3], "relevant_chunks": []}]
    pred = [{"id": 1, "relevant_docs": [1], "relevant_chunks": []}]
    r = metric.evaluate(pred, gold)
    assert r["doc_f2"] == pytest.approx(metric.f2(1.0, 0.5) / 2)  # query 2 missing -> 0
    assert metric.f2(1, 0.5) == pytest.approx(5 * 0.5 / (4 + 0.5))
