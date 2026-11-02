import json
import os

import pandas as pd
import pytest

from conftest import FakeEncoder
from retrieve import judge, llm, plan, select
from retrieve.config import Config, for_mode
from test_v0 import chunks_of


class FakeLLM:
    """Plans: zh_q = the Chinese text the test planted; judge: rel 3 for the doc whose title is in the query map."""

    def __init__(self, zh_for=None, good_titles=()):
        self.zh_for, self.good, self.calls = zh_for or {}, set(good_titles), 0

    def map_json(self, prompts):
        out = []
        for m in prompts:
            self.calls += 1
            u = m[-1]["content"]
            if "Trả về JSON đúng các khoá" in u:
                q = u.split("\n")[1]
                out.append({"type": "other", "vi_kw": q, "zh_q": self.zh_for.get(q, ""), "zh_kw": "", "en_kw": "",
                            "entities": [], "subqs": []})
            else:
                title = u.split("Tài liệu: ")[1].split("\n")[0]
                out.append({"rel": 3 if title in self.good else 0, "evidence": [0]})
        return out


def test_extract_json_tolerates_fences_and_think():
    assert llm.extract_json('<think>x</think>```json\n{"a": 1}\n```') == {"a": 1}
    assert llm.extract_json("Kết quả: {\"rel\": 2} hết") == {"rel": 2}
    assert llm.extract_json("không có json") is None


def test_plan_normalize_and_fallback():
    p = plan.normalize("đau đầu", {"vi_kw": " đau  đầu ", "zh_q": "头痛怎么办", "zh_kw": "头痛", "en_kw": "headache",
                                   "entities": [{"vi": "Panadol", "en": "paracetamol", "zh": "扑热息痛"}], "subqs": ["a", "b"]})
    assert p["ok"] and p["vi_kw"] == "đau đầu" and "扑热息痛" in p["zh_kw"] and "paracetamol" in p["en_kw"]
    assert p["subqs"] == []  # short question: sub-questions ignored
    bad = plan.normalize("đau đầu", None)
    assert not bad["ok"] and bad["zh_q"] == ""


def test_f2_cut_math():
    assert select.f2_cut([0.98, 0.05, 0.03], 0.2, 5) == 1            # confident query -> 1 doc
    assert select.f2_cut([0.5, 0.45, 0.4, 0.35, 0.3], 0.2, 5) == 5   # ambiguous query -> many
    assert select.f2_cut([0.95, 0.8, 0.4, 0.3, 0.1, 0.05], 0.2, 6) == 4  # the reference artifact's example
    assert select.f2_cut([], 0.5, 5) == 0


def test_judge_parse():
    assert judge.parse({"rel": "2", "evidence": ["c1", 0, 9]}, 3) == (2, [1, 0])
    assert judge.parse({"rel": 7}, 3) == (None, [])
    assert judge.parse(None, 3) == (None, [])


def test_three_modes_end_to_end(ingest_root, tmp_path):
    from retrieve.__main__ import run_modes
    ch = chunks_of(ingest_root)
    zh = ch[ch.lang == "zh"].iloc[0]
    vi = ch[ch.lang == "vi"].iloc[0]
    q_zh_target = "câu hỏi tiếng Việt không có chữ chung với tài liệu"
    queries = pd.DataFrame({"id": [1, 2], "query": [vi.text[:300], q_zh_target]})
    qp = str(tmp_path / "q.parquet")
    queries.to_parquet(qp)
    fake = FakeLLM(zh_for={q_zh_target: zh.text[:200]}, good_titles={zh.title})
    res = run_modes(qp, ingest_root, str(tmp_path / "run"), FakeEncoder(), fake, ["v0", "v1", "v2"],
                    Config(k_dense=3, k_sparse=3))  # tiny corpus: short lists keep RRF selective
    assert set(res) == {"v0", "v1", "v2"}
    for m in res:
        data = json.load(open(os.path.join(tmp_path, "run", m, "submission.json"), encoding="utf-8"))
        assert [d["id"] for d in data] == [1, 2]
    v1 = json.load(open(os.path.join(tmp_path, "run", "v1", "submission.json"), encoding="utf-8"))
    assert zh.doc_id in v1[1]["relevant_docs"]  # only the translated zh variant can find it
    assert json.load(open(os.path.join(tmp_path, "run", "v2", "config.json")))["use_judge"] is True


def test_modes_without_llm_refuse_plan_modes(ingest_root, tmp_path):
    from retrieve.__main__ import run_modes
    qp = str(tmp_path / "q.parquet")
    pd.DataFrame({"id": [1], "query": ["x"]}).to_parquet(qp)
    with pytest.raises(SystemExit):
        run_modes(qp, ingest_root, str(tmp_path / "run"), FakeEncoder(), None, ["v1"])


def test_for_mode_overrides():
    assert for_mode("v0").select == "topk" and for_mode("v1").select == "learned" and for_mode("v2").use_judge
    assert for_mode("v1", Config(max_docs=3)).max_docs == 3


def test_plan_keywords_cleaned_and_deduped():
    p = plan.normalize("q", {"zh_q": "x", "zh_kw": "龋齿 牙龈出血", "en_kw": '"esophagitis" AND (gastritis OR "ulcer")',
                             "entities": [{"vi": "a", "en": "gastritis", "zh": "龋齿"}]})
    assert p["en_kw"] == "esophagitis gastritis ulcer" and p["zh_kw"] == "龋齿 牙龈出血"


def test_cpu_dense_rerank_adds_dense_lists_over_candidates(ingest_root, tmp_path):
    import pickle
    from retrieve.__main__ import run_modes
    ch = chunks_of(ingest_root)
    vi = ch[ch.lang == "vi"].iloc[0]
    queries = pd.DataFrame({"id": [1], "query": [vi.text[:300]]})
    qp = str(tmp_path / "q.parquet")
    queries.to_parquet(qp)
    run = str(tmp_path / "run")
    run_modes(qp, ingest_root, run, FakeEncoder(), None, ["v0"],
              Config(dense_langs=[], k_sparse=5, rerank_docs=4), chunk_encoder=FakeEncoder())
    lists = pickle.load(open(os.path.join(run, "lists.pkl"), "rb"))
    dense = [r for r in lists if r["channel"] == "dense" and r["ids"]]
    assert dense and all(len(r["ids"]) <= 4 and r["scores"] == sorted(r["scores"], reverse=True) for r in dense)
    assert dense[0]["ids"][0] == vi.chunk_id  # its own text is the closest candidate
    assert os.path.exists(os.path.join(run, "chunk_vec.sqlite"))


def test_plans_file_is_reused_without_llm(ingest_root, tmp_path):
    from retrieve.__main__ import run_modes
    ch = chunks_of(ingest_root)
    queries = pd.DataFrame({"id": [1], "query": [ch.iloc[0].text[:200]]})
    qp = str(tmp_path / "q.parquet")
    queries.to_parquet(qp)
    pp = str(tmp_path / "plans.json")
    json.dump({"1": {"ok": True, "vi_kw": "a", "zh_q": "", "zh_kw": "", "en_kw": "", "subqs": []}}, open(pp, "w"))
    res = run_modes(qp, ingest_root, str(tmp_path / "run"), FakeEncoder(), None, ["v1"], Config(k_dense=3, k_sparse=3),
                    plans_path=pp)
    assert res["v1"]["queries"] == 1
    json.dump({"2": {"ok": True}}, open(pp, "w"))  # no plan for query 1 -> refuse instead of silently running v0
    with pytest.raises(SystemExit):
        run_modes(qp, ingest_root, str(tmp_path / "run2"), FakeEncoder(), None, ["v1"], plans_path=pp)


def test_lang_weight_drops_and_demotes_and_chunk_docs_limits():
    from retrieve import search, select
    lists = [{"query_id": 1, "variant": "orig", "channel": "sparse_vi", "ids": ["1:0", "2:0", "3:0"], "langs": ["en", "zh", "vi"]}]
    cfg = Config(select="topk", k_docs=3, cand_docs=3, lang_weight={"en": 0.0, "zh": 0.1}, chunk_docs=1)
    cand = search.fuse(lists, 60, cfg=cfg)
    assert list(cand.doc_id) == [3, 2]  # en dropped, zh pushed below vi
    sel = select.select(cand, cfg)
    assert sel[1]["docs"] == [3, 2] and list(sel[1]["chunks"]) == [3]


def test_listwise_judge_reorders_by_grade(ingest_root, tmp_path):
    from retrieve.__main__ import run_listwise
    ch = chunks_of(ingest_root)
    docs = ch.drop_duplicates("doc_id").head(4)
    cand = pd.DataFrame({"query_id": 1, "chunk_id": docs.chunk_id.tolist(), "doc_id": docs.doc_id.tolist(), "lang": "vi",
                         "score": [0.04, 0.03, 0.02, 0.01], "n_lists": 1, "lists_total": 1})

    class Grader:
        def map_json(self, prompts):
            return [{"d0": 0, "d1": 3} for _ in prompts]  # batches of 2: first doc irrelevant, second answers

    out = run_listwise(pd.DataFrame({"id": [1], "query": ["q"]}), cand, ingest_root, Grader(),
                       Config(deep_judge=4, judge_batch=2), {})
    order = select.doc_table(out).doc_id.tolist()
    d = docs.doc_id.tolist()
    assert order == [d[1], d[3], d[0], d[2]]  # graded 3 first (fused order breaks the tie), graded 0 last


def test_list_parse_keyed_and_partial():
    from retrieve import judge
    assert judge.list_parse({"d0": 3, "d1": "1", "d2": 0}, 3) == [3, 1, 0]
    assert judge.list_parse({"d0": 3, "d1": 2}, 3) == [3, 2, None]  # one missing: keep the others
    assert judge.list_parse({"d0": 3}, 3) is None and judge.list_parse({"d0": 9, "d1": 2, "d2": 7}, 3) is None
    assert '"d2": ?' in judge.list_prompt("q", [("t", "x")] * 3)[1]["content"]


def test_doc_agg_prefers_docs_with_several_matching_chunks():
    from retrieve import search
    lists = [{"query_id": 1, "variant": "orig", "channel": "dense", "ids": ["1:0", "2:0", "2:1", "3:0"], "langs": ["vi"] * 4},
             {"query_id": 1, "variant": "orig", "channel": "sparse_vi", "ids": ["1:0", "2:1", "2:0"], "langs": ["vi"] * 3}]
    top = lambda n: search.fuse(lists, 60, cfg=Config(doc_agg=n, cand_docs=3)).drop_duplicates("doc_id").doc_id.tolist()
    assert top(1)[0] == 1 and top(2)[0] == 2  # doc 2 has two strong chunks


def test_variant_weight_overrides_kind_factor():
    from retrieve.search import list_weight
    cfg = Config(w_trans=0.7, variant_weight={"hyde_vi": 0.3})
    assert list_weight({"channel": "dense", "variant": "hyde_vi"}, cfg) == 0.3
    assert list_weight({"channel": "dense", "variant": "zh_q"}, cfg) == 0.7
    assert list_weight({"channel": "dense", "variant": "orig"}, cfg) == 1.0
