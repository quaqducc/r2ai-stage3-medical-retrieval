import numpy as np
import pandas as pd

from retrieve import learn, select
from retrieve.config import Config

LISTS = [  # doc 2 is weak on rank (vi lists agree on doc 1) but has the best dense cosine and a strong en hit
    {"query_id": 1, "variant": "orig", "channel": "dense", "ids": ["1:0", "2:0"], "langs": ["vi", "en"], "scores": [0.50, 0.80]},
    {"query_id": 1, "variant": "orig", "channel": "sparse_vi", "ids": ["1:0", "1:1"], "langs": ["vi", "vi"], "scores": [9.0, 8.0]},
    {"query_id": 1, "variant": "en_kw", "channel": "sparse_en", "ids": ["2:0"], "langs": ["en"], "scores": [5.0]},
]


def test_features_and_ranker_roundtrip(tmp_path):
    df = learn.features(LISTS)
    assert set(df.doc_id) == {1, 2} and list(df.columns).count("d_gap") == 1
    assert df.set_index("doc_id").loc[2, "d_max"] == 0.80 and df.set_index("doc_id").loc[1, "n_lists"] == 3
    r = learn.Ranker(np.zeros(len(learn.FEATURES)), np.ones(len(learn.FEATURES)),
                     [1.0 if f == "d_max" else 0.0 for f in learn.FEATURES], 0.0)
    r.save(tmp_path / "m.json")
    dp = learn.doc_p(LISTS, lambda rec: True, learn.Ranker.load(tmp_path / "m.json"))
    assert dp.iloc[0].doc_id == 2 and dp.iloc[0]["rank"] == 1


def test_select_learned_follows_ranker_p_not_rrf():
    cand = pd.DataFrame([{"query_id": 1, "chunk_id": "1:0", "doc_id": 1, "lang": "vi", "score": 0.9, "n_lists": 2, "lists_total": 3},
                         {"query_id": 1, "chunk_id": "2:0", "doc_id": 2, "lang": "en", "score": 0.1, "n_lists": 2, "lists_total": 3}])
    learned = pd.DataFrame({"query_id": [1, 1], "doc_id": [1, 2], "p": [0.05, 0.9]})
    sel = select.select(cand, Config(select="learned", f2_missing=0.1), learned=learned)
    assert sel[1]["docs"] == [2] and sel[1]["chunks"] == {2: ["2:0"]}
