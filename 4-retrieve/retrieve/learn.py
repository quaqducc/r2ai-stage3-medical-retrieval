"""Learned doc scorer: per (query, doc) features from the ranked lists (absolute dense cosine, lexical score per
language, ranks, agreement, doc language) -> logistic regression -> p(doc relevant). Fixes rank-only RRF, which lets
a language with more correlated lists (vi: orig + vi_kw + sub-questions) outvote a strong single en/zh hit.
The model is a small JSON (feature names, scaler, coefficients) so retrieval needs no sklearn at run time."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

LANGS = ("vi", "zh", "en")
FEATURES = ["d_max", "d_orig", "d_zhq", "d_subq", "d_gap", "s_vi", "s_zh", "s_en", "s_vi_rel", "s_zh_rel", "s_en_rel",
            "log_best_rank", "n_lists", "lang_vi", "lang_zh", "lang_en", "rrf"]


def features(lists: list[dict], keep=lambda r: True, depth: int = 50, rrf_k: int = 60) -> pd.DataFrame:
    """One row per (query_id, doc_id) among the top `depth` of any kept list."""
    acc = {}
    top_dense = {}
    for rec in lists:
        if not keep(rec) or not rec["ids"]:
            continue
        q, ch, v = rec["query_id"], rec["channel"], rec["variant"]
        scores = rec.get("scores") or [0.0] * len(rec["ids"])
        top = scores[0] if scores else 0.0
        if ch == "dense":
            top_dense[q] = max(top_dense.get(q, -1.0), top)
        for r, (c, sc, lang) in enumerate(zip(rec["ids"][:depth], scores[:depth], (rec.get("langs") or [None] * depth)[:depth]), 1):
            d = int(str(c).split(":")[0])
            f = acc.setdefault((q, d), {"query_id": q, "doc_id": d, "lang": lang, "d_max": 0.0, "d_orig": 0.0, "d_zhq": 0.0,
                                        "d_subq": 0.0, "s_vi": 0.0, "s_zh": 0.0, "s_en": 0.0, "s_vi_rel": 0.0,
                                        "s_zh_rel": 0.0, "s_en_rel": 0.0, "best_rank": 10**6, "n_lists": 0, "rrf": 0.0})
            f["best_rank"] = min(f["best_rank"], r)
            f["n_lists"] += 1
            f["rrf"] += 1.0 / (rrf_k + r)
            if ch == "dense":
                f["d_max"] = max(f["d_max"], sc)
                key = "d_orig" if v == "orig" else "d_zhq" if v == "zh_q" else "d_subq" if v.startswith("subq") else None
                if key:
                    f[key] = max(f[key], sc)
            else:
                lang_ = ch.split("_", 1)[1]
                f[f"s_{lang_}"] = max(f[f"s_{lang_}"], sc)
                f[f"s_{lang_}_rel"] = max(f[f"s_{lang_}_rel"], sc / top if top else 0.0)
    df = pd.DataFrame(list(acc.values()))
    if df.empty:
        return pd.DataFrame(columns=["query_id", "doc_id"] + FEATURES)
    df["d_gap"] = df.d_max - df.query_id.map(top_dense).fillna(0.0)
    df["log_best_rank"] = np.log(df.best_rank.astype(float))
    for l in LANGS:
        df[f"lang_{l}"] = (df["lang"] == l).astype(float)
    return df


class Ranker:
    def __init__(self, mean, scale, coef, intercept, features=FEATURES):
        self.mean, self.scale = np.asarray(mean), np.asarray(scale)
        self.coef, self.intercept, self.features = np.asarray(coef), float(intercept), list(features)

    def p(self, df: pd.DataFrame) -> np.ndarray:
        x = (df[self.features].to_numpy(float) - self.mean) / self.scale
        z = np.clip(x @ self.coef + self.intercept, -50, 50)
        return 1 / (1 + np.exp(-z))

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"features": self.features, "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                       "coef": self.coef.tolist(), "intercept": self.intercept}, f, indent=1)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        return cls(m["mean"], m["scale"], m["coef"], m["intercept"], m["features"])

    @classmethod
    def fit(cls, df: pd.DataFrame, y: np.ndarray, C: float = 1.0) -> "Ranker":
        from sklearn.linear_model import LogisticRegression
        x = df[FEATURES].to_numpy(float)
        mean, scale = x.mean(0), x.std(0)
        scale[scale == 0] = 1.0
        lr = LogisticRegression(C=C, max_iter=2000).fit((x - mean) / scale, y)
        return cls(mean, scale, lr.coef_[0], lr.intercept_[0])


def doc_p(lists: list[dict], keep, ranker: Ranker) -> pd.DataFrame:
    """(query_id, doc_id, p, rank) with rank by p within each query."""
    df = features(lists, keep)
    if df.empty:
        return df.assign(p=[], rank=[])
    df["p"] = ranker.p(df)
    df = df.sort_values(["query_id", "p"], ascending=[True, False], kind="stable")
    df["rank"] = df.groupby("query_id").cumcount() + 1
    return df
