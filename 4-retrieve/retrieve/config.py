"""Every knob of a run in one place; each mode's run stores the config it used (runs/<id>/<mode>/config.json)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class Config:
    tag: str = "bge-m3"                                   # embedding model tag in <ingest>/rag/{dense,sparse_index}/<tag>
    dense_langs: list = field(default_factory=lambda: ["vi", "zh", "en"])
    sparse_langs: list = field(default_factory=lambda: ["vi", "zh", "en"])  # zh/en lexical needs translated variants
    k_dense: int = 200                                     # per language and variant
    k_sparse: int = 200
    rrf_k: int = 60
    cand_docs: int = 50                                    # candidates kept per query after fusion
    cand_chunks: int = 3                                   # candidate chunks kept per doc
    doc_agg: int = 1                                       # doc score = sum of its n best fused chunk scores (1 = max)
    rerank_docs: int = 0                                   # >0 (BM25 index, no dense): bge-m3 on CPU over this many fused docs
    w_dense: float = 1.0                                   # weighted RRF: weight of a list = w_<channel kind> * w_<variant kind>
    w_sparse: float = 1.0
    w_trans: float = 1.0                                   # translated / keyword variants (vi_kw, zh_q, zh_kw, en_kw)
    w_subq: float = 0.5
    variant_weight: dict = field(default_factory=dict)    # per-variant list weight, overrides w_trans / w_subq
    # --- which ranked lists a mode fuses
    use_plan: bool = False                                 # v1+: translated / keyword variants
    use_subqs: bool = False                                # v2: sub-question variants
    # --- doc selection
    select: str = "topk"                                   # "topk" (v0) | "f2" (label-free p) | "learned" (ranker p)
    model_path: str = ""                                   # learned: ranker JSON; a bare name resolves in 4-retrieve/models
    k_docs: int = 2                                        # topk: docs per query
    max_docs: int = 5                                      # f2: cap
    f2_missing: float = 0.5                                # f2: expected relevant docs not retrieved (m in E|G| = sum p + m)
    p_bias: float = 1.5                                    # f2: p = sigmoid(p_bias - p_rank*(rank-1) + p_lists*lists_frac)
    p_rank: float = 1.0
    p_lists: float = 2.0
    # --- chunk selection
    k_chunks: int = 1                                      # topk: chunks per doc
    chunk_docs: int = 0                                    # >0: chunks only for the first N submitted docs
    lang_weight: dict = field(default_factory=dict)        # fused-score factor per doc language; 0 drops it
    chunk_tau: float = 0.8                                 # f2: keep chunks scoring >= tau * best chunk of the doc
    k_chunks_max: int = 3
    merge_adjacent: bool = True
    # --- judge (v2)
    use_judge: bool = False
    judge_low: float = 0.15                                # judge docs with p in [low, high] among the top judge_max_docs
    judge_high: float = 0.95                               # (high enough that a shaky rank-1 doc is judged too)
    judge_max_docs: int = 5
    judge_chunks: int = 3                                  # chunks shown per judged doc (CPU LLM: 1)
    judge_chars: int = 3000                                # chars per chunk in the judge prompt (CPU LLM: ~1500)
    deep_judge: int = 0                                    # deep mode: LLM grades this many fused docs per query (listwise)
    judge_batch: int = 10                                  # docs per listwise call
    judge_rel_p: list = field(default_factory=lambda: [0.03, 0.25, 0.7, 0.95])  # rel 0..3 -> p

    # --- guards: fail loudly instead of writing a silently bad submission
    allow_degraded: bool = False
    min_plan_ok: float = 0.8                               # share of queries with a usable LLM plan (plan modes)
    min_judge_ok: float = 0.5                              # share of judged docs with a parseable verdict
    max_empty_queries: float = 0.01                        # share of queries allowed to end with no doc

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=1)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f))


MODES = {
    "v0": dict(),                                                            # orig query lists, fixed top-k
    "v1": dict(use_plan=True, select="learned", model_path="ranker_v1.json"),   # + translations, ranker p, F2 cut
    "v2": dict(use_plan=True, use_subqs=True, select="learned", model_path="ranker_v2.json",
               use_judge=True),                                                 # + sub-questions, LLM judge (grey zone)
    # leaderboard probes (21/10): ~45 gold docs/query, no PubMed, few zh -> deep weighted-RRF top-k, no silver ranker
    # depth (k_docs = cand_docs), chunk_docs and lang_weight come from the base config: configs/deep_*.json
    "deep": dict(use_plan=True, use_subqs=True, select="topk"),
}


def for_mode(mode: str, base: Config | None = None) -> Config:
    c = Config(**asdict(base)) if base else Config()
    for k, v in MODES[mode].items():
        setattr(c, k, v)
    return c
