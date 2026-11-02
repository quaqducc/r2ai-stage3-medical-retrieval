"""Submission variants from one run's lists.pkl (no re-retrieval): each changes one thing (depth, chunk count, which
lists are fused), so leaderboard deltas are attributable. Usage:
python bench/variants.py <run_dir with lists.pkl> <ingest> <queries.parquet> <out_dir> name[,name...]"""
import os
import pickle
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from retrieve import search, select, submit  # noqa: E402
from retrieve.config import Config  # noqa: E402

ALL = (lambda r: True)
ORIG = (lambda r: r["variant"] == "orig")
ORIGZHQ = (lambda r: r["variant"] in ("orig", "zh_q"))
PRF = (lambda r: r["variant"] in ("orig", "zh_q", "orig_prf", "zh_q_prf"))
BASE = dict(select="topk", k_chunks=1, cand_chunks=3, lang_weight={"en": 0.0, "zh": 0.5}, allow_degraded=True)
B = dict(k_docs=150, cand_docs=150, k_chunks=3, chunk_docs=150, w_sparse=0.25, lang_weight={"en": 0.0, "zh": 0.4},
         chunk_w_sparse=0.25, chunk_global=120)  # best1 (25/10, .3321) without its chunk_keep
C9 = dict(B, lang_weight={"en": 0.0, "zh": 0.3}, chunk_global=130, chunk_keep=PRF)  # round 9 base (27/10)
C10 = dict(C9, w_trans=0.7, chunk_global=135, lang_weight={"en": 0.0, "zh": 0.25})  # round 10 base (28/10)
C11 = dict(C10, lang_weight={"en": 0.0, "zh": 0.3}, rrf_k=80)  # round 11 base (29/10)
C12 = dict(C11, lang_weight={"en": 0.0, "zh": 0.35})  # round 11 best (29/10, .3471): base for the e5 runs
HV = ("hyde_vi", "hyde_zh", "title")
with_v = (lambda *vs: (lambda r: r["variant"] in ("orig", "zh_q", "orig_prf", "zh_q_prf") + vs))
HYDE = with_v(*HV)
E5K = with_v("orig_e5", "zh_q_e5")
DOCL = with_v("orig_head", "zh_q_head", "orig_mean", "zh_q_mean")
dw = (lambda w: {"orig_head": w, "orig_mean": w, "zh_q_head": round(w * 2 / 3, 3), "zh_q_mean": round(w * 2 / 3, 3)})
F1 = dict(C12, variant_weight=dw(1.0), chunk_keep=PRF)  # 31/10 best f_w1 (.3536)
HH = {"orig_head": 1.25, "zh_q_head": 0.833, "orig_mean": 0.75, "zh_q_mean": 0.5}  # g_headheavy (.3539)
PRFW = {"orig_prf_head": 1.0, "zh_q_prf_head": 0.667, "orig_prf_mean": 1.0, "zh_q_prf_mean": 0.667}
DOCLP = with_v("orig_head", "zh_q_head", "orig_mean", "zh_q_mean", *PRFW)
PRF_ZHKW = (lambda r: r["variant"] in ("orig", "zh_q", "orig_prf", "zh_q_prf", "zh_kw"))
VARIANTS = {  # name -> (keep, config overrides); reference = deep_bge_k100: all lists, k 100, chunks for 10 docs
    "hy_k200_c20": (ALL, dict(k_docs=200, cand_docs=200, chunk_docs=20)),
    "hy_k100_c40": (ALL, dict(k_docs=100, cand_docs=100, chunk_docs=40)),
    "hy_dense_k100": (lambda r: r["channel"] == "dense", dict(k_docs=100, cand_docs=100, chunk_docs=10)),
    "hy_orig_k100": (lambda r: r["variant"] == "orig", dict(k_docs=100, cand_docs=100, chunk_docs=10)),
    # round 2 (22/10): round 1 -> more chunks >> depth > 100; LLM rewrites hurt docs; dense-only worse than hybrid
    "hy_orig_k150_c40": (ORIG, dict(k_docs=150, cand_docs=150, chunk_docs=40)),
    "hy_orig_k150_c80_t1500": (ORIG, dict(k_docs=150, cand_docs=150, chunk_docs=80, trim=1500)),
    "hy_orig_k150_c80": (ORIG, dict(k_docs=150, cand_docs=150, chunk_docs=80)),
    # round 2 result: k150 c40 .2364, c80 trimmed to 1500 chars .1933 (never trim: chunk P .20 -> .08), c80 .2605
    "hy_orig_k150_c150": (ORIG, dict(k_docs=150, cand_docs=150, chunk_docs=150)),
    # round 3: from the vi+zh collect (subs/deep_bge_vizh_k100): does adding zh docs help, at which weight
    "vz_orig_k150_c80_zh50": (ORIG, dict(k_docs=150, cand_docs=150, chunk_docs=80, lang_weight={"en": 0.0, "zh": 0.5})),
    # orig-only keeps zh near 0 (2.4% of docs at zh 0.5, 0% at 0.25): zh docs come from the translated zh_q query
    "vz_origzhq_k150_c80": (lambda r: r["variant"] in ("orig", "zh_q"),
                            dict(k_docs=150, cand_docs=150, chunk_docs=80, lang_weight={"en": 0.0, "zh": 0.5})),
    "vz_origzhq_k150_c80_zh20": (lambda r: r["variant"] in ("orig", "zh_q"),
                                 dict(k_docs=150, cand_docs=150, chunk_docs=80, lang_weight={"en": 0.0, "zh": 0.2})),
    # round 4: chunk gold ~58/query > doc gold ~45 -> several gold chunks per doc; depth, fusion weights
    "hy_orig_k150_2x75": (ORIG, dict(k_docs=150, cand_docs=150, k_chunks=2, chunk_docs=75)),
    "hy_orig_k200_c80": (ORIG, dict(k_docs=200, cand_docs=200, chunk_docs=80)),
    "hy_orig_k150_c80_sp2": (ORIG, dict(k_docs=150, cand_docs=150, chunk_docs=80, w_sparse=2.0)),
    "hy_orig_k150_c80_sp05": (ORIG, dict(k_docs=150, cand_docs=150, chunk_docs=80, w_sparse=0.5)),
    "hy_orig_k150_c80_rrf20": (ORIG, dict(k_docs=150, cand_docs=150, chunk_docs=80, rrf_k=20)),
    # round 4 result (23/10): 2x75 .2739, k200 .2610, sparse x2 .2203, sparse x0.5 .2864 (docs F2 .325), rrf20 .2593
    "combo_sp05_2x75": (ORIG, dict(k_docs=150, cand_docs=150, k_chunks=2, chunk_docs=75, w_sparse=0.5)),
    "combo_sp025_2x75": (ORIG, dict(k_docs=150, cand_docs=150, k_chunks=2, chunk_docs=75, w_sparse=0.25)),
    "combo_sp05_2x90": (ORIG, dict(k_docs=150, cand_docs=150, k_chunks=2, chunk_docs=90, w_sparse=0.5)),
    # zh result: orig + zh_q at zh 0.5 (28% zh docs) .2706 vs .2605 vi-only; zh 0.2 .2635 -> combos on the vi+zh lists
    "vzc_sp05_2x75": (ORIGZHQ, dict(k_docs=150, cand_docs=150, k_chunks=2, chunk_docs=75, w_sparse=0.5)),
    "vzc_sp025_2x75": (ORIGZHQ, dict(k_docs=150, cand_docs=150, k_chunks=2, chunk_docs=75, w_sparse=0.25)),
    "vzc_sp05_2x90": (ORIGZHQ, dict(k_docs=150, cand_docs=150, k_chunks=2, chunk_docs=90, w_sparse=0.5)),
    # round 5 result: vzc_sp05_2x75 .3130, vzc_sp025_2x75 .3212 (best), vzc_sp05_2x90 .3192 -> less sparse, more chunks
    "vzc_sp025_2x90": (ORIGZHQ, dict(k_docs=150, cand_docs=150, k_chunks=2, chunk_docs=90, w_sparse=0.25)),
    "vzc_sp01_2x90": (ORIGZHQ, dict(k_docs=150, cand_docs=150, k_chunks=2, chunk_docs=90, w_sparse=0.1)),
    "vzc_sp0_2x90": (ORIGZHQ, dict(k_docs=150, cand_docs=150, k_chunks=2, chunk_docs=90, w_sparse=0.0)),
    "vzc_sp025_3x60": (ORIGZHQ, dict(k_docs=150, cand_docs=150, k_chunks=3, chunk_docs=60, w_sparse=0.25)),
    "vzc_sp025_2x90_zh70": (ORIGZHQ, dict(k_docs=150, cand_docs=150, k_chunks=2, chunk_docs=90, w_sparse=0.25,
                                          lang_weight={"en": 0.0, "zh": 0.7})),
    "vzc_sp025_2x90_zh100": (ORIGZHQ, dict(k_docs=150, cand_docs=150, k_chunks=2, chunk_docs=90, w_sparse=0.25,
                                           lang_weight={"en": 0.0, "zh": 1.0})),
    "vzc_sp025_k200_2x90": (ORIGZHQ, dict(k_docs=200, cand_docs=200, k_chunks=2, chunk_docs=90, w_sparse=0.25)),
    "vzc_sp025_2x90_rrf100": (ORIGZHQ, dict(k_docs=150, cand_docs=150, k_chunks=2, chunk_docs=90, w_sparse=0.25, rrf_k=100)),
    # pipeline (not scoring knobs): doc score from several chunks; dense pseudo-relevance feedback (needs LISTS=lists_prf.pkl)
    "vzc_sp025_2x90_agg2": (ORIGZHQ, dict(k_docs=150, cand_docs=150, k_chunks=2, chunk_docs=90, w_sparse=0.25, doc_agg=2)),
    "vzc_sp025_2x90_agg3": (ORIGZHQ, dict(k_docs=150, cand_docs=150, k_chunks=2, chunk_docs=90, w_sparse=0.25, doc_agg=3)),
    "prf_add_sp025_2x90": (lambda r: r["variant"] in ("orig", "zh_q", "orig_prf", "zh_q_prf"),
                           dict(k_docs=150, cand_docs=150, k_chunks=2, chunk_docs=90, w_sparse=0.25)),
    "prf_swap_sp025_2x90": (lambda r: r["variant"] in ("orig_prf", "zh_q_prf") or
                            (r["channel"] != "dense" and r["variant"] in ("orig", "zh_q")),
                            dict(k_docs=150, cand_docs=150, k_chunks=2, chunk_docs=90, w_sparse=0.25)),
    # round 6 result (24/10): prf_add .3229 (best, docs F2 .358), prf_swap .3197, agg2/3 worse, sp0.1/0 ~ sp0.25,
    # 3x60 .3038, zh 1.0 .2679, k200 ~ k150; chunks: sp0.5 picks better chunks (.298) than sp0.25 (.283), 2x75 > 2x90.
    # -> chunk choice decoupled from doc ranking (chunk_w_sparse) and chunks allocated by global chunk rank (chunk_global)
    "prf_2x75": (PRF, dict(k_docs=150, cand_docs=150, k_chunks=2, chunk_docs=75, w_sparse=0.25)),
    "prf_2x75_csp05": (PRF, dict(k_docs=150, cand_docs=150, k_chunks=2, chunk_docs=75, w_sparse=0.25, chunk_w_sparse=0.5)),
    "prf_2x75_csp1": (PRF, dict(k_docs=150, cand_docs=150, k_chunks=2, chunk_docs=75, w_sparse=0.25, chunk_w_sparse=1.0)),
    "prf_g150_csp05": (PRF, dict(k_docs=150, cand_docs=150, k_chunks=3, chunk_docs=150, w_sparse=0.25,
                                 chunk_w_sparse=0.5, chunk_global=150)),
    "prf_g120_csp05": (PRF, dict(k_docs=150, cand_docs=150, k_chunks=3, chunk_docs=150, w_sparse=0.25,
                                 chunk_w_sparse=0.5, chunk_global=120)),
    "prf_2x60_csp05": (PRF, dict(k_docs=150, cand_docs=150, k_chunks=2, chunk_docs=60, w_sparse=0.25, chunk_w_sparse=0.5)),
    "prf_2x75_csp05_zh40": (PRF, dict(k_docs=150, cand_docs=150, k_chunks=2, chunk_docs=75, w_sparse=0.25,
                                      chunk_w_sparse=0.5, lang_weight={"en": 0.0, "zh": 0.4})),
    "prf_2x75_csp05_zh60": (PRF, dict(k_docs=150, cand_docs=150, k_chunks=2, chunk_docs=75, w_sparse=0.25,
                                      chunk_w_sparse=0.5, lang_weight={"en": 0.0, "zh": 0.6})),
    # same as prf_2x75_csp05, built from other PRF settings (LISTS=lists_prf_a1.pkl: alpha 1.0; lists_prf_n20.pkl: top-20)
    "prfa1_2x75_csp05": (PRF, dict(k_docs=150, cand_docs=150, k_chunks=2, chunk_docs=75, w_sparse=0.25, chunk_w_sparse=0.5)),
    "prfn20_2x75_csp05": (PRF, dict(k_docs=150, cand_docs=150, k_chunks=2, chunk_docs=75, w_sparse=0.25, chunk_w_sparse=0.5)),
    # round 7 result (25/10): g120 .3266 (best; chunks F2 .295), zh .4 .3233 (docs F2 .366), zh .6 .3068, 2x60 .3078,
    # chunk_w_sparse no effect, PRF alpha 1 / top-20 no gain; PRF lists help docs but hurt chunk choice (2x75: .275 vs .290)
    # -> docs from the PRF lists, chunks from the non-PRF lists (chunk_keep)
    "best1_prfdocs_g120_zh40": (PRF, dict(k_docs=150, cand_docs=150, k_chunks=3, chunk_docs=150, w_sparse=0.25,
                                          lang_weight={"en": 0.0, "zh": 0.4}, chunk_w_sparse=0.25, chunk_global=120,
                                          chunk_keep=ORIGZHQ)),
    # round 8 (26/10): one change each around best1 (.3321)
    "b1_g130": (PRF, dict(B, chunk_global=130, chunk_keep=ORIGZHQ)),
    "b1_zh30": (PRF, dict(B, lang_weight={"en": 0.0, "zh": 0.3}, chunk_keep=ORIGZHQ)),
    "b1_cdense": (PRF, dict(B, chunk_keep=lambda r: r["channel"] == "dense" and r["variant"] in ("orig", "zh_q"))),
    "b1_g120_max2": (PRF, dict(B, k_chunks=2, chunk_keep=ORIGZHQ)),
    "b1_g120_max4": (PRF, dict(B, k_chunks=4, cand_chunks=4, chunk_keep=ORIGZHQ)),
    "b1_k130": (PRF, dict(B, k_docs=130, cand_docs=130, chunk_docs=130, chunk_keep=ORIGZHQ)),
    "b1_k170": (PRF, dict(B, k_docs=170, cand_docs=170, chunk_docs=170, chunk_keep=ORIGZHQ)),
    "b1_sp015": (PRF, dict(B, w_sparse=0.15, chunk_keep=ORIGZHQ)),
    "b1_csp05": (PRF, dict(B, chunk_w_sparse=0.5, chunk_keep=ORIGZHQ)),
    "b1_cprf": (PRF, dict(B, chunk_keep=PRF)),
    # round 8 result (26/10): zh .3 .3375 (best, docs F2 .373), chunks from PRF lists .3348, g130 .3346; max2/max4,
    # dense-only chunks, sparse .15, chunk sparse .5, k130/k170 neutral -> combine the three gains, then explore
    "c9_base": (PRF, dict(C9)),
    "c9_zh25": (PRF, dict(C9, lang_weight={"en": 0.0, "zh": 0.25})),
    "c9_zh20": (PRF, dict(C9, lang_weight={"en": 0.0, "zh": 0.2})),
    "c9_g135": (PRF, dict(C9, chunk_global=135)),
    "c9_zhkw": (PRF_ZHKW, dict(C9, chunk_keep=PRF_ZHKW)),
    "c9_tr07": (PRF, dict(C9, w_trans=0.7)),
    "c9_tr13": (PRF, dict(C9, w_trans=1.3)),
    "c9_k160": (PRF, dict(C9, k_docs=160, cand_docs=160, chunk_docs=160)),
    "c9_zh25_g135": (PRF, dict(C9, chunk_global=135, lang_weight={"en": 0.0, "zh": 0.25})),
    "c9_zh30_tr13_g135": (PRF, dict(C9, chunk_global=135, w_trans=1.3)),
    # round 9 result (27/10): w_trans .7 .3445 (best), zh .25+g135 .3441, g135 .3435, zh .25 .3426, base .3425, k160
    # .3424, zh_kw .3415, zh .2 .3409, w_trans 1.3 .3392 -> round 10 around C10 = C9 + w_trans .7 + g135 + zh .25
    "c10_base": (PRF, dict(C10)),
    "c10_tr05": (PRF, dict(C10, w_trans=0.5)),
    "c10_tr04": (PRF, dict(C10, w_trans=0.4)),
    "c10_tr06": (PRF, dict(C10, w_trans=0.6)),
    "c10_rrf40": (PRF, dict(C10, rrf_k=40)),
    "c10_rrf80": (PRF, dict(C10, rrf_k=80)),
    "c10_sp02": (PRF, dict(C10, w_sparse=0.2)),
    "c10_sp035": (PRF, dict(C10, w_sparse=0.35)),
    "c10_zh30": (PRF, dict(C10, lang_weight={"en": 0.0, "zh": 0.3})),
    "c10_tr05_zh30": (PRF, dict(C10, w_trans=0.5, lang_weight={"en": 0.0, "zh": 0.3})),
    # query-generation agent (bench/hyde.py, LISTS=lists_hyde.pkl): LLM-written passages/titles ADDED as dense lists
    "h_vi05": (with_v("hyde_vi"), dict(C10, variant_weight={"hyde_vi": 0.5}, chunk_keep=with_v("hyde_vi"))),
    "h_title05": (with_v("title"), dict(C10, variant_weight={"title": 0.5}, chunk_keep=with_v("title"))),
    "h_all03": (HYDE, dict(C10, variant_weight={v: 0.3 for v in HV}, chunk_keep=HYDE)),
    "h_all05": (HYDE, dict(C10, variant_weight={v: 0.5 for v in HV}, chunk_keep=HYDE)),
    "h_all1": (HYDE, dict(C10, variant_weight={v: 1.0 for v in HV}, chunk_keep=HYDE)),
    # round 10 result (28/10): zh .3 .3458 (best), rrf80 .3450 (docs F2 .379), base .3445; HyDE neutral at .3, hurts
    # above (.5 .3395, 1.0 .3261) -> dropped. Round 11 around C11 = C10 + zh .3 + rrf 80
    "c11_base": (PRF, dict(C11)),
    "c11_rrf100": (PRF, dict(C11, rrf_k=100)),
    "c11_rrf120": (PRF, dict(C11, rrf_k=120)),
    "c11_rrf60": (PRF, dict(C11, rrf_k=60)),
    "c11_zh35": (PRF, dict(C11, lang_weight={"en": 0.0, "zh": 0.35})),
    "c11_tr08": (PRF, dict(C11, w_trans=0.8)),
    "c11_tr06": (PRF, dict(C11, w_trans=0.6)),
    "c11_k140": (PRF, dict(C11, k_docs=140, cand_docs=140, chunk_docs=140)),
    "c11_sp03": (PRF, dict(C11, w_sparse=0.3)),
    "c11_g140_lzma": (PRF, dict(C11, chunk_global=140)),
    # second dense model (bench/add_e5.py, LISTS=lists_e5.pkl): e5 lists added to the C11 lists
    "e5_w05": (E5K, dict(C12, variant_weight={"orig_e5": 0.5, "zh_q_e5": 0.35}, chunk_keep=E5K)),
    "e5_w1": (E5K, dict(C12, variant_weight={"orig_e5": 1.0, "zh_q_e5": 0.7}, chunk_keep=E5K)),
    "e5_w1_docsonly": (E5K, dict(C12, variant_weight={"orig_e5": 1.0, "zh_q_e5": 0.7}, chunk_keep=PRF)),
    "e5_w15": (E5K, dict(C12, variant_weight={"orig_e5": 1.5, "zh_q_e5": 1.0}, chunk_keep=E5K)),
    # e5 result (29/10): w .5 .3472 (= base .3471), w 1 .3368, docs-only .3405, w 1.5 .3221 -> gold follows bge-m3.
    # round 12 (30/10, last public day): doc-level bge-m3 lists (bench/doc_dense.py, LISTS=lists_doc.pkl), docs only
    "d12_head05": (with_v("orig_head", "zh_q_head"), dict(C12, variant_weight={"orig_head": 0.5, "zh_q_head": 0.35}, chunk_keep=PRF)),
    "d12_head1": (with_v("orig_head", "zh_q_head"), dict(C12, variant_weight={"orig_head": 1.0, "zh_q_head": 0.7}, chunk_keep=PRF)),
    "d12_mean05": (with_v("orig_mean", "zh_q_mean"), dict(C12, variant_weight={"orig_mean": 0.5, "zh_q_mean": 0.35}, chunk_keep=PRF)),
    "d12_mean1": (with_v("orig_mean", "zh_q_mean"), dict(C12, variant_weight={"orig_mean": 1.0, "zh_q_mean": 0.7}, chunk_keep=PRF)),
    "d12_both05": (with_v("orig_head", "zh_q_head", "orig_mean", "zh_q_mean"),
                   dict(C12, variant_weight={"orig_head": 0.5, "zh_q_head": 0.35, "orig_mean": 0.5, "zh_q_mean": 0.35},
                        chunk_keep=PRF)),
    "d12_prf1": (PRF, dict(C12, variant_weight={"orig_prf": 1.0, "zh_q_prf": 0.7})),
    "d12_rrf100": (PRF, dict(C12, rrf_k=100)),
    "d12_zh40": (PRF, dict(C12, lang_weight={"en": 0.0, "zh": 0.4})),
    # long chunk_text (LCS/|gold| >= .4 only bounds the gold side): one text per doc / whole docs (team baconvit: chunk P .96)
    "L_merge": (PRF, dict(C12, chunk_mode="merge")),
    "L_doc40": (PRF, dict(C12, chunk_mode="doc", doc_n=40)),
    "L_doc60": (PRF, dict(C12, chunk_mode="doc", doc_n=60)),
    "L_doc68": (PRF, dict(C12, chunk_mode="doc", doc_n=68)),
    # round 12 result (30/10): head .5 .3495, mean .5 .3485, both .5 .3504 (best); long chunk_text hurts (merge .339,
    # whole docs .28-.31: the scorer matches one submitted chunk to one gold chunk) -> last slot: both lists at .75
    "d12_both075": (with_v("orig_head", "zh_q_head", "orig_mean", "zh_q_mean"),
                    dict(C12, variant_weight={"orig_head": 0.75, "zh_q_head": 0.5, "orig_mean": 0.75, "zh_q_mean": 0.5},
                         chunk_keep=PRF)),
    # 31/10 (public extended to 16:59Z, 10 last slots, one at a time): around d12_both075 (.3526)
    "f_w1": (DOCL, dict(C12, variant_weight=dw(1.0), chunk_keep=PRF)),
    "f_w125": (DOCL, dict(C12, variant_weight=dw(1.25), chunk_keep=PRF)),
    "f_w15": (DOCL, dict(C12, variant_weight=dw(1.5), chunk_keep=PRF)),
    "f_w075_head1": (DOCL, dict(C12, variant_weight={**dw(0.75), "orig_head": 1.0, "zh_q_head": 0.7}, chunk_keep=PRF)),
    "f_w075_mean1": (DOCL, dict(C12, variant_weight={**dw(0.75), "orig_mean": 1.0, "zh_q_mean": 0.7}, chunk_keep=PRF)),
    "f_w075_zh40": (DOCL, dict(C12, variant_weight=dw(0.75), lang_weight={"en": 0.0, "zh": 0.4}, chunk_keep=PRF)),
    "f_w075_zh30": (DOCL, dict(C12, variant_weight=dw(0.75), lang_weight={"en": 0.0, "zh": 0.3}, chunk_keep=PRF)),
    "f_w075_rrf100": (DOCL, dict(C12, variant_weight=dw(0.75), rrf_k=100, chunk_keep=PRF)),
    "f_w075_tr06": (DOCL, dict(C12, variant_weight=dw(0.75), w_trans=0.6, chunk_keep=PRF)),
    "f_w075_k160": (DOCL, dict(C12, variant_weight=dw(0.75), k_docs=160, cand_docs=160, chunk_docs=160, chunk_keep=PRF)),
    # f_w1 .3536, f_w125 .3536 (docs .3844 vs .3848, chunks .3228 vs .3224): weight plateau -> base F1 = C12 + dw(1.0)
    "g_headheavy": (DOCL, dict(F1, variant_weight={"orig_head": 1.25, "zh_q_head": 0.833, "orig_mean": 0.75, "zh_q_mean": 0.5})),
    "g_meanheavy": (DOCL, dict(F1, variant_weight={"orig_head": 0.75, "zh_q_head": 0.5, "orig_mean": 1.25, "zh_q_mean": 0.833})),
    "g_zh40": (DOCL, dict(F1, lang_weight={"en": 0.0, "zh": 0.4})),
    "g_rrf100": (DOCL, dict(F1, rrf_k=100)),
    "g_k160": (DOCL, dict(F1, k_docs=160, cand_docs=160, chunk_docs=160)),
    "g_tr06": (DOCL, dict(F1, w_trans=0.6)),
    "g_g140": (DOCL, dict(F1, chunk_global=140)),
    # g_headheavy .3539 (best). doc-level lists of the PRF query vectors (bench/doc_dense_prf.py, LISTS=lists_docprf.pkl)
    "h_dprf05": (DOCLP, dict(F1, variant_weight={**HH, **{v: w * 0.5 for v, w in PRFW.items()}})),
    "h_dprf1": (DOCLP, dict(F1, variant_weight={**HH, **PRFW})),
    # g_rrf100 .3544 (best, docs F2 .3866) -> I = headheavy weights + rrf 100; then + doc-level PRF lists
    "i_base": (DOCL, dict(F1, rrf_k=100, variant_weight=dict(HH))),
    "i_dprf05": (DOCLP, dict(F1, rrf_k=100, variant_weight={**HH, **{v: w * 0.5 for v, w in PRFW.items()}})),
    "i_dprf1": (DOCLP, dict(F1, rrf_k=100, variant_weight={**HH, **PRFW})),
    "i_rrf120": (DOCL, dict(F1, rrf_k=120, variant_weight=dict(HH))),
    # i_base .3547, i_dprf05 .3570 (best: doc-level PRF lists at .5)
    "i_dprf075": (DOCLP, dict(F1, rrf_k=100, variant_weight={**HH, **{v: w * 0.75 for v, w in PRFW.items()}})),
    "i_dprf15": (DOCLP, dict(F1, rrf_k=100, variant_weight={**HH, **{v: w * 1.5 for v, w in PRFW.items()}})),
    "i_dprf05_rrf120": (DOCLP, dict(F1, rrf_k=120, variant_weight={**HH, **{v: w * 0.5 for v, w in PRFW.items()}})),
    "i_dprf1_rrf120": (DOCLP, dict(F1, rrf_k=120, variant_weight={**HH, **PRFW})),
    # i_dprf1 .3575, i_dprf15 .3577 (best) -> scale the head/mean lists too
    "j_hh125": (DOCLP, dict(F1, rrf_k=100, variant_weight={**{v: w * 1.25 for v, w in HH.items()},
                                                         **{v: w * 1.5 for v, w in PRFW.items()}})),
    "j_hh15": (DOCLP, dict(F1, rrf_k=100, variant_weight={**{v: w * 1.5 for v, w in HH.items()},
                                                        **{v: w * 1.75 for v, w in PRFW.items()}})),
    "j_prf2": (DOCLP, dict(F1, rrf_k=100, variant_weight={**HH, **{v: w * 2.0 for v, w in PRFW.items()}})),
    # j_hh125 .3577 = i_dprf15 .3577: weight plateau. Last slot: doc-level lists also rank the chunks (they point at
    # each doc's first chunk), chunk side has been flat at ~.325
    "k_chunkdoc": (DOCLP, dict(F1, rrf_k=100, variant_weight={**HH, **{v: w * 1.5 for v, w in PRFW.items()}},
                               chunk_keep=DOCLP)),
}


def load_doc_chunks(ingest, doc_ids: set) -> dict:
    """{doc_id: [chunk dicts in idx order]} for every chunk of the given docs (select.merge_chunks rebuilds the text)."""
    import glob
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    out, ids = {}, pa.array(sorted(int(d) for d in doc_ids), pa.int64())
    for p in glob.glob(os.path.join(ingest, "rag", "chunks", "*", "*.parquet")):
        t = pq.read_table(p, columns=["doc_id", "idx", "spans", "text", "title"])
        t = t.filter(pc.is_in(t.column("doc_id"), value_set=ids))
        for r in t.to_pylist():
            out.setdefault(int(r["doc_id"]), []).append({"idx": int(r["idx"]), "spans": [tuple(int(x) for x in sp) for sp in r["spans"]],
                                                       "text": r["text"], "title": r["title"]})
    return {d: sorted(v, key=lambda c: c["idx"]) for d, v in out.items()}


def main(run_dir, ingest, queries_path, out_dir, names):
    with open(os.path.join(run_dir, os.environ.get("LISTS", "lists.pkl")), "rb") as f:
        lists = pickle.load(f)
    queries = pd.read_parquet(queries_path, columns=["id", "query"])
    for name in names.split(","):
        keep, over = VARIANTS[name]
        over = dict(over)
        trim = over.pop("trim", 0)  # chunk text cut to its first N chars (still a verbatim substring), smaller file
        csp = over.pop("chunk_w_sparse", None)  # chunks ranked with their own sparse weight (docs keep w_sparse)
        glob_n = over.pop("chunk_global", 0)  # N chunks per query by global chunk rank (max k_chunks per doc)
        ckeep = over.pop("chunk_keep", keep)  # lists used to rank chunks (may differ from the doc-ranking lists)
        mode = over.pop("chunk_mode", "chunks")  # chunks | merge (one text per doc) | doc (whole doc text, doc_n docs)
        doc_n = over.pop("doc_n", 0)
        cfg = Config(**{**BASE, **over})
        cand = search.fuse(lists, cfg.rrf_k, keep=keep, cfg=cfg)  # cfg carries the list weights (w_dense, w_sparse)
        sel = select.select(cand, cfg)
        if csp is not None or glob_n:
            ccfg = Config(**{**BASE, **over, "w_sparse": csp if csp is not None else cfg.w_sparse, "cand_docs": 10 ** 6})
            cc = search.fuse(lists, cfg.rrf_k, keep=ckeep, cfg=ccfg).sort_values(["query_id", "score"], ascending=[True, False])
            by_q = {int(q): g for q, g in cc.groupby("query_id")}
            for q, sq in sel.items():
                g = by_q.get(q)
                if g is None:
                    continue
                g = g[g.doc_id.isin(sq["docs"][:cfg.chunk_docs])]
                if glob_n:
                    g = g.groupby("doc_id", sort=False).head(cfg.k_chunks).head(glob_n)
                    order = {d: i for i, d in enumerate(sq["docs"])}
                    sq["chunks"] = {int(d): x.chunk_id.tolist() for d, x in sorted(g.groupby("doc_id"), key=lambda kv: order[kv[0]])}
                else:
                    sq["chunks"] = {int(d): x.chunk_id.head(cfg.k_chunks).tolist() for d, x in g.groupby("doc_id")}
        if mode == "doc":  # whole doc text (all its chunks, document order) for the first doc_n docs
            need = {d for sq in sel.values() for d in sq["docs"][:doc_n]}
            full = load_doc_chunks(ingest, need)
            texts = {q: [(d, select.merge_chunks(full[d])) for d in sq["docs"][:doc_n] if d in full] for q, sq in sel.items()}
        else:
            store = select.load_chunks(ingest, {c for s in sel.values() for cs in s["chunks"].values() for c in cs})
            if mode == "merge":  # one verbatim text per doc from its selected chunks
                texts = {q: [(d, select.merge_chunks([store[c] for c in sq["chunks"].get(d, []) if c in store]))
                             for d in sq["docs"] if any(c in store for c in sq["chunks"].get(d, []))] for q, sq in sel.items()}
            else:
                texts = select.chunk_texts(sel, store, False)
        entries = {q: {"docs": s["docs"], "chunk_texts": [(d, t[:trim] if trim else t) for d, t in texts.get(q, [])]}
                   for q, s in sel.items()}
        print(name, submit.write(queries, entries, os.path.join(out_dir, name), 0.01), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
