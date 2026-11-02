"""CLI: python -m retrieve run --modes v0,v1,v2 --queries query.parquet --ingest <local ingest root> --run-dir runs/r1
          --server <bge-m3 url> [--llm-url http://.../v1 --llm-model <name>]   (LLM key: env LLM_API_KEY)
All modes share the expensive work: one plan per query (cached), one encoding per distinct query text (cached),
one pass per retrieval channel; each mode then fuses its subset of lists and selects."""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from dataclasses import replace

import pandas as pd

from . import judge, learn, plan, search, select, submit
from .search import DegradedRun
from .config import MODES, Config, for_mode
from .encode_cache import CachedEncoder

log = logging.getLogger("retrieve")
KEEP = {
    "v0": lambda r: r["variant"] == "orig",
    "v1": lambda r: not r["variant"].startswith("subq"),
    "v2": lambda r: True,
    "deep": lambda r: True,
}


MODELS = os.environ.get("RETRIEVE_MODELS") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def learned_candidates(lists: list[dict], keep, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ranker p per (query, doc) over every kept list; the top cfg.cand_docs docs by p and their fused chunks.
    RRF still picks the chunks inside a doc, it no longer decides which docs make the cut."""
    path = cfg.model_path if os.path.dirname(cfg.model_path) else os.path.join(MODELS, cfg.model_path)
    dp = learn.doc_p(lists, keep, learn.Ranker.load(path))
    dp = dp[dp["rank"] <= cfg.cand_docs]
    cand = search.fuse(lists, cfg.rrf_k, keep=keep, cfg=replace(cfg, cand_docs=10**9))
    return cand.merge(dp[["query_id", "doc_id"]]), dp


def run_listwise(queries: pd.DataFrame, cand: pd.DataFrame, ingest: str, llm, cfg, store: dict) -> pd.DataFrame:
    """deep mode: the LLM grades the first cfg.deep_judge fused docs of each query, cfg.judge_batch per call (title +
    best chunk, cfg.judge_chars chars). New order: grade 3 > 2 > 1 > not graded (fused order) > 0; keep cand_docs above
    k_docs so docs graded 0 fall past the cut. Returns cand with the scores shifted by tier."""
    qtext = dict(zip(queries["id"].astype(int), queries["query"]))
    d = select.doc_table(cand)
    top = d[d["rank"] <= cfg.deep_judge]
    best = cand.sort_values("score", ascending=False).drop_duplicates(["query_id", "doc_id"])
    best_chunk = {(int(q), int(doc)): c for q, doc, c in zip(best.query_id, best.doc_id, best.chunk_id)}
    store.update(select.load_chunks(ingest, {best_chunk[(int(q), int(x))] for q, x in zip(top.query_id, top.doc_id)}
                                    - store.keys()))
    keys, prompts = [], []
    for q, g in top.groupby("query_id"):
        docs = [int(x) for x in g.sort_values("rank").doc_id if best_chunk[(int(q), int(x))] in store]
        for i in range(0, len(docs), cfg.judge_batch):
            b = docs[i:i + cfg.judge_batch]
            cs = [store[best_chunk[(int(q), x)]] for x in b]
            keys.append((int(q), b))
            prompts.append(judge.list_prompt(qtext[int(q)], [(c["title"], c["text"]) for c in cs], cfg.judge_chars))
    grade, ok = {}, 0
    for (q, b), raw in zip(keys, llm.map_json(prompts)):
        s = judge.list_parse(raw, len(b))
        if s is not None:
            ok += 1
            grade.update({(q, x): g for x, g in zip(b, s) if g is not None})
    log.info("listwise judge: %d / %d batches parsed, %d docs graded", ok, len(keys), len(grade))
    if keys and ok / len(keys) < cfg.min_judge_ok:
        msg = f"listwise judge: only {ok}/{len(keys)} batches parsed (LLM down or replies truncated)"
        if not cfg.allow_degraded:
            raise DegradedRun(msg)
        log.warning(msg)
    tier = {3: 4, 2: 3, 1: 2, 0: 0}  # not graded: 1
    shift = [10.0 * tier.get(grade.get((int(q), int(x))), 1) for q, x in zip(cand.query_id, cand.doc_id)]
    return cand.assign(score=cand.score + shift)


def run_judge(queries: pd.DataFrame, cand: pd.DataFrame, ingest: str, llm, cfg, store: dict,
              learned: pd.DataFrame | None = None) -> tuple[dict, dict]:
    """Grey-zone docs (p in [low, high] among the top judge_max_docs of a query) -> p override + evidence chunks."""
    qtext = dict(zip(queries["id"].astype(int), queries["query"]))
    d = select.doc_probs(cand, cfg, learned)
    grey = d[(d.p >= cfg.judge_low) & (d.p <= cfg.judge_high) & (d["rank"] <= cfg.judge_max_docs)]
    if grey.empty:
        return {}, {}
    top = cand.sort_values("score", ascending=False).groupby(["query_id", "doc_id"]).head(cfg.judge_chunks)
    cids = top.merge(grey[["query_id", "doc_id"]]).groupby(["query_id", "doc_id"]).chunk_id.apply(list)
    missing = {c for cs in cids for c in cs} - store.keys()
    store.update(select.load_chunks(ingest, missing))
    keys, prompts, chunk_lists = [], [], []
    for (q, doc), cs in cids.items():
        cs = [c for c in cs if c in store]
        if not cs:
            continue
        keys.append((int(q), int(doc)))
        chunk_lists.append(cs)
        prompts.append(judge.prompt(qtext[int(q)], store[cs[0]]["title"], [store[c]["text"] for c in cs],
                                    cfg.judge_chars))
    p_over, evidence = {}, {}
    raws = llm.map_json(prompts)
    for key, cs, raw in zip(keys, chunk_lists, raws):
        rel, ev = judge.parse(raw, len(cs))
        if rel is None:
            continue  # unparseable: keep the calibrated p
        p_over[key] = cfg.judge_rel_p[rel]
        if ev:
            evidence[key] = [cs[i] for i in ev]
    log.info("judge: %d grey docs, %d judged", len(keys), len(p_over))
    if keys and len(p_over) / len(keys) < cfg.min_judge_ok:
        msg = f"judge: only {len(p_over)}/{len(keys)} verdicts parsed (LLM down or replies truncated)"
        if not cfg.allow_degraded:
            raise DegradedRun(msg)
        log.warning(msg)
    return p_over, evidence


def run_modes(queries_path: str, ingest: str, run_dir: str, encoder, llm, modes: list[str], base: Config | None = None,
              cache_dir: str | None = None, chunk_encoder=None, plans_path: str | None = None) -> dict:
    """plans_path: plans.json of an earlier run (same queries) reused as is, e.g. to keep one plan set across runs."""
    os.makedirs(run_dir, exist_ok=True)
    queries = pd.read_parquet(queries_path, columns=["id", "query"])
    cfgs = {m: for_mode(m, base) for m in modes}
    need_plan = any(c.use_plan for c in cfgs.values())
    plans = None
    if need_plan:
        if plans_path:
            with open(plans_path, encoding="utf-8") as f:
                plans = {int(k): v for k, v in json.load(f).items()}
            missing = set(queries["id"].astype(int)) - plans.keys()
            if missing:
                raise SystemExit(f"--plans {plans_path} has no plan for {len(missing)} queries")
        elif llm is None:
            raise SystemExit(f"modes {modes} need an LLM (--llm-url)")
        else:
            plans = plan.run(queries, llm)
        with open(os.path.join(run_dir, "plans.json"), "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in plans.items()}, f, ensure_ascii=False, indent=1)
        ok = sum(p["ok"] for p in plans.values())
        log.info("plans: %d ok / %d", ok, len(plans))
        guard = Config(**{**(base.__dict__ if base else {})})
        if plans and ok / len(plans) < guard.min_plan_ok:
            msg = f"plans: only {ok}/{len(plans)} usable (LLM down?); plan modes would silently fall back to v0"
            if not guard.allow_degraded:
                raise DegradedRun(msg)
            log.warning(msg)
    union = Config(**{**(base.__dict__ if base else {})})
    union.sparse_langs = sorted({l for c in cfgs.values() for l in c.sparse_langs})
    lists = search.collect(queries, plans, ingest, encoder, union)
    if union.rerank_docs > 0:
        from . import rerank
        from ingest.rag import BgeM3Encoder
        cv = rerank.ChunkVectors(chunk_encoder or BgeM3Encoder(devices=["cpu"], batch_size=16, max_length=512, fp16=False),
                                 os.path.join(cache_dir or run_dir, "chunk_vec.sqlite"))
        lists += rerank.dense_lists(lists, queries, plans, ingest, encoder, cv, union)
    with open(os.path.join(run_dir, "lists.pkl"), "wb") as f:
        pickle.dump(lists, f)
    out, store = {}, {}  # chunk texts loaded once, shared by all modes
    for m, cfg in cfgs.items():
        mdir = os.path.join(run_dir, m)
        os.makedirs(mdir, exist_ok=True)
        cfg.save(os.path.join(mdir, "config.json"))
        learned = None
        if cfg.select == "learned":
            cand, learned = learned_candidates(lists, KEEP[m], cfg)
        else:
            cand = search.fuse(lists, cfg.rrf_k, keep=KEEP[m], cfg=cfg)
        if cfg.select == "topk" and cfg.deep_judge > 0 and llm is not None:
            cand = run_listwise(queries, cand, ingest, llm, cfg, store)
        cand.to_parquet(os.path.join(mdir, "candidates.parquet"))
        p_over, ev = (run_judge(queries, cand, ingest, llm, cfg, store, learned) if cfg.use_judge and llm is not None
                      else ({}, {}))
        sel = select.select(cand, cfg, p_over, ev, learned)
        need = {c for s in sel.values() for cs in s["chunks"].values() for c in cs} - store.keys()
        store.update(select.load_chunks(ingest, need))
        texts = select.chunk_texts(sel, store, cfg.merge_adjacent)
        entries = {q: {"docs": s["docs"], "chunk_texts": texts.get(q, [])} for q, s in sel.items()}
        zp = submit.write(queries, entries, mdir, 1.0 if cfg.allow_degraded else cfg.max_empty_queries)
        with open(os.path.join(mdir, "selection.json"), "w", encoding="utf-8") as f:
            json.dump({str(q): s for q, s in sel.items()}, f, ensure_ascii=False)
        out[m] = {"queries": len(queries), "with_docs": sum(1 for s in sel.values() if s["docs"]),
                  "docs_per_query": round(sum(len(s["docs"]) for s in sel.values()) / max(len(queries), 1), 2),
                  "chunks_per_query": round(sum(len(v) for v in texts.values()) / max(len(queries), 1), 2),
                  "judged": len(p_over), "zip": zp}
        with open(os.path.join(mdir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(out[m], f, ensure_ascii=False, indent=1)
    return out


def run_v0(queries_path: str, ingest: str, run_dir: str, encoder, cfg: Config) -> dict:
    """Single v0 run written straight into run_dir (kept for the simple path and its tests)."""
    res = run_modes(queries_path, ingest, run_dir, encoder, None, ["v0"], cfg)["v0"]
    for f in ("submission.json", "submission.zip", "config.json", "summary.json", "candidates.parquet", "selection.json"):
        os.replace(os.path.join(run_dir, "v0", f), os.path.join(run_dir, f))
    return res


def query_encoder(server: str | None):
    """HTTP bge-m3 server, bge-m3 loaded on CPU (None / "cpu": short queries take minutes), or "bm25": tokenizer
    only, for the CPU-only BM25 index (config tag "bm25", dense_langs [])."""
    from ingest.rag import BgeM3Encoder, HttpEncoder
    if server in ("bm25", "bm25+cpu"):  # bm25+cpu: also a bge-m3 dense vector, for the CPU dense rerank
        from ingest.bm25 import QueryEncoder
        if server == "bm25":
            return QueryEncoder()
        from .rerank import HybridQueryEncoder
        return HybridQueryEncoder(QueryEncoder(), BgeM3Encoder(devices=["cpu"], batch_size=16, max_length=512, fp16=False))
    if server and server != "cpu":
        return HttpEncoder(server)
    return BgeM3Encoder(devices=["cpu"], batch_size=16, max_length=512, fp16=False)


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(prog="retrieve")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run one or more modes (v0, v1, v2) sharing plan/encoding/retrieval")
    r.add_argument("--modes", default="v0", help=f"comma list of {sorted(MODES)}")
    r.add_argument("--queries", required=True)
    r.add_argument("--ingest", required=True, help="local ingest root (gcloud storage rsync it first)")
    r.add_argument("--run-dir", required=True)
    r.add_argument("--server", help='bge-m3 HTTP server (POST /encode_batch); omit for bge-m3 on CPU; "bm25" for the BM25 index')
    r.add_argument("--llm-url", help="OpenAI-compatible base URL, e.g. http://host:8000/v1 (key: env LLM_API_KEY)")
    r.add_argument("--llm-model", default="Qwen/Qwen3-8B")
    r.add_argument("--llm-workers", type=int, default=8)
    r.add_argument("--llm-no-think", action="store_true", help="Qwen3 on vLLM: disable thinking (chat_template_kwargs)")
    r.add_argument("--llm-extra", help='JSON merged into every request, e.g. OpenRouter: \'{"reasoning": {"enabled": false}}\'')
    r.add_argument("--cache-dir", default=".cache", help="query-embedding + LLM caches, shared across runs")
    r.add_argument("--config", help="base config.json; mode settings are applied on top")
    r.add_argument("--plans", help="plans.json of an earlier run to reuse (no plan LLM calls; the LLM still judges)")
    a = p.parse_args(argv)
    base = Config.load(a.config) if a.config else None
    enc = CachedEncoder(query_encoder(a.server), os.path.join(a.cache_dir, "query_enc.sqlite"))
    llm = None
    if a.llm_url:
        from .llm import LLM
        extra = json.loads(a.llm_extra) if a.llm_extra else {}
        if a.llm_no_think:
            extra["chat_template_kwargs"] = {"enable_thinking": False}
        llm = LLM(a.llm_url, a.llm_model, os.path.join(a.cache_dir, "llm.sqlite"), workers=a.llm_workers,
                  extra_body=extra or None)
    res = run_modes(a.queries, a.ingest, a.run_dir, enc, llm, a.modes.split(","), base, a.cache_dir, plans_path=a.plans)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
