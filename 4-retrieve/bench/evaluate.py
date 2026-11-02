"""Run modes on the silver dev set and score them with the competition metric (overall + per stratum).
Usage: LLM_API_KEY=... python bench/evaluate.py <bench_dir> <bge-m3 url> <llm_url> <llm_model> [modes] [config.json]"""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from retrieve.__main__ import run_modes  # noqa: E402
from retrieve.config import Config  # noqa: E402
from retrieve.__main__ import query_encoder  # noqa: E402
from retrieve.encode_cache import CachedEncoder  # noqa: E402
from retrieve.llm import LLM  # noqa: E402
from retrieve.metric import evaluate  # noqa: E402


def score_run(bench: str, run_dir: str, modes: list[str]) -> dict:
    dev = pd.read_parquet(os.path.join(bench, "dev.parquet"))
    gold = [{"id": int(r.id), **json.loads(r.gold)} for r in dev.itertuples()]
    strata = {int(r.id): f"{r.lang}/{r.doc_type}" for r in dev.itertuples()}
    rep = {}
    for m in modes:
        pred = json.load(open(os.path.join(run_dir, m, "submission.json"), encoding="utf-8"))
        res = evaluate(pred, gold)
        for s in sorted(set(strata.values())):
            res[s] = round(evaluate(pred, [g for g in gold if strata[g["id"]] == s])["final"], 3)
        rep[m] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}
    return rep


def main(bench, server, llm_url, llm_model, modes="v0,v1,v2", config=None):
    modes = modes.split(",")
    run_dir = os.path.join(bench, "runs", os.environ.get("RUN", "latest"))
    enc = CachedEncoder(query_encoder(server), os.path.join(bench, "cache", "query_enc.sqlite"))
    extra = json.loads(os.environ["LLM_EXTRA"]) if os.environ.get("LLM_EXTRA") else None  # e.g. reasoning off
    llm = LLM(llm_url, llm_model, os.path.join(bench, "cache", "llm.sqlite"), workers=int(os.environ.get("LLM_WORKERS", 8)),
              extra_body=extra)
    dev = pd.read_parquet(os.path.join(bench, "dev.parquet"))
    qp = os.path.join(bench, "dev_queries.parquet")
    dev[["id", "query"]].to_parquet(qp)
    run_modes(qp, os.path.join(bench, "ingest"), run_dir, enc, llm, modes, Config.load(config) if config else None,
              os.path.join(bench, "cache"))
    rep = score_run(bench, run_dir, modes)
    with open(os.path.join(run_dir, "eval.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
