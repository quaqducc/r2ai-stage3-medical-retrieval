"""Query-generation agent (HyDE): per query one LLM call writes a hypothetical Vietnamese answer passage, the same in
Chinese, and the question re-phrased as an article title. Each text is embedded (bge-m3) and searched over the dense
shards; the results are ADDED as extra lists (variants hyde_vi, hyde_zh, title) — the LLM never filters or reorders
candidates (that hurt on the leaderboard). Usage (LLM key: env LLM_API_KEY):
python bench/hyde.py <run_dir with LISTS> <ingest> <queries.parquet> <cache_dir> <out lists.pkl> [llm_url model]"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ingest import dense  # noqa: E402
from retrieve import search  # noqa: E402
from retrieve.__main__ import query_encoder  # noqa: E402
from retrieve.encode_cache import CachedEncoder  # noqa: E402
from retrieve.llm import LLM  # noqa: E402

SYSTEM = "Bạn là bác sĩ viết nội dung y khoa. Chỉ trả về MỘT object JSON, không giải thích."
USER = """Câu hỏi của người bệnh: {q}

Viết:
- "vi_doc": một đoạn 100-160 từ tiếng Việt, như trích từ một bài viết y khoa trên báo/website sức khỏe Việt Nam, trả lời trực tiếp câu hỏi (nêu bệnh, nguyên nhân, triệu chứng, cách xử trí liên quan).
- "zh_doc": một đoạn tương tự 150-250 chữ bằng tiếng Trung giản thể, như trích từ trang hỏi đáp y khoa Trung Quốc.
- "title": câu hỏi viết lại thành tiêu đề bài viết tiếng Việt ngắn gọn (dưới 20 từ).
Trả về: {{"vi_doc": "...", "zh_doc": "...", "title": "..."}}"""
KEYS = {"vi_doc": "hyde_vi", "zh_doc": "hyde_zh", "title": "title"}


def main(run_dir, ingest, queries_path, cache_dir, out_path, url="https://openrouter.ai/api/v1", model="qwen/qwen3.5-9b"):
    lists = pickle.load(open(os.path.join(run_dir, os.environ.get("LISTS", "lists.pkl")), "rb"))
    queries = pd.read_parquet(queries_path, columns=["id", "query"])
    llm = LLM(url, model, os.path.join(cache_dir, "llm_hyde.sqlite"), workers=16, temperature=0.0, max_tokens=900,
              extra_body={"reasoning": {"enabled": False}})
    raws = llm.map_json([[{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER.format(q=q)}]
                         for q in queries["query"]])
    reqs = []
    for qid, r in zip(queries["id"].astype(int), raws):
        for k, v in KEYS.items():
            t = (r or {}).get(k)
            if isinstance(t, str) and len(t.strip()) > 10:
                reqs.append((qid, v, t.strip()))
    ok = sum(1 for r in raws if r)
    print(f"hyde: {ok}/{len(raws)} replies parsed, {len(reqs)} texts", flush=True)
    if ok < 0.8 * len(raws):
        raise SystemExit("too few LLM replies parsed")
    enc = CachedEncoder(query_encoder("cpu"), os.path.join(cache_dir, "query_enc.sqlite"))
    qv, _ = enc.encode_queries([t for _, _, t in reqs])
    per_lang = [(lang, *dense.search(np.asarray(qv, np.float32), search.dense_shards(ingest, "bge-m3", lang), k=200))
                for lang in ("vi", "zh")]
    added = []
    for j, (qid, v, _) in enumerate(reqs):
        merged = sorted(((float(s), c, lang) for lang, ids, sc in per_lang for c, s in zip(ids[j], sc[j])), reverse=True)[:200]
        added.append({"query_id": qid, "variant": v, "channel": "dense", "ids": [c for _, c, _ in merged],
                      "langs": [l for _, _, l in merged], "scores": [s for s, _, _ in merged]})
    with open(out_path, "wb") as f:
        pickle.dump(lists + added, f)
    print("hyde lists added:", len(added), "->", out_path, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
