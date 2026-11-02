"""Silver dev set from the benchmark corpus: for sampled chunks, an LLM writes the Vietnamese patient question the
passage answers and quotes the answer verbatim. Gold doc = that doc (+ exact-duplicate docs); gold chunk = the quote.
Usage: LLM_API_KEY=... python bench/make_dev.py <bench_dir> <llm_url> <llm_model> [n]
Caveat: synthetic questions have exactly one source doc, so they measure ranking/chunk quality, not how many docs
real questions have; tune the number of submitted docs on the leaderboard, not here."""
import glob
import json
import os
import random
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from retrieve.llm import LLM  # noqa: E402

SYSTEM = "Bạn tạo dữ liệu đánh giá cho hệ thống tìm kiếm y khoa. Chỉ trả về MỘT object JSON."
USER = """Đoạn trích từ một tài liệu y khoa (có thể tiếng Việt, Trung hoặc Anh):
<<<
{passage}
>>>

Đóng vai một người bệnh Việt Nam, viết MỘT câu hỏi bằng tiếng Việt mà đoạn trên trả lời được.
- Phong cách: {style}
- Không chép nguyên văn đoạn trích, không nhắc tên website hay bác sĩ trong đoạn.
Sau đó chép NGUYÊN VĂN (giữ đúng ngôn ngữ gốc, không dịch, không sửa chữ) 1–3 câu liên tiếp trong đoạn chứa câu trả lời.
Trả về: {{"query": "<câu hỏi tiếng Việt>", "evidence": "<trích nguyên văn>"}}"""
STYLES = ["ngắn gọn, dưới 20 từ, như tiêu đề câu hỏi trên diễn đàn",
          "kể tình trạng cụ thể (tuổi, triệu chứng, đã dùng thuốc gì) rồi hỏi, khoảng 40–80 từ"]
# stratified by what the corpus holds: (lang, doc_type) -> share of the dev set
STRATA = {("zh", "qa"): 0.28, ("zh", "article"): 0.12, ("vi", "article"): 0.30, ("vi", "qa"): 0.08, ("en", "pubmed"): 0.22}


def squash(s):
    return " ".join((s or "").split())


def main(bench, url, model, n=240):
    ing = os.path.join(bench, "ingest")
    cols = ["chunk_id", "doc_id", "lang", "doc_type", "host", "n_words", "text"]
    ch = pd.concat(pd.read_parquet(f, columns=cols) for f in glob.glob(os.path.join(ing, "rag", "chunks", "*", "*.parquet")))
    ch = ch[ch.n_words >= 40]
    rng = random.Random(7)
    picks = []
    for (lang, kind), share in STRATA.items():
        pool = ch[(ch.lang == lang) & (ch.doc_type == kind)].drop_duplicates("doc_id")
        k = min(len(pool), round(n * share))
        picks += pool.sample(k, random_state=rng.randint(0, 10**6)).to_dict("records")
    extra = json.loads(os.environ["LLM_EXTRA"]) if os.environ.get("LLM_EXTRA") else None  # e.g. reasoning off
    llm = LLM(url, model, os.path.join(bench, "cache", "llm_dev.sqlite"), workers=8, temperature=0.7, extra_body=extra)
    prompts = [[{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER.format(passage=c["text"][:2500], style=STYLES[i % 2])}]
               for i, c in enumerate(picks)]
    if os.environ.get("CLEAN_DOCS"):  # full corpus: the 2-extract store on GCS, e.g. gs://aiguru_crawler/clean/v1/docs/
        clean = pd.read_parquet(os.environ["CLEAN_DOCS"], columns=["id", "content_sha1", "status"])
    else:
        clean = pd.concat(pd.read_parquet(f, columns=["id", "content_sha1", "status"])
                          for f in glob.glob(os.path.join(bench, "clean", "docs", "*", "*.parquet")))
    clean = clean[clean.status == "ok"]
    dupes = clean.groupby("content_sha1").id.apply(list).to_dict()
    sha = dict(zip(clean.id, clean.content_sha1))
    rows, dropped = [], 0
    for i, (c, r) in enumerate(zip(picks, llm.map_json(prompts))):
        q, ev = (r or {}).get("query"), (r or {}).get("evidence")
        if not (isinstance(q, str) and isinstance(ev, str) and q.strip() and ev.strip()):
            dropped += 1
            continue
        if squash(ev) not in squash(c["text"]):  # the quote must be verbatim, otherwise it is not a valid gold chunk
            dropped += 1
            continue
        docs = sorted(set([int(c["doc_id"])] + [int(x) for x in dupes.get(sha.get(c["doc_id"]), [])]))
        rows.append({"id": len(rows) + 1, "query": q.strip(), "lang": c["lang"], "doc_type": c["doc_type"],
                     "host": c["host"], "style": i % 2, "gold": json.dumps(
                         {"relevant_docs": docs, "relevant_chunks": [{"doc_id": int(c["doc_id"]), "chunk_text": ev.strip()}]},
                         ensure_ascii=False)})
    dev = pd.DataFrame(rows)
    dev.to_parquet(os.path.join(bench, "dev.parquet"))
    print(f"dev: {len(dev)} queries, dropped {dropped}; by stratum:",
          dev.groupby(["lang", "doc_type"]).size().to_dict(), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 240)
