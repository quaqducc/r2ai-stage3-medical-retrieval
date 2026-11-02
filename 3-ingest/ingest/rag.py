"""Flow A commands. Layout under <out>/rag/:
  chunks/<lang>/<name>.parquet          chunks of one clean-store part (<name> = <host>__<part>)
  docs/<lang>/<name>.parquet            doc cards (Docs F2 + pubmed metadata field)
  _done/chunk/<name>.json               marker: that part is chunked (resume)
  dense/<tag>/<lang>/<name>.{npy,ids.parquet}     bge-m3 dense, float16
  sparse/<tag>/<lang>/<name>.{npz,ids.parquet}    bge-m3 lexical weights, CSR
  sparse_index/<tag>/<lang>/            stacked CSC index (sparse.build)
  reports/chunk_stats.json"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import shutil
import zlib
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from extract import storage

from . import dense, sparse
from .chunk import chunk_doc, count_words, doc_type, pubmed_meta, usable, PUBMED

log = logging.getLogger("ingest")
DOC_COLS = ["id", "url", "host", "lang", "text_lang", "status", "title", "text", "markdown", "n_chars", "flags"]


def _name(path: str) -> str:
    host, part = path.replace("\\", "/").split("/")[-2:]
    return f"{host}__{part[:-len('.parquet')]}"


def _read(path: str, columns=None) -> pd.DataFrame:
    fs, p = storage.filesystem(path)
    return pd.read_parquet(p, columns=columns, filesystem=fs)


def _write(df: pd.DataFrame, dst: str, workdir: str) -> None:
    tmp = os.path.join(workdir, os.path.basename(dst))
    os.makedirs(workdir, exist_ok=True)
    df.to_parquet(tmp, index=False)
    storage.put(tmp, dst)
    os.remove(tmp)


# ------------------------------------------------------------------ chunk
def chunk_part(args) -> dict:
    src, out, workdir, limit = args
    name = _name(src)
    df = _read(src, DOC_COLS)
    if limit:
        df = df.head(limit)
    chunks, cards = [], []
    for doc in df.to_dict("records"):
        doc["flags"] = list(doc["flags"]) if doc["flags"] is not None else []
        if not usable(doc):
            continue
        chs = chunk_doc(doc)
        chunks += chs
        cards.append({"doc_id": int(doc["id"]), "url": doc["url"], "host": doc["host"], "lang": doc["lang"],
                      "text_lang": doc["text_lang"], "title": doc["title"], "doc_type": doc_type(doc),
                      "n_chunks": len(chs), "n_words": round(count_words(doc["text"]), 1),
                      "meta_text": pubmed_meta(doc) if doc["host"] == PUBMED else "", "doc_flags": doc["flags"]})
    st = {"part": name, "docs_in": len(df), "docs_chunked": len(cards), "chunks": len(chunks),
          "kinds": dict(Counter(f"{c['doc_type']}/{c['unit_kind']}" for c in chunks)),
          "words": [c["n_words"] for c in chunks]}
    if chunks:
        cdf = pd.DataFrame(chunks)
        for lang, g in cdf.groupby("lang"):
            _write(g, storage.join(out, "rag", "chunks", lang or "xx", name + ".parquet"), workdir)
        for lang, g in pd.DataFrame(cards).fillna({"lang": "xx"}).groupby("lang"):
            _write(g, storage.join(out, "rag", "docs", lang or "xx", name + ".parquet"), workdir)
    if not limit:  # a smoke run over a truncated part must not mark the part as done
        storage.write_json(storage.join(out, "rag", "_done", "chunk", name + ".json"), st)
    return st


def run_chunk(clean: str, out: str, workers: int = 1, hosts: list[str] | None = None, workdir: str = "/tmp/ingest",
              limit: int | None = None, force: bool = False) -> dict:
    parts = storage.glob(storage.join(clean, "docs", "*", "*.parquet"))
    if hosts:
        parts = [p for p in parts if p.replace("\\", "/").split("/")[-2] in set(hosts)]
    todo = [p for p in parts if force or limit or not storage.exists(storage.join(out, "rag", "_done", "chunk", _name(p) + ".json"))]
    log.info("chunk: %d parts (%d already done)", len(todo), len(parts) - len(todo))
    jobs = [(p, out, os.path.join(workdir, f"w{i}"), limit) for i, p in enumerate(todo)]
    if workers > 1:
        with mp.get_context("spawn").Pool(workers) as pool:
            stats = list(pool.imap_unordered(chunk_part, jobs))
    else:
        stats = [chunk_part(j) for j in jobs]
    if not limit:  # the report covers every finished part, not only the ones processed by this run
        stats = [storage.read_json(p) for p in storage.glob(storage.join(out, "rag", "_done", "chunk", "*.json"))]
    words = [w for s in stats for w in s.get("words", [])]
    rep = {"parts": len(stats), "parts_this_run": len(todo), "docs_in": sum(s["docs_in"] for s in stats), "docs_chunked": sum(s["docs_chunked"] for s in stats),
           "chunks": len(words), "kinds": dict(sum((Counter(s["kinds"]) for s in stats), Counter())),
           "words_p50": float(np.median(words)) if words else 0, "words_p90": float(np.percentile(words, 90)) if words else 0,
           "words_max": float(max(words)) if words else 0}
    storage.write_json(storage.join(out, "rag", "reports", "chunk_stats.json"), rep)
    return rep


# ------------------------------------------------------------------ embed
def embed_input(c: dict) -> str:
    """Model input = title + heading path + chunk text (context for the encoder; the chunk text itself stays verbatim)."""
    return "\n".join(x for x in (c.get("title"), c.get("heading_path"), c["text"]) if x)


class BgeM3Encoder:
    """bge-m3 via FlagEmbedding: one forward pass gives the dense vector and the sparse lexical weights."""

    def __init__(self, model: str = "BAAI/bge-m3", devices: list[str] | None = None, batch_size: int = 32,
                 max_length: int = 2048, fp16: bool = True):
        from FlagEmbedding import BGEM3FlagModel
        self.m = BGEM3FlagModel(model, use_fp16=fp16, devices=devices, batch_size=batch_size,
                                passage_max_length=max_length, query_max_length=512)
        self.batch_size, self.max_length = batch_size, max_length

    def encode(self, texts: list[str]):
        r = self.m.encode_corpus(texts, batch_size=self.batch_size, max_length=self.max_length,
                                 return_dense=True, return_sparse=True, return_colbert_vecs=False)
        return np.asarray(r["dense_vecs"], np.float32).reshape(len(texts), -1), _plain(r["lexical_weights"])

    def encode_queries(self, texts: list[str]):
        r = self.m.encode_queries(texts, return_dense=True, return_sparse=True, return_colbert_vecs=False)
        return np.asarray(r["dense_vecs"], np.float32).reshape(len(texts), -1), _plain(r["lexical_weights"])


def _plain(lex) -> list[dict]:
    """FlagEmbedding gives {token: np.float16/32} (a bare dict for one text); HttpEncoder gives {str: float}. Same type
    for both so caches can JSON them."""
    return [{str(k): float(v) for k, v in d.items()} for d in ([lex] if isinstance(lex, dict) else lex)]


class E5Encoder:
    """multilingual-e5 (dense only) as a second, differently trained retriever: "passage: " / "query: " prefixes,
    mean pooling, L2-normalised. Sparse side is empty (run_embed still writes an empty sparse shard per part)."""

    def __init__(self, model: str = "intfloat/multilingual-e5-large", device: str | None = None, batch_size: int = 64,
                 max_length: int = 512, fp16: bool = True):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model)
        self.m = AutoModel.from_pretrained(model, torch_dtype=torch.float16 if fp16 and self.device != "cpu" else None)
        self.m.to(self.device).eval()
        self.batch_size, self.max_length = batch_size, max_length

    def _enc(self, texts: list[str]) -> np.ndarray:
        out = []
        order = np.argsort([len(t) for t in texts])
        with self.torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                part = [texts[j] for j in order[i:i + self.batch_size]]
                b = self.tok(part, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt").to(self.device)
                h = self.m(**b).last_hidden_state
                mask = b["attention_mask"].unsqueeze(-1).to(h.dtype)
                v = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
                out.append(self.torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy())
        vecs = np.concatenate(out) if out else np.zeros((0, 1024), np.float32)
        res = np.empty_like(vecs)
        res[order] = vecs
        return res

    def encode(self, texts: list[str]):
        return self._enc(["passage: " + t for t in texts]), [{} for _ in texts]

    def encode_queries(self, texts: list[str]):
        return self._enc(["query: " + t for t in texts]), [{} for _ in texts]

    name = "e5-large"


class HttpEncoder:
    """bge-m3 behind an HTTP server: POST <url>/encode_batch {"input": [...]} ->
    {"data": [{"dense": [1024 floats, L2-normalised], "sparse": {token_id: weight}}]}. Used for queries / small runs."""

    def __init__(self, url: str, batch_size: int = 16, timeout: int = 600, retries: int = 8):
        import requests
        self.url = url.rstrip("/") + ("" if url.rstrip("/").endswith("/encode_batch") else "/encode_batch")
        self.batch_size, self.timeout, self.retries = batch_size, timeout, retries
        self.s = requests.Session()
        self.s.headers["ngrok-skip-browser-warning"] = "1"

    def _post(self, texts: list[str]) -> list[dict]:
        import time
        for k in range(self.retries):
            try:
                r = self.s.post(self.url, json={"input": texts}, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()["data"]
                if len(data) != len(texts):
                    raise ValueError(f"server returned {len(data)} items for {len(texts)} inputs")
                return data
            except Exception as e:  # transient tunnel/server errors: back off and retry, then fail loudly
                if k == self.retries - 1:
                    raise
                log.warning("encode request failed (%s), retry %d", e, k + 1)
                time.sleep(min(2 ** k, 60))  # ~5 min of tunnel outage before giving up

    def encode(self, texts: list[str]):
        vecs, lex = [], []
        for i in range(0, len(texts), self.batch_size):
            for item in self._post(texts[i:i + self.batch_size]):
                vecs.append(np.asarray(item["dense"], np.float32))
                lex.append(item["sparse"])
        return (np.stack(vecs) if vecs else np.zeros((0, 1024), np.float32)), lex

    encode_queries = encode  # bge-m3 uses no query instruction


def run_embed(out: str, encoder, tag: str = "bge-m3", langs: list[str] | None = None, workdir: str = "/tmp/ingest",
              max_parts: int | None = None, shard: tuple[int, int] = (0, 1)) -> dict:
    """Embeds every chunk part not embedded yet (a part is done when both its dense and sparse shards exist).
    shard=(i, n): only parts with crc32(lang/name) % n == i, so n parallel sessions never embed the same part."""
    parts = storage.glob(storage.join(out, "rag", "chunks", "*", "*.parquet"))
    done = n_chunks = 0
    for p in parts:
        lang, name = p.replace("\\", "/").split("/")[-2], os.path.basename(p)[:-len(".parquet")]
        if langs and lang not in langs:
            continue
        if zlib.crc32(f"{lang}/{name}".encode()) % shard[1] != shard[0]:
            continue
        d_dst = storage.join(out, "rag", "dense", tag, lang, name)
        s_dst = storage.join(out, "rag", "sparse", tag, lang, name)
        if storage.exists(d_dst + ".ids.parquet") and storage.exists(s_dst + ".ids.parquet"):
            continue
        if max_parts is not None and done >= max_parts:
            break
        df = _read(p, ["chunk_id", "title", "heading_path", "text"])
        texts = [embed_input(c) for c in df.to_dict("records")]
        order = np.argsort([len(t) for t in texts])  # similar lengths per batch -> less padding
        vecs, lex = encoder.encode([texts[i] for i in order])
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        vecs, lex = vecs[inv], [lex[i] for i in inv]
        ids = df.chunk_id.tolist()
        tmp = os.path.join(workdir, "embed", lang)
        dense.save_shard(os.path.join(tmp, name), ids, vecs)
        sparse.save_shard(os.path.join(tmp, name + "_s"), ids, sparse.lexical_to_csr(lex))
        for ext in (".npy", ".ids.parquet"):
            storage.put(os.path.join(tmp, name + ext), d_dst + ext)
        storage.put(os.path.join(tmp, name + "_s.npz"), s_dst + ".npz")
        storage.put(os.path.join(tmp, name + "_s.ids.parquet"), s_dst + ".ids.parquet")  # written last = shard done
        shutil.rmtree(tmp, ignore_errors=True)
        done += 1
        n_chunks += len(ids)
        log.info("embedded %s/%s: %d chunks", lang, name, len(ids))
    return {"parts_embedded": done, "chunks": n_chunks}


def run_bm25(out: str, langs: list[str] | None = None, workdir: str = "/tmp/ingest") -> dict:
    """CPU fallback: BM25 index per language over the same model input as `rag embed` (title + headings + text),
    written as rag/sparse_index/bm25/<lang>/ (retrieval: tag "bm25", no dense)."""
    from . import bm25
    tok = bm25.Tokenizer()
    by_lang = defaultdict(list)
    for p in storage.glob(storage.join(out, "rag", "chunks", "*", "*.parquet")):
        by_lang[p.replace("\\", "/").split("/")[-2]].append(p)
    res = {}
    for lang, parts in sorted(by_lang.items()):
        if langs and lang not in langs:
            continue
        data = []
        for p in sorted(parts):
            df = _read(p, ["chunk_id", "title", "heading_path", "text"])
            data.append((df.chunk_id.tolist(), [embed_input(c) for c in df.to_dict("records")]))
        tmp = os.path.join(workdir, "bm25", lang)
        res[lang] = bm25.build_lang(data, tmp, tok)
        for f in os.listdir(tmp):
            storage.put(os.path.join(tmp, f), storage.join(out, "rag", "sparse_index", bm25.TAG, lang, f))
        shutil.rmtree(tmp, ignore_errors=True)
        log.info("bm25 %s: %s", lang, res[lang])
    return res


def run_index(out: str, tag: str = "bge-m3", workdir: str = "/tmp/ingest") -> dict:
    """Stacks the sparse shards of each language into one CSC index (dense stays sharded: exact search streams it)."""
    res = {}
    by_lang = defaultdict(list)
    for p in storage.glob(storage.join(out, "rag", "sparse", tag, "*", "*.ids.parquet")):
        by_lang[p.replace("\\", "/").split("/")[-2]].append(p[:-len(".ids.parquet")])
    for lang, shards in sorted(by_lang.items()):
        local = []
        for s in sorted(shards):
            d = os.path.join(workdir, "shards", lang)
            local.append(os.path.join(d, os.path.basename(s)))
            for ext in (".npz", ".ids.parquet"):
                p = storage.download(s + ext, d)
                if p != local[-1] + ext:  # local source: copy next to the others
                    os.makedirs(d, exist_ok=True)
                    shutil.copyfile(p, local[-1] + ext)
        tmp = os.path.join(workdir, "index", lang)
        res[lang] = sparse.build(local, tmp)
        for f in os.listdir(tmp):
            storage.put(os.path.join(tmp, f), storage.join(out, "rag", "sparse_index", tag, lang, f))
        shutil.rmtree(os.path.join(workdir, "shards", lang), ignore_errors=True)
        shutil.rmtree(tmp, ignore_errors=True)
    return res
