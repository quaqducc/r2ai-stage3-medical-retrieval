"""OpenAI-compatible chat client (vLLM, or any /v1/chat/completions) with an on-disk cache keyed by the request,
parallel batch calls, and tolerant JSON extraction. The API key comes from the environment (LLM_API_KEY), never files."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor

log = logging.getLogger("retrieve")
_JSON = re.compile(r"\{.*\}", re.S)


def extract_json(text: str) -> dict | None:
    """First JSON object in a model reply (tolerates ```json fences, prose and <think> blocks around it)."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = _JSON.search(text)
    if not m:
        return None
    s = m.group()
    for end in range(len(s), 0, -1):  # trailing junk after the object: shrink to the last parseable '}'
        if s[end - 1] == "}":
            try:
                v = json.loads(s[:end])
                return v if isinstance(v, dict) else None
            except ValueError:
                continue
    return None


class Cache:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.con = sqlite3.connect(path, check_same_thread=False)
        self.con.execute("create table if not exists c (k text primary key, v text)")
        self.lock = threading.Lock()

    def get(self, k):
        with self.lock:
            r = self.con.execute("select v from c where k=?", (k,)).fetchone()
        return r[0] if r else None

    def delete(self, k):
        with self.lock:
            self.con.execute("delete from c where k=?", (k,))
            self.con.commit()

    def put(self, k, v):
        with self.lock:
            self.con.execute("insert or replace into c values (?,?)", (k, v))
            self.con.commit()


class LLM:
    def __init__(self, url: str, model: str, cache_path: str, api_key: str | None = None, workers: int = 8,
                 timeout: int = 180, retries: int = 3, temperature: float = 0.0, max_tokens: int = 1200,
                 extra_body: dict | None = None):
        import requests
        self.url = url.rstrip("/") + "/chat/completions"
        self.model, self.workers, self.timeout, self.retries = model, workers, timeout, retries
        self.temperature, self.max_tokens = temperature, max_tokens
        self.extra_body = extra_body or {}  # e.g. {"chat_template_kwargs": {"enable_thinking": False}} for Qwen3 on vLLM
        self.s = requests.Session()
        key = api_key if api_key is not None else os.environ.get("LLM_API_KEY", "")
        if key:
            self.s.headers["Authorization"] = f"Bearer {key}"
        self.cache = Cache(cache_path)

    def _key(self, messages):
        return hashlib.sha1(json.dumps([self.model, self.temperature, messages], ensure_ascii=False).encode()).hexdigest()

    def chat(self, messages: list[dict]) -> str:
        k = self._key(messages)
        hit = self.cache.get(k)
        if hit is not None:
            return hit
        body = {"model": self.model, "messages": messages, "temperature": self.temperature, "max_tokens": self.max_tokens,
                **self.extra_body}
        for i in range(self.retries):
            try:
                r = self.s.post(self.url, json=body, timeout=self.timeout)
                r.raise_for_status()
                out = r.json()["choices"][0]["message"]["content"] or ""
                self.cache.put(k, out)
                return out
            except Exception as e:  # transient: retry; the caller falls back when it finally fails
                if i == self.retries - 1:
                    raise
                log.warning("llm call failed (%s), retry %d", e, i + 1)
                time.sleep(2 ** i)

    def map_json(self, prompts: list[list[dict]]) -> list[dict | None]:
        """Parallel calls; a failed call or unparseable reply yields None (callers fall back, never drop a query)."""
        def one(m):
            try:
                out = extract_json(self.chat(m))
                if out is None:  # truncated / unparseable reply: do not keep it, a rerun should ask again
                    self.cache.delete(self._key(m))
                return out
            except Exception as e:
                log.warning("llm gave up: %s", e)
                return None
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(one, prompts))
