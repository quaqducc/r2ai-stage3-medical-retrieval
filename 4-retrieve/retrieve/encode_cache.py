"""Query encodings cached on disk by text: re-running retrieval with new settings never re-calls the GPU server."""
from __future__ import annotations

import json
import os
import sqlite3

import numpy as np


class CachedEncoder:
    def __init__(self, encoder, path: str, namespace: str | None = None, batch: int = 256):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        namespace = namespace or getattr(encoder, "name", "bge-m3")  # model tag, so switching model never reuses
        self.enc, self.ns, self.batch = encoder, namespace, batch
        self.con = sqlite3.connect(path)
        self.con.execute("create table if not exists e (t text primary key, dense blob, sparse text)")

    def encode_queries(self, texts: list[str]):
        have, key = {}, (lambda t: f"{self.ns}\x1f{t}")
        uniq = list(dict.fromkeys(texts))
        for i in range(0, len(uniq), 500):
            part = [key(t) for t in uniq[i:i + 500]]
            q = f"select t, dense, sparse from e where t in ({','.join('?' * len(part))})"
            for t, d, s in self.con.execute(q, part):
                have[t.split("\x1f", 1)[1]] = (np.frombuffer(d, np.float32), json.loads(s))
        todo = [t for t in uniq if t not in have]
        for i in range(0, len(todo), self.batch):  # commit per batch: a tunnel failure late in a run keeps the rest
            part = todo[i:i + self.batch]
            vecs, lex = self.enc.encode_queries(part)
            for t, v, l in zip(part, vecs, lex):
                v = np.asarray(v, np.float32)
                self.con.execute("insert or replace into e values (?,?,?)", (key(t), v.tobytes(), json.dumps(l)))
                have[t] = (v, l)
            self.con.commit()
        return np.stack([have[t][0] for t in texts]) if texts else np.zeros((0, 1024), np.float32), [have[t][1] for t in texts]

    encode = encode_queries
