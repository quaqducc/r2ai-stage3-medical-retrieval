"""Parquet shard writer for the clean store (one directory per host)."""
from __future__ import annotations

import os

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = pa.schema([
    ("id", pa.int64()), ("url", pa.string()), ("host", pa.string()), ("lang", pa.string()), ("text_lang", pa.string()), ("lane", pa.string()),
    ("status", pa.string()), ("source", pa.string()), ("fetched_at", pa.float64()), ("title", pa.string()),
    ("markdown", pa.string()), ("text", pa.string()), ("n_chars", pa.int32()), ("parser", pa.string()),
    ("flags", pa.list_(pa.string())), ("content_sha1", pa.string()), ("pipeline_version", pa.string()),
])


class ShardWriter:
    """Buffers rows, writes zstd parquet files part-00000.parquet, part-00001.parquet, ... into a local dir."""

    def __init__(self, local_dir: str, rows_per_shard: int = 50_000, max_bytes: int = 256 << 20):
        self.dir, self.n, self.max_bytes, self.buf, self.files, self.bytes = local_dir, rows_per_shard, max_bytes, [], [], 0
        os.makedirs(local_dir, exist_ok=True)

    def add(self, row: dict) -> None:
        self.buf.append(row)
        self.bytes += len(row["markdown"]) + len(row["text"]) + len(row["url"])
        if len(self.buf) >= self.n or self.bytes >= self.max_bytes:
            self._flush()

    def _flush(self) -> None:
        if not self.buf:
            return
        path = os.path.join(self.dir, f"part-{len(self.files):05d}.parquet")
        pq.write_table(pa.Table.from_pylist(self.buf, schema=SCHEMA), path, compression="zstd", row_group_size=10_000)
        self.files.append(path)
        self.buf, self.bytes = [], 0

    def close(self) -> list[str]:
        self._flush()
        return self.files
