"""Clean store (2-extract) -> RAG artifacts: verbatim chunks, per-language BM25, dense embeddings."""
import os
import sys

try:  # reuse 2-extract (storage, clean.to_text); in Docker both packages sit in /app
    import extract  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "2-extract"))

VERSION = "v1"
