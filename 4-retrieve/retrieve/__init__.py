"""Queries -> submission.json over the 3-ingest artifacts (bge-m3 dense + sparse)."""
import os
import sys

_MP = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _pkg, _dir in (("ingest", "3-ingest"), ("extract", "2-extract")):
    try:
        __import__(_pkg)
    except ImportError:  # dev checkout: sibling folders; in Docker all packages sit in /app
        sys.path.insert(0, os.path.join(_MP, _dir))
