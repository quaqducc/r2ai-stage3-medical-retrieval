import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRACT = os.path.join(os.path.dirname(ROOT), "2-extract")
sys.path[:0] = [ROOT, EXTRACT]
FIXTURES = os.path.join(EXTRACT, "tests", "fixtures", "db")


def clean_doc(host, doc_id=None):
    """A clean-store row (as written by 2-extract) built from a committed 2-extract fixture page."""
    from extract.pipeline import build_record
    from extract.source import iter_rows
    rows = [r for r in iter_rows(os.path.join(FIXTURES, f"{host}.sqlite")) if r.kind == "ok"]
    r = next(x for x in rows if doc_id is None or x.id == doc_id)
    return build_record(r, host, (r.url, {"pubmed.ncbi.nlm.nih.gov": "en"}.get(host, "zh" if host.endswith((".cn", "39.net", "120ask.com")) else "vi"), "lane"))
