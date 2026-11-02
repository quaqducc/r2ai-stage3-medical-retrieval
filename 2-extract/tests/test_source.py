from extract.source import decode, iter_rows

import zstandard


def z(b):
    return zstandard.ZstdCompressor().compress(b)


def test_decode_gb2312_label_with_gb18030_only_char():
    # '镕' is not in GB2312 but is in GB18030; servers often label such pages gb2312
    raw = "<p>朱镕基 医生</p>".encode("gb18030")
    assert "朱镕基" in decode(z(raw), "gb2312", "text/html")


def test_decode_charset_from_ctype_and_meta():
    raw = "<html><head><meta charset='gbk'></head><p>头痛</p>".encode("gbk")
    assert "头痛" in decode(z(raw), None, "text/html; charset=GBK")
    assert "头痛" in decode(z(raw), None, "text/html")  # sniffed from <meta>


def test_decode_bad_bytes_never_raise():
    out = decode(z(b"<p>ok \xff\xfe broken</p>"), "utf-8", None)
    assert "ok" in out and "broken" in out


def test_decode_strips_wayback_toolbar():
    html = b"<html><body><!-- BEGIN WAYBACK TOOLBAR INSERT --><div>WAYBACK JUNK</div><!-- END WAYBACK TOOLBAR INSERT --><p>real</p></body></html>"
    out = decode(z(html), "utf-8", None)
    assert "WAYBACK JUNK" not in out and "real" in out


def test_iter_rows_ordered_with_limit(db_factory):
    db = db_factory("h.sqlite", [
        {"id": 3, "url": "u3", "kind": "ok", "html": "<p>c</p>"},
        {"id": 1, "url": "u1", "kind": "notfound"},
        {"id": 2, "url": "u2", "kind": "ok", "html": "<p>b</p>", "source": "wayback_direct"},
    ])
    rows = list(iter_rows(db))
    assert [r.id for r in rows] == [1, 2, 3]
    assert rows[0].blob is None and rows[1].source == "wayback_direct"
    assert [r.id for r in iter_rows(db, limit=2)] == [1, 2]
