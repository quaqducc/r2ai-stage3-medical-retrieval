"""Read a SQLite file on GCS without downloading it: apsw VFS that serves pages via HTTP Range requests.
Only for dev tooling (audit, fixture, --remote smoke runs); full runs download the db first."""
from __future__ import annotations

import threading
import urllib.parse

import apsw
import requests

from . import storage

CHUNK = 256 << 10
_S = requests.Session()
_S.mount("https://", requests.adapters.HTTPAdapter(
    max_retries=requests.adapters.Retry(total=5, backoff_factor=1, allowed_methods=None, status_forcelist=(429, 500, 502, 503, 504)), pool_maxsize=16))


def _url(gs: str) -> str:
    bucket, obj = gs[5:].split("/", 1)
    return f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{urllib.parse.quote(obj, safe='')}"


class _File:
    def __init__(self, gs: str):
        self.url, self.cache, self.bytes_read = _url(gs), {}, 0
        r = _S.get(self.url, headers=self._h(), timeout=60)
        r.raise_for_status()
        self.size = int(r.json()["size"])

    @staticmethod
    def _h(extra=None):
        return {"Authorization": "Bearer " + storage.access_token(), **(extra or {})}

    def _chunk(self, i: int) -> bytes:
        if i not in self.cache:
            lo, hi = i * CHUNK, min((i + 1) * CHUNK, self.size) - 1
            r = _S.get(self.url + "?alt=media", headers=self._h({"Range": f"bytes={lo}-{hi}"}), timeout=120)
            r.raise_for_status()
            if len(self.cache) > 512:  # ~128 MB cap
                self.cache.clear()
            self.cache[i] = r.content
            self.bytes_read += len(r.content)
        return self.cache[i]

    def xRead(self, amount, offset):
        out = bytearray()
        while amount > 0 and offset < self.size:
            i = offset // CHUNK
            part = self._chunk(i)[offset - i * CHUNK:][:amount]
            out += part
            offset += len(part)
            amount -= len(part)
        return bytes(out)

    def xFileSize(self): return self.size
    def xClose(self): pass
    def xLock(self, level): pass
    def xUnlock(self, level): pass
    def xCheckReservedLock(self): return False
    def xFileControl(self, op, ptr): return False
    def xSectorSize(self): return 4096
    def xDeviceCharacteristics(self): return 0
    def xSync(self, flags): pass
    def xWrite(self, data, offset): raise apsw.ReadOnlyError("read-only")
    def xTruncate(self, size): raise apsw.ReadOnlyError("read-only")


class _VFS(apsw.VFS):
    def __init__(self):
        self.urls, self.lock = {}, threading.Lock()
        super().__init__("gcsrange", "")

    def xFullPathname(self, name): return name

    def xOpen(self, name, flags):
        n = name.filename() if hasattr(name, "filename") else name
        return _File(self.urls[n.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]])

    def xAccess(self, name, flags): return False
    def xDelete(self, name, syncdir): pass


_vfs: _VFS | None = None
_INIT = threading.Lock()


def connect(gs: str) -> apsw.Connection:
    global _vfs
    with _INIT:
        _vfs = _vfs or _VFS()
    with _vfs.lock:
        key = f"r{len(_vfs.urls)}"
        _vfs.urls[key] = gs
    return apsw.Connection(f"file:{key}?immutable=1", flags=apsw.SQLITE_OPEN_READONLY | apsw.SQLITE_OPEN_URI, vfs="gcsrange")
