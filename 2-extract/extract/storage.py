"""Paths are either local or gs://bucket/key. GCS auth: Application Default Credentials (VM service
account), else the active gcloud user's access token (dev laptops without ADC)."""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time

_FS = None
_TOKEN: tuple[str, float] | None = None


def is_gcs(path: str) -> bool:
    return path.startswith("gs://")


def access_token() -> str:
    """Bearer token for raw GCS HTTP calls (remote_sqlite). Cached for 45 minutes."""
    global _TOKEN
    if _TOKEN and time.time() - _TOKEN[1] < 2700:
        return _TOKEN[0]
    if os.environ.get("GCS_ACCESS_TOKEN"):  # dev: short-lived token passed at runtime, e.g. into a local container
        return os.environ["GCS_ACCESS_TOKEN"]
    try:
        import google.auth
        import google.auth.transport.requests
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/devstorage.read_write"])
        creds.refresh(google.auth.transport.requests.Request())
        tok = creds.token
    except Exception as e:  # no ADC on this machine -> gcloud user login
        logging.getLogger("extract").info("no ADC (%s: %s); using gcloud user token", type(e).__name__, e)
        tok = subprocess.check_output("gcloud auth print-access-token", shell=True, text=True).strip()
    _TOKEN = (tok, time.time())
    return tok


def _gcs():
    global _FS
    if _FS is None:
        import gcsfs
        if os.environ.get("GCS_ACCESS_TOKEN"):
            from google.oauth2.credentials import Credentials
            _FS = gcsfs.GCSFileSystem(token=Credentials(os.environ["GCS_ACCESS_TOKEN"]))
            return _FS
        try:
            import google.auth
            google.auth.default()
            _FS = gcsfs.GCSFileSystem(token="google_default")
        except Exception:
            from google.oauth2.credentials import Credentials
            _FS = gcsfs.GCSFileSystem(token=Credentials(access_token()))
    return _FS


def _strip(path: str) -> str:
    return path[5:] if is_gcs(path) else path


def join(*parts: str) -> str:
    return "/".join(p.rstrip("/") for p in parts)


def exists(path: str) -> bool:
    return _gcs().exists(_strip(path)) if is_gcs(path) else os.path.exists(path)


def read_text(path: str) -> str:
    if is_gcs(path):
        with _gcs().open(_strip(path), "r", encoding="utf-8") as f:
            return f.read()
    with open(path, encoding="utf-8") as f:
        return f.read()


def write_text(path: str, text: str) -> None:
    if is_gcs(path):
        with _gcs().open(_strip(path), "w", encoding="utf-8") as f:
            f.write(text)
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def write_json(path: str, obj) -> None:
    write_text(path, json.dumps(obj, ensure_ascii=False, indent=1, default=str))


def read_json(path: str):
    return json.loads(read_text(path))


def rm_tree(path: str) -> None:
    if is_gcs(path):
        fs = _gcs()
        if fs.exists(_strip(path)):
            fs.rm(_strip(path), recursive=True)
    elif os.path.isfile(path):
        os.remove(path)
    elif os.path.exists(path):
        shutil.rmtree(path)


def put(local: str, dst: str) -> None:
    if is_gcs(dst):
        _gcs().put(local, _strip(dst))
    else:
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        shutil.copyfile(local, dst)


def download(src: str, dst_dir: str) -> str:
    """Local path of src: local files are used in place, gs:// objects are copied into dst_dir."""
    if not is_gcs(src):
        return src
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(src))
    try:
        _gcs().get(_strip(src), dst + ".part")
        want, got = size(src), os.path.getsize(dst + ".part")
        if want != got:
            raise IOError(f"truncated download {src}: {got} of {want} bytes")
        os.replace(dst + ".part", dst)
    finally:
        if os.path.exists(dst + ".part"):
            os.remove(dst + ".part")
    return dst


def glob(pattern: str) -> list[str]:
    if is_gcs(pattern):
        return ["gs://" + p for p in _gcs().glob(_strip(pattern))]
    import glob as g
    return sorted(g.glob(pattern))


def size(path: str) -> int:
    return _gcs().size(_strip(path)) if is_gcs(path) else os.path.getsize(path)


def filesystem(path: str):
    """(pyarrow-compatible filesystem or None, path) for pyarrow readers."""
    return (_gcs(), _strip(path)) if is_gcs(path) else (None, path)
