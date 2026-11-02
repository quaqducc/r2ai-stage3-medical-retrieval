import glob
import json
import os

import pandas as pd
import pytest

from conftest import FIXTURES
from extract.__main__ import _hosts_for, main

FIXTURE_HOSTS = sorted(os.path.basename(p)[:-7] for p in glob.glob(os.path.join(FIXTURES, "db", "*.sqlite")))


@pytest.fixture
def mini_manifest(tmp_path):
    """Manifest = every row of every committed fixture db + one id per host that no db has."""
    import sqlite3
    rows = []
    for h in FIXTURE_HOSTS:
        con = sqlite3.connect(os.path.join(FIXTURES, "db", f"{h}.sqlite"))
        ids = [r[0] for r in con.execute("select id from pages")]
        rows += [(i, f"u{i}", h, "vi", "lane") for i in ids] + [(10**9 + len(rows), "gone", h, "vi", "lane")]
    path = str(tmp_path / "manifest.parquet")
    pd.DataFrame(rows, columns=["id", "url", "host", "lang", "lane"]).to_parquet(path)
    return path


def test_run_then_validate_complete(mini_manifest, tmp_path):
    out = str(tmp_path / "out")
    main(["run", "--db-root", os.path.join(FIXTURES, "db"), "--manifest", mini_manifest, "--out", out,
          "--workers", "1", "--workdir", str(tmp_path / "w")])
    main(["validate", "--manifest", mini_manifest, "--out", out])  # exits 1 on any gap
    s = json.load(open(os.path.join(out, "reports", "summary.json")))
    assert s["complete"] and s["parse_errors"] == 0 and s["ok"] > 0
    cov = pd.read_csv(os.path.join(out, "reports", "coverage.csv"))
    assert set(cov.host) == set(FIXTURE_HOSTS)


def test_validate_fails_on_missing_row(mini_manifest, tmp_path):
    out = str(tmp_path / "out")
    host = FIXTURE_HOSTS[0]
    main(["run", "--db-root", os.path.join(FIXTURES, "db"), "--manifest", mini_manifest, "--out", out,
          "--workers", "1", "--workdir", str(tmp_path / "w"), "--hosts", host])
    with pytest.raises(SystemExit) as e:
        main(["validate", "--manifest", mini_manifest, "--out", out])
    assert e.value.code == 1


def test_shards_are_balanced_and_disjoint():
    man = pd.DataFrame({"host": ["a"] * 50 + ["b"] * 30 + ["c"] * 20 + ["d"] * 10})
    parts = [_hosts_for(man, None, f"{i}/2") for i in range(2)]
    assert sorted(parts[0] + parts[1]) == ["a", "b", "c", "d"] and not set(parts[0]) & set(parts[1])
    assert parts[0] == ["a", "d"] and parts[1] == ["b", "c"]


def test_unknown_host_rejected():
    with pytest.raises(SystemExit):
        _hosts_for(pd.DataFrame({"host": ["a"]}), "a,zzz", None)
