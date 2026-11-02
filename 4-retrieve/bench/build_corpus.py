"""Small but real benchmark corpus: random ok pages of the biggest hosts (remote HTTP-range reads, no db download),
then the real pipeline: 2-extract -> 3-ingest chunk -> embed (bge-m3 server) -> sparse index.
Usage: python bench/build_corpus.py <out_dir> <bge-m3 server url>"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MP = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(MP, "2-extract"), os.path.join(MP, "3-ingest")]

from extract import storage  # noqa: E402
from extract.__main__ import main as extract_main  # noqa: E402
from extract.fixture import copy_rows  # noqa: E402
from ingest import rag  # noqa: E402

# roughly the corpus mix (vi 61% / zh 20% / en 20%), capped by what the embedding server can do in about an hour
SAMPLE = {
    "pubmed.ncbi.nlm.nih.gov": 250, "suckhoecongdongonline.vn": 150, "www.120ask.com": 200, "suckhoedoisong.vn": 100,
    "nhathuoclongchau.com.vn": 100, "thanhnien.vn": 80, "www.cnkang.com": 80, "ask.39.net": 150, "laodong.vn": 40,
    "phunusuckhoe.giadinhonline.vn": 40, "www.familydoctor.com.cn": 60, "www.vinmec.com": 60, "medlatec.vn": 40,
    "hellobacsi.com": 40, "www.msdmanuals.cn": 30, "www.youlai.cn": 30, "zhongyibaodian.net": 30, "jbk.39.net": 30,
    "tamanhhospital.vn": 20, "vnexpress.net": 20,
}
MANIFEST = os.path.join(MP, "..", "data", "crawl_report", "manifest.parquet")


def main(out, server):
    db_root = os.path.join(out, "db")
    for h, n in SAMPLE.items():
        if not os.path.exists(os.path.join(db_root, f"{h}.sqlite")):
            copy_rows(h, "gs://aiguru_crawler/crawl/db", db_root, n=n, with_bad=False)
            print("sampled", h, n, flush=True)
    import sqlite3
    ids = {i for h in SAMPLE for (i,) in sqlite3.connect(os.path.join(db_root, f"{h}.sqlite")).execute("select id from pages")}
    man = pd.read_parquet(MANIFEST)
    mini = os.path.join(out, "manifest.parquet")
    man[man.id.isin(ids)].to_parquet(mini)
    extract_main(["run", "--db-root", db_root, "--manifest", mini, "--out", os.path.join(out, "clean"),
                  "--workers", "4", "--workdir", os.path.join(out, "w")])
    ing = os.path.join(out, "ingest")
    print(rag.run_chunk(os.path.join(out, "clean"), ing, workers=4, workdir=os.path.join(out, "w2")), flush=True)
    print(rag.run_embed(ing, rag.HttpEncoder(server, batch_size=16), workdir=os.path.join(out, "w3")), flush=True)
    print(rag.run_index(ing, workdir=os.path.join(out, "w4")), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
