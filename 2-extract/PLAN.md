# Extract: crawl warehouse to clean markdown store. Implementation plan

> **For agentic workers:** core tasks (1–6) are built inline by the lead; host-rule tasks (7a–7d) run as parallel subagents; Task 8 is review + end-to-end verification. Not a git repo: there are no commit steps.

**Goal:** Turn `gs://aiguru_crawler/crawl/db/<host>.sqlite` (94 merged per-host SQLite files, 34.8 GB, raw zstd HTML/XML) into a clean store: parquet shards in which every corpus id has exactly one row with clean markdown and plain text, ready for chunking, BM25/dense indexing and agent retrieval.

**Architecture:**
- A Python package `extract`, run as `python -m extract`, ships in a Docker image and runs on a GCP VM.
- For each host, the pipeline downloads the db to local disk, streams rows, and parses each page in a process pool.
- Parser order: a per-host rule (XPath) if one exists, otherwise a format parser (PubMed XML, Next.js `__NEXT_DATA__`), otherwise trafilatura, then an lxml fallback.
- Output is cleaned, then written to zstd parquet shards with one stats JSON per host (used to resume).
- `validate` proves corpus coverage against `crawl/manifest.parquet`.
- `audit` measures, on a few remote pages per host, which long paragraphs of the HTML the extraction lost. It is how rules get written and verified without downloading the warehouse.

**Tech stack:** Python 3.9+ (local 3.9, image 3.11), trafilatura 2.0, lxml, zstandard, pyarrow, pandas, gcsfs, apsw (remote range reads for audit/smoke only), pytest.

**Spec:** the conversation decisions, recorded here:
- The clean store lives at `gs://aiguru_crawler/clean/<version>/`. Admin access is pending, so local runs write to a local dir.
- Format is parquet, not one `.md` per document.
- The laptop only builds and tests on small samples. It never runs the full corpus.
- The full run is done by the team, on a GCP VM, with Docker.

## Global constraints
- Every manifest id (1,220,598) appears exactly once in `docs/`; ids absent from the db get `status="missing"`.
- Never rewrite content: `markdown`/`text` are extracted verbatim. Competition `chunk_text` must be verbatim from the source.
- Local runs never download a db over 200 MB; local smoke uses `--remote --limit`.
- Code runs on Python 3.9 (`from __future__ import annotations`, no `match`).
- Output schema (pyarrow), in this order: `id int64, url string, host string, lang string, lane string, status string, source string, fetched_at float64, title string, markdown string, text string, n_chars int32, parser string, flags list<string>, content_sha1 string, pipeline_version string`.
- Flags vocabulary: `empty` (n_chars == 0), `short` (n_chars < 300), `fallback` (generic parser used after a rule/format parser failed), `archive` (source in wayback/wayback_direct/commoncrawl/mirror).

## Review focus
1. A gb2312-labelled page holding gb18030-only characters: decode with gb18030 and never crash; `errors="replace"` is the last resort.
2. A host rule whose XPath matches nothing (template changed): fall back to generic with flag `fallback`, never return empty silently. Test: rule test with a non-matching page.
3. A page whose parser raises: the row is still emitted (`status="ok"`, empty text, flags `empty`, parser `error:<Type>`) and counted in host stats. One bad page must not kill a host.
4. An interrupted host (VM preempted mid-host): rerunning must not leave duplicate or partial shards. Shards are written to a temp prefix and renamed only after the host completes; the stats JSON is written last.
5. A duplicate id in the db (same id twice): emit one row, keeping the `ok` one. The validate test covers duplicates.

---

### Task 1: Storage + source
**Files:** `extract/storage.py`, `extract/source.py`, `tests/test_source.py`.
**Produces:**
- `storage.fs_for(path) -> (fsspec fs, path)`. gs:// uses gcsfs with ADC, falling back to a `gcloud auth print-access-token` token.
- `storage.download(src, dst_dir) -> str`
- `storage.put(local, dst)`
- `storage.exists(path) -> bool`
- `storage.write_text(path, s)`, `storage.read_parquet(path, columns)`
- `source.Row` (dataclass: id, url, final_url, status, kind, source, charset, ctype, fetched_at, blob: bytes|None)
- `source.iter_rows(db: str, limit: int|None = None, remote: bool = False) -> Iterator[Row]`. Ordered by id, deduped by id, preferring kind `ok`. Remote mode uses `extract.remote_sqlite` (apsw HTTP-range VFS).
- `source.decode(blob, charset, ctype) -> str`. zstd-decompress, then the charset map gb2312/gbk → gb18030, utf-8 default, `replace` as the last resort.
**Tests:** decode a gb2312 fixture; ids are deduped preferring ok; `limit` respected.

### Task 2: HTML→markdown + parsers
**Files:** `extract/htmlmd.py`, `extract/parsers.py`, `extract/rules.py`, `tests/test_parsers.py`.
**Produces:**
- `htmlmd.to_markdown(nodes: list[lxml element]) -> str`. Handles h1–h6 → `#`, p/div/br blocks, li → `- `, table → pipe rows, and inline text. It drops script/style/noscript/iframe/form/button/svg.
- `rules.Rule(body: tuple[str,...], title: str|None=None, drop: tuple[str,...]=(), drop_lines: tuple[str,...]=(), min_chars: int=50)`. XPath selectors.
- `rules.RULES: dict[str, Rule]`, merged from `extract/hosts/*.py` (each module exports `RULES`).
- `parsers.Doc(title: str, markdown: str, parser: str, fallback: bool)`
- `parsers.parse(host: str, url: str, html: str) -> Doc`. Order:
  1. PubMed (`<PubmedArticle`)
  2. A rule, if the host has one (yields < `min_chars` → continue with `fallback=True`)
  3. Next.js (`__NEXT_DATA__`) when the generic result is under 300 chars
  4. trafilatura markdown (`favor_recall`, tables, comments)
  5. lxml body fallback
- Title resolution: rule title, then og:title, then h1, then `<title>`.
**Tests:**
- PubMed XML yields title, abstract and MeSH.
- msdmanuals yields the Next.js path.
- A matching rule yields its body and drops its `drop` nodes.
- A non-matching rule falls back with `fallback=True`.
- Broken HTML (baotayninh fixture) is non-empty.

### Task 3: Clean
**Files:** `extract/clean.py`, `tests/test_clean.py`.
**Produces:**
- `clean.clean_markdown(md: str, drop_lines: tuple[str,...]=()) -> str`. NFC, zero-width removal, whitespace collapse, global and per-rule boilerplate line regexes, consecutive-duplicate removal, at most one blank line.
- `clean.to_text(md: str) -> str`. Strips markdown syntax (`#`, `**`, `- `, table pipes), keeping words in order.
- `clean.flags(text, source, fallback) -> list[str]`
**Tests:** boilerplate lines removed; markdown emphasis stripped in text; flags thresholds.

### Task 4: Pipeline + writer
**Files:** `extract/writer.py`, `extract/pipeline.py`, `tests/test_pipeline.py`.
**Produces:**
- `writer.SCHEMA`
- `writer.ShardWriter(out_dir, host, rows_per_shard=50_000)`, with `.add(dict)`, `.close() -> list[str]`. It writes `local shards that are uploaded to docs/<host>/` and then moves them into place on close.
- `pipeline.process_host(db, host, manifest: DataFrame, out: str, workers: int, limit: int|None, remote: bool) -> dict`. The stats dict has total, ok, statuses, parsers, flags, seconds.
  - Emits one row per db id plus a `missing` row per manifest id absent from the db (only when `limit is None`).
  - Writes `hosts/<host>.json` last.
- A parse error becomes a row with parser `error:<Type>`.
**Tests:** a fixture db and a mini manifest produce one row per id; a missing id is emitted; an error page is emitted; a rerun of a completed host is skipped.

### Task 5: Validate + CLI
**Files:** `extract/validate.py`, `extract/__main__.py`, `tests/test_cli.py`.
**Produces:**
- `validate.run(manifest, out) -> dict`. Writes `reports/coverage.csv` (per host: manifest, rows, ok, empty, short, missing, duplicate ids) and `reports/summary.json`. The CLI exits 1 on any missing or duplicate id.
- CLI subcommands:
  - `run --db-root --manifest --out [--hosts] [--shard i/n] [--workers] [--workdir] [--limit] [--remote] [--force]`
  - `validate --manifest --out`
  - `audit --hosts --n --db-root`
  - `fixture --host --n --db-root --dest`
**Tests:** CLI `run` and then `validate` on fixtures exit 0; deleting one row makes validate exit 1.

### Task 6: Audit + fixtures tooling, Docker
**Files:** `extract/remote_sqlite.py`, `extract/audit.py`, `Dockerfile`, `requirements.txt`, `.dockerignore`, `README.md` (Vietnamese).
**Produces:**
- `audit.paragraph_recall(html, text) -> (recall: float, missed: list[str])`. Blocks are p/li/td/dd/div direct text of at least 40 chars, or 20 CJK chars, excluding nav/header/footer/aside/script.
- `audit.run(hosts, n, db_root) -> DataFrame`, with per-host columns median recall, median chars, parser mix, and worst missed block.
- `fixture`: copies n ok rows (+1 non-ok) of a host from remote into `tests/fixtures/db/<host>.sqlite` with the same schema.
- Docker: `python:3.11-slim`, `ENTRYPOINT ["python","-m","extract"]`. Build it locally and run `run` and `validate` on the fixtures inside the container.

### Tasks 7a–7d: Host rules (parallel subagents)
Each agent owns one module and its tests: `extract/hosts/zh_qa.py`, `zh_kb.py`, `vi_health.py`, `vi_news.py`, with `tests/test_hosts_<family>.py` and fixtures under `tests/fixtures/db/`.

Procedure per host:
1. `audit` with n ≥ 6.
2. If median recall < 0.85 or boilerplate remains, write a `Rule` (XPath on the real page), add a fixture, and add a test asserting a known answer/body sentence is present and a known junk string is absent.
3. Re-audit until recall ≥ 0.85, or document why not.

Agents never edit core files; they report core changes they need.

### Task 8: Review + end-to-end verification
- Run python-reviewer, silent-failure-hunter and rag-pipeline-reviewer in parallel; fix what they confirm.
- Run the full pytest suite.
- Run `run --remote --limit 20` across all 94 hosts to a local dir, then the audit summary.
- Write `main-pipeline/2-extract/REPORT.md` (Vietnamese) covering results, per-host recall, known gaps, and the VM run command.
