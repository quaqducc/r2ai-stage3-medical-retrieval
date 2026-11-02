# Brief for host-rule agents (Task 7 of PLAN.md)

Project root: `D:\AI Guru SV\main-pipeline\2-extract` (Windows, Git Bash available, Python 3.9). Run every command from this dir.
The package `extract/` converts crawled pages into clean markdown/text for a RAG system. The text is used for retrieval
and for **verbatim** answer chunks in a Vietnamese medical retrieval competition. Losing real content (forum answers, article body,
tables, Q&A) is the worst failure. Keeping site chrome (menus, "related questions" lists, disclaimers, footers, addresses,
share buttons, ads, comment boxes, "hot news") is the second worst.

## Your job
You own ONE family of hosts: a rule module `extract/hosts/<family>.py`, a test file `tests/test_hosts_<family>.py`,
and fixture dbs `tests/fixtures/db/<host>.sqlite` for your hosts. Do not edit anything else. If the core needs a change
(e.g. a new parser mode), describe it in your final report instead of editing it.

For each of your hosts (biggest first; the doc counts are in your task):
1. Audit: `python -m extract audit --hosts <h1,h2> --n 6 --dump ../../_audit/<family>` and read the dumps
   (`<host>_<id>.html`, `.md` = current clean output, `.missed.txt` = long HTML text blocks absent from output).
   `recall_med` is a heuristic: missed blocks may be junk (footers, related lists), so read them and judge.
2. If the generic output (trafilatura) loses content or keeps junk, write a `Rule` for the host from the real HTML
   (look at several pages: templates vary, e.g. QA pages vs article pages on the same host, old vs new layouts).
   Order `body` selectors so the document reads naturally (question/title context first, then answers/body).
   Prefer stable ids/classes. `drop` removes junk nested inside the body. `drop_lines` (regex, whole line)
   removes leftover chrome lines. Keep `min_chars` so a page where the rule matches nothing falls back to generic.
3. Re-audit with a different sample size (e.g. `--n 8`) until the output is complete and clean, or explain why not.
4. Fixture + test: for every host where you wrote a rule, and for your 3 biggest hosts regardless:
   `python -m extract fixture --host <h> --n 2` (keep each fixture db under ~400 KB; use `--ids` to pick small pages),
   then add tests in `tests/test_hosts_<family>.py` that parse each fixture page with `extract.parsers.parse` plus
   `extract.clean.clean_markdown(doc.markdown, doc.drop_lines)` and assert: the parser used (`rule` when you wrote one),
   at least one exact content sentence you saw in the page is present (for QA: an answer sentence), and at least one known
   junk string is absent. Use `conftest.fixture_rows(host)` to read fixture pages. Run `python -m pytest -q tests/test_hosts_<family>.py`.
5. Never download a whole db; only `audit` and `fixture` (they read a few pages over HTTP range). Keep total
   audit pages per host <= 20.

## API
```python
from ..rules import Rule
RULES = {
    "www.example.com": Rule(
        body=('//h1[@id="title"]', '//div[@class="question"]', '//div[contains(@class,"answer-content")]'),
        title='//h1[@id="title"]',
        drop=('//div[@class="related"]', '//span[@class="report"]'),
        drop_lines=(r"^声明：", r"^Xem thêm"),
        min_chars=80,
    ),
}
```
- XPath 1.0 via lxml (`contains(concat(" ", normalize-space(@class), " "), " x ")` for exact class tokens).
- Body nodes are converted by `extract/htmlmd.py` (headings, paragraphs, lists, tables; script/style/nav/form/button dropped).
- Lookup is by exact host. Do not use suffix keys (leading dot) unless every subdomain shares the template;
  never let a suffix key capture a host owned by another family (e.g. `ask.39.net` belongs to zh_qa, `jbk.39.net` to zh_kb).
- A rule object can be reused for several hosts (`for h in (...): RULES[h] = R`).
- Python 3.9: no `match`, no `X | Y` types at runtime.

## Final report (your last message), concise
Per host: rule yes/no, recall before→after (audit), one line of what was wrong, anything left unsolved. Then: files you
created/changed, test result line, and any core change you need (with the reason and an example id).
