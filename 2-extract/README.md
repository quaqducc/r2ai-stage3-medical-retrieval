# 2 · Extract: warehouse crawl → kho markdown sạch

Pipeline này chuyển 94 file SQLite thô (`gs://aiguru_crawler/crawl/db/<host>.sqlite`, tổng 34,8 GB, HTML/XML nén zstd) thành một kho parquet sạch. Trong kho, **mỗi id của corpus (1.220.598 id) có đúng một dòng**, gồm markdown và text đã làm sạch, sẵn sàng cho bước chia chunk, đánh index BM25/dense và cho agent truy hồi.

Laptop chỉ dùng để phát triển và test trên mẫu nhỏ. Bản chạy đầy đủ dùng Docker trên VM GCP.

## Đầu ra

```
<out>/                          vd. gs://aiguru_crawler/clean/v1
  docs/<host>/part-00000.parquet   ≤ 50.000 dòng/shard, nén zstd
  hosts/<host>.json                thống kê host (status, parser, flags, rates, suspect); có file này = host đã xong
  hosts/<host>.failed.json         chỉ có khi host lỗi: traceback để xem lại sau
  reports/coverage.csv             mỗi host: manifest, rows, ok, usable, empty, short, lỗi parse, id thiếu, id trùng
  reports/summary.json             tổng hợp; "complete": true nghĩa là phủ đủ corpus, "suspect_hosts": host có dấu hiệu hỏng hàng loạt
```

Schema của mỗi dòng:

| Cột | Ý nghĩa |
|---|---|
| `id` | id trong `links_corpus` (khóa dùng để nộp bài) |
| `url`, `host`, `lang`, `lane` | lấy từ `crawl/manifest.parquet` (`lang` tính theo host) |
| `text_lang` | ngôn ngữ thực của nội dung, đoán theo loại ký tự: `vi`, `zh`, `en`, `mixed` |
| `status` | `ok`, hoặc trạng thái crawl (`notfound`, `blocked`, `error`, `invalid`, `home`, `robots`…), hoặc `missing` (id có trong corpus nhưng không có trong db) |
| `source` | nguồn crawl: `direct`, `stealth`, `wayback_direct`, `wayback`, `commoncrawl`, `mirror` |
| `title` | tiêu đề trang |
| `markdown` | nội dung chính dạng markdown (heading, đoạn, danh sách, bảng) |
| `text` | `markdown` đã bỏ ký hiệu markdown, giữ nguyên chữ và thứ tự. **Dùng cột này để cắt `chunk_text`**, vì đề bài yêu cầu đoạn trích nguyên văn |
| `n_chars` | độ dài `text` |
| `parser` | bộ trích đã dùng: `rule` (luật riêng của host), `trafilatura`, `next_data`, `pubmed_xml`, `lxml`, `meta` (chỉ còn tiêu đề và mô tả của trang), `none` (trang không có nội dung), `error:<Loại>` |
| `flags` | `empty`; `short` (dưới 300 ký tự); `fallback` (luật của host không khớp, phải dùng cách trích chung); `archive` (trang lấy từ bản lưu); `meta_only` (thân bài bị mã hoá phía client, chỉ giữ tiêu đề và sapo); `http_<mã>` (lúc crawl nhận mã HTTP khác 200); `replacement_chars` (nghi sai charset); `heavy_drop` (luật bỏ dòng xoá hơn 30% trang); `rule_error`, `tree_none`, `extractor_error` (lý do rơi khỏi đường trích chính) |
| `content_sha1` | sha1 của `text`, dùng để khử trùng lặp về sau |
| `pipeline_version` | phiên bản pipeline (`v1`) |

Đọc dữ liệu:

```python
import pandas as pd
df = pd.read_parquet("gs://aiguru_crawler/clean/v1/docs/")                 # toàn bộ (cần gcsfs)
use = df[(df.status == "ok") & ~df["flags"].map(lambda f: "empty" in f or "short" in f)]
```

Gợi ý cho bước chunking và index:
- Cắt `chunk_text` từ `text`. Ranh giới đoạn là `\n\n`; heading vẫn còn trong `markdown`.
- Doc có cờ `fallback` hoặc `meta_only` thì giảm trọng số.
- Với PubMed, dòng đầu của `text` là tiêu đề bài.
- Đoạn văn lặp lại trong cùng một doc đã được bỏ: chỉ giữ lần xuất hiện đầu.

## Chạy đầy đủ trên VM GCP

Cấu hình VM:
- Đề xuất 16 vCPU, 32 GB RAM, ổ đĩa 100 GB, cùng region với bucket.
- Service account của VM cần quyền đọc `crawl/` và quyền ghi `clean/`.
- Chi phí CPU khoảng 35 ms mỗi tài liệu, tức khoảng 12 giờ CPU cho cả corpus. Ước tính chạy khoảng 1 giờ trên 16 vCPU, hoặc chia cho nhiều VM bằng `--shard`.

```bash
# 1. Build image ngay trên VM (copy thư mục main-pipeline/2-extract lên)
docker build -t r2ai-extract:v1 .
sudo mkdir -p /mnt/work && sudo chown 1000:1000 /mnt/work     # container chạy với uid 1000

# 2. Kiểm thử trên VM trước: test + vài host nhỏ
docker run --rm --entrypoint python r2ai-extract:v1 -m pytest -q -p no:cacheprovider
docker run --rm -v /mnt/work:/work r2ai-extract:v1 run --out gs://aiguru_crawler/clean/v1 \
  --workdir /work --hosts special.vietnamplus.vn,www.familydoctor.cn

# 3. Chạy toàn bộ (để chạy nền: thêm -d hoặc dùng tmux)
docker run --rm -v /mnt/work:/work r2ai-extract:v1 run --out gs://aiguru_crawler/clean/v1 --workdir /work
#    Chạy trên 2 VM: thêm --shard 0/2 trên máy thứ nhất, --shard 1/2 trên máy thứ hai (host được chia cân theo kích thước)

# 4. Kiểm định độ phủ (thoát mã 1 nếu thiếu hoặc trùng id)
docker run --rm r2ai-extract:v1 validate --out gs://aiguru_crawler/clean/v1
```

Cách chạy tiếp và chạy lại:
- Host nào đã có `hosts/<host>.json` thì được bỏ qua. Nếu VM bị dừng giữa chừng, chạy lại đúng lệnh cũ: host đang dở sẽ bị xóa và làm lại từ đầu.
- `--force` buộc làm lại cả những host đã xong, ví dụ sau khi sửa luật trích của host đó.
- Muốn tạo một phiên bản mới mà không ghi đè, đổi `--out` (vd. `clean/v2`).

## Phát triển và test trên local

```bash
pip install -r requirements-dev.txt
python -m pytest -q                                        # toàn bộ test, dùng fixture trong tests/fixtures/db
python -m extract audit --hosts www.120ask.com --n 6 --dump ../../_audit   # đo chất lượng trên trang thật, đọc từ xa
python -m extract run --remote --limit 20 --out ../../_smoke --hosts vnexpress.net   # chạy thử nhỏ, không tải db
```

- `audit` lấy ngẫu nhiên vài trang của host qua HTTP range (không tải cả db), chạy pipeline và đo **recall**: tỷ lệ các đoạn văn dài trong HTML còn giữ lại trong `text`. Với `--dump`, nó ghi ra `.html`, `.md` và `.missed.txt` để xem tận mắt.
- `fixture --host <h> --n 2` chép vài trang thật vào `tests/fixtures/db/<h>.sqlite` để viết test.
- Trên máy không có ADC, quyền GCS lấy từ `gcloud auth print-access-token`. Khi chạy container ở local, truyền token qua `-e GCS_ACCESS_TOKEN=$(gcloud auth print-access-token)`.

## Thêm hoặc sửa luật cho một host

Luật nằm trong `extract/hosts/{zh_qa,zh_kb,vi_health,vi_news}.py`. Mỗi luật là một `Rule(body=(xpath…), title=…, drop=(…), drop_lines=(regex…))`.

Thứ tự trích như sau:
1. PubMed XML
2. luật riêng của host
3. trafilatura, kèm JSON `__NEXT_DATA__` cho site Next.js
4. lxml

Nếu luật của host không khớp với trang (site đổi giao diện), pipeline tự chuyển sang trafilatura và gắn cờ `fallback`. Nhờ vậy có thể đếm cờ này trong `hosts/<host>.json` để biết luật nào cần sửa.

Quy trình chi tiết cho người viết luật nằm trong `AGENT_BRIEF.md`. Kết quả đánh giá chất lượng nằm trong `REPORT.md`.
