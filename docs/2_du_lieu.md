# 2. Tài liệu mô tả dữ liệu

## 2.1 Nguồn dữ liệu

Hệ thống **chỉ dùng dữ liệu BTC cung cấp**. Không dùng dữ liệu ngoài hay nhãn tự gán.

| Dữ liệu | Nội dung | Nguồn |
|---|---|---|
| `links_corpus` | 1.220.598 cặp (`id`, `url`) trên 94 domain: vi 740.598, zh 240.000, en (PubMed) 240.000 | BTC |
| `query.parquet` | 1.200 câu hỏi tiếng Việt (`id`, `query`), dùng cho cả hai vòng public và private | BTC |
| Nội dung trang | Do nhóm tự crawl từ các URL của `links_corpus` | Bước 1 |

## 2.2 Thu thập (bước 1, crawl)

- **Cách tải trang:**
  - Mỗi host tải vào một file SQLite: HTML hoặc XML nén zstd, kèm trạng thái HTTP và nguồn.
  - Thứ tự thử nguồn: tải trực tiếp → trình duyệt headless cho site chặn bot → bản lưu Wayback / CommonCrawl / mirror khi site gốc không còn.
  - Với PubMed, lấy bản ghi XML qua NCBI efetch theo PMID.
- **Độ phủ:**
  - Lấy được 1.217.229 / 1.220.598 URL (99,72%). Chi tiết theo domain: [1-crawl/crawl_stats.csv](../1-crawl/crawl_stats.csv).
  - Phần thiếu chủ yếu ở zysjonline (bị Cloudflare chặn) và article.iiyi (lỗi HTTP 521).
- **Lưu trữ:** `gs://aiguru_crawler/crawl/db/<host>.sqlite` (32 GB) và `crawl/manifest.parquet`.

## 2.3 Làm sạch (bước 2, extract)

Mã nguồn ở `2-extract/`. Tài liệu chi tiết: [2-extract/README.md](../2-extract/README.md), [2-extract/REPORT.md](../2-extract/REPORT.md).

**Thứ tự trích:**
1. parser PubMed XML: tiêu đề, abstract, MeSH, từ khoá;
2. luật XPath riêng cho từng host: thân bài, tiêu đề, phần cần bỏ, mẫu dòng rác;
3. trafilatura, kèm JSON `__NEXT_DATA__` cho site Next.js;
4. lxml.

**Hậu xử lý:**
- Bỏ menu, khối "tin liên quan" và đoạn lặp trong cùng trang.
- Giữ nguyên văn, kể cả mọi câu trả lời của trang hỏi đáp.
- Đoán ngôn ngữ thực của văn bản theo loại ký tự.
- Gắn cờ chất lượng: `empty`, `short` (< 300 ký tự), `fallback`, `archive`, `meta_only`, `replacement_chars`…

**Đầu ra:**
- Mỗi id corpus có đúng một dòng: `id`, `url`, `host`, `lang`, `text_lang`, `status`, `title`, `markdown`, `text`, `flags`, `content_sha1`.
- Lưu ở `gs://aiguru_crawler/clean/v1/docs/<host>/part-*.parquet` (2,9 GB).
- `validate` kiểm tra phủ đủ 1.220.598 id và không có id trùng.

**Kiểm thử:** 159 test trên fixture thật của 84 host. Audit chất lượng đo recall đoạn văn so với HTML gốc.

## 2.4 Phân đoạn (bước 3, chunk)

Đặc tả đầy đủ ở [3-ingest/CHUNKING.md](../3-ingest/CHUNKING.md). Tóm tắt:

- **Nguyên văn tuyệt đối:**
  - `chunk_text == join_spans(text, spans)`, nghĩa là mọi chunk nộp đi là chuỗi con của tài liệu.
  - Có test chặn mọi lỗi lệch offset.
- **Phân loại tài liệu bằng luật:** `qa` (diễn đàn hỏi đáp), `pubmed`, `article`.
- **Ranh giới cắt theo thứ tự ưu tiên:** heading → hỏi/đáp → đoạn văn → câu. Không cắt giữa câu.
- **Độ dài:**
  - `article`: 500–1.000 từ (zh: 1,6 chữ Hán = 1 từ).
  - `qa`: câu hỏi + câu trả lời. Trang nhiều câu trả lời dài (> 600 từ) thì tách mỗi câu trả lời một chunk, chunk nào cũng kèm câu hỏi.
  - `pubmed`: tiêu đề + abstract.
- **Bỏ** tài liệu rỗng hoặc lỗi crawl.
- **Kết quả:** 1.572.721 chunk.
- **Lưu ở** `gs://aiguru_crawler/ingest/v1/rag/`:
  - `chunks/<lang>/*.parquet`: `chunk_id` = `<doc_id>:<idx>`, `doc_id`, `idx`, `text`, `spans`, `title`, `heading_path`, `doc_type`…;
  - `docs/<lang>/*.parquet`: thẻ tài liệu;
  - `dense/bge-m3/<lang>/*.npy` + `*.ids.parquet`: vector float16 đã chuẩn hoá;
  - `sparse_index/bge-m3/<lang>/`: chỉ mục sparse CSC.

## 2.5 Hướng dẫn truy cập dữ liệu

Toàn bộ dữ liệu nằm trên Google Cloud Storage, bucket `gs://aiguru_crawler` (project `aiguru26`, region asia-southeast1). Nhóm cấp quyền đọc (Storage Object Viewer) cho tài khoản Google mà BTC cung cấp.

| Đường dẫn | Kích thước | Nội dung | Cần cho |
|---|---|---|---|
| `crawl/db/`, `crawl/manifest.parquet` | 32 GB | trang thô theo host | chạy lại từ bước 2 |
| `clean/v1/` | 2,9 GB | kho văn bản sạch | chạy lại từ bước 3 |
| `ingest/v1/rag/` | 15 GB | chunk, vector dense, chỉ mục sparse | chạy lại từ bước 4 |
| `ingest/v1/rag/chunks/` | phần của dòng trên | chunk nguyên văn | dựng nhanh bài nộp |
| `release/v1/run/plans.json` | 0,6 MB | plan của Qwen3.5-9B cho 1.200 câu hỏi | tái hiện chính xác các danh sách |
| `release/v1/run/lists.pkl` | 32 MB | danh sách top-200 cấp chunk (S2) | |
| `release/v1/run/lists_docprf.pkl` | 92 MB | toàn bộ danh sách dùng cho bài nộp (S2 + PRF + cấp tài liệu) | dựng nhanh bài nộp |
| `release/v1/submissions/*.zip` | 97 MB mỗi file | các file đã nộp: `i_dprf15` (bài báo cáo), `j_hh125`, `i_dprf1`, `d12_both075`, `i_dprf05` | đối chiếu |

**Tải về:**

```bash
gcloud auth login                     # tài khoản đã được cấp quyền đọc
gcloud storage ls gs://aiguru_crawler/release/v1/
gcloud storage cp -r gs://aiguru_crawler/release/v1 ./release
gcloud storage rsync -r gs://aiguru_crawler/ingest/v1/rag/chunks ./ingest/rag/chunks
```

**Đọc trong Python:**

```python
import pandas as pd
docs = pd.read_parquet("gs://aiguru_crawler/clean/v1/docs/www.vinmec.com/")   # cần gcsfs
chunks = pd.read_parquet("ingest/rag/chunks/vi/")                            # sau khi rsync
```
