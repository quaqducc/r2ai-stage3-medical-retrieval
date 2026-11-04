# 1 · Crawl: tình trạng (17/10/2026)

> **Mã nguồn crawler:** thành viên phụ trách crawl sẽ bổ sung vào thư mục này (`1-crawl/`).
> Đầu ra của bước này là `gs://aiguru_crawler/crawl/db/<host>.sqlite` (HTML/XML nén zstd, mỗi host một file) và
> `crawl/manifest.parquet`. Đây là đầu vào của bước 2 (`2-extract`), xem [docs/THUYET_MINH.md](../docs/THUYET_MINH.md) mục 3.3.1 và README mục 3.

**Crawl đã xong. Bước tiếp theo là trích văn bản và đánh index.**

- **Đã lấy được 1.217.229 / 1.220.598 URL (99,72%).** Theo mẫu, khoảng 1,206 triệu trang có văn bản dùng được.
- **Domain:** 94 domain, trong đó 92 domain crawl đủ, 2 domain còn thiếu. Về nội dung: 78 tốt, 13 cần xử lý riêng, 3 hỏng.
- **Còn thiếu 3.369 URL**, chủ yếu ở zysjonline (thiếu 1.953 URL, bị Cloudflare chặn) và article.iiyi (632 URL, lỗi HTTP 521). Số này quá nhỏ, không cần cố lấy trước lần nộp bài đầu tiên.

| Ngôn ngữ | Số URL | Lấy được | Chưa crawl | Lỗi khi tải |
|---|---|---|---|---|
| Việt | 740.598 | 739.934 | 0 | 664 |
| Trung | 240.000 | 237.356 | 2.116 | 528 |
| Anh (PubMed) | 240.000 | 239.939 | 0 | 61 |

## Cần xử lý trước khi đánh index

- **Cần parser riêng:**
  - PubMed: dữ liệu lưu dạng XML efetch, chỉ lấy tiêu đề, abstract và MeSH.
  - msdmanuals.cn: nội dung nằm trong JSON `__NEXT_DATA__`.
  - baotayninh: HTML lỗi cấu trúc nên trafilatura trả về rỗng, cần dùng lxml.
  - Hàm xử lý PubMed và msdmanuals đã có sẵn trong `analysis/sample_to_md.py`.
- **Cần cắt boilerplate:** 120ask, vov, baocantho. Các trang này lặp lại menu hoặc khối tin liên quan.
- **Trang ngắn (dưới 300 ký tự):** chiếm 13–20% ở ask.39.net, a-hospital, pmphai, zydcd. Bỏ các trang này, không cần sửa.
- **Lấy từ bản lưu:** phần lớn trang của longchau, dantri, thaythuocvietnam lấy từ Wayback. Vẫn dùng được, nhưng nội dung có thể cũ hơn site thật.
- **Vinmec đã crawl lại** và giờ có nội dung đầy đủ. Lỗi "trang chặn" trước đây không còn.

## File

- `crawl_stats.csv`: mỗi dòng là một domain, gồm các cột:
  - `language`, `domain`, `urls`
  - `crawled` (yes/partial), `crawled_ok`, `ok_pct`, `not_crawled`
  - `fetch_errors`: bị chặn, lỗi tải, link chết
  - `content_ok` (yes / needs work / broken) và `issue`
- **Nguồn số liệu:**
  - Báo cáo độ phủ của nhóm: `gs://aiguru_crawler/crawl/report/coverage.csv` (16/10/2026).
  - Mẫu chất lượng nội dung: `analysis/crawl_quality.py`.
  - Hai nguồn trên được gộp bằng `analysis/crawl_report.py`.

Dashboard: https://claude.ai/artifact/2vD49DN5DZg9fzCRayyJWp
