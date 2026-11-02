# Báo cáo bước 2 · Extract (18/10/2026)

**Pipeline trích nội dung sạch đã dựng xong và đã kiểm thử. Bản chạy đầy đủ chỉ còn chờ quyền ghi vào bucket.**

Pipeline đọc 94 file SQLite thô trong `gs://aiguru_crawler/crawl/db/` và ghi ra kho parquet sạch (`clean/v1`):
- Mỗi id của corpus có đúng một dòng.
- Mỗi dòng có `markdown` và `text` nguyên văn, kèm các cờ chất lượng.

Laptop không xử lý corpus: chỉ kéo vài trang mẫu cho mỗi host qua HTTP range.

## Đã kiểm chứng (số liệu đo ngày 18/10)

| Kiểm tra | Kết quả |
|---|---|
| Bộ test (local Python 3.9 và trong container Python 3.11) | 159/159 pass, trên fixture thật của 84 host |
| Audit: 5 trang ngẫu nhiên × 94 host, đọc từ GCS | 94/94 host chạy được, 0 lỗi. 403/449 trang đi qua luật riêng của host, 9 trang rơi về cách trích chung. Chi tiết: `reports/audit_2026-10-18.csv` |
| Smoke end-to-end: 15 dòng × 94 host | 1.329 dòng, 0 lỗi parse, 0 host thất bại, ngôn ngữ phát hiện khớp 100% với ngôn ngữ của host |
| Chạy trong Docker theo luồng VM: tải db thật từ GCS, đọc manifest 1.220.598 id | 4 host chạy trọn: 208 dòng, 0 lỗi. `validate` báo đúng là chưa phủ đủ, vì mới chạy 4 host |
| Tốc độ | trung bình 35 ms/trang (p95 98 ms), tức khoảng 12 giờ CPU cho cả corpus |

## Cách trích theo loại nguồn

| Loại nguồn | Số tài liệu | Cách trích |
|---|---|---|
| PubMed (XML efetch) | 240.000 | parser XML. `text` mở đầu bằng tiêu đề, rồi đến abstract, MeSH, keywords |
| Báo và cổng sức khỏe tiếng Việt (50 host) | 740.598 | luật XPath riêng cho 44 host. Các host còn lại (dantri, baochinhphu, vov2, baoquangtri…) dùng trafilatura vì đã đủ sạch |
| Diễn đàn hỏi đáp và bài viết tiếng Trung (120ask, ask.39.net, cnkang, familydoctor…) | ~200.000 | luật riêng cho 8/12 host (4 host nhỏ dùng trafilatura), **giữ mọi câu trả lời**. Bản trích chung trước đây làm mất câu trả lời của bác sĩ |
| Tri thức và sách Đông y tiếng Trung (zysj, MediaWiki, *.39.net, msdmanuals…) | ~37.000 | luật riêng cho 31 host. msdmanuals lấy từ HTML đã render |
| Trang Next.js (Long Châu `/benh/`) | ~1.200 | luật cộng thêm các mục chỉ có trong JSON `__NEXT_DATA__` (nguy cơ, chẩn đoán, điều trị, phòng ngừa) |
| Trang không có nội dung (lỗi, bị chặn, chưa crawl) | ~3.400 | vẫn có dòng, với `status` tương ứng và text rỗng |

Về con số recall trong file audit: đây là heuristic tính *mọi* đoạn văn dài trong HTML là nội dung, kể cả menu, tin liên quan và footer. Vì vậy host nào bỏ được nhiều phần thừa thì recall càng thấp. Các agent viết luật đã đọc từng file `.missed.txt`. Tôi cũng tự kiểm lại 120ask và ask.39.net: phần bị bỏ chỉ là breadcrumb và câu hỏi liên quan; câu hỏi và câu trả lời của bác sĩ còn đủ.

## Cơ chế chống lỗi âm thầm (đã có test)

- **Một trang hỏng không làm hỏng cả host:** trang đó vẫn có dòng, với `parser=error:<Loại>`. Mỗi trang có timeout 120 giây; nếu cả host đứng quá 30 phút thì host đó báo lỗi rõ ràng thay vì treo.
- **Chạy tiếp được:** một host chỉ được coi là xong khi đã có `hosts/<host>.json`, file này được ghi sau cùng và có đối chiếu số dòng. Host đang dở sẽ bị xoá và làm lại. Lần chạy thử (`--limit`) không bao giờ được tính là xong.
- **Cảnh báo `suspect`** khi một host có tỷ lệ lỗi, rỗng, fallback hoặc thiếu id vượt ngưỡng, hoặc khi luật của host không được dùng. Cảnh báo hiện trong `hosts/*.json` và `reports/summary.json`. Smoke test đã bắt được một template của zysjonline mà agent chưa gặp lúc audit (699 trang `/articles/misc/`); template này đã được sửa.
- **Các cờ chi tiết:** `fallback`, `meta_only`, `http_<mã>`, `replacement_chars` (nghi sai charset), `heavy_drop`, `rule_error`.
- **Bảo vệ nội dung nguyên văn:**
  - charset: ưu tiên UTF-8, trừ khi nhãn là charset đa byte CJK;
  - `\*` và `> 140/90` trong nguồn được giữ nguyên;
  - bảng dàn trang (layout table) được đọc theo khối;
  - đoạn văn lặp lại (tiêu đề, sapo) chỉ giữ lần xuất hiện đầu.

## Hạn chế đã biết

- **baophutho, baothanhhoa (~2.000 tài liệu):** CMS dhcn mã hoá thân bài phía client. Hiện chỉ giữ được tiêu đề và sapo (`og:description`), có cờ `meta_only`. Muốn lấy đủ thân bài phải viết lại bộ giải mã JS của site.
- **health.people.com.cn:** một số bài chỉ có nội dung dạng ảnh.
- **zysjonline:** thiếu 1.953 URL do lỗi crawl, đã có từ bước 1. Các id này có dòng `status=missing`.
- **jbk.39.net:** chưa gặp trang `/zhengzhuang/` khi audit. Nếu luật trượt thì sẽ có cờ `fallback` và cảnh báo `suspect`.
- **Chưa có đo lường truy hồi thật**, tức chưa đo chunk LCS ≥ 0,4 trên câu hỏi thật. Việc này thuộc bước 3.

## Bước tiếp theo

1. Cấp quyền ghi `gs://aiguru_crawler/clean/` cho service account của VM.
2. Trên VM (16 vCPU, 32 GB RAM, ổ 100 GB, cùng region với bucket):
   - chạy `docker build`, rồi `run --out gs://aiguru_crawler/clean/v1 --workdir /work`, khoảng 1 giờ;
   - sau đó chạy `validate` và xem `summary.json`: `complete` phải là `true`, và `suspect_hosts` cần được xem qua.

   Lệnh chi tiết nằm trong `README.md`.
3. Bước 3, chunking và index:
   - cắt chunk từ `text` theo đoạn `\n\n`, gộp thành khối khoảng 800–1500 ký tự (metric chỉ phạt chunk thiếu chữ, không phạt chunk dài);
   - dựng BM25 riêng cho từng ngôn ngữ, cộng dense đa ngữ;
   - giảm trọng số doc có cờ `fallback` hoặc `meta_only`.
