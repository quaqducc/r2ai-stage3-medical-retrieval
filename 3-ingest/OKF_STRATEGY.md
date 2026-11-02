# Chiến lược OKF (luồng B) · bản nháp để thảo luận

**Trạng thái: HOÃN (18/10/2026).** Ưu tiên xây pipeline RAG thường trước. Chỉ làm OKF nếu RAG chưa đủ tốt. Thiết kế dự án đã chừa sẵn chỗ để cắm OKF vào (xem mục "Chỗ chừa cho OKF" trong [README.md](README.md)). Các mục **[Bàn]** sẽ quyết định khi mở lại.

## 1. OKF là gì (tóm tắt SPEC v0.2)

[Open Knowledge Format](https://github.com/GoogleCloudPlatform/open-knowledge-format) là định dạng mở của Google Cloud (v0.1 ra tháng 6/2026, hiện là v0.2). OKF đóng gói tri thức thành **một thư mục file markdown có frontmatter YAML**:

- **Bundle** là một cây thư mục. Mỗi file `.md` là một **khái niệm**; định danh của khái niệm (concept id) là đường dẫn file bỏ đuôi `.md`.
- **Frontmatter** chỉ bắt buộc một trường là `type`. Các trường khuyến nghị: `title`, `description` (một câu), `resource` (URI của đối tượng mà khái niệm mô tả), `tags`. Producer được thêm key riêng.
- **Provenance và độ tin cậy:**
  - `sources` là danh sách nguồn, mỗi mục có `id`, `resource`, `title`, `author`, `usage_count`, `last_modified`;
  - footnote `[^id]` gắn từng nhận định với nguồn của nó;
  - `generated: {by, at}` cho biết ai hoặc cái gì tạo ra nội dung;
  - `verified` cho biết ai đã xác minh, từ đó suy ra tầng tin cậy: chưa xác minh / máy xác nhận / người duyệt;
  - `status` nhận một trong `draft`, `stable`, `deprecated`; `stale_after` là thời điểm nội dung hết hạn.
- **Link:** khái niệm nối với nhau bằng link markdown thường (khuyến nghị dạng `/path.md` tính từ gốc bundle). Link hỏng không bị coi là lỗi.
- **`index.md`:** mục lục mỗi thư mục, phục vụ **progressive disclosure**: agent đọc mục lục trước, chỉ mở doc khi cần. **`log.md`:** lịch sử thay đổi.
- **Tuân thủ:** mọi file `.md` không trùng tên dành riêng phải có frontmatter với `type`. Ngoài điều đó ra, consumer không được từ chối bundle vì bất cứ lý do nào khác.

## 2. Vì sao cần OKF cho bài toán này

Câu hỏi của đề là tiếng Việt, ngôn ngữ đời thường, nhiều thực thể. Kho tài liệu lại có 1/3 là tiếng Trung và tiếng Anh. RAG thuần dựa vào embedding đa ngữ để bắc cầu giữa các ngôn ngữ, nên dễ hỏng ở **tên riêng y khoa**: tên biệt dược, tên bệnh dân gian, tên vị thuốc Đông y. Ví dụ trong [architecture.txt](../../architecture.txt): Duphaston → dydrogesterone → 地屈孕酮.

Bundle OKF đóng vai **bản đồ khái niệm** cho agent:
1. **Chuẩn hoá câu hỏi:** tìm bí danh trong câu hỏi, ánh xạ sang khái niệm, rồi lấy các tên tương ứng ở ngôn ngữ khác để mở rộng truy vấn BM25 tiếng Trung và tiếng Anh.
2. **Gợi ý câu hỏi con:** từ link "liên quan" của khái niệm, ví dụ thuốc → bệnh chỉ định.
3. **Dẫn tới bằng chứng:** `sources` của khái niệm trỏ thẳng tới doc id trong corpus.

**[Bàn] Giá trị cần đo được:** so điểm Docs F2 và Chunks F2 của luồng A đơn thuần với luồng A cộng mở rộng truy vấn từ OKF, trên public leaderboard. Nếu không tăng thì không mở rộng luồng B.

## 3. Phạm vi **[Bàn]**

| Phương án | Khái niệm khởi đầu | Quy mô ước tính | Chi phí LLM | Ưu / nhược |
|---|---|---|---|---|
| **(a) Theo corpus** | mọi mục từ của các site tri thức (MediaWiki, jbk.39, msdmanuals, trang bệnh Long Châu/Vinmec, MeSH) | vài chục nghìn | hàng chục triệu token, nhiều giờ GPU | tổng quát, dùng lại được; tốn kém và chậm |
| **(b) Theo câu hỏi** | thực thể trích từ 1.200 câu hỏi, cộng hàng xóm 1 bước | 3–6 nghìn | cỡ 1/10 | rẻ, phục vụ đúng đề; ít tổng quát |
| (c) Kết hợp | (b) trước, rồi mở rộng dần theo (a) | tuỳ chọn | tuỳ chọn | |

Đề xuất của tôi: **(b) trước, sau đó (c) nếu (b) giúp tăng điểm.**

## 4. Cấu trúc bundle (đề xuất)

```
okf/bundle/
  index.md                    okf_version: "0.2"
  log.md
  diseases/  symptoms/  drugs/  tests/  procedures/  tcm/      ← loại khái niệm y khoa
  sources/<host>.md           type: Corpus Source (ngôn ngữ, loại nội dung, chất lượng trích của host)
  skills/search-corpus.md     type: Skill (cách agent gọi BM25/dense của luồng A)
```

- **[Bàn]** Danh sách `type`: Disease, Symptom, Drug, Diagnostic Test, Procedure, TCM Herb/Formula (vị thuốc, bài thuốc Đông y), Corpus Source, Skill. Cần thêm hay bớt loại nào?
- **[Bàn]** Slug file: lấy theo tên chuẩn tiếng Anh, ví dụ `drugs/dydrogesterone.md`. Khái niệm không có tên tiếng Anh (thường gặp với Đông y) thì dùng pinyin.

## 5. Quy ước frontmatter (đề xuất)

```yaml
type: Drug
title: Dydrogesterone
description: Progestogen đường uống dùng trong dọa sảy thai.   # một câu, viết bằng tiếng Việt
tags: [pregnancy, progestogen]
aliases: { vi: [...], zh: [...], en: [...] }                 # key mở rộng: lõi của việc chuẩn hoá câu hỏi
codes: { mesh: D004394 }                                     # key mở rộng, tuỳ chọn (MeSH lấy từ PubMed)
status: draft                                                # mọi khái niệm sinh bằng máy đều là draft
generated: { by: ingest-okf/<model>, at: <ISO-8601> }
sources:
  - { id: d282674, resource: <url>, title: <tiêu đề doc>, author: "host:<host>", doc_id: 282674 }
```

- **[Bàn]** `description` và phần thân viết bằng tiếng Việt (ngôn ngữ của câu hỏi) hay tiếng Anh (trung lập)? Đề xuất: tiếng Việt.
- `verified` để trống. Nếu sau này có người kiểm một mẫu thì thêm `verified: {by: human:<id>}` cho các khái niệm đã kiểm.

## 6. Luồng ingest

Tham khảo cách bản mẫu của Google chia lượt (lượt metadata tất định, rồi lượt LLM làm giàu nội dung). Code tự viết.

| Lượt | Lệnh | Việc | LLM |
|---|---|---|---|
| 1 | `okf seed` | khái niệm khởi đầu: (b) thực thể trong câu hỏi; (a) tiêu đề mục từ ở các site tri thức. Tạo thêm `sources/<host>.md` | (b): có, để trích thực thể từ câu hỏi |
| 2 | `okf extract` | với mỗi khái niệm: đọc vài doc bằng chứng, trích bí danh vi/zh/en, loại, định nghĩa, khái niệm liên quan. Mỗi trường hợp chọn **bổ sung / tạo mới / bỏ qua** | có |
| 2 | `okf resolve` | gộp khái niệm trùng giữa các ngôn ngữ (tên chuẩn hoặc bí danh trùng, mã MeSH trùng) | không |
| 3 | `okf evidence` | tìm doc chứa bí danh bằng BM25 của luồng A, ghi top-k vào `sources` | không |
| 4 | `okf index` / `validate` | sinh `index.md` và `log.md`, `aliases.json` cho phía consumer; kiểm tra tuân thủ SPEC, link hỏng, doc mồ côi | không |

**Guard nằm trong code ghi file** (rút kinh nghiệm từ bản mẫu: nhắc trong prompt thì không đủ):
1. Ghi đè một khái niệm không được làm mất key frontmatter hay heading `#`. `sources` và `aliases` không bao giờ ngắn đi.
2. **Bí danh phải xuất hiện nguyên văn trong ít nhất một doc nguồn.** Quy tắc này chặn LLM bịa tên.
3. Khái niệm không có bằng chứng thì gắn tag `no-evidence`, không xoá.
4. Không có khái niệm mồ côi: mỗi khái niệm phải có ít nhất một link tới, hoặc nằm trong `index.md`.

## 7. LLM **[Bàn]**

- Điều kiện của đề: open-weight, ≤ 15B tham số, phát hành trước 1/8/2026.
- Ứng viên: Qwen3-8B hoặc Qwen3-14B (tiếng Trung tốt, tiếng Việt khá); Gemma 3 12B.
- Chạy bằng vLLM trên GPU (Kaggle 2×T4 hoặc VM), gọi qua endpoint OpenAI-compatible.
- Ước tính cho phương án (b): khoảng 5 nghìn khái niệm × khoảng 3 nghìn token mỗi lượt extract, tức cỡ 15 triệu token. Với 8B trên T4 thì mất vài giờ.

## 8. Agent ở bước 4 dùng bundle thế nào

1. Trích thực thể từ câu hỏi, tra `aliases.json` để ra khái niệm.
2. Mở doc khái niệm để lấy bí danh zh/en, khái niệm liên quan và doc bằng chứng.
3. Mở rộng truy vấn BM25 zh/en, sau đó cộng điểm cho doc bằng chứng.
4. Khi tra cứu không thấy thì dùng progressive disclosure: đọc `index.md` theo loại khái niệm.

## 9. Rủi ro

- **LLM bịa bí danh hoặc quan hệ:** đã có guard 2; ngoài ra mọi khái niệm để `status: draft`.
- **Thời gian:** vòng public đóng sáng 31/10. Luồng B chỉ làm sau khi luồng A đã có bài nộp đầu tiên.
- **Chuẩn hoá sai còn tệ hơn không chuẩn hoá:** mở rộng truy vấn phải là *thêm* vào truy vấn gốc, không *thay* truy vấn gốc.

## Câu hỏi cần chốt

1. Phạm vi: (a), (b) hay (c)?
2. Danh sách `type` và cách đặt slug.
3. Ngôn ngữ của `description` và phần thân.
4. Chọn LLM và nơi chạy.
5. Tiêu chí đánh giá: dùng leaderboard để quyết định có mở rộng luồng B hay không.
