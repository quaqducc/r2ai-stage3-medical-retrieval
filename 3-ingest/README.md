# 3 · Ingest: kho markdown sạch → (A) chỉ mục RAG + (B) bundle tri thức OKF

**Trạng thái:** đang triển khai luồng A (RAG). Luồng B (OKF) **hoãn**, chỉ làm khi RAG chưa đủ tốt, nhưng thiết kế đã chừa sẵn chỗ để cắm vào.

Bước này đọc kho sạch của bước 2 (`clean/v1/docs/<host>/*.parquet`, 1,22 triệu tài liệu) và dựng hai thứ mà agent ở bước 4 sẽ dùng song song:

| Luồng | Tạo ra | Agent dùng để |
|---|---|---|
| **A · RAG** | chunk nguyên văn, chỉ mục sparse + dense từ bge-m3 | tìm tài liệu và đoạn để nộp (`relevant_docs`, `relevant_chunks`) |
| **B · OKF** | bundle [Open Knowledge Format v0.2](https://github.com/GoogleCloudPlatform/open-knowledge-format): mỗi khái niệm y khoa (bệnh, triệu chứng, thuốc, xét nghiệm, thủ thuật) là một file markdown có frontmatter YAML, link với nhau thành đồ thị | hiểu và chuẩn hoá câu hỏi trước khi tìm. Ví dụ "Duphaston" → dydrogesterone → 地屈孕酮; tìm ra bệnh liên quan; biết nên tra tài liệu nào |

Hai luồng bổ trợ nhau: luồng B dùng chỉ mục sparse/dense của luồng A để gắn bằng chứng (doc id trong corpus) vào từng khái niệm. Luồng A thì dùng bí danh đa ngữ của luồng B để mở rộng câu hỏi ở bước 4.

Thiết kế dựa trên một nhận xét quan trọng: **đề chỉ có 1.200 câu hỏi cố định, vòng private dùng lại đúng các câu hỏi đó.** Vì vậy truy hồi chạy offline theo lô, không cần server hay vector database. Mọi đầu ra chỉ là file trên bucket.

---

## Luồng A · RAG (đã có code, đã test)

Lexical và dense **dùng chung một model: `BAAI/bge-m3`**. Một lần chạy model cho ra cả vector dense (1.024 chiều) lẫn trọng số sparse (lexical weights theo token XLM-R, khoảng 250 nghìn cột). Bản cũ dùng BM25 tự viết, nay đã bỏ.

### Đầu ra

```
<out>/rag/
  chunks/<lang>/<host>__<part>.parquet          chunk nguyên văn (schema trong CHUNKING.md)
  docs/<lang>/<host>__<part>.parquet            thẻ tài liệu: doc_id, title, url, doc_type, n_chunks, meta_text (MeSH/PMID của PubMed)
  dense/bge-m3/<lang>/<name>.{npy,ids.parquet}  vector float16, đã chuẩn hoá
  sparse/bge-m3/<lang>/<name>.{npz,ids.parquet} trọng số lexical, CSR
  sparse_index/bge-m3/<lang>/                   chỉ mục sparse đã gộp (CSC, nạp bằng mmap)
  reports/chunk_stats.json
  _done/chunk/<name>.json                       đánh dấu phần đã chunk (để chạy tiếp)
```

### Lệnh

| Lệnh | Việc | Máy |
|---|---|---|
| `python -m ingest rag chunk --clean <clean> --out <out>` | kho sạch → chunk + thẻ tài liệu theo [CHUNKING.md](CHUNKING.md); chạy tiếp được, mỗi phần của kho sạch là một đơn vị | CPU (Docker `r2ai-ingest`) |
| `python -m ingest rag embed --out <out>` | bge-m3 dense + sparse cho các phần chưa embed; mỗi phần ghi shard riêng nên chạy tiếp được; `--max-parts N` cho phiên có giới hạn thời gian (Kaggle) | **GPU**, chạy ngay trên máy có GPU (`pip install -r requirements-gpu.txt`) |
| `python -m ingest rag embed --out <out> --server <url>` | như trên, nhưng gọi bge-m3 qua server HTTP (`POST /encode_batch`) | dùng cho câu hỏi hoặc lô nhỏ (xem lưu ý) |
| `python -m ingest rag index --out <out>` | gộp các shard sparse thành một chỉ mục mỗi ngôn ngữ | CPU |

Bước 4 dùng `ingest.sparse.load(<out>/rag/sparse_index/bge-m3/<lang>).search(query_lexical_weights, k)` và `ingest.dense.search(query_vecs, shards, k)`. Tìm dense là nhân ma trận trực tiếp, cho kết quả chính xác tuyệt đối.

### Số liệu đã đo (18/10/2026)

- **Chunk trên fixture thật** (152 doc của 84 host): 194 chunk. Không có lỗi nguyên văn hay phủ thiếu chữ. Chunk bài viết có trung vị 520 từ, tối đa 997 từ.
- **Smoke test qua server bge-m3 của nhóm:** câu hỏi tiếng Việt tìm ra đúng tài liệu tiếng Trung ở vị trí 1 bằng dense. Ví dụ "Quan hệ 21 ngày thử que có chính xác không…" ra `同房21天验孕棒准确吗`, điểm 0,72.
- **Sparse chỉ khớp được cùng ngôn ngữ:** câu hỏi tiếng Việt chấm trên chỉ mục zh/en được 0 điểm, vì đây vẫn là so khớp token. Muốn dùng kênh sparse cho zh/en thì **bước 4 phải dịch câu hỏi** sang zh/en trước.
- **Lưu ý về server HTTP:**
  - Mỗi đoạn dài cỡ một chunk mất khoảng 1,9 giây; response khoảng 22 KB mỗi đoạn (JSON).
  - Toàn bộ corpus (khoảng 2 triệu chunk) qua server sẽ mất hơn 40 ngày và tải về khoảng 44 GB.
  - **Embed corpus phải chạy `rag embed` (không có `--server`) ngay trên máy GPU.** Server dùng cho 1.200 câu hỏi và cho test.

## Chỗ chừa cho OKF

Hiện chưa có code OKF. Ba quy ước dưới đây giữ để sau này thêm OKF không phải sửa luồng A:
1. **Đường dẫn đầu ra tách riêng:** luồng A ghi vào `<ingest>/rag/…`; `<ingest>/okf/…` để dành cho OKF.
2. **CLI có nhóm lệnh:** `python -m ingest rag <lệnh>`; nhóm `okf` sẽ thêm cạnh nhóm `rag`.
3. **Khóa chung là `doc_id` và `chunk_id`:** `sources` của khái niệm OKF trỏ thẳng tới doc của luồng A. Ở bước 4, phần mở rộng truy vấn là một hàm cắm vào được (`expand(query) -> [query…]`), nên bí danh từ OKF chỉ cần cắm thêm vào đó.

## Luồng B · OKF (hoãn, thiết kế để tham khảo)

### Bundle trông như thế nào

```
<ingest>/okf/bundle/
  index.md                       okf_version: "0.2" + mục lục cấp 1 (sinh tự động)
  log.md                         lịch sử các lần ingest
  diseases/<slug>.md             type: Disease
  symptoms/<slug>.md             type: Symptom
  drugs/<slug>.md                type: Drug
  tests/<slug>.md                type: Diagnostic Test
  procedures/<slug>.md           type: Procedure
  sources/<host>.md              type: Corpus Source   (mỗi host: ngôn ngữ, loại nội dung, chất lượng trích)
  skills/search-corpus.md        type: Skill           (cách agent gọi công cụ tìm kiếm của luồng A)
  */index.md                     mục lục từng thư mục, sinh tự động
```

Ví dụ một khái niệm:

```markdown
---
type: Drug
title: Dydrogesterone
description: Progestogen đường uống, dùng trong dọa sảy thai và rối loạn kinh nguyệt.
tags: [pregnancy, progestogen]
aliases:                                   # key mở rộng (OKF cho phép), lõi của việc chuẩn hoá câu hỏi
  vi: [Duphaston, dydrogesteron]
  zh: [地屈孕酮, 达芙通]
  en: [dydrogesterone, Duphaston]
status: draft
generated: { by: ingest-okf/qwen3-8b, at: 2026-10-20T10:00:00Z }
sources:                                   # bằng chứng = tài liệu trong corpus
  - { id: d3861954, resource: "https://...", title: "地屈孕酮的作用", author: "host:www.120ask.com", doc_id: 3861954 }
  - { id: d282674, resource: "https://nhathuoclongchau.com.vn/...", author: "host:nhathuoclongchau.com.vn", doc_id: 282674 }
---

# Định nghĩa
Dydrogesterone là progestogen tổng hợp...[^d282674]

# Liên quan
- [Dọa sảy thai](/diseases/threatened-miscarriage.md)
- [Drotaverine](/drugs/drotaverine.md) - thường được kê cùng

[^d282674]: Nhà thuốc Long Châu
```

Các điểm này theo đúng SPEC v0.2: `type` bắt buộc; `title`/`description`/`tags` khuyến nghị; `sources` có `id` dùng làm khoá footnote; `generated`/`status` cho phép phân tầng tin cậy (mọi khái niệm sinh bằng máy đều là `draft`, chưa ai xác minh). `aliases` và `doc_id` là key mở rộng; SPEC cho phép producer thêm key riêng.

### Luồng ingest OKF (tham khảo cách chạy 2 lượt của bản mẫu Google, không chép)

| Lệnh | Lượt | Việc | LLM? |
|---|---|---|---|
| `okf seed` | 1 | tạo khái niệm khởi đầu từ các nguồn **vốn đã có cấu trúc**: tiêu đề mục từ MediaWiki (zhongyibaodian, a-hospital), jbk.39.net, msdmanuals, trang bệnh Long Châu và Vinmec, MeSH trong PubMed. Tạo thêm `sources/<host>.md` từ thống kê bước 2 | không |
| `okf extract` | 2 | với mỗi khái niệm khởi đầu và doc nguồn của nó: LLM trích tên chuẩn tiếng Anh, bí danh vi/zh/en, loại, định nghĩa một câu, các khái niệm liên quan. Mỗi trường hợp chọn một trong ba: **bổ sung** vào khái niệm có sẵn, **tạo** khái niệm mới, hoặc **bỏ qua** | có (open-weight ≤ 15B, ví dụ Qwen3-8B qua vLLM) |
| `okf resolve` | 2 | gộp khái niệm trùng giữa các ngôn ngữ, dựa trên tên chuẩn tiếng Anh và bí danh trùng nhau; sửa link cho trỏ về khái niệm còn lại | không |
| `okf evidence` | 3 | với mỗi khái niệm, tìm doc chứa bí danh của nó bằng chỉ mục sparse của luồng A, rồi ghi top-k vào `sources` (kèm `doc_id`, `author: host:…`) | không |
| `okf index` | 4 | sinh `index.md` cho mọi thư mục và ghi `log.md`. Mô tả thư mục tạo bằng template, không cần LLM | không |
| `okf validate` | 4 | kiểm tra tuân thủ SPEC (mọi file có frontmatter và `type`), kiểm tra link hỏng và doc mồ côi, thống kê | không |

**Các guard nằm trong code ghi file** (rút kinh nghiệm từ bản mẫu: nhắc trong prompt thì không đủ):
- Ghi đè một khái niệm thì không được làm mất key frontmatter hay heading `#` đã có.
- `sources` và `aliases` không bao giờ được ngắn đi.
- Bí danh phải xuất hiện nguyên văn trong ít nhất một doc nguồn. Quy tắc này chặn LLM bịa tên thuốc hay tên bệnh.
- Khái niệm không có bằng chứng nào trong corpus thì giữ `status: draft` và gắn tag `no-evidence`.

Agent ở bước 4 đọc bundle theo hai cách:
1. **Tra cứu nhanh:** `okf index` sinh thêm `aliases.json` (bí danh → khái niệm). Đây là artifact phái sinh phía consumer, không thuộc bundle.
2. **Đi theo link:** đọc `index.md` rồi mở doc khái niệm, đúng kiểu progressive disclosure của OKF.

---

## Cấu trúc thư mục code

```
3-ingest/
  README.md
  Dockerfile               build từ main-pipeline/ để dùng chung extract.storage của bước 2
  requirements.txt
  ingest/
    __main__.py            CLI: rag {chunk,bm25,embed,stats} | okf {seed,extract,resolve,evidence,index,validate}
    rag/
      chunk.py  tokenize.py  bm25.py  embed.py
    okf/
      document.py          đọc/ghi concept (frontmatter + body), slug, guard bổ sung
      seed.py              lượt 1: khái niệm từ nguồn có cấu trúc
      extract.py           lượt 2: LLM trích và chuẩn hoá (endpoint OpenAI-compatible: vLLM/Ollama)
      resolve.py           gộp trùng đa ngữ
      evidence.py          lượt 3: gắn doc bằng chứng qua chỉ mục sparse của luồng A
      index.py             index.md, log.md, aliases.json, validate
      prompts/extract.md
  tests/
    rag/ ...               offset nguyên văn, chỉ mục sparse/dense nhỏ
    okf/ ...               guard không làm mất nội dung, tuân thủ SPEC, gộp trùng, link hỏng/mồ côi
```

Mỗi lệnh đọc và ghi bằng đường dẫn local hoặc `gs://`, có `--limit` để test nhỏ trên laptop. Bản đầy đủ chạy trên VM hoặc Kaggle.

## Quyết định cần bạn xác nhận

1. **Phạm vi bundle OKF** (thảo luận chi tiết trong **[OKF_STRATEGY.md](OKF_STRATEGY.md)**). Có hai lựa chọn:
   - (a) *Theo corpus*: khái niệm khởi đầu là toàn bộ mục từ của các site tri thức, ước tính vài chục nghìn khái niệm. Chi phí LLM lớn: hàng chục triệu token, nhiều giờ GPU.
   - (b) *Theo câu hỏi*: LLM trích thực thể từ 1.200 câu hỏi trước, rồi chỉ dựng khái niệm cho các thực thể đó và hàng xóm 1 bước, ước tính 3–6 nghìn khái niệm. Rẻ hơn cỡ 10 lần và phục vụ đúng đề.

   Tôi đề xuất **(b) trước, (a) sau** nếu còn thời gian.
2. **LLM cho `okf extract`:** Qwen3-8B (hoặc 14B) chạy vLLM trên GPU Kaggle. Thoả điều kiện đề: open-weight, ≤ 15B, phát hành trước 1/8/2026.
3. **Chunking** (luồng A): đã chốt, xem [CHUNKING.md](CHUNKING.md).
4. **Thứ tự làm:** luồng A trước, vì nó là đường nộp bài và luồng B cần chỉ mục của nó. Luồng B làm song song phần `seed`, `document`, `index`, `validate`, vì các phần này không cần LLM.

## Embed toàn corpus trên Colab ([colab/embed_colab.ipynb](colab/embed_colab.ipynb))

Notebook chạy `rag.run_embed` ngay trên GPU Colab: đọc chunk từ GCS và ghi shard dense/sparse trở lại GCS. Không đi qua ngrok.

1. Đóng gói code rồi đưa lên bucket. Làm lại mỗi khi sửa `ingest/` hoặc `extract/storage.py`:
   ```
   python 3-ingest/colab/pack.py
   gcloud storage cp 3-ingest/colab/dist/r2ai_ingest.zip gs://aiguru_crawler/code/r2ai_ingest.zip
   ```
2. Mở notebook trên Colab với runtime T4, đặt `SHARD`, rồi Run all.
   - Có n phiên song song thì đặt `"0/n"` … `"n-1/n"`. Mỗi phần chunk thuộc đúng một phiên, chia theo `crc32(lang/name) % n`. CLI tương đương là `rag embed --shard i/n`.
   - Bị ngắt thì chạy lại với cùng `SHARD`: phần đã xong được bỏ qua.
3. Khi cell cuối báo `todo = 0`, chạy `rag index` trên VM.

Ước lượng: khoảng 1,5 triệu chunk × 0,065 s ≈ 27 giờ trên một T4, tức khoảng 9–10 giờ với 3 phiên.

## Dự phòng khi không có GPU: `rag bm25`

`python -m ingest rag bm25 --out <out>` dựng chỉ mục BM25 (k1 1,2, b 0,75) trên token XLM-R của bge-m3. Lệnh chỉ dùng tokenizer, không chạy model, nên chạy CPU được: 1,57 triệu chunk mất vài phút trên 16 vCPU. Kết quả nằm ở `rag/sparse_index/bm25/<lang>/`, cùng định dạng với chỉ mục sparse của bge-m3, nên bước retrieve coi nó là một kênh sparse bình thường (config `tag: "bm25"`, `dense_langs: []`).

Bối cảnh (21/10): embed toàn corpus qua server ngrok dừng sau khoảng 44k chunk vì tài khoản ngrok free hết băng thông tháng (`ERR_NGROK_725`). Muốn có dense đầy đủ, chạy [colab/embed_colab.ipynb](colab/embed_colab.ipynb): notebook ghi thẳng GCS, không đi qua ngrok.
