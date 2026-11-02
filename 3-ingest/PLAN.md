# Ingest luồng A (RAG): kế hoạch triển khai

**Goal:** chuyển kho sạch `clean/v1/docs` thành chunk nguyên văn, chỉ mục sparse và vector dense từ bge-m3, đúng theo [CHUNKING.md](CHUNKING.md) và [README.md](README.md).

**Architecture:**
- Package `ingest`, chạy bằng `python -m ingest rag {chunk,embed,index}`.
- Đọc và ghi qua `extract.storage` của bước 2 (local hoặc gs://). Ghi ra `<out>/rag/...`.
- Chunk là contiguous span trong `text`. Chunk hỏi đáp tách theo câu trả lời thì gồm 2 span (câu hỏi + câu trả lời).
- Lexical dùng trọng số sparse của bge-m3 (quyết định ngày 18/10, thay BM25).

**Tech stack:** Python 3.9+ (image 3.11), pandas/pyarrow, scipy (chỉ mục sparse CSC), FlagEmbedding 1.4.2 (`BGEM3FlagModel`, chỉ cài trên máy GPU), hoặc server HTTP của nhóm (`POST /encode_batch`).

## Trạng thái các task (18/10/2026)

| # | Task | Trạng thái |
|---|---|---|
| 1 | `chunk.py`: đếm từ, phân loại doc, tách hỏi–đáp theo site, chunk bài viết theo heading (min 500 / max 1000), chunk PubMed | ✅ có test trên fixture thật |
| 2 | `rag.py run_chunk`: kho sạch → chunk + thẻ tài liệu, chạy tiếp được, song song theo phần | ✅ test end-to-end |
| 3 | `sparse.py` (chỉ mục lexical từ trọng số bge-m3) + `dense.py` (tìm kiếm chính xác theo shard) | ✅ |
| 4 | `rag.py run_embed` (`BgeM3Encoder` local GPU / `HttpEncoder` qua server), `run_index` | ✅ test với encoder giả; smoke test qua server thật |
| 5 | CLI, Dockerfile (context `main-pipeline/`), README | ✅ |
| 6 | Review (python-reviewer, rag-pipeline-reviewer) | ⏳ |
| 7 | Chạy thật: chunk trên VM khi có kho sạch; embed trên máy GPU | ⏳ chờ bước 2 chạy đầy đủ |
