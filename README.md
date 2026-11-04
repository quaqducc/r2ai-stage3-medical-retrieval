# R2AI 2026 – Stage 3: truy hồi tài liệu y khoa đa ngữ

Hệ thống truy hồi lai dựa trên **BGE-M3** (dense + sparse), mở rộng câu hỏi bằng bản dịch tiếng Trung do **Qwen3.5-9B** sinh ra, PRF Rocchio và truy hồi cấp tài liệu. Các danh sách được gộp bằng RRF có trọng số.

Bài nộp được báo cáo là **`i_dprf15`**: FINAL **0,3577** trên vòng public (Docs F2 0,3900, Chunks F2 0,3255), hạng 7/50.

| Tài liệu | Nội dung |
|---|---|
| [docs/THUYET_MINH.md](docs/THUYET_MINH.md) (+ bản `.docx`) | **Tài liệu thuyết minh**: đội thi, mô hình và prompt, tổng quan hệ thống, phương pháp (ingest, retrieve), sơ đồ, cấu hình và ví dụ |
| [docs/4_ket_qua.md](docs/4_ket_qua.md) | Điểm từng bài nộp, tiến trình, các thí nghiệm không hiệu quả |
| [3-ingest/CHUNKING.md](3-ingest/CHUNKING.md), [2-extract/REPORT.md](2-extract/REPORT.md) | Đặc tả phân đoạn; báo cáo chất lượng bước trích |

## 1. Cấu trúc repo

```
1-crawl/       thống kê độ phủ crawl (crawl_stats.csv); mã crawler: xem 1-crawl/README.md
2-extract/     HTML/XML thô -> markdown + text sạch     (Python package `extract`, Dockerfile, 159 test)
3-ingest/      chunk nguyên văn, embed bge-m3, chỉ mục sparse   (package `ingest`, Dockerfile, notebook Colab)
4-retrieve/    1.200 câu hỏi -> submission.json            (package `retrieve`, bench/, configs/, Dockerfile)
scripts/reproduce.sh   chạy lại toàn bộ hoặc từng bước
docs/          tài liệu thuyết minh
```

## 2. Yêu cầu môi trường

| Mục | Phiên bản / cấu hình |
|---|---|
| Máy chạy chính | Linux x86-64; nhóm dùng GCP e2-standard-16 (16 vCPU, 64 GB RAM), đĩa ≥ 100 GB |
| GPU (chỉ cho bước `embed`) | 1 × NVIDIA T4 16 GB (Colab hoặc Kaggle là đủ), khoảng 10 giờ |
| Docker | ≥ 24 |
| Google Cloud SDK | `gcloud` ≥ 450, có quyền đọc `gs://aiguru_crawler` (xem mục 3) |
| Python (trong image) | 3.11.11 (`python:3.11.11-slim-bookworm`) |

**Thư viện chính** (pin trong các Dockerfile và `requirements*.txt`):

| Bước | Thư viện |
|---|---|
| 2-extract | pyarrow 18.1.0, pandas 2.2.3, trafilatura 2.0.0, lxml 5.3.0, zstandard 0.23.0, apsw 3.51.0.0, gcsfs 2025.10.0 |
| 3-ingest | pandas 2.2.3, pyarrow 18.1.0, numpy < 2.1, scipy 1.11–1.14, gcsfs 2025.10.0; GPU: FlagEmbedding 1.4.2 |
| 4-retrieve | torch 2.6.0 (CPU), FlagEmbedding 1.4.2, transformers 4.57.1, pandas 2.2.3, numpy < 2.1, scipy, scikit-learn 1.5.2, rapidfuzz 3.9.7, requests 2.32.3 |
| LLM server | `ghcr.io/ggml-org/llama.cpp:server` + `Qwen3.5-9B-Q4_K_M.gguf` |

## 3. Dữ liệu và checkpoint

**Dữ liệu** nằm trên Google Cloud Storage, bucket `gs://aiguru_crawler` (project `aiguru26`, region asia-southeast1). Đội cấp quyền đọc (*Storage Object Viewer*) cho tài khoản Google của người chấm.
- Trình duyệt: <https://console.cloud.google.com/storage/browser/aiguru_crawler>. Link tải từng file có dạng `https://storage.cloud.google.com/aiguru_crawler/<đường dẫn>`.
- Dòng lệnh: dùng `gcloud storage`.

| Đường dẫn | Kích thước | Nội dung | Dùng để |
|---|---|---|---|
| `crawl/db/`, `crawl/manifest.parquet` | 32 GB | trang thô, một SQLite cho mỗi host (HTML/XML nén zstd) | chạy lại từ bước 2 |
| `clean/v1/docs/<host>/*.parquet` | 2,9 GB | kho văn bản sạch, mỗi id corpus một dòng | chạy lại từ bước 3 |
| `ingest/v1/rag/` | 15 GB | `chunks/`, `docs/`, `dense/bge-m3/`, `sparse_index/bge-m3/` | chạy lại từ bước 6 |
| `release/v1/run/plans.json` | 0,6 MB | plan của Qwen3.5-9B cho 1.200 câu hỏi | tái hiện chính xác danh sách |
| `release/v1/run/lists.pkl`, `lists_docprf.pkl` | 32 MB, 92 MB | danh sách top-200 (cấp chunk; toàn bộ 14 nhóm của bài nộp) | dựng nhanh bài nộp |
| `release/v1/submissions/*.zip` | 97 MB mỗi file | 5 file đã nộp ở vòng private | đối chiếu |

```bash
gcloud auth login
gcloud storage cp -r gs://aiguru_crawler/release/v1 ./release
gcloud storage rsync -r gs://aiguru_crawler/ingest/v1/rag/chunks ./ingest/rag/chunks
```

Schema của chunk: `chunk_id` (`<doc_id>:<idx>`), `doc_id`, `idx`, `text`, `spans`, `title`, `heading_path`, `doc_type`, `unit_kind`. Đặc tả phân đoạn: [3-ingest/CHUNKING.md](3-ingest/CHUNKING.md).

**Checkpoint** (bản chính thức, không fine-tune):

```bash
# BGE-M3: FlagEmbedding tự tải vào HF_HOME khi chạy; hoặc tải trước:
huggingface-cli download BAAI/bge-m3 --local-dir /mnt/hf/bge-m3
# Qwen3.5-9B GGUF Q4_K_M + llama.cpp server (bước `llm` của scripts/reproduce.sh)
curl -L -o /mnt/llm/qwen3.5-9b-q4km.gguf   https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf
docker run -d --name qwen --network host -v /mnt/llm:/models ghcr.io/ggml-org/llama.cpp:server   -m /models/qwen3.5-9b-q4km.gguf -c 16384 -np 4 -t 16 --jinja --reasoning-budget 0 --host 0.0.0.0 --port 8080
```

Client LLM (`4-retrieve/retrieve/llm.py`) gọi endpoint OpenAI-compatible với `temperature=0`, `max_tokens=1200`. Có thể thay bằng vLLM hoặc server tương đương qua `--llm-url`, `--llm-model` (key qua biến môi trường `LLM_API_KEY`).

## 4. Tái hiện nhanh bài nộp (khoảng 15 phút, không cần GPU hay LLM)

Bước này dùng các danh sách truy hồi đã công bố (`release/v1/run/lists_docprf.pkl`) và chunk nguyên văn (`ingest/v1/rag/chunks`). Nó chạy lại phần gộp RRF, lựa chọn và ghi file, rồi so với file đã nộp.

```bash
git clone <repo> && cd <repo>
gcloud auth login                                        # tài khoản đã được cấp quyền đọc bucket
mkdir -p ~/data && cp /path/to/query.parquet ~/data/     # file câu hỏi của BTC
sudo docker build -f 4-retrieve/Dockerfile -t r2ai-retrieve:v1 .
scripts/reproduce.sh fast
# -> /mnt/full/variants/i_dprf15/submission.zip
# -> "IDENTICAL to the released i_dprf15"
```

Đổi thư mục làm việc bằng biến môi trường `WORK=/path INGEST=/path DATA=/path`; giá trị mặc định ghi ở đầu `scripts/reproduce.sh`. Nhóm đã chạy thử bước `build` từ image dựng lại theo repo này: kết quả trùng khớp hoàn toàn với file đã nộp (1.200/1.200 câu), mất khoảng 2 phút.

## 5. Chạy lại từ đầu

```bash
scripts/reproduce.sh full          # hoặc từng bước theo thứ tự dưới đây
```

| # | Bước (`scripts/reproduce.sh <bước>`) | Lệnh chính | Máy / thời gian | Đầu ra |
|---|---|---|---|---|
| 0 | `images` | `docker build` 3 image | CPU, ~10 phút | `r2ai-extract:v1`, `r2ai-ingest:v1`, `r2ai-retrieve:v1` |
| 1 | (crawl) | xem `1-crawl/README.md` | | `gs://…/crawl/db/<host>.sqlite` |
| 2 | `extract` | `r2ai-extract run` + `validate` | CPU 16 vCPU, ~1 giờ | `gs://…/clean/v1/docs/` |
| 3 | `chunk` | `r2ai-ingest rag chunk` | CPU | `gs://…/ingest/v1/rag/chunks/`, `docs/` |
| 4 | `embed` | `python -m ingest rag embed`, hoặc notebook [3-ingest/colab/embed_colab.ipynb](3-ingest/colab/embed_colab.ipynb) | GPU T4, ~10 giờ (chia nhiều phiên bằng `SHARD`) | `rag/dense/bge-m3/`, `rag/sparse/bge-m3/` |
| 5 | `index` | `r2ai-ingest rag index` | CPU, vài phút | `rag/sparse_index/bge-m3/<lang>/` |
| 6 | `sync` | `gcloud storage rsync` về đĩa local | | `$INGEST/rag/` |
| 7 | `llm` | tải GGUF và chạy llama.cpp server | CPU | `localhost:8080` |
| 8 | `retrieve` | `python -m retrieve run --modes deep --config configs/deep_bge_vizh_k100.json` | CPU (plan LLM cho 1.200 câu là phần lâu nhất; truy hồi khi đã có plan ~3 phút) | `lists.pkl`, `plans.json` |
| 9 | `prf` | `bench/prf.py … 10 0.5` | CPU | `lists_prf.pkl` |
| 10 | `doc` | `bench/doc_dense.py` | CPU | `lists_doc.pkl` |
| 11 | `docprf` | `bench/doc_dense_prf.py` | CPU | `lists_docprf.pkl` |
| 12 | `build` | `bench/variants.py … i_dprf15` | CPU, ~2 phút | `variants/i_dprf15/submission.zip` |
| 13 | `verify` | so `submission.json` với file đã nộp | | IDENTICAL / DIFFERENT |

**Ghi chú:**
- **Quyền GCS khi chạy từ đầu.** Bước 2–5 ghi lên bucket qua Application Default Credentials của VM (service account cần quyền đọc `crawl/` và ghi `clean/`, `ingest/`). Muốn ghi vào bucket khác, đặt `BUCKET=gs://<bucket>`.
- **Tái hiện đúng từng danh sách.** Suy luận LLM trên CPU có thể lệch nhẹ. Để có đúng các danh sách đã dùng, chạy bước 8 với `PLANS=<đường dẫn tới release/v1/run/plans.json>` thay vì gọi LLM.
- **Dựng các bài nộp khác.** Mọi cấu hình thí nghiệm nằm trong `4-retrieve/bench/variants.py` (`VARIANTS`). Mỗi bài dựng lại trong vài phút từ cùng `lists_docprf.pkl`: `LISTS=lists_docprf.pkl python bench/variants.py <run_dir> <ingest> query.parquet <out> j_hh125,i_dprf1`.

## 6. Kiểm thử

```bash
pip install -r 2-extract/requirements-dev.txt -r 3-ingest/requirements.txt scikit-learn==1.5.2 rapidfuzz==3.9.7
(cd 2-extract && python -m pytest -q)    # 159 test: luật trích theo host, trên fixture thật của 84 host
(cd 3-ingest && python -m pytest -q)     # 27 test: chunk nguyên văn (offset/spans), chỉ mục sparse/dense
(cd 4-retrieve && python -m pytest -q)   # 34 test: end-to-end trên fixture (extract -> chunk -> retrieve -> submit)
```

Các test end-to-end của `4-retrieve` cần thư mục `2-extract/tests/fixtures/` có trong repo. Vì vậy chạy test từ repo, không chạy trong image.

## 7. Định dạng đầu ra

`submission.json` nén zip (bzip2, đúng một file, không có thư mục con):

```json
[{"id": 1, "relevant_docs": [123, 456],
  "relevant_chunks": [{"doc_id": 123, "chunk_text": "<đoạn trích nguyên văn>"}]}]
```

`retrieve/submit.py` kiểm tra trước khi ghi:
- đủ 1.200 câu;
- id là số nguyên;
- `doc_id` của mọi chunk thuộc `relevant_docs`;
- không có chunk rỗng;
- không quá 1% số câu thiếu doc.
