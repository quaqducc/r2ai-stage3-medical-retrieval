# 3. Mô hình sử dụng

Bài nộp cuối chỉ dùng **hai mô hình mã nguồn mở**. Cả hai đều ≤ 15B tham số và phát hành trước 01/08/2026. Không dùng mô hình đóng, và **không có bước fine-tune**: dùng đúng checkpoint chính thức.

| Mô hình | Vai trò | Định danh / checkpoint | Chạy ở đâu |
|---|---|---|---|
| **BGE-M3** (568M) | Nhúng chunk và câu hỏi. Dense 1.024 chiều cho truy hồi ngữ nghĩa xuyên ngôn ngữ; sparse (lexical weights) cho truy hồi từ khoá | [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3), tải qua `FlagEmbedding==1.4.2` | Corpus: GPU T4 (Colab); câu hỏi: CPU |
| **Qwen3.5-9B** (9B, open weights, phát hành 10/03/2026) | S1 Plan: dịch câu hỏi sang tiếng Trung (`zh_q`), sinh từ khoá vi/zh/en và thực thể | GGUF Q4_K_M: [`unsloth/Qwen3.5-9B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF), file `Qwen3.5-9B-Q4_K_M.gguf` (5,7 GB); bản gốc [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B) | CPU, llama.cpp server (`ghcr.io/ggml-org/llama.cpp:server`) |

## 3.1 BGE-M3

**Mã hoá corpus** (`3-ingest/ingest/rag.py`, `BGEM3Encoder`):
- `BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)`, `passage_max_length=2048`, `batch_size=32`.
- Đầu vào = `title + "\n" + heading_path + "\n" + chunk_text`.
- Lưu vector dense đã chuẩn hoá L2 (float16) và trọng số sparse (CSR).

**Mã hoá câu hỏi** (`4-retrieve/retrieve/__main__.py: query_encoder`): cùng model, `query_max_length=512`, chạy CPU, có cache SQLite theo text.

**Cách tải:** tự động khi chạy, vào thư mục cache Hugging Face (`HF_HOME=/work/hf`, khoảng 2,3 GB). Muốn tải trước thì dùng:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download BAAI/bge-m3 --local-dir /mnt/hf/bge-m3
```

## 3.2 Qwen3.5-9B (S1 Plan)

**Khởi động server** (bước `llm` của `scripts/reproduce.sh`):

```bash
curl -L -o /mnt/llm/qwen3.5-9b-q4km.gguf \
  https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf
docker run -d --name qwen --network host -v /mnt/llm:/models ghcr.io/ggml-org/llama.cpp:server \
  -m /models/qwen3.5-9b-q4km.gguf -c 16384 -np 4 -t 16 --jinja --reasoning-budget 0 --host 0.0.0.0 --port 8080
```

**Cấu hình gọi** (`retrieve/llm.py`):
- Endpoint OpenAI-compatible `POST /v1/chat/completions`, `model=qwen3.5-9b-q4km`.
- `temperature=0`, `max_tokens=1200`, 4 luồng song song.
- Tắt chế độ suy luận (`--reasoning-budget 0`).
- Kết quả được cache theo hash(model, temperature, messages).
- Có thể thay bằng bất kỳ server OpenAI-compatible nào phục vụ cùng model, ví dụ vLLM. Khi đó truyền `--llm-url`, `--llm-model`, và key nếu có qua biến môi trường `LLM_API_KEY`.

**Prompt** (`retrieve/plan.py`):

System:
```
Bạn là chuyên gia y khoa song ngữ Việt–Trung–Anh, chuẩn bị truy vấn tìm kiếm cho một kho tài liệu y khoa gồm
bài báo/hỏi đáp tiếng Việt, diễn đàn hỏi đáp và bách khoa tiếng Trung, và tóm tắt PubMed tiếng Anh.
Chỉ trả về MỘT object JSON, không giải thích.
```

User:
```
Câu hỏi của người bệnh (tiếng Việt):
{q}

Trả về JSON đúng các khoá sau:
{"type": "<một trong: symptom, disease, drug, pregnancy, pediatric, lab_result, nutrition, mental, procedure, other>",
 "vi_kw": "<từ khoá y khoa tiếng Việt, bỏ phần kể lể, giữ chỉ số/tên thuốc>",
 "zh_q": "<câu hỏi bằng tiếng Trung giản thể, viết như tiêu đề diễn đàn hỏi đáp y khoa (120ask, 39问医生)>",
 "zh_kw": "<thuật ngữ y khoa tiếng Trung, cách nhau bằng dấu cách>",
 "en_kw": "<medical English keywords, space separated, no AND/OR/quotes/brackets>",
 "entities": [{"vi": "<tên trong câu hỏi>", "en": "<tên chuẩn/hoạt chất>", "zh": "<tên tiếng Trung>"}],
 "subqs": [<chỉ khi câu hỏi có nhiều ý: mỗi ý là một câu hỏi tiếng Việt ngắn; ngược lại để []>]}
```

**Hậu xử lý** (`plan.normalize`):
- Chỉ giữ các trường chuỗi không rỗng; bỏ cú pháp toán tử tìm kiếm.
- JSON lỗi thì dùng câu hỏi gốc, không bỏ câu nào.
- Kết quả: 1.186/1.200 plan hợp lệ.
- Bài nộp cuối **chỉ dùng trường `zh_q`**. Các trường khác có trong `plans.json` nhưng không đóng góp vào danh sách được gộp.

**Tái hiện chính xác:** suy luận LLM trên CPU có thể lệch nhẹ giữa các máy. Vì vậy chúng tôi công bố `plans.json` đã dùng (`gs://aiguru_crawler/release/v1/run/plans.json`). Chạy `PLANS=release/plans.json scripts/reproduce.sh retrieve` để bỏ qua bước gọi LLM.

## 3.3 Các mô hình đã thử nhưng không dùng trong bài nộp cuối

| Mô hình | Thử ở | Kết quả |
|---|---|---|
| `intfloat/multilingual-e5-large` (nhúng thứ hai, `bench/add_e5.py`, notebook `3-ingest/colab/embed_e5_colab.ipynb`) | Thêm danh sách e5 vào RRF | Ngang hoặc giảm (0,3472 ở trọng số 0,5 so với 0,3471 khi không dùng; 0,3221 ở trọng số 1,5) |
| Qwen3.5-9B làm LLM judge listwise (`retrieve/judge.py`) | Chấm 60 doc đầu, thang 0–3 | Giảm (0,1677 so với 0,1756) |
| Qwen3.5-9B sinh HyDE (`bench/hyde.py`) | Đoạn văn giả định bằng vi/zh và tiêu đề, dùng làm truy vấn dense | Ngang ở trọng số 0,3, giảm khi trọng số cao hơn |
