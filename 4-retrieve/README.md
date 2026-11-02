# 4 · Retrieve: 1.200 câu hỏi → `submission.json`

**Trạng thái:** bản thiết kế, đang viết v0. **Chưa có tập dev (quyết định 19/10):** mọi bước phụ thuộc tập dev đều bỏ qua, việc tinh chỉnh chỉ dựa vào dò leaderboard. Có tham khảo harness A1–A5 của nhóm ([artifact](https://claude.ai/artifact/Naw1GcxmUE3L7qSnnvjn8o)), nhưng đã chỉnh lại theo dữ liệu và quyết định hiện tại.

Đầu vào: chunk, chỉ mục sparse và vector dense bge-m3 của bước 3 (`<ingest>/rag/…`), cùng `query.parquet` (1.200 câu).
Đầu ra: `submission.json` (nén zip), đúng định dạng của đề.

---

## 1. Những điều đề bài buộc thiết kế phải theo

| Sự thật | Hệ quả |
|---|---|
| **Score = (Docs F2 + Chunks F2) / 2**, tính macro theo từng câu hỏi. F2 = 5h / (k + 4\|G\|) | Tối ưu độc lập **từng câu hỏi**. Recall nặng gấp 4 precision, nhưng nộp thừa vẫn bị phạt qua k |
| Chunk đúng khi **LCS(chunk, gold) / \|gold\| ≥ 0,4**, chỉ so với gold *cùng doc_id* | Chunk dài không bị phạt. Một chunk phủ nhiều gold vẫn tăng recall mà chỉ tính 1 lần vào k, nên **gộp các đoạn liền nhau** có lợi. Chunk thuộc doc sai thì chắc chắn sai |
| Baseline BM25 của BTC: Docs P = 0,88, R = 0,51, F2 = 0,56, đứng trên mọi đội | Số doc đúng mỗi câu hỏi (\|G\|) nhiều khả năng **nhỏ (1–3)**. Nộp ít mà trúng thì thắng; các đội nộp nhiều thì mất precision |
| 1.200 câu hỏi cố định; vòng private dùng lại đúng các câu đó và chỉ được nộp 5 lần | Chạy **offline theo lô**, không giới hạn latency. Mỗi lần nộp ở vòng private chỉ là chọn một cấu hình |
| Model open-weight ≤ 15B, phát hành trước 1/8/2026; cấm model đóng | Phải tự host LLM, reranker, embedding, dịch máy. Hạ tầng GPU (Kaggle/Colab + server HTTP) là giới hạn thật |
| Corpus: **61% vi, 20% zh, 20% en (PubMed)** | Không chỉ bắc cầu Việt–Trung mà **cả Việt–Anh**; và tiếng Việt chiếm đa số |
| Sparse bge-m3 chỉ khớp được cùng ngôn ngữ (đã đo ở bước 3) | Muốn dùng kênh lexical cho zh/en thì **phải dịch câu hỏi**. Dense tự bắc cầu được xuyên ngôn ngữ |

## 2. Đánh giá bản tham khảo: giữ, sửa, bỏ

| Thành phần tham khảo | Quyết định | Lý do |
|---|---|---|
| Chạy theo lô từng bước cho cả 1.200 câu; cache theo hash; khi lỗi thì lùi về baseline | **Giữ nguyên** | Đúng cho bài toán offline. Cache giúp chỉnh bước cuối mà không tốn GPU |
| Số liệu "4,4M URL, 83% tiếng Trung, không có tiếng Anh" | **Sửa** | Đã lỗi thời. Corpus hiện tại là 1,22M URL: vi 61%, zh 20%, en 20%. Phải thêm nhánh tiếng Anh (PubMed) |
| A1 Planner: chuẩn hoá thực thể, câu hỏi con, các biến thể theo ngôn ngữ | **Giữ, thu gọn** | Dịch là *bắt buộc* cho kênh sparse zh/en. Biến thể tiếng Trung "giống tiêu đề diễn đàn" rất đáng giá vì corpus có ~130k trang hỏi đáp. Nhưng **câu hỏi con chỉ bật khi câu hỏi dài và nhiều ý**, không áp cho mọi câu |
| HyDE (sinh câu trả lời mẫu rồi tìm bằng dense) | **Để thí nghiệm, mặc định tắt** | Dense bge-m3 đã bắc cầu Việt–Trung tốt (đo được 0,72 cho đúng trang 120ask). HyDE tốn token và dễ đưa thông tin sai vào truy vấn |
| BM25 với nhiều trường (title, body, forum), tách từ jieba | **Thay** | Đã chốt dùng sparse bge-m3. Trường tiêu đề đã được ghép vào đầu vào embedding; chunk hỏi đáp chứa sẵn câu hỏi |
| Dense ở cấp tài liệu (title + đầu bài) | **Thay** | Index ở cấp chunk, điểm doc = tổng hợp điểm chunk. Cùng một lần tìm cho ra cả doc lẫn chunk |
| RRF → top-200 doc → rerank đoạn bằng cross-encoder | **Giữ** | Reranker cross-encoder đa ngôn ngữ (bge-reranker-v2-m3) chấm được trực tiếp câu Việt với đoạn Trung hoặc Anh. Đây là bước tăng độ chính xác rẻ nhất |
| A3 Judge bằng LLM: chấm 20 thẻ cho **mọi** câu hỏi | **Thu hẹp: chỉ "vùng xám"** | 1.200 × 20 thẻ ≈ 24k lượt đọc, khoảng 35M token trên GPU tự host, tức nhiều giờ. Chỉ gọi judge cho doc có xác suất mơ hồ (0,2–0,8), nơi quyết định nộp hay không thật sự đổi điểm |
| A4 Refine (tối đa 2 vòng, khoảng 40% câu hỏi) | **Hoãn (v3)** | Phức tạp và tốn nhất. Chỉ làm khi leaderboard cho thấy recall hụt vì *thiếu truy vấn* (ví dụ thêm kênh hay bản dịch vẫn còn tăng điểm) |
| A5 Select: hiệu chỉnh thành xác suất rồi **chọn k theo kỳ vọng F2** (thêm doc khi p > F2/5) | **Giữ, là lõi của thiết kế** | Đúng công thức chấm và tối ưu được từng câu hỏi. Đây là chỗ thắng được baseline: nộp 1 doc khi chắc chắn, nộp nhiều khi mơ hồ |
| Hiệu chỉnh bằng LogisticRegression trên 60 câu dev gán nhãn tay | **Bỏ qua (chưa có dev)** | Thay bằng sigmoid trên điểm rerank, có 2 tham số, chỉnh bằng leaderboard |
| Gộp đoạn gần nhau thành span (cho phép trống 1 đoạn) | **Giữ** | Phù hợp với metric (xem mục 1). Chunk bước 3 đã dài 500–1.000 từ, nên việc gộp chủ yếu dùng cho các chunk liền nhau |

## 3. Luồng đề xuất

```
query.parquet
  │
  S1  plan        LLM, 1 lần/câu, có cache      → bản dịch zh/en, từ khoá, thực thể chuẩn hoá, (câu hỏi con nếu dài)
  S2  retrieve    bge-m3 server + chỉ mục bước 3 → dense (câu gốc, mọi ngôn ngữ) + sparse (vi/zh/en theo bản dịch)
                                                  → RRF ở cấp chunk → gom về doc → top-N doc + chunk ứng viên
  S3  rerank      cross-encoder                  → điểm (câu hỏi, chunk); điểm doc = max chunk
  S4  judge       LLM, chỉ vùng xám (v2)          → rel 0–3 + chunk làm bằng chứng
  S5  select      toán, không LLM                → p(doc đúng) → k theo E[F2] → chunk theo ngưỡng + gộp span
  S6  submit      ghi + kiểm định dạng           → submission.json / .zip (đủ 1.200 id; doc_id của chunk ⊆ relevant_docs)
  (S7 eval        bỏ qua: chưa có tập dev; đánh giá bằng leaderboard)
```

### S1 · Plan (LLM)

Đầu ra JSON có schema cố định. Parse lỗi thì lùi về chỉ dùng câu gốc, không bỏ trống câu nào.

```json
{"type": "drug|pregnancy|lab_result|symptom|...",
 "vi_kw": "duphaston nospa mang thai ảnh hưởng thai nhi",
 "zh_q":  "怀孕7周吃地屈孕酮和屈他维林对胎儿有影响吗",
 "zh_kw": "地屈孕酮 屈他维林 孕早期 安全性",
 "en_kw": "dydrogesterone drotaverine first trimester pregnancy safety",
 "entities": [{"vi": "Duphaston", "en": "dydrogesterone", "zh": "地屈孕酮"}],
 "subqs": []}
```

- `zh_q` viết theo **phong cách tiêu đề diễn đàn** (để khớp phần câu hỏi trong chunk của 120ask/39); `zh_kw` và `en_kw` viết theo kiểu thuật ngữ, cho bài viết, bách khoa và PubMed.
- `subqs` chỉ sinh khi câu hỏi dài hơn khoảng 60 từ hoặc có nhiều dấu hỏi. Mỗi câu hỏi con có bộ biến thể riêng.
- **Hook mở rộng truy vấn** `expand(query) -> list[str]`: hiện là no-op. Đây là chỗ để sau này cắm bí danh từ OKF vào.

### S2 · Retrieve

| Kênh | Truy vấn | Chỉ mục | k |
|---|---|---|---|
| dense | câu gốc vi (và từng câu hỏi con) | dense vi + zh + en | 200 mỗi ngôn ngữ |
| sparse vi | câu gốc + `vi_kw` | sparse vi | 200 |
| sparse zh | `zh_q`, `zh_kw` | sparse zh | 200 |
| sparse en | `en_kw` | sparse en | 200 |

- Embed truy vấn qua server bge-m3 của nhóm: khoảng 1.200 × 6 câu ngắn, mất vài phút.
- **RRF** (k = 60) trên các danh sách chunk. Điểm doc = RRF của chunk tốt nhất, cộng một chút cho doc có nhiều chunk cùng trúng.
- Lấy **top-50 doc**, mỗi doc tối đa 3 chunk ứng viên, chuyển sang S3.
- Ghi lại thứ hạng theo từng kênh để làm đặc trưng cho S5.

### S3 · Rerank

- Model mặc định **`BAAI/bge-reranker-v2-m3`** (568M, đa ngôn ngữ); phương án so sánh là `Qwen3-Reranker-4B`. Chạy trên máy GPU, gọi qua server HTTP giống bge-m3.
- Chấm theo cặp (câu hỏi gốc tiếng Việt, chunk). Nếu chunk là zh/en mà điểm thấp, chấm thêm với bản dịch (`zh_q` hoặc `en_kw`) và lấy max.
- Khoảng 1.200 × 150 = 180k cặp, mất khoảng 1–2 giờ trên một GPU T4.

### S4 · Judge (v2, chỉ vùng xám)

- Chỉ đưa vào judge các doc có p ∈ [0,2; 0,8] sau S5 lần 1, tối đa 5 doc mỗi câu.
- Thẻ đưa vào gồm: tiêu đề, `doc_type`, tối đa 3 chunk (cắt ngắn). Đầu ra là `rel` 0–3 và `evidence_chunks`.
- `rel` trở thành một đặc trưng của S5, không phải quyết định cuối.

### S5 · Select (toán, chạy trong vài giây)

1. **Hiệu chỉnh** điểm thành p(doc đúng) bằng các đặc trưng: điểm rerank, hạng RRF, số kênh trúng, `doc_type`, `doc_flags` (fallback, meta_only…), và `rel` nếu có.
   - Hàm sigmoid trên điểm rerank với 2 tham số, có cộng thêm theo số kênh trúng và `rel` của judge. Các tham số chỉnh bằng leaderboard (không có tập dev).
2. **Chọn k cho từng câu hỏi:** xếp doc theo p, thêm doc kế tiếp khi `p > E[F2 hiện tại] / 5`, với `E|G| = Σp + m` (m là phần đúng chưa tìm thấy, ước lượng từ leaderboard). Câu nào cũng nộp ít nhất 1 doc.
3. **Chọn chunk:** với mỗi doc đã chọn, lấy chunk có điểm rerank ≥ τ·max, cộng chunk mà judge chỉ ra. Chunk liền nhau trong cùng doc thì **gộp thành một span**. Chunk hỏi đáp có chung phần câu hỏi thì gộp theo hợp các `spans` của bước 3, không lặp câu hỏi. Text lấy lại nguyên văn từ `spans`.
4. Mọi tham số (m, τ, các tham số sigmoid, giới hạn số chunk) đều nằm trong `config`. Đổi tham số chỉ cần chạy lại S5 + S6, mất vài giây.

### S6 · Submit

- Ghi JSON đúng schema của đề: `id` là int, `relevant_docs` là danh sách id số nguyên, `relevant_chunks` gồm `{doc_id, chunk_text}`.
- Kiểm tra trước khi ghi: đủ 1.200 câu; id là số nguyên; `doc_id` của mọi chunk nằm trong `relevant_docs`; không có chunk rỗng; không quá 1% số câu thiếu doc (vượt ngưỡng nghĩa là truy hồi hỏng). Việc `chunk_text` là nguyên văn được đảm bảo ở bước 3 (`join_spans`, có test trên fixture thật) và khi gộp span ở S5.
- Nén zip đúng một file, không có thư mục con.

### S7 · Eval: bỏ qua

Chưa có tập dev nên chưa có gì để chấm offline. Khi nào có nhãn thì thêm `metric.py`, cài lại P/R/F2 và luật LCS ≥ 0,4.

## 4. Dò leaderboard (thay cho tập dev)

Chưa có tập dev, nên mọi tham số của S5 được chỉnh bằng các lần nộp có chủ đích (vòng public cho 10 lần mỗi ngày). Mỗi lần nộp chỉ đổi **một** yếu tố so với lần trước:
  1. *Chỉ top-1 doc* → biết precision@1. Kết hợp với recall sẽ ước lượng được trung bình \|G\|, dùng làm `m` cho S5.
  2. *Cùng danh sách doc nhưng đổi độ dài chunk* (1 chunk / gộp span / cả doc) → biết chunk gold dài cỡ nào.
  3. *Chỉ kênh dense* so với *dense + sparse* → biết sparse có đáng giữ không.

## 5. Cấu trúc thư mục

```
4-retrieve/
  README.md                 tài liệu này
  retrieve/
    __main__.py             CLI: plan | search | rerank | judge | select | submit | eval | run (chạy cả luồng)
    config.py               mọi tham số trong một file; mỗi lần chạy lưu lại bản config đã dùng
    cache.py                cache theo hash(bước, đầu vào, config): chạy lại không tốn GPU
    clients.py              client HTTP cho LLM (OpenAI-compatible, ví dụ vLLM), reranker, bge-m3 (dùng lại HttpEncoder của bước 3)
    plan.py                 S1: prompt, parse JSON, lùi về câu gốc khi lỗi, hook expand()
    search.py               S2: gọi ingest.dense / ingest.sparse, RRF, gom về doc
    rerank.py               S3
    judge.py                S4 (v2)
    select.py               S5: hiệu chỉnh, chọn k theo E[F2], chọn và gộp chunk
    submit.py               S6: ghi, kiểm tra, nén zip
  prompts/plan.md, judge.md
  runs/<run_id>/            đầu ra từng bước (parquet/jsonl), config.json, submission.zip, eval.json
  tests/
```

Mỗi bước đọc đầu ra của bước trước trong `runs/<run_id>/`, nên chạy lại được từng bước và so sánh được các lần chạy với nhau.

## 6. Lộ trình theo phiên bản

| Phiên bản | Gồm | Cần GPU | Mục tiêu |
|---|---|---|---|
| **v0** | S2 chỉ dense + sparse vi, RRF, cố định top-2 doc và top-1 chunk mỗi doc, S6 | bge-m3 (truy vấn) | **bài nộp đầu tiên**: kiểm định dạng và lấy mốc điểm |
| **v1** | + S1 (dịch zh/en) + sparse zh/en + S3 rerank + S5 theo E[F2] (hiệu chỉnh bằng sigmoid) | + LLM, reranker | vượt baseline BTC (0,376) |
| **v2** | + S4 judge vùng xám, câu hỏi con; chỉnh tham số S5 bằng dò leaderboard | + LLM nhiều hơn | tối ưu precision và độ dài chunk |
| v3 | + vòng refine, HyDE, mở rộng truy vấn bằng OKF | | chỉ làm nếu leaderboard cho thấy recall hụt |

## 7. Rủi ro và phụ thuộc

- **Đường găng (critical path) nằm ở dữ liệu, không ở bước 4.** Kho sạch (bước 2) chưa chạy đầy đủ vì vướng quyền bucket, và embedding corpus (bước 3) cần GPU chạy trực tiếp. Vòng public đóng **sáng 31/10**. Bước 4 viết và test được trên dữ liệu mẫu, nhưng muốn có bài nộp đầu tiên thì phải chạy xong bước 2 và 3.
- **Tốc độ server GPU:** bge-m3 qua ngrok đo được khoảng 1,9 giây cho mỗi đoạn dài. Truy vấn thì đủ dùng; reranker 180k cặp mà chạy qua server chậm như vậy thì không xong. **Reranker phải chạy batch ngay trên máy GPU** giống `rag embed`.
- **LLM ≤ 15B đọc tiếng Trung và tiếng Việt:** cần thử trên khoảng 20 câu trước khi chạy cả 1.200 câu (chất lượng dịch y khoa, JSON hợp lệ).
- **Overfit leaderboard:** vòng public chỉ chấm một phần các câu hỏi, và không có tập dev để đối chiếu. Mỗi lần nộp chỉ đổi một yếu tố, và chỉ giữ những thay đổi tăng điểm rõ ràng.

## 8. Cần nhóm quyết định

1. **LLM cho S1/S4** và nơi chạy: Qwen3-8B hay Qwen3-14B (AWQ) bằng vLLM trên Kaggle 2×T4, mở qua server giống bge-m3?
2. **Reranker:** bge-reranker-v2-m3 (nhẹ) hay Qwen3-Reranker-4B (mạnh hơn nhưng chậm hơn)?
3. Đồng ý **lộ trình v0 → v1 → v2** và đưa refine/HyDE xuống v3?

## 9. Chạy v0 (đã có code, đã test)

```bash
# ingest root phải nằm trên đĩa local (shard và chỉ mục được đọc bằng mmap):
gcloud storage rsync -r gs://<bucket>/ingest/v1/rag ./ingest/rag
python -m retrieve run --queries ../../data/query_v2.parquet --ingest ./ingest \
    --run-dir runs/v0 --server <bge-m3 server URL> --k-docs 2 --k-chunks 1
# → runs/v0/{config.json, candidates.parquet, submission.json, submission.zip, summary.json}
```

- v0 gồm: dense (câu gốc tiếng Việt, tìm trên vi/zh/en) + sparse vi → RRF ở cấp chunk → top `k_docs` doc → top `k_chunks` chunk mỗi doc. Chunk liền nhau được gộp thành một span nguyên văn; chunk hỏi đáp có chung câu hỏi thì câu hỏi chỉ xuất hiện một lần.
- `submit.validate` chặn trước những lỗi khiến bài không được chấm hoặc mất điểm: thiếu câu, id không phải số nguyên, chunk thuộc doc không có trong `relevant_docs`, chunk rỗng.
- Đổi `k_docs`/`k_chunks` chỉ cần chạy lại lệnh, không phải dựng lại chỉ mục. Đây là thí nghiệm dò leaderboard số 1 và 2 ở mục 4.

## 10. Bảo vệ khỏi lần chạy hỏng mà không báo lỗi (sau review 20/10)

Các tình huống sau làm lệnh **dừng với lỗi** (`DegradedRun` / `ValueError`) thay vì ra một file nộp "hợp lệ" nhưng gần rỗng. Thêm `allow_degraded: true` trong config nếu cố ý muốn chạy tiếp.
- `--ingest` sai đường dẫn, hoặc không có danh sách kết quả nào.
- Chỉ mục cũ: số shard dense khác số phần chunk, hoặc chỉ mục sparse không chứa đủ chunk.
- LLM sập: dưới 80% plan hợp lệ (mode v1/v2 sẽ lặng lẽ thành v0); dưới 50% phán quyết judge parse được.
- Quá 1% câu hỏi không có doc nào.

Ngoài ra:
- Dense gộp các ngôn ngữ thành **một danh sách theo cosine** cho mỗi biến thể câu hỏi. Ứng viên được cắt còn 50 doc × 3 chunk mỗi câu.
- Reply LLM không parse được thì không được cache.
- Embedding câu hỏi được cache theo tên model, commit theo từng lô.

## 11. Bộ xếp hạng học (20/10)

Kết quả bench r1 cho thấy RRF chỉ dựa trên thứ hạng nên các danh sách tiếng Việt, vốn nhiều và tương quan với nhau, lấn át một kết quả en/zh rất mạnh. Vì vậy PubMed và zh gần như bằng 0.

- `retrieve/learn.py` tạo đặc trưng cho từng cặp (câu hỏi, doc) gồm:
  - cosine dense tuyệt đối và khoảng cách tới doc đứng đầu;
  - điểm sparse của từng ngôn ngữ, cả tuyệt đối lẫn tương đối;
  - thứ hạng, số danh sách trúng, ngôn ngữ của doc.
- Từ các đặc trưng đó, một logistic regression ước lượng p(doc đúng). Model chỉ là một file JSON (`models/ranker_v1.json`, `ranker_v2.json`), nên lúc chạy không cần sklearn.
- Mode v1 và v2 dùng `select="learned"`:
  - p của ranker quyết định doc nào được chọn và cắt bao nhiêu doc theo kỳ vọng F2;
  - RRF chỉ còn dùng để chọn chunk bên trong mỗi doc;
  - judge cũng lấy vùng xám theo p của ranker.
- `select="f2"` (p không cần nhãn) vẫn còn để dự phòng.
- Huấn luyện lại: `python bench/train_ranker.py <bench> <server> <run_dir có plans.json>`. Script chạy CV 5 fold chia theo câu hỏi rồi lưu model.

Kết quả trên dev silver (237 câu):

| | v0 | v1 | v2 (+judge) |
|---|---|---|---|
| r1 (RRF) | 0.294 | 0.407 | 0.534 |
| CV ngoài fold (ranker) | – | 0.848 | 0.838 (chưa có judge) |
| r2 end-to-end | 0.294 | 0.844 | 0.853 |

Lưu ý:
- Số r2 là in-sample, vì model được học trên chính tập này. Số trung thực là số CV.
- Câu silver được sinh từ một chunk, nên mỗi câu có đúng 1 doc gold và câu hỏi khá sát văn bản. Đề thật nhiều khả năng có nhiều doc đúng: baseline BM25 của ban tổ chức có P .88 / R .51. Vì vậy cần chỉnh `f2_missing` (tăng lên để nộp nhiều doc hơn) bằng leaderboard.
- Judge hiện chạy trên LLM demo đóng, chỉ dùng để thử. Khi nộp thật phải thay bằng model mở ≤15B.

## 12. Chạy trên VM: LLM Qwen3.5-9B qua OpenRouter, encode câu hỏi bằng CPU (21/10)

- LLM: `qwen/qwen3.5-9b` trên OpenRouter (open weights, 9B, phát hành 10/3/2026, hợp luật ≤15B và trước 1/8). Key chỉ truyền qua env `LLM_API_KEY`. Tắt reasoning cho nhanh và rẻ (bật reasoning tốn khoảng 800 token cho một câu dịch):
  ```
  --llm-url https://openrouter.ai/api/v1 --llm-model qwen/qwen3.5-9b --llm-extra '{"reasoning": {"enabled": false}}'
  ```
- Encoder câu hỏi: bỏ `--server` (hoặc `--server cpu`) thì bge-m3 chạy trên CPU. Câu hỏi ngắn nên 1.200 câu × khoảng 6 biến thể chỉ mất vài phút, và không còn phụ thuộc server ngrok.
- Image: `docker build -f 4-retrieve/Dockerfile -t r2ai-retrieve:v1 .` (chạy từ `main-pipeline/`). GCS dùng ADC.

## 13. Chạy không cần GPU: BM25 + rerank dense trên CPU (21/10)

- `--server bm25 --config configs/bm25.json`: chỉ kênh BM25. Plan dịch zh/en, ranker, judge và F2 cut vẫn dùng như cũ. Ranker riêng nằm ở `models/bm25/` (chọn bằng env `RETRIEVE_MODELS`).
- `--server bm25+cpu --config configs/bm25_rerank.json` (`rerank_docs: 30`):
  - Lấy 30 doc đứng đầu sau khi gộp kết quả BM25, mỗi doc một chunk tốt nhất. Chỉ những chunk này được embed bằng bge-m3 trên CPU (cắt 2.000 ký tự).
  - Cosine với các biến thể câu hỏi dense (orig, zh_q, subq) thành các danh sách "dense" bình thường, nên ranker lấy lại được đặc trưng dense.
  - Vector chunk được cache theo `chunk_id` trong `<cache-dir>/chunk_vec.sqlite`.
- Toàn bộ chạy tự động trên VM: `bench/full_run.sh bm25`.

Bench silver (237 câu, Qwen3.5-9B):

| | v0 | v1 | v2 |
|---|---|---|---|
| bge-m3 đầy đủ | 0,294 | 0,817 | 0,861 |
| BM25 (ranker BM25) | 0,278 | 0,583 | 0,723 |

## 14. Thứ tự chạy trên VM khi notebook embed đang chạy (21/10)

- `bench/final_cpu.sh`: BM25 + rerank CPU + Qwen3.5-9B Q4_K_M trên llama.cpp (`localhost:8080`, 4 slot, `--reasoning-budget 0`). Đây là bản không cần GPU và đến sớm hơn. Chạy v0/v1 trước, sau đó kiểm tra lại trên bench, rồi đến v2.
- `bench/after_embed.sh`: cứ 10 phút đếm shard của notebook [3-ingest/colab/embed_colab.ipynb](../3-ingest/colab/embed_colab.ipynb). Khi đủ 109/109 phần, script dựng index sparse bge-m3, kéo dense + index về VM, rồi chạy v0/v1/v2 với nhánh bge-m3 đầy đủ (`configs/bge_cpu.json`: judge 2 doc × 1 chunk × 1.500 ký tự). Script dùng chung cache với lượt BM25, nên plan Qwen không phải sinh lại.
- Tốc độ notebook trên T4: khoảng 45 chunk/s, tức khoảng 10 giờ cho 1,57 triệu chunk. Nếu phiên Colab bị ngắt, chạy lại notebook với cùng `SHARD`.

## 15. Điều chỉnh theo leaderboard: hybrid sâu + LLM judge listwise (21/10)

Các lần nộp thăm dò cho thấy gold khác hẳn bộ silver:
- Mỗi câu có khoảng 45 doc gold.
- Gold không có PubMed và có ít doc tiếng Trung.
- Nộp càng sâu điểm càng cao.

Vì vậy bước cắt F2 và ranker học trên silver bị bỏ. Chi tiết xem memory `leaderboard-gold-facts`.

- Mode `deep` (`select="topk"`): gộp các danh sách bằng RRF có trọng số rồi lấy top-k doc. Độ sâu, số doc có chunk và trọng số ngôn ngữ nằm trong `configs/deep_*.json`:
  - `k_docs` / `cand_docs`: số doc nộp / số ứng viên giữ lại;
  - `chunk_docs`: chỉ gắn chunk cho N doc đầu;
  - `lang_weight`: `{"en": 0, "zh": 0.5}`.
- Judge listwise (`deep_judge`, `judge_batch`, `judge_chars`): LLM chấm 60 doc đầu mỗi câu, mỗi lần gọi 10 doc (tiêu đề + 700 ký tự của chunk tốt nhất).
  - Thứ tự mới: điểm 3 > 2 > 1 > chưa chấm (giữ thứ tự gộp) > điểm 0.
  - `cand_docs > k_docs`, để doc bị chấm 0 rơi ra ngoài phần nộp.
- `bench/hybrid.sh`: chờ notebook embed đủ các phần, dựng index bge-m3, rồi chạy `deep_bge_k100`, `deep_bge_judge_k100`, `deep_bge_judge_k200`. Plan dùng lại từ `plans.json` qua `--plans`; judge gọi Qwen3.5-9B trên OpenRouter.
