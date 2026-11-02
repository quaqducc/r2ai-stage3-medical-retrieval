# Chiến lược chunking (luồng A · RAG)

**Trạng thái:** đã chốt ngày 18/10/2026.

Đầu vào là cột `text` của kho sạch (bước 2). Đầu ra là các chunk dùng cho cả index (sparse + dense của bge-m3) lẫn `chunk_text` khi nộp bài.

## 1. Nguyên tắc chung

1. **Nguyên văn.** Mọi chữ trong chunk được cắt nguyên văn từ `text` của tài liệu: không viết lại, không sinh thêm. Mỗi chunk lưu danh sách đoạn nguồn `spans = [(start, end), …]`, và `chunk_text == join_spans(text, spans)`. Hàm này nối thẳng hai span chạm nhau, còn hai span cách nhau thì nối bằng một dòng trống.
   - Span của các chunk liên tiếp **phủ kín văn bản không chừa khe hở**: chunk này kết thúc đúng chỗ chunk sau bắt đầu. Vì vậy chunk bài viết, PubMed và hỏi đáp một câu trả lời (`qa_single`) là **chuỗi con nguyên văn** của doc, và gộp các chunk liền nhau vẫn ra chuỗi con nguyên văn.
   - Ngoại lệ duy nhất là chunk câu trả lời thứ 2 trở đi của trang hỏi đáp đã tách (`qa_answer`): chunk này gồm câu hỏi và câu trả lời đó, là 2 span không liền nhau (theo quyết định ở mục 3). Rủi ro là nếu bộ chấm đòi chunk phải là chuỗi con của doc, những chunk này sẽ không được tính. Nên kiểm chứng điều này bằng một lần nộp thử.
2. **Đo độ dài bằng từ, không bằng ký tự.**
   - vi, en: 1 từ là 1 token tách bởi khoảng trắng. Với tiếng Việt, đó là 1 âm tiết: "sốt xuất huyết" tính 3 từ.
   - zh: không có khoảng trắng nên quy ước **1,6 chữ Hán = 1 từ**, tức 500 từ ≈ 800 chữ Hán.
   - Dấu câu, kể cả dấu toàn khổ (`，` `。`), không được tính là từ.
   - Số, ký hiệu và chữ Latin xen trong văn bản tiếng Trung được đếm theo khoảng trắng như vi/en.
3. **Cắt theo cấu trúc trước, theo độ dài sau.** Thứ tự ưu tiên của ranh giới:
   1. heading markdown (`#`, `##`, `###`…)
   2. ranh giới câu hỏi / câu trả lời
   3. đoạn văn (`\n\n`)
   4. câu (`.` `?` `!` cho vi/en; `。` `？` `！` cho zh)

   Không bao giờ cắt giữa câu.
4. **Tiêu đề không chép vào chunk.** Tiêu đề và heading cha được ghép vào **đầu vào khi tạo embedding** để thêm ngữ cảnh, nhưng không nằm trong `chunk_text`. Ngoại lệ là câu hỏi ở trang hỏi đáp (mục 3), vì câu hỏi là một phần nội dung trang.

## 2. Phân loại tài liệu

Mỗi doc được gán `doc_type` dựa trên host, mẫu URL và dấu hiệu trong markdown (không dùng mô hình phân loại):

| `doc_type` | Nhận biết | Ví dụ host |
|---|---|---|
| `qa` | host hỏi đáp, URL dạng `/question/`, hoặc có dấu hiệu hỏi–đáp | 120ask, ask.39.net, familydoctor QA, vinmec "Hỏi bác sĩ", vnexpress "Trả lời" |
| `pubmed` | host pubmed | pubmed.ncbi.nlm.nih.gov |
| `article` | mọi loại còn lại: blog, bài chia sẻ, báo, trang bệnh/thuốc, bách khoa, sách | thanhnien, suckhoedoisong, Long Châu, hellobacsi, msdmanuals, zysjonline… |

## 3. Trang hỏi đáp (`qa`)

**Bố cục:** 1 trang = 1 câu hỏi + 0..N câu trả lời. Không có trang nào chứa nhiều câu hỏi khác nhau.

**Tách phần hỏi và phần trả lời theo dấu hiệu của từng site:**

| Site | Phần hỏi | Ranh giới giữa các câu trả lời |
|---|---|---|
| ask.39.net | tiêu đề + nội dung + `补充说明：` (gộp các dòng lặp lại thành một) | dòng tên bác sĩ: `<tên> <chức danh> <bệnh viện> <hạng BV>`, chức danh ∈ {医师, 主治医师, 副主任医师, 主任医师} |
| 120ask | `健康咨询描述：`, `曾经的治疗情况和效果：`, `想得到怎样的帮助：` | mỗi câu trả lời là một khối `病情分析：… 指导意见：… 生活护理：`; câu trả lời kiểu cũ thì là đoạn văn bắt đầu bằng "您好" |
| vinmec / vnexpress | đoạn câu hỏi (vnexpress mở đầu bằng "Độc giả hỏi") | dòng "Trả lời:" hoặc dòng tên bác sĩ trả lời |
| FAQ trong bài (Long Châu, hellobacsi) | heading câu hỏi | mỗi cặp hỏi–đáp xử lý như một trang hỏi đáp nhỏ |

**Quy tắc cắt chunk:**

| Trường hợp | Chunk |
|---|---|
| Có **1 câu trả lời** | **1 chunk = câu hỏi + câu trả lời** |
| Có **nhiều câu trả lời**, tổng (hỏi + tất cả trả lời) ≤ 600 từ | 1 chunk = câu hỏi + tất cả câu trả lời |
| Có **nhiều câu trả lời**, tổng > 600 từ | **mỗi câu trả lời là 1 chunk, và chunk nào cũng chép toàn bộ câu hỏi lên đầu** |
| Trang có câu hỏi nhưng **không có câu trả lời** | 1 chunk = câu hỏi |

- Ngưỡng 600 từ chỉ dùng để quyết định có tách theo câu trả lời hay không.
- **Không giới hạn độ dài chunk hỏi đáp.** Đã tách thì mỗi chunk là **toàn bộ câu hỏi** cộng **toàn bộ một câu trả lời**, không cắt nhỏ thêm, dù dài bao nhiêu. Trang chỉ có 1 câu trả lời thì luôn là 1 chunk.
- Mỗi chunk có thêm cột `unit_kind`, nhận một trong các giá trị: `qa_single`, `qa_answer`, `qa_question`. Cột `answer_idx` cho biết thứ tự câu trả lời.

Lý do chép câu hỏi lên đầu: câu trả lời thường không nhắc lại vấn đề, ví dụ chỉ viết "你好，可以看出来的" (chào bạn, có thể nhận ra được). Thiếu câu hỏi thì embedding không biết câu trả lời đang nói về chuyện gì.

## 4. PubMed (`pubmed`)

- 1 chunk = **toàn bộ abstract** (một span liền mạch). Tiêu đề không nằm trong chunk mà được ghép vào đầu vào embedding, vì trong `text` tiêu đề và abstract bị ngăn cách bởi dòng tạp chí/PMID. Abstract thường dài 200–400 từ, ngắn hơn ngưỡng tối thiểu, nên không cắt.
- Abstract có nhãn từng phần và dài hơn 1.000 từ (hiếm): cắt theo nhãn (Background / Methods / Results / Conclusions), áp dụng ngưỡng min/max ở mục 5.
- Dòng tạp chí/PMID và dòng MeSH/keywords **không** vào chunk. Chúng được lưu ở cột `meta_text` của thẻ tài liệu.

## 5. Bài viết, blog, bài chia sẻ, sách, trang bệnh/thuốc, bách khoa (`article`)

**Chia theo heading markdown, ngưỡng `min_words = 500`, ⚠️ `max_words = 1000`.**

Thuật toán:

1. **Tách section:** cắt `text` (đối chiếu heading từ cột `markdown`) thành các section. Mỗi section bắt đầu từ một heading và kéo tới heading kế tiếp, bất kể cấp. Phần trước heading đầu tiên (sapo, đoạn mở đầu) là section 0.
2. **Cắt section quá dài:** section nào > `max_words` thì cắt theo đoạn văn. Đoạn nào vẫn > `max_words` thì cắt tiếp theo câu. Mảnh tạo ra ≤ `max_words`.
3. **Gộp tham lam:** đi từ đầu tới cuối, cộng dồn các section/mảnh liền nhau vào chunk hiện tại cho tới khi đạt ≥ `min_words`. Nếu thêm section kế tiếp mà vượt `max_words` thì đóng chunk trước khi thêm.
4. **Phần dư cuối bài:** nếu chunk cuối < `min_words` thì gộp vào chunk trước khi tổng ≤ `max_words`; nếu không gộp được thì giữ thành chunk riêng.
5. **Doc ngắn:** doc có tổng số từ < `min_words` thì cả doc là 1 chunk. Doc có cờ `meta_only` (chỉ còn sapo) cũng xử lý như vậy.
6. Mỗi chunk lưu `heading_path`, là chuỗi heading cha của section đầu tiên trong chunk (ví dụ `Dương vật cong > Chẩn đoán`), dùng làm ngữ cảnh khi index.

Heading nằm trong `text` (chữ nguyên văn của heading), nên chunk bắt đầu bằng heading của section là đúng nguyên văn.

## 6. Schema chunk

| Cột | Ý nghĩa |
|---|---|
| `chunk_id` | `"<doc_id>:<idx>"` |
| `doc_id`, `idx` | id corpus, thứ tự chunk trong doc |
| `doc_type`, `unit_kind`, `answer_idx` | `qa` / `pubmed` / `article`; loại đơn vị; thứ tự câu trả lời (chỉ có ở qa) |
| `spans` | danh sách `(start, end)` trong `text` của doc |
| `text` | nội dung chunk (nối các span) |
| `n_words` | độ dài theo quy ước ở mục 1 |
| `title`, `heading_path` | ngữ cảnh khi index, không nằm trong `text` |
| `host`, `lang`, `text_lang`, `doc_flags` | metadata |

## 7. Kiểm tra bắt buộc (test)

- Mọi chunk: `join_spans(text, spans) == chunk_text`. Mọi chunk trừ `qa_answer` thứ 2 trở đi là chuỗi con nguyên văn của `text` (đã kiểm trên toàn bộ fixture thật: 188/188 chunk).
- Mọi chữ của doc `article` và `pubmed` thuộc ít nhất một chunk (không rơi mất đoạn nào). Riêng dòng metadata PubMed được miễn.
- Chunk `article` có độ dài trong [500, 1000] từ. Ngoại lệ: doc ngắn hơn 500 từ; phần dư cuối bài không gộp được; chunk phải đóng sớm vì section kế tiếp lớn tới mức gộp vào sẽ vượt 1.000 từ; hoặc một câu đơn lẻ đã dài hơn 1.000 từ.
- Chunk `qa`: chunk nào sinh ra khi cắt nhiều câu trả lời đều bắt đầu bằng câu hỏi.
- Tách hỏi–đáp chạy đúng trên fixture thật của ask.39.net (2 câu trả lời), 120ask, vinmec.

## 8. Cách điều chỉnh sau này

Ban tổ chức không công bố độ dài chunk gold. Tham số `max_words` / `min_words` và kích thước chunk hỏi đáp được đặt trong cấu hình. Sau lần nộp đầu tiên, so `CHUNKS F2` giữa 2–3 cấu hình để chọn. Một chunk được tính đúng khi LCS(chunk, gold)/|gold| ≥ 0,4, tức metric không phạt chunk dài. Chỉ có điều chunk nộp sai làm giảm precision.

## Các quyết định đã chốt (18/10/2026)

- `article`: `min_words = 500`, `max_words = 1000`.
- Tiếng Trung: 1,6 chữ Hán = 1 từ.
- Hỏi đáp: chép toàn bộ câu hỏi lên đầu mỗi chunk câu trả lời; không giới hạn độ dài chunk hỏi đáp.
