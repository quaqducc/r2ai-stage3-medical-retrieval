# 4. Kết quả và các thí nghiệm

Đề không có tập dev có nhãn, nên mọi lựa chọn đều được kiểm chứng bằng các lần nộp ở vòng public. Mỗi lần nộp chỉ đổi một yếu tố. Điểm là FINAL = (Docs F2 + Chunks F2) / 2, macro theo câu hỏi.

## 4.1 Bài nộp được báo cáo

| Bài | FINAL | Docs F2 (P / R) | Chunks F2 (P / R) | Ghi chú |
|---|---|---|---|---|
| **`i_dprf15`** | **0,3577** | 0,3900 (0,242 / 0,577) | 0,3255 (0,228 / 0,448) | bài tốt nhất; hạng 7/50 trên leaderboard public |
| `j_hh125` | 0,3577 | 0,3902 (0,242 / 0,577) | 0,3253 (0,228 / 0,448) | như trên, các danh sách head/mean nhân thêm × 1,25 |
| `i_dprf1` | 0,3575 | 0,3897 (0,241 / 0,577) | 0,3253 (0,228 / 0,448) | danh sách doc-level PRF ở trọng số 1,0 |
| `i_dprf05` | 0,3570 | 0,3894 (0,241 / 0,577) | 0,3247 (0,227 / 0,447) | doc-level PRF ở trọng số 0,5 |
| `d12_both075` | 0,3526 | 0,3847 (0,238 / 0,573) | 0,3205 (0,224 / 0,442) | không có doc-level PRF |

Năm bài trên là các bài nộp ở vòng private. File của chúng nằm ở `gs://aiguru_crawler/release/v1/submissions/`. Cấu hình từng bài nằm trong `4-retrieve/bench/variants.py`.

## 4.2 Tiến trình

| Mốc | FINAL | Thay đổi |
|---|---|---|
| bài đầu tiên | 0,009 | bộ xếp hạng học trên dev silver tự sinh, chỉ nộp 2–5 doc. Dev silver lệch hẳn gold thật |
| BM25 top-100 (vi) | 0,108 | nộp sâu: mỗi câu có khoảng 45 doc gold, nên F2 thưởng recall |
| hybrid bge-m3 dense + sparse | 0,1756 | |
| gắn chunk cho 80 doc | 0,2605 | số chunk là đòn bẩy lớn nhất |
| + danh sách `zh_q`, sparse × 0,25, 2 chunk × 75 doc | 0,3212 | |
| + PRF Rocchio, phân bổ chunk toàn cục (120 chunk, ≤ 3/doc) | 0,3321 | |
| zh × 0,3, chunk xếp từ danh sách PRF, 135 chunk, `w_trans` 0,7, `rrf_k` 80 | 0,3471 | |
| + danh sách cấp tài liệu (head + mean) | 0,3526 | |
| trọng số head-heavy, `rrf_k` 100 | 0,3547 | |
| + danh sách cấp tài liệu của vector PRF (× 1,5) | **0,3577** | |

## 4.3 Những gì không hiệu quả

| Thí nghiệm | Kết quả | Kết luận |
|---|---|---|
| LLM judge listwise (Qwen3.5-9B chấm 0–3) | 0,1677 so với 0,1756 | gold bám theo truy hồi, không theo phán đoán độ liên quan của LLM |
| BM25 thay hoặc thêm vào sparse bge-m3 | thấp hơn | |
| Thêm danh sách từ bản viết lại của LLM (từ khoá, câu hỏi con) | Docs F2 0,272 so với 0,286 | chỉ giữ bản dịch `zh_q` |
| HyDE (đoạn văn giả định do LLM viết) | ngang ở trọng số 0,3; 0,3261 ở trọng số 1,0 | |
| multilingual-e5-large làm nhúng thứ hai | ngang ở 0,5; giảm ở 1,0–1,5 | chỉ các tín hiệu từ bge-m3 có ích |
| Cắt chunk còn 1.500 ký tự | chunk P 0,20 → 0,08 | không bao giờ cắt chunk |
| Ghép chunk thành text dài (cả doc / gộp theo doc) | 0,28–0,34 | bộ chấm ghép một chunk nộp với một chunk gold |
| Điểm doc = tổng nhiều chunk (`doc_agg`) | thấp hơn | |
| Dùng danh sách cấp tài liệu để xếp hạng chunk | 0,3506 | danh sách cấp tài liệu chỉ dùng cho doc |

## 4.4 Nhận xét về gold (suy ra từ leaderboard)

- Mỗi câu có khoảng 45 doc gold và 58 chunk gold. Recall quan trọng hơn precision, nên nộp 150 doc.
- Gold không có tài liệu PubMed: bài chỉ gồm doc tiếng Anh được đúng 0 điểm. Doc tiếng Trung cũng ít, nên trọng số zh là 0,35.
- Gold có vẻ được dựng từ một pipeline dựa trên bge-m3. Các tín hiệu khác bge-m3 (BM25, e5, LLM judge, HyDE) đều không cải thiện điểm.
