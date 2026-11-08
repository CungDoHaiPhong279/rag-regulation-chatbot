# Bộ câu hỏi đánh giá

`questions.jsonl`: mỗi dòng một câu hỏi (JSON). Hiện chỉ có 3 câu mẫu (`sample-*`, split `dev`) để kiểm tra
công cụ; **70 câu thật do người dùng tự viết và gán nhãn**.

## Định dạng

```json
{"id": "q07", "question": "...", "in_scope": true, "split": "test",
 "gold": [{"source_file": "thong_tu_08_2021.pdf", "article": "Điều 11", "pages": [13, 14]}]}
```

| Trường | Ý nghĩa |
|---|---|
| `id` | Duy nhất trong file |
| `question` | Câu hỏi như sinh viên thật sẽ gõ |
| `in_scope` | `true` nếu tài liệu có câu trả lời; `false` nếu chatbot phải từ chối |
| `split` | `test` (chỉ dùng để báo cáo) hoặc `dev` (dùng để chỉnh ngưỡng, chọn `chunk_size`) |
| `gold` | Các Điều chứa câu trả lời; `pages` là các trang của Điều đó có nội dung trả lời. Câu `in_scope=false` để `[]` |

Một chunk được tính là **trúng** khi cùng `source_file`, cùng số Điều và `page_number` nằm trong `pages`.
Câu cần ghép 2 Điều thì ghi 2 phần tử `gold`; Recall của câu đó là tỉ lệ Điều tìm thấy.

## Quy tắc gán nhãn

- **70 câu:** 50 `test` + 20 `dev`. Mỗi split có câu ngoài phạm vi (khoảng 10 câu trong `test`).
- `dev` có **2 câu gõ không dấu** (ví dụ `sinh vien bi canh bao hoc tap khi nao`) để đo độ bền với cách gõ thật.
- Chưa có quy chế trường thì các câu về học phí, học bổng của trường là **ngoài phạm vi**.
- Viết câu hỏi **trước** khi tra cứu, bằng lời của sinh viên, không chép nguyên câu trong văn bản: chép nguyên văn
  làm BM25 trúng dễ và Recall bị thổi phồng.
- Không sửa câu hỏi hay nhãn của split `test` sau khi đã xem kết quả đánh giá; chỉnh hệ thống chỉ dựa trên `dev`.

## Công cụ hỗ trợ

Cần Qdrant đã bật và đã index (`chatbot.ps1 start` hoặc `docker compose up -d`):

```powershell
.venv\Scripts\python -m eval.label_helper "Sinh viên bị buộc thôi học trong những trường hợp nào?"
```

In top-10 ứng viên hybrid (file, trang, Điều, 200 ký tự đầu). Công cụ **không** ghi vào file: bạn đọc văn bản gốc
rồi tự quyết định Điều và trang nào là đáp án, kể cả khi đáp án không nằm trong top-10 (đó chính là lỗi cần đo).

Kiểm tra file và chạy đánh giá:

```powershell
.venv\Scripts\python -m eval.evaluate_retrieval --split dev
```

Báo cáo JSON ghi vào `eval/reports/` kèm model, collection, `chunk_size`, commit và ngày chạy.
