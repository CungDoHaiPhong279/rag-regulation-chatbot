# Bộ câu hỏi đánh giá

`questions.jsonl`: mỗi dòng một câu hỏi (JSON), 100 câu (70 `test`, 30 `dev`): q01–q70 về Thông tư 08/2021
(`thong_tu_08_2021.pdf`), q71–q100 về Quy chế đào tạo ĐH Bách khoa Hà Nội 2025 (`quy_che_hust_2025.pdf`).

**Nguồn gốc (ghi rõ trong mọi báo cáo):** câu hỏi và nhãn do AI (Gemini) soạn nháp từ văn bản đã index, một worker
AI thứ hai soát lại (sửa 20 câu), Claude kiểm tra tự động (nhãn có thật, không chép ≥5 chữ liên tiếp của văn bản,
mỗi Điều tối đa 3 câu, phủ 22/22 Điều), người dùng duyệt ngày 2026-11-08. Nhãn chưa được đối chiếu từng câu với PDF
gốc. Trường `answer_hint` (tóm tắt đáp án) chỉ để duyệt, bộ đọc dữ liệu bỏ qua.

q71–q100 (HUST): worker AI (Gemini) soạn nháp; Claude kiểm tra tự động (nhãn có thật, không trùng câu cũ, phủ 30
Điều, mỗi Điều tối đa 2 câu) và rút 5 câu hai ý còn một ý; người dùng duyệt ngày 2026-11-08. 8 câu về thạc sĩ/tiến sĩ
vì quy chế HUST áp dụng cả sau đại học. 16/30 câu có cụm ≥5 âm tiết trùng văn bản (phần lớn là thuật ngữ, như
"bảo vệ luận văn thạc sĩ"), có thể làm BM25 trúng dễ hơn một chút. 14 câu ngoài phạm vi đã được soát lại với quy
chế HUST: vẫn ngoài phạm vi. File HUST trên trang trường để trống số và ngày quyết định (tên file ghi số 5445).

## Định dạng

```json
{"id": "q07", "question": "...", "in_scope": true, "split": "test",
 "gold": [{"source_file": "thong_tu_08_2021.pdf", "article": "Điều 11", "pages": [13, 14]}],
 "answer_hint": "Tóm tắt đáp án (tùy chọn, chỉ để duyệt)"}
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

- **100 câu:** 70 `test` + 30 `dev`. Mỗi split có câu ngoài phạm vi (10 câu trong `test`, 4 trong `dev`).
- `dev` có **3 câu gõ không dấu** (2 về Thông tư 08, 1 về HUST) (ví dụ `sinh vien bi canh bao hoc tap khi nao`) để đo độ bền với cách gõ thật.
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

## Hội thoại đánh giá bước viết lại (`dialogues.jsonl`)

15 hội thoại: d01–d11 về Thông tư 08/2021 (đã bỏ d08 vì nhãn phải suy luận), d12–d16 về quy chế HUST. Mỗi dòng: `history` (2 hoặc 4 lượt
user/assistant), `question` (câu nối tiếp, phụ thuộc ngữ cảnh), `standalone` (câu độc lập viết tay), `gold` (như trên),
`answer_hint` (chỉ để duyệt). **Nguồn gốc:** d01 do Claude soạn từ câu q32; d02–d11 do worker AI (Gemini) soạn nháp,
Claude kiểm tra tự động (định dạng, Điều/trang có thật), người dùng duyệt ngày 2026-11-08; d12–d16 làm tương tự. Chưa đối chiếu với PDF gốc.

```powershell
.venv\Scripts\python -m eval.evaluate_rewrite
```
