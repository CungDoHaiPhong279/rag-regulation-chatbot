# Project Spec: Chatbot RAG Hỏi Đáp Quy Chế Trường (Có Trích Nguồn)

> Dự án CV của một sinh viên IT, làm một mình trong khoảng 2–3 tuần.
> Người dùng sẽ bị hỏi phỏng vấn về từng quyết định thiết kế, nên **hiểu code quan trọng hơn chạy nhanh**.

## 1. Kiến trúc & Tech Stack bắt buộc

- **Luồng xử lý:** PDF quy chế -> Chunk + Embed (BGE-M3) -> Qdrant -> LLM đọc & trả lời -> Trả lời kèm trích dẫn đúng số trang.
- **Tech stack:** Python, FastAPI (backend), Qdrant (vector DB), BAAI/bge-m3 (embedding), một LLM API (Qwen / GPT / Gemini), Streamlit (giao diện).
- **LLM mặc định:** chọn MỘT model làm mặc định, đặt trong `config.yaml` (hoặc `.env`), cho phép đổi mà không sửa code. Ghi rõ tên model đã dùng khi báo cáo kết quả đánh giá.
- **Phạm vi tài liệu:** quy chế / quy định của trường (ví dụ Thông tư 08/2021 về quy chế đào tạo đại học, và/hoặc quy chế đào tạo của chính trường). Không đưa tài liệu không liên quan vào (ví dụ Nghị định 13/2023 về dữ liệu cá nhân).

## 2. Tính năng ăn điểm (Key Requirements)

1. **Trích dẫn đúng số trang:** giữ nguyên metadata `source_file` và `page_number` xuyên suốt từ lúc cắt chunk đến lúc trả lời.
2. **Hybrid Search + Rerank:** kết hợp dense vector và sparse vector (BM25), sau đó rerank.
   - Khi tạo collection Qdrant, khai báo sẵn **named vectors** (dense và sparse) từ Tuần 1 để Tuần 3 không phải tạo lại collection và index lại.
   - BM25 cho tiếng Việt cần tách từ (word segmentation); cân nhắc từ đầu.
3. **Chống ảo giác (Anti-Hallucination), hai lớp:**
   - Lớp 1 (không cần gọi LLM): nếu điểm tương đồng của các chunk truy xuất được thấp hơn ngưỡng cấu hình thì trả lời ngay câu từ chối.
   - Lớp 2 (prompt): LLM chỉ được dùng nội dung trong context; nếu context không đủ thì trả lời câu từ chối.
   - Câu từ chối là **một chuỗi cố định duy nhất**, đặt thành hằng số trong code: `Không tìm thấy thông tin trong tài liệu quy chế.` (để dễ kiểm thử).
4. **Bộ đánh giá tự động 50 câu hỏi (đo Recall@5):**
   - Mỗi câu có đáp án chuẩn do người dùng gán tay: file, số trang, số Điều.
   - Có khoảng 5–10 câu **ngoài phạm vi tài liệu** để kiểm tra khả năng từ chối.
   - Đây là cơ sở để chọn `chunk_size` / `chunk_overlap` ở Tuần 3 (thử 400 / 800 / 1200 và so Recall@5).

## 3. Quy tắc làm việc (áp dụng cho mọi agent)

- **Mặc định: làm từng task một.** Xong một task thì dừng lại và chờ người dùng nói "tiếp". Không tự làm sang task kế tiếp.
- **Không commit/push** khi chưa được người dùng đồng ý.
- **Giải thích sau mỗi task** bằng tiếng Việt đơn giản: mỗi file làm gì, vì sao thiết kế như vậy, và lệnh để người dùng tự chạy kiểm tra. Giải thích thuật ngữ ở lần đầu xuất hiện.
- **Mơ hồ thì hỏi trước.** Nếu định làm khác plan, hoặc có nhiều cách hiểu, hỏi người dùng trước khi làm.
- **Không tự thêm thư viện** ngoài plan, không đổi cấu trúc thư mục khi chưa hỏi.
- **Môi trường Windows + venv:** luôn dùng `.venv\Scripts\python` và `.venv\Scripts\python -m pip`. Không dùng `pip` trần (máy có `python` 3.12 nhưng `pip` thuộc 3.11).
- **Tiêu chí "xong":** mỗi task phải nêu rõ cách kiểm tra (lệnh chạy, kết quả mong đợi). Task chưa có test hoặc chưa chạy được thì chưa tính là xong.

## 4. Quy trình [TEAM] (mô hình Planner / Builder / Critic)

Quy trình này **không tự kích hoạt** khi làm việc theo Tuần. Chỉ dùng khi người dùng gõ `[TEAM]`, hoặc khi kế hoạch của task ghi rõ có Critic review (hiện tại: chunker và indexer của Tuần 1).

Ý tưởng (theo bài harness design của Anthropic): tách người **làm** và người **chấm**, vì agent hay khen sản phẩm do chính nó viết. Critic phải là một bên độc lập và khắt khe.

1. **Planner:** chia nhỏ task, chọn chiến lược chunking, định nghĩa schema dữ liệu (xem mục 5). Chỉ nêu *cái gì cần đạt*, không áp đặt chi tiết cài đặt không cần thiết.
2. **Hợp đồng nghiệm thu (TRƯỚC khi viết code):**
   - Builder đề xuất checklist "xong là gì" cho task (hành vi mong đợi + lệnh kiểm tra).
   - Critic bổ sung chỗ thiếu (ca biên, lỗi hay gặp). Lặp tối đa **2 vòng** rồi chốt.
   - Ghi vào `docs/contracts/<tên-task>.md`. Builder chỉ được báo "xong" khi mọi mục trong hợp đồng đạt.
3. **Builder:** viết code hoàn chỉnh vào đúng các file trong `src/`, xử lý lỗi đầy đủ, kèm test.
4. **Critic:** đánh giá độc lập theo hợp đồng và checklist dưới đây, trả về `APPROVED` hoặc `REJECTED`.
   - Mặc định **nghi ngờ**; không khen chung chung.
   - Bắt buộc **chạy thật**: `pytest` và các lệnh CLI liên quan (chunker, indexer) trên PDF mẫu, rồi dán kết quả thật vào báo cáo. Không được chỉ đọc code rồi nhận xét.
   - `APPROVED` phải kèm bằng chứng cho từng mục của hợp đồng (lệnh đã chạy, kết quả).
   - `REJECTED` bắt buộc nêu lý do cụ thể, file/dòng liên quan và cách tái hiện lỗi.
   - Tối đa **3 vòng** sửa. Quá 3 vòng thì dừng lại và báo người dùng quyết định.

**Checklist của Critic:**
- [ ] Có làm mất hoặc làm lệch `page_number` / `source_file` ở bước nào không (chunk, embed, upsert, truy xuất, trả lời)?
- [ ] Chunk kéo qua nhiều trang có gắn đúng số trang và đúng Điều không?
- [ ] Chạy lại nhiều lần có sinh dữ liệu trùng không (ID cố định)?
- [ ] Prompt và ngưỡng chống bịa đã chặt chưa; câu từ chối có đúng chuỗi cố định không?
- [ ] Có lộ API key trong code, log hoặc commit không?
- [ ] Lỗi hay gặp (Qdrant chưa bật, PDF scan không có lớp chữ, file hỏng) có thông báo dễ hiểu không?
- [ ] Có test cho phần vừa viết, gồm test "không mất `page_number`"?

**Chọn model cho từng vai** (công cụ `invoke_subagent` của Antigravity chỉ nhận tham số `Model` là `inherit` / `flash_lite` / `flash` / `pro`; tên model cụ thể chỉ đổi được trên giao diện chính):
- Agent chính (Planner / Builder): dùng model người dùng chọn trên giao diện (hoặc subagent `Model='flash'` / `inherit`).
- Subagent Critic: bắt buộc gọi qua `invoke_subagent` trong ngữ cảnh hội thoại độc lập với `Model='pro'` (hoặc `inherit` khi giao diện đang ở model khác với Builder), tự chạy `pytest` + CLI và thử phá code có chủ đích (mutation testing) để kiểm chứng test thực sự bắt được lỗi rồi hoàn tác.
**Quy tắc giao việc cho Critic:**
- Agent chính chỉ đưa cho Critic đường dẫn hợp đồng và mã nguồn. Không chỉ định phép phá,
  không nói trước kết quả mong đợi, không nêu số test.
- Critic tự chọn phép thử phá code và đầu vào xấu, làm việc trên bản sao, không sửa file thật.
- Báo cáo của Critic phải được dán nguyên văn cho người dùng.

## 5. Quy ước kỹ thuật

### Dữ liệu & chunking
- Chuẩn hóa **Unicode NFC** cho mọi text tiếng Việt trước khi chunk (tránh lỗi "ế" lưu dạng 1 ký tự và 2 ký tự ghép).
- Chunk theo từng trang để số trang không bao giờ lệch. Chèn tiêu đề Điều vào đầu mỗi chunk (ví dụ `Điều 5. Học phí: ...`) để chunk tự mang ngữ cảnh, kể cả khi một Điều kéo sang trang sau.
- PDF scan (không có lớp chữ) phải bị bỏ qua và **báo rõ**, không được index ra dữ liệu rác.
- Nên loại bỏ header/footer lặp lại của PDF thật khi làm sạch văn bản.
- `chunk_size` và `chunk_overlap` là **giá trị khởi đầu** (ban đầu 800 / 150), ghi rõ đơn vị là ký tự hay token, đặt trong file cấu hình, không hard-code. Chỉ chốt sau khi đo Recall@5.

### Schema metadata của mỗi chunk
| Trường | Ý nghĩa |
|---|---|
| `source_file` | Tên file PDF gốc |
| `page_number` | Số trang trong PDF (bắt đầu từ 1) |
| `article` | Số/tên Điều chứa chunk (nếu xác định được) |
| `chunk_index` | Thứ tự chunk trong trang |
| `text` | Nội dung chunk |

### Indexing (Qdrant)
- ID chunk cố định, băm từ `source_file + page_number + chunk_index`.
- Index lại một file: **upsert chunk mới trước** (ghi đè point trùng ID), rồi **xóa các point cũ có ID không còn trong tập mới** (tránh mất sạch dữ liệu nếu upsert lỗi giữa chừng).
- Collection dùng named vectors (dense + sparse), xem mục 2.
- Nếu Qdrant chưa chạy: báo lỗi dễ hiểu, gợi ý cách bật. Phương án dự phòng khi Docker trục trặc: Qdrant local mode (`qdrant-client` chạy bằng file, không cần server).

### Bảo mật & cấu hình
- API key chỉ đặt trong `.env` (đã nằm trong `.gitignore`), có file mẫu `.env.example` không chứa key thật.
- Tuyệt đối không ghi key vào code, log hoặc tin nhắn commit.
- Ghim phiên bản cụ thể cho từng thư viện trong `requirements.txt`.

### Kiểm thử
- Có PDF quy chế mẫu tự sinh, biết trước Điều nào ở trang nào, để kiểm tra trích dẫn số trang.
- Mọi hàm xử lý chunk và trích dẫn phải có test, bao gồm test "không mất `page_number`".
- Chạy thử sớm trên PDF thật (Thông tư 08/2021 hoặc quy chế trường) để phát hiện lỗi font tiếng Việt, bảng biểu, header/footer lặp.

## 6. Lộ trình tham chiếu

| Tuần | Nội dung |
|---|---|
| 1 | Gom PDF, làm sạch, cắt chunk, index vào Qdrant (named vectors) |
| 2 | API FastAPI + giao diện chat Streamlit, hiện nguồn trích dẫn (file, trang, Điều) |
| 3 | Bộ 50 câu đánh giá, đo Recall@5, thêm hybrid search + rerank, chọn chunk_size dựa trên số liệu |

## 7. Câu hỏi phỏng vấn cần trả lời được (bằng số liệu thật)

- Chunk size chọn thế nào, vì sao? -> trả lời bằng bảng Recall@5 của các cấu hình đã thử.
- Làm sao biết bot đang bịa? -> ngưỡng tương đồng, prompt chống bịa, và các câu ngoài phạm vi trong bộ đánh giá.
- Vì sao dùng hybrid search? -> so Recall@5 giữa dense-only và hybrid.
