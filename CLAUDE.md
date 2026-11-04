@AGENTS.md

# CLAUDE.md — Hướng dẫn cho Claude Code

> **`AGENTS.md` (import ở trên) là nguồn quy tắc có thẩm quyền.** File này chỉ bổ sung
> bối cảnh đã kiểm chứng cho Claude Code. Nếu hai file mâu thuẫn, theo `AGENTS.md` và hỏi người dùng.
> Trạng thái ghi dưới đây kiểm chứng ngày 2026-11-04 tại commit `9beee48`; đọc lại code trước khi tin.

## 1. Mục đích & phạm vi

Chatbot RAG hỏi đáp quy chế đào tạo đại học (Thông tư 08/2021 và/hoặc quy chế của trường),
trả lời tiếng Việt kèm trích dẫn **file / trang / Điều** chính xác. Dự án CV làm một mình:
người dùng phải giải thích được mọi quyết định khi phỏng vấn → luôn giải thích **vì sao**, không chỉ **cái gì**.

## 2. Kiến trúc: đã làm vs. dự kiến

| Thành phần | Trạng thái | Vị trí |
|---|---|---|
| Sinh PDF mẫu (biết trước Điều ↔ trang) | ✅ Đã làm | `src/ingestion/sample_pdf.py` |
| Đọc PDF → NFC → lọc header/footer → tách Điều → nối dòng mềm → cắt chunk | ✅ Đã làm | `src/ingestion/chunker.py` |
| In UTF-8 trên console Windows | ✅ Đã làm | `src/utils.py` |
| Embedder BGE-M3 + Indexer Qdrant (named vectors, local mode) | ⏳ Task 4 — chưa có code | `embedder.py` (chưa tạo), `indexer.py` (0 byte) |
| Hybrid search (dense + BM25), rerank | ⏳ Tuần 3 | `src/retrieval/*.py` (0 byte) |
| Chống bịa 2 lớp, LLM, FastAPI | ⏳ Tuần 2 | `src/api/main.py` (0 byte) |
| Streamlit UI | ⏳ Tuần 2 | `src/ui/app.py` (0 byte) |
| Bộ đánh giá 50 câu, Recall@5 | ⏳ Tuần 3 | `eval/evaluate_recall.py` (0 byte) |
| README | ⏳ Task 5 | `README.md` (0 byte) |

File 0 byte là **chỗ giữ theo plan**, không phải code bị mất. Đừng mô tả chúng như đã hoạt động.
Thư viện đã ghim (`requirements.txt`): pymupdf 1.24.14, langchain-text-splitters 0.3.2,
sentence-transformers 3.3.1, qdrant-client 1.12.1, python-dotenv 1.0.1, pytest 8.3.4.
Docker: `qdrant/qdrant:v1.12.5`. Chưa có: fastapi, streamlit, SDK LLM, thư viện tách từ tiếng Việt, reranker.

## 3. Pipeline chunking (đã làm — `chunker.py`)

1. `page.get_text("text")` từng trang; trang không chữ → `SkippedPage` (`scan`/`blank`) + cảnh báo stderr;
   cả file không chữ → `PdfNoTextError` (`chunk_directory` bỏ qua file, ghi vào `skipped_files`).
2. `normalize_text`: Unicode NFC + gộp khoảng trắng.
3. Lọc dòng đầu/cuối trang lặp ở ≥ 60% số trang (khi ≥ 3 trang có chữ). Chỉ dòng số trang
   (`Trang N`, `N/M`, `- N -`, `[N]`, `N`) được gom mẫu; dòng khác phải trùng nguyên văn.
4. `split_by_article`: tách theo tiêu đề Điều (regex bắt buộc `.` hoặc `:` sau số), gộp tiêu đề 2 dòng,
   mang Điều cuối trang sang trang sau; gặp `Chương/Mục/Phần` thì ngắt về `article=None`.
5. `join_soft_lines` rồi `RecursiveCharacterTextSplitter` với `body_size = chunk_size - len(prefix)`,
   separators `["\n\n", "\n", ". ", "; ", " ", ""]`.
6. `text = "<article>: " + body`; `chunk_id = uuid5(NAMESPACE_URL, f"{source_file}:{page_number}:{chunk_index}")`.

**Bất biến không được phá (đều có test):** mỗi chunk thuộc đúng 1 trang và tối đa 1 Điều;
`len(text) <= chunk_size` kể cả tiền tố; tiền tố không lặp; `chunk_index` liên tục 0..n trong trang;
ID tất định. `chunk_size/overlap` = 800/150 **ký tự**, đọc từ `.env` (`CHUNK_SIZE`, `CHUNK_OVERLAP`) —
là giá trị khởi đầu, chỉ chốt sau khi đo Recall@5.

## 4. Metadata & trích dẫn

Mỗi chunk/point phải mang `source_file` (basename), `page_number` (1-based), `article`
(`"Điều 5. Học phí"` hoặc `None`), `chunk_index`, `text`. Không bước nào (embed, upsert, truy xuất,
trả lời) được làm rơi hoặc tính lại `page_number`. Câu từ chối cố định:
`Không tìm thấy thông tin trong tài liệu quy chế.`

## 5. Qdrant & embedding (thiết kế đã chốt, CHƯA cài đặt)

Nguồn chi tiết: `docs/superpowers/plans/2026-11-02-week1-ingestion.md` (Task 4) và `docs/contracts/task-4.md`.
- Named vectors: `dense` (1024, Cosine — BGE-M3) + `sparse` (`SparseVectorParams`), payload index keyword `source_file`.
- Re-index theo file: embed hết → upsert (`wait=True`) → xóa point thừa bằng scroll phân trang lọc theo `source_file`.
  Lỗi embed/upsert thì **không** xóa gì. File không có chunk: cảnh báo, không xóa.
- Kiểm tra Qdrant (`get_collections()`) trước khi tải BGE-M3; lỗi → hướng dẫn `docker compose up -d` hoặc `--local`.
- `sentence_transformers` chỉ import bên trong `BgeM3Embedder.__init__`; unit test dùng `FakeEmbedder` + `QdrantClient(":memory:")`.

## 6. Môi trường & lệnh đã kiểm chứng (Windows)

- **Venv thực tế là Python 3.11.9**, trong khi plan (dòng Tech Stack) ghi Python 3.12. Code dùng
  `from __future__ import annotations` nên chạy được trên 3.11. Không tự tạo lại venv; hỏi người dùng nếu cần đổi.
- Luôn dùng `.venv\Scripts\python` / `.venv\Scripts\python -m pip`; không dùng `pip` trần.
- Chạy test từ **thư mục gốc repo** (`test_chunker.py` dùng đường dẫn font tương đối):

```powershell
.venv\Scripts\python -m pytest -v                                   # kỳ vọng: 56 passed (~4 s)
.venv\Scripts\python -m src.ingestion.sample_pdf                    # sinh data/pdfs/quy_che_mau.pdf
.venv\Scripts\python -m src.ingestion.chunker data/pdfs/quy_che_mau.pdf
.venv\Scripts\python -m src.ingestion.chunker data/pdfs/thong_tu_08_2021.pdf
```

Kết quả đã quan sát trên Thông tư 08 thật: 77 chunk, Điều 1→22 liên tục, trang 1–2 là ảnh scan bị bỏ qua.

## 7. Quy trình làm việc với Superpowers

- **Không tạo plan cạnh tranh.** Plan hiện hành là `docs/superpowers/plans/2026-11-02-week1-ingestion.md`;
  dùng `superpowers:executing-plans` để thực hiện **từng task một**, dừng sau mỗi task chờ "tiếp".
- Chỉ dùng `superpowers:brainstorming` / `writing-plans` cho phần việc **chưa có plan** (Tuần 2–3), và hỏi trước.
- Code mới: `superpowers:test-driven-development` (viết test đỏ trước, như plan mô tả).
- Trước khi báo "xong": `superpowers:verification-before-completion` — chạy lệnh thật, dán kết quả.
- Lỗi/test đỏ: `superpowers:systematic-debugging` trước khi sửa.
- Sau mỗi task: giải thích tiếng Việt đơn giản (file làm gì, vì sao, thuật ngữ mới, lệnh tự kiểm tra).

### Critic / subagent
- Quy trình Critic định nghĩa trong `AGENTS.md` mục 4 và hợp đồng `docs/contracts/`.
- `.agent/agents/builder.md`, `critic.md` được viết cho **Antigravity** (`invoke_subagent`, model Gemini).
  **Không mặc định chúng là subagent của Claude Code** và **chưa di chuyển/chuyển đổi** chúng.
- Chỉ gọi Critic khi task ghi rõ có review (Task 2, Task 4) hoặc người dùng gõ `[TEAM]`; hỏi người dùng
  cách chạy Critic trong Claude Code trước khi gọi. Không tự gọi Critic cho task mới.

## 8. Git & bảo mật

- Không commit/push, không đổi nhánh, không `reset`/`checkout`/`pull` khi chưa được đồng ý.
- Không đọc `.env` hay bất kỳ secret nào; chỉ xem `.env.example`. Không ghi key vào code, log, commit.
- PDF trong `data/pdfs/` bị ignore — không commit; không bulk-đọc PDF, `.venv`, dữ liệu Qdrant, cache model.
- Không tự thêm thư viện hay đổi cấu trúc thư mục khi chưa hỏi; giữ ghim `==`.
- Không ghi đè/xóa thay đổi chưa commit của người dùng.

## 9. Trạng thái hiện tại & rủi ro mở

| Task (plan Tuần 1) | Trạng thái |
|---|---|
| 1. Môi trường, font, PDF mẫu | DONE |
| 2. Chunker | DONE về code; **báo cáo Critic không được lưu trong repo** (vòng 3–4 chưa có bằng chứng review lại) |
| 3. Kiểm chứng PDF thật | PARTIAL — đã chạy, chưa được người dùng chốt |
| 4. Embedder + Indexer | NOT STARTED — hợp đồng `task-4.md` đã commit, chưa tick |
| 5. E2E + README | NOT STARTED |

Rủi ro/việc chưa giải quyết (cần hỏi người dùng trước khi xử lý):
1. Plan còn ghi thứ tự cũ "embed → xóa → upsert" ở phần Architecture/Review Focus/README (trái với `AGENTS.md` mục Indexing).
2. Plan cho chunk trước rồi mới kiểm tra Qdrant; `task-4.md` mục 3.4 yêu cầu kiểm tra Qdrant trước.
3. `chunk_directory` chỉ bắt `PdfNoTextError` → PDF hỏng làm dừng cả thư mục (AGENTS yêu cầu báo lỗi dễ hiểu).
4. Trang 1–2 Thông tư 08 là ảnh scan → phần thân Thông tư không được index; bản có lớp chữ sẽ có hai "Điều 1".
5. Đổi tên/xóa PDF để lại point mồ côi (dọn dẹp chỉ theo file đang có).
6. ID không chứa `chunk_size` → so sánh 400/800/1200 cần collection riêng cho từng cấu hình.
7. BM25 tiếng Việt chưa chọn cách tách từ; `SparseVectorParams()` chưa đặt `modifier=IDF`.
8. Ngưỡng chống bịa Lớp 1 phải hiệu chỉnh bằng câu ngoài phạm vi; định nghĩa "trúng" cho Recall@5 (trang/Điều) chưa chốt.
