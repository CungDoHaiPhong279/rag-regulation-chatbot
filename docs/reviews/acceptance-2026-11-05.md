# Báo cáo nghiệm thu — từ onboarding đến hết Task 6

> Ngày: 2026-11-05 · Nhánh: `feat/rag-pipeline` (chưa push) · HEAD: `b826b7e`
> Điểm khôi phục: tag `checkpoint/pre-redesign` (= `c4101b1`) · Tests: **124 passed**, 1 deselected (test model thật, đánh dấu `slow`)

## 1. Tóm tắt

| Hạng mục | Trạng thái |
|---|---|
| Khảo sát dự án, CLAUDE.md, yêu cầu sản phẩm, spec, plan tổng | ✅ Xong, đã được duyệt |
| Task 1 Quy trình + môi trường | ✅ |
| Task 2 Cấu hình tập trung | ✅ |
| Task 3 Chunker: PDF hỏng/khóa/cắt cụt | ✅ Code + test; Critic APPROVED ở lượt đầy đủ gần nhất |
| Task 4 BM25 + pyvi | ✅ |
| Task 5 Embedder BGE-M3 | ✅ |
| Task 6 Indexer Qdrant | ✅ Code + test; lượt Critic gần nhất REJECTED → đã sửa 4 lỗi có test, **chưa có Critic xác nhận lại** (hết số vòng) |
| Task 7–15 | Chưa làm |

## 2. Dòng thời gian

1. **Audit chỉ đọc** (2026-11-04): xác định Task 1–2 đã có code, Task 3 làm một phần, Task 4+ chưa có; phát hiện tài liệu mâu thuẫn về thứ tự re-index, PDF hỏng làm dừng cả thư mục, venv là Python 3.11 trong khi plan ghi 3.12.
2. **CLAUDE.md** đầu tiên (`c4101b1`), commit cùng tag `checkpoint/pre-redesign`.
3. **Khảo sát yêu cầu** (Superpowers brainstorming), chốt 6 quyết định: demo phỏng vấn, chạy local, Gemini, TT08 + quy chế trường, hội thoại nhiều lượt có viết lại câu hỏi, ~2 tuần.
4. **Kiểm chứng kỹ thuật (spike, code bỏ đi)**: Qdrant IDF/RRF, BM25, pyvi, BGE-M3, reranker CPU (số liệu ở mục 5).
5. **Spec** (`3262801`) — 4 phần thiết kế được duyệt lần lượt.
6. **Plan tổng 15 task** (`9718e5f`), chọn thực thi Native.
7. **Task 1–6** + 5 lượt Critic (mục 4).

## 3. Quyết định đã chốt (và vì sao — dùng khi phỏng vấn)

| Quyết định | Lý do |
|---|---|
| Mỗi chunk thuộc đúng 1 trang và tối đa 1 Điều, tiền tố `"Điều X. Tên: "` | `page_number` đúng theo cấu trúc, không đoán; chunk tự mang ngữ cảnh khi embed |
| `chunk_size` tính bằng ký tự, gồm cả tiền tố; 800/150 chỉ là giá trị khởi đầu | So 400/800/1200 công bằng; chốt bằng Recall@5 trên tập dev |
| BM25 = phần tf bão hòa + chuẩn hóa độ dài ở client, IDF do Qdrant (`Modifier.IDF`) | Đúng BM25 đầy đủ (khớp tham chiếu sai số 9.2e-7), không phải TF-IDF; IDF tự cập nhật theo kho |
| Tách từ bằng pyvi (`học phí` → `học_phí`) | Khớp cụm từ tiếng Việt; nhẹ, offline |
| Token ID = 32 bit đầu `blake2b` | Ổn định giữa các lần chạy (khác `hash()`), không cần file từ điển; 0 va chạm trên 716 từ |
| Qdrant Docker là môi trường chính, local mode chỉ cho test | Local mode khóa file, không cho API + indexer + eval chạy cùng lúc; lỗi khóa file trên Windows |
| Index toàn kho một lượt; upsert trước, xóa point thừa sau; giữ point của file bị bỏ qua | Lỗi giữa chừng không mất dữ liệu; file bị xóa/đổi tên được dọn; file hỏng tạm thời không bị xóa |
| Số trang trong câu trả lời lấy từ metadata, LLM chỉ trích `[n]` | Không thể bịa số trang về mặt cấu trúc (áp dụng ở Task 9–10) |
| Embedder/reranker/LLM sau Protocol, test dùng bản giả | 124 test chạy ~14 giây, không tải model 6,5 GB |
| Reranker chạy CPU làm baseline (14 s cho top-20) | Không đổi PyTorch trước khi có số liệu Recall (quyết định ở Task 15) |
| Đánh giá: 50 câu test + 20 câu dev | Chỉnh ngưỡng và chunk_size trên dev, báo cáo trên test — tránh overfit |
| CLAUDE.md là nguồn quy tắc duy nhất; AGENTS.md cũ, plan Tuần 1, cấu hình Antigravity lưu `docs/archive/` | Hết mâu thuẫn giữa 4 nguồn chỉ dẫn; không mất nội dung |
| Critic là subagent chạy model rẻ (Sonnet), phạm vi đầy đủ nhưng có ngân sách lệnh | Theo yêu cầu của bạn: tiết kiệm token bằng model, không bằng bỏ độ sâu |

## 4. Kết quả từng task

| Task | Thay đổi chính | File | Test | Commit |
|---|---|---|---|---|
| 1 | CLAUDE.md viết lại; AGENTS.md thành stub; `git mv` tài liệu cũ vào `docs/archive/`; `.claude/agents/critic.md`; `requirements.lock`; `.gitattributes`; Qdrant chỉ mở `127.0.0.1`; marker `slow`; cài `pyvi==0.1.1` | `CLAUDE.md`, `AGENTS.md`, `.claude/agents/critic.md`, `pytest.ini`, `docker-compose.yml`, `requirements.*` | 56 test cũ vẫn pass | `4f8d6d4` |
| 2 | `Settings` 17 tham số từ `.env`; key ẩn khi in | `src/config.py`, `.env.example` | 5 | `dfcc321` |
| 3 | Bỏ qua PDF hỏng / có mật khẩu / MuPDF phải tự sửa / 0 trang / có chữ nhưng 0 chunk; tắt log MuPDF; tham số chunk qua `config.py`; **PDF mẫu hư cấu dời khỏi `data/pdfs/` sang `data/samples/`** | `src/ingestion/chunker.py`, `sample_pdf.py` | 60 (chunker) + 7 (sample) | `90a8f16`, `0294972`, `363ee88`, `b826b7e` |
| 4 | `tokenize`, `token_id`, `bm25_doc_vector`, `query_vector` | `src/ingestion/sparse.py` | 15 | `d6aedd6` |
| 5 | `Embedder` Protocol, `BgeM3Embedder`, `tests/fakes.py` | `src/ingestion/embedder.py` | 3 + 1 slow (model thật: 1024 chiều, pass) | `68eb04a` |
| 6 | Collection dense Cosine + sparse IDF + payload index; `index_corpus`; kiểm cấu trúc collection; CLI báo lỗi thân thiện; hợp đồng `task-4.md` mục 5 | `src/ingestion/indexer.py`, `docs/contracts/task-4.md` | 31 | `b1e7daa`, `38b190e`, `b826b7e` |

**Chạy thật (ngoài test):** indexer + BGE-M3 trên Thông tư 08 → 77 chunk (75 s); chạy lại không trùng. Critic chạy thêm trên Qdrant Docker 1.12.5 → đúng; truy vấn "Sinh viên bị cảnh báo học tập khi nào?" trả trang 13, Điều 11. TT08 cắt cụt 50% trước sửa: 129 chunk rác → sau sửa: bị bỏ qua, 0 chunk.

## 5. Số liệu kiểm chứng kỹ thuật (spike)

| Hạng mục | Kết quả |
|---|---|
| Qdrant Docker 1.12.5 vs local mode | Sparse IDF, prefetch, RRF chạy giống hệt; local khóa file |
| BM25 qua Qdrant vs tham chiếu | Sai số tối đa 9.2e-7 |
| pyvi | `tín_chỉ` nhất quán 38/39 lần trong TT08; "cảnh báo học tập" tách đúng mọi chỗ |
| "học phí" trong TT08 | **0 lần** → câu hỏi học phí cần quy chế trường |
| BGE-M3 CPU | Câu hỏi 0.27 s; 77 chunk 67 s |
| Reranker CPU top-20 / top-10 | 14.2 s / 7.3 s |

## 6. Lịch sử Critic

| Lượt | Model | Phạm vi | Kết quả | Token |
|---|---|---|---|---|
| Chunker r1 | Opus | Đầy đủ | REJECTED — PDF cắt cụt báo "scan"; lỗ hổng test (M9) | ~169k |
| Indexer r1 | Opus | Đầy đủ | REJECTED — 5 phép phá sống (preserve, avgdl, payload index, chunk_index, wait) | ~138k |
| Chunker r2 | Opus | Đầy đủ | REJECTED — **TT08 cắt cụt ra 129 chunk rác** | ~122k |
| Verify r3 | Sonnet | Chỉ kiểm bản sửa | Cả hai APPROVED | ~74k |
| Full r4 | Sonnet, ngân sách 30 lệnh | Đầy đủ | Chunker APPROVED; **Indexer REJECTED — file 0 chunk bị xóa im lặng dữ liệu cũ** | ~130k |

Báo cáo đầy đủ: `docs/reviews/task-3-chunker-r1.md`, `task-3-chunker-r2.md`, `task-6-indexer-r1.md`, `task-3-6-verify-r3.md`, `task-3-6-full-r4.md`.
Mọi lỗi chặn đã sửa kèm test đỏ→xanh. Bản sửa r4 (`b826b7e`) **chưa có Critic độc lập xác nhận** — đã đạt giới hạn 3 vòng nên chờ bạn quyết.

## 7. Quyết định tôi tự đưa ra khi thực thi (rulings)

| Quyết định | Rủi ro nếu sai |
|---|---|
| PDF mẫu hư cấu dời từ `data/pdfs/` sang `data/samples/` (chuyển, không xóa) | Phải chạy lại `python -m src.ingestion.sample_pdf` nếu cần |
| Mọi PDF MuPDF phải tự sửa đều bị bỏ qua, kể cả khi chữ còn đúng | File hỏng nhẹ nhưng đọc được sẽ không được index đến khi tải lại |
| Test tắt log MuPDF kiểm cờ của PyMuPDF thay vì output (log C bị đệm, không quan sát được) | Test kiểm cờ, không kiểm trực tiếp dòng log |
| Indexer truyền `chunk_size` từ `Settings` vào chunker | Không đáng kể |
| `--recreate` chỉ thêm cảnh báo trong `--help` | Embed lỗi sau `--recreate` → collection trống đến khi chạy lại |
| Critic chạy bằng subagent tổng quát đọc `critic.md` (subagent `critic` chỉ nạp được ở phiên mới) | Không đáng kể |
| `torch==2.15.0` trong lock (pip bỏ hậu tố `+cpu`) | Máy khác có thể cài bản torch khác |

## 8. Việc còn tồn / rủi ro mở

| Mục | Mức | Kế hoạch |
|---|---|---|
| Bản sửa r4 chưa có Critic xác nhận | Trung bình | Bạn chọn: chấp nhận hoặc thêm 1 vòng Sonnet |
| Chưa có PDF quy chế trường | Cao (cho demo) | Task 15 |
| Tiêu đề `Ðiều` (font cũ) / `ĐIỀU` (in hoa) không nhận ra | Thấp | Kiểm lại khi có PDF trường |
| `dense_dim = 1024` cứng dù model đổi được | Thấp | Ghi chú; xử lý nếu đổi model |
| Lỗi đọc từng trang chưa bọc | Thấp | Chưa kiểm chứng |
| Trang có luồng nội dung hỏng bị gọi "trang trắng" | Thấp | Ghi chú |
| Test 4 file indexer ~10 s, hợp đồng ghi < 5 s | Thấp | Do import thư viện; đã ghi trong hợp đồng |
| Chưa push lên GitHub | — | Chờ bạn đồng ý |

## 9. Lệnh tự kiểm tra

```powershell
git log --oneline checkpoint/pre-redesign..feat/rag-pipeline
.venv\Scripts\python -m pytest                                     # 124 passed
.venv\Scripts\python -m pytest -m slow                             # model thật (cần cache BGE-M3)
.venv\Scripts\python -m src.ingestion.sample_pdf                   # sinh data/samples/quy_che_mau.pdf
.venv\Scripts\python -m src.ingestion.chunker data/pdfs/           # 22 Điều, trang 1-2 scan
docker compose up -d
.venv\Scripts\python -m src.ingestion.indexer --recreate           # Đã index 77 chunks
```
