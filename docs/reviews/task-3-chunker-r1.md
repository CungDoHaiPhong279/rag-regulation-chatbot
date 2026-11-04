# Báo cáo Critic — Task 2 (Chunker) @ commit 90a8f16 — vòng 1

> Lưu nguyên văn kết luận và phát hiện của subagent Critic (chạy 2026-11-04). Kết quả: **REJECTED**.

**Kết luận: REJECTED** — 1 lỗi chặn đã xác nhận bằng chạy thật (PDF cắt cụt bị báo "PDF scan — cần OCR" thay vì "file PDF hỏng hoặc bị khóa" theo spec mục 6), cùng 4 điểm không chặn. Bộ test hiện có xanh 61/61 và bắt được 21/22 phép phá.

Phạm vi review: `src/ingestion/chunker.py`, `src/ingestion/sample_pdf.py`, `src/utils.py`, `tests/ingestion/test_chunker.py`, `tests/ingestion/test_sample_pdf.py`, `tests/test_utils.py`. Trong lúc review HEAD đã chuyển sang 68eb04a (agent khác commit sparse/embedder); `git diff --stat 90a8f16 HEAD` trên 6 file trên rỗng. Không sửa file nào trong repo, không commit, không đọc `.env`. Mọi phép phá làm trên bản sao trong scratchpad.

## 1. Bằng chứng chạy thật

- `pytest` trên 3 file giao: `61 passed, 5 warnings in 3.58s` (warnings là SWIG/PyMuPDF DeprecationWarning).
- `pytest -q` toàn repo lúc đó lỗi collect `tests/ingestion/test_sparse.py` (file của agent khác đang làm dở); bỏ qua file đó: `66 passed in 4.11s`.
- CLI trên PDF mẫu `data/samples/quy_che_mau.pdf`: exit 0; 6 trang, 14 chunk, 10 Điều; 12 dòng header/footer bị lọc; 3 chunk mẫu: `page_number`=1 đúng trang vật lý, chunk mở đầu `article=None` không tiền tố, 2 chunk Điều có tiền tố đúng 1 lần, len ≤ 800, không còn header/footer.
- CLI theo hợp đồng mục 3.2 `data/pdfs/quy_che_mau.pdf`: exit 1 `Không tìm thấy đường dẫn` — đúng vì PDF mẫu đã được dời; hợp đồng chưa cập nhật (phát hiện 5).
- CLI trên `data/pdfs/thong_tu_08_2021.pdf`: exit 0; trang 1–2 scan ảnh; 20 trang có chunk, 77 chunk, đủ 22 Điều; Điều 16 tiêu đề 2 dòng gộp đúng; header số trang đầu trang lọc đúng.
- Thư mục PDF xấu tự tạo:

| File | Loại | Kết quả | Đánh giá |
|---|---|---|---|
| `b_bad.pdf` | 9 byte rác | hỏng/khóa | Đúng |
| `c_empty.pdf` | 0 byte | hỏng/khóa | Đúng |
| `d_trunc.pdf` | PDF mẫu cắt còn 50% | "PDF scan — cần OCR" | **Sai** |
| `e_locked.pdf` | AES-256 user pw | hỏng/khóa | Đúng |
| `f_owneronly.pdf` | chỉ owner pw (mở được) | 1 chunk | Hợp lý |
| `g_scan.pdf` | trang trắng | scan | Đúng |
| `a_ok.pdf`, `z_UPPER.PDF`, `h_badstream.pdf` | hợp lệ | 14 chunk mỗi file | Đúng, vẫn xử lý tiếp |

`fitz.open(d_trunc.pdf)` → `page_count=6`, `is_repaired=True`.

## 2. Mutation testing (bản sao, 3 file test được giao; trước 61 passed, sau hoàn tác 61 passed)

| # | Phép phá | Kết quả |
|---|---|---|
| M1 | `page_number + 1` | 11 FAIL |
| M2 | Tắt lọc header/footer | 2 FAIL |
| M3 | `chunk_directory` bỏ `except PdfUnreadableError` | 1 FAIL |
| M4 | Bỏ kiểm tra `needs_pass` | 1 FAIL |
| M5 | Không cập nhật `last_seen_article` khi gặp Điều | 5 FAIL |
| M6 | Mọi dòng có chữ số = số trang | 2 FAIL |
| M7 | `chunk_id` dùng uuid4 | 1 FAIL |
| M8 | `body_size` không trừ tiền tố | 1 FAIL |
| **M9** | **Xóa `last_seen_article = None` khi gặp Chương/Mục/Phần** | **61 passed — SỐNG SÓT** |
| M10 | UTF-8 chỉ stdout | 1 FAIL |
| M11 | PDF mẫu vào `data/pdfs` | 1 FAIL |
| M12 | Tắt `join_soft_lines` | 1 FAIL |
| M13 | Bỏ NFC | 1 FAIL |
| M14 | Không mang Điều sang trang sau | 4 FAIL |
| M15 | Đảo scan/blank | 1 FAIL |
| M16 | `chunk_index` không reset theo trang | 27 FAIL |
| M17 | Bỏ try/except quanh `fitz.open` | 3 FAIL |
| M18 | Regex Điều không bắt buộc `.`/`:` | 1 FAIL |
| M19 | Ngưỡng lặp 60% → 101% | 2 FAIL |
| M20 | Gộp tiêu đề 2 dòng cả khi dòng sau viết hoa | 10 FAIL |
| M21 | Bỏ `ValueError` chunk_size quá nhỏ | 1 FAIL |
| M22 | `source_file` thành đường dẫn đầy đủ | 3 FAIL |

**21/22 bị bắt. M9 sống sót.**

Fuzz: PDF cắt cụt ở mọi mức 5%–99% → `PdfNoTextError` ("scan"); 40 file bị flip 200 byte ngẫu nhiên → 39 OK, 1 "scan". Không có exception nào lọt ra.

## 3. Phát hiện

**#1 (CHẶN, Trung bình, đã xác nhận): PDF cắt cụt bị báo "PDF scan — cần OCR".**
`chunker.py:528-535` chỉ coi là hỏng khi `fitz.open` ném lỗi hoặc `needs_pass`. Với file cắt cụt MuPDF tự sửa (`is_repaired == True`) nhưng mất chữ → rơi xuống `PdfNoTextError`. Tái hiện: cắt `quy_che_mau.pdf` còn 50%, chạy CLI trên thư mục → `Bỏ qua d_trunc.pdf: không có lớp chữ (PDF scan?) — cần OCR`. Spec mục 6 yêu cầu `Bỏ qua <file>: file PDF hỏng hoặc bị khóa`; file tải về bị đứt là dạng hỏng phổ biến nhất, thông báo hiện tại đẩy người dùng đi OCR thay vì tải lại. Mong đợi: nếu `doc.is_repaired` và không trang nào có chữ → `PdfUnreadableError`; thêm `test_truncated_pdf_reported_as_unreadable`.

**#2 (Không chặn, Thấp–Trung bình, lỗ hổng test): M9 sống sót.**
`chunker.py:318`. Đầu vào `["Điều 4. Đánh giá\nNội dung Điều 4.\nChương II", "TỔ CHỨC ĐÀO TẠO\nNội dung mở đầu chương.\nĐiều 5. Học phí\nMức học phí."]` — code thật: trang 2 `article=None` (đúng); bản phá M9: trang 2 bị gán `Điều 4. Đánh giá` với tiền tố, mọi test vẫn xanh. "Chương II" ở cuối trang, tên chương đầu trang sau là bố cục phổ biến. Mong đợi: thêm `test_chapter_heading_at_page_bottom_resets_next_page`.

**#3 (Không chặn, Thấp, đã xác nhận): Nhãn CLI sai + log MuPDF lẫn vào stdout.**
`chunker.py:656` in `File bị bỏ qua (scan) : [...]` cho cả file hỏng/khóa. PyMuPDF in 323 dòng `MuPDF error: ...` ra stdout; không gọi `fitz.TOOLS.mupdf_display_errors(False)`.

**#4 (Không chặn, Thấp): Chunker đọc cấu hình riêng.**
`_resolve_chunk_params` (`chunker.py:396-412`) tự `load_dotenv()` + `os.getenv("CHUNK_SIZE")` thay vì `src/config.py`; test gọi `chunk_pdf(path)` không tham số phụ thuộc `.env` máy cục bộ (chưa kiểm chứng ảnh hưởng vì không đọc `.env`).

**#5 (Không chặn, tài liệu):** `docs/contracts/task-2.md` mục 3.2 còn `data/pdfs/quy_che_mau.pdf` → nên đổi `data/samples/quy_che_mau.pdf`.

Quan sát (không phải lỗi): `join_soft_lines` nối tiêu đề in hoa / tên chương với dòng kế tiếp (chỉ ảnh hưởng hình thức); bộ lọc header/footer chỉ bỏ 1 dòng đầu/cuối — header 2 dòng sẽ sót 1 dòng.

## 4. Checklist Critic

- [x] Không mất/lệch `page_number`/`source_file` (M1: 11 FAIL; M22: 3 FAIL; CLI khớp trang).
- [x] Điều kéo trang đúng trang + đúng Điều (M5, M14) — trừ ca Chương cuối trang (#2).
- [x] ID cố định (M7).
- [—] Prompt/ngưỡng chống bịa — ngoài phạm vi.
- [x] Không lộ API key (`"secret-xyz"` trong `tests/test_config.py:45` là giá trị giả).
- [~] Thông báo lỗi dễ hiểu: đúng với rác/0 byte/khóa/scan; **sai với PDF cắt cụt** (#1).
- [x] Có test "không mất `page_number`" (`test_page_number_matches_exact_pdf_page`, bị M1 làm đỏ).

## 5. Hợp đồng (mục 4): TC1–TC18 Đạt; **TC19 chưa đạt** do #1.
Yêu cầu bổ sung: PDF mẫu ngoài `data/pdfs/` — Đạt; PDF hỏng/khóa bỏ qua với thông báo cố định và xử lý tiếp — Đạt một phần (cắt cụt báo sai loại).

## 6. Yêu cầu để APPROVED
1. Sửa #1 + test PDF cắt cụt.
2. Thêm test ca Chương cuối trang (#2) để giết M9.
3. (Khuyến nghị) Sửa nhãn "(scan)", tắt log MuPDF ra stdout, chunker dùng `src/config.py`, cập nhật đường dẫn trong hợp đồng.
