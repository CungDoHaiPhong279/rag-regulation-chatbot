# Báo cáo Critic — Task 2 (Chunker) @ commit 0294972 — vòng 2

> Lưu nội dung kết luận và phát hiện của subagent Critic (chạy 2026-11-04). Kết quả: **REJECTED**.

**Kết luận: REJECTED** — 1 lỗi chặn mới đã xác nhận bằng chạy thật (PDF thật bị cắt cụt vẫn bị index thành 129 chunk rác thay vì bị bỏ qua như spec mục 6). Hai phát hiện chặn/lỗ hổng của vòng 1 (#1, #2) đã được sửa và tự kiểm chứng; 4 điểm nhỏ không chặn.

Phạm vi: `src/ingestion/chunker.py`, `src/ingestion/sample_pdf.py`, `src/utils.py` + 3 file test. `git diff --stat 90a8f16 HEAD`: chỉ `chunker.py` (+24/−10), `test_chunker.py` (+34), `docs/contracts/task-2.md` (1 dòng). Không sửa repo, không commit, không đọc `.env`. Mutation trên bản sao trong scratchpad.

## 1. Bằng chứng chạy thật
- `pytest -q` toàn repo: `106 passed, 1 deselected, 5 warnings in 15.37s`.
- 3 file test giao: `64 passed`. `test_chunker.py -v`: `54 passed` (gồm 3 test mới).
- CLI `data/samples/quy_che_mau.pdf`: EXIT=0, 6 trang, 14 chunk, 10 Điều, lọc 12 dòng; 3 chunk mẫu đúng trang, tiền tố đúng 1 lần, len ≤ 800.
- Thư mục PDF xấu: EXIT=0, 0 dòng `MuPDF` ở stdout/stderr, nhãn `File bị bỏ qua (scan / hỏng / bị khóa)`; 3 file hợp lệ → 42 chunk.

| File | Loại | Thông báo | Đánh giá |
|---|---|---|---|
| b_junk.pdf | 9 byte rác | hỏng hoặc bị khóa | Đúng |
| c_empty.pdf | 0 byte | hỏng hoặc bị khóa | Đúng |
| d_trunc.pdf | PDF mẫu cắt 50% | hỏng hoặc bị khóa | **Đúng (đã sửa #1)** |
| e_locked.pdf | AES-256 user pw | hỏng hoặc bị khóa | Đúng |
| g_blank.pdf | trang trắng | cần OCR | Đúng |
| h_scanimg.pdf | chỉ ảnh | cần OCR | Đúng |
| i_scan_badxref.pdf | ảnh + startxref sai | hỏng hoặc bị khóa | Chấp nhận được |
| j_text_badxref.pdf | PDF mẫu + startxref sai | cảnh báo "đã được tự sửa", 14 chunk đúng | Đúng |

- Fuzz: cắt cụt PDF mẫu 5%–95%: 19/19 UNREADABLE; flip 200 byte ×40: 39 OK, 1 NOTEXT (#N3). Không exception lọt ra.

## 2. Kiểm chứng phát hiện vòng 1
| # | Trạng thái |
|---|---|
| #1 PDF cắt cụt báo "scan" | Đã sửa với PDF mẫu — **chưa đủ** (#N1) |
| #2 M9 sống sót | **Đã sửa** (N2 = M9 làm test mới đỏ) |
| #3 nhãn + log MuPDF | Code đã sửa; **test không bảo vệ phần log** (#N2) |
| #4 chunker đọc cấu hình riêng | **Chưa sửa** (#N4) |
| #5 đường dẫn hợp đồng | Đã sửa |

## 3. Mutation testing (trước 64 passed; sau hoàn tác 64 passed)
| # | Phép phá | Kết quả |
|---|---|---|
| N1 | Bỏ nhánh `if repaired: raise PdfUnreadableError` | 1 FAIL |
| N2 | (M9) bỏ `last_seen_article = None` | 1 FAIL |
| **N3** | **Bỏ `fitz.TOOLS.mupdf_display_errors(False)`** | **64 passed — SỐNG SÓT** |
| N4 | Nhãn CLI về `(scan)` | 1 FAIL |
| N5 | Mọi `PdfNoTextError` → Unreadable | 2 FAIL |
| **N6** | **Tắt cảnh báo "đã được tự sửa"** | **64 passed — SỐNG SÓT** |
| N7 | `page_number + 1` | 12 FAIL |
| N8 | Không mang Điều sang trang sau | 4 FAIL |
| N9 | Bỏ `needs_pass` | 1 FAIL |
| N10 | `repaired = False` | 1 FAIL |
| N11 | Unreadable không ghi `skipped_files` | 2 FAIL |

9/11 bị bắt. N3 xác nhận là lỗ hổng thật: bản phá N3 chạy CLI trên thư mục `bad/` in **323 dòng `MuPDF error`** ra stdout, nhưng fixture của test (`b"not a pdf"`) không sinh log MuPDF nên `assert "MuPDF error" not in out` luôn đúng.

## 4. Phát hiện

**#N1 (CHẶN, Cao, đã xác nhận): PDF thật bị cắt cụt vẫn bị index thành chunk rác.**
`chunker.py:549-568`: bản sửa #1 chỉ đổi sang `PdfUnreadableError` khi `repaired` **và** không còn chữ. Khi MuPDF sửa được cấu trúc nhưng mất ToUnicode/font, `get_text()` trả chuỗi rác khác rỗng → chỉ in cảnh báo rồi trả chunk.
Tái hiện: cắt `data/pdfs/thong_tu_08_2021.pdf` (717446 byte) còn 50%, chạy CLI → EXIT=0, `Tổng số chunk sinh ra : 129` (gốc 77), `Số Điều nhận diện được: 0`, không nằm trong danh sách bỏ qua; 125/129 chunk chứa ký tự điều khiển (`'3. \x004\x00X\x00\\\x00\x03…'`); trang 3–22 đều khác bản gốc. Theo mức cắt: 30/50/70/80% → 129 chunk rác; 90–99% → 77 chunk; 10% → UNREADABLE.
Tác động: indexer gọi thẳng `chunk_directory` → upsert rác vào Qdrant; vi phạm spec mục 6 và quy tắc "không index dữ liệu rác"; cảnh báo "có thể thiếu nội dung" gây hiểu nhầm (nội dung bị hỏng, không phải thiếu).
Mong đợi: file loại này vào `skipped_files` với đúng thông báo spec. Đơn giản nhất: mọi `doc.is_repaired` → `PdfUnreadableError` (đánh đổi: ca chỉ hỏng startxref nhưng chữ đúng cũng bị bỏ qua, người dùng được báo tải lại). Hoặc heuristic tỉ lệ ký tự điều khiển. Cần test với PDF có font nhúng, cắt cụt mà vẫn còn chữ rác.

**#N2 (Không chặn, Thấp, lỗ hổng test): test chống log MuPDF vô hiệu (N3 sống sót).** Fixture phải dùng file thật sự sinh log; kiểm chứng test chuyển đỏ với bản phá N3 (lưu ý `capsys` chỉ bắt được log đi qua `sys.stdout`).

**#N3 (Không chặn, Thấp, đã xác nhận): PDF có cây trang hỏng `page_count == 0` bị báo "scan".** File flip seed=1: `is_repaired=False`, `page_count=0`, `mupdf_warnings()` báo `non-page object in page tree` → "cần OCR". Mong đợi: `len(doc) == 0` → `PdfUnreadableError`.

**#N4 (Không chặn, Thấp, tồn từ vòng 1): chunker đọc cấu hình riêng, lệch với `src/config.py`.** `chunker.py:396-412`. `CHUNK_SIZE=` (rỗng): `load_settings(None).chunk_size` = 800 nhưng `chunk_pages(['abc'], 'x.pdf')` → `ValueError: invalid literal for int()`; `CHUNK_OVERLAP=abc` → CLI chunker traceback thô, 17 test đỏ. Indexer không bị ảnh hưởng (truyền `settings.chunk_size`). Mong đợi: dùng `load_settings()` làm nguồn duy nhất.

Quan sát: không có test cho cảnh báo "đã được tự sửa" (N6) — nếu sửa #N1 bằng cách bỏ qua mọi file repaired thì cảnh báo không cần nữa; `mupdf_display_errors(False)` là cài đặt toàn cục (chấp nhận được); CLI chunker exit 0 khi mọi file hỏng (quy tắc exit 1 thuộc indexer).

## 5. Checklist
- [x] Không mất/lệch `page_number`/`source_file` (N7: 12 FAIL).
- [x] Điều kéo trang + ca Chương cuối trang (N2, N8).
- [x] ID cố định.
- [—] Chống bịa: ngoài phạm vi.
- [x] Không lộ key.
- [~] Thông báo lỗi: đúng với rác/0 byte/khóa/scan/PDF mẫu cắt cụt; **sai với PDF thật cắt cụt (#N1)** và file 0 trang (#N3).
- [x] Có test "không mất `page_number`".

## 6. Hợp đồng
TC1–4, 7–18 Đạt; TC5 đạt với PDF đọc được; TC6/15 Đạt; **TC19 chưa đạt** do #N1. PDF mẫu ngoài `data/pdfs/`: Đạt. PDF hỏng/khóa bỏ qua: đạt một phần.

## 7. Yêu cầu để APPROVED
1. Sửa #N1 + test PDF có font nhúng cắt cụt vẫn sinh chữ rác → nằm trong `skipped_files`, in đúng thông báo spec.
2. Sửa fixture test log MuPDF (#N2) để N3 bị bắt.
3. (Khuyến nghị) #N3 (0 trang → hỏng) và #N4 (dùng `src/config.py`).
