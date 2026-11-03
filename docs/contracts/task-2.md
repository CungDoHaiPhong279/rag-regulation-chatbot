# Hợp đồng nghiệm thu: Task 2 — Module Chunker (Cắt Chunk Giữ Số Trang & Tiêu Đề Điều)

> **File:** `docs/contracts/task-2.md`  
> **Các bên tham gia:** Builder (Thiết kế & Lập trình) & Critic (Phản biện & Kiểm thử độc lập)  
> **Tài liệu căn chiếu:** `AGENTS.md`, `docs/superpowers/plans/2026-11-02-week1-ingestion.md`

---

## 1. Mục tiêu của Task 2
Xây dựng module `src/ingestion/chunker.py` để:
- Đọc file PDF quy chế tiếng Việt, chuẩn hóa Unicode NFC, loại bỏ header/footer lặp.
- Tách văn bản theo từng trang và theo từng Điều trước khi cắt chunk.
- Đảm bảo **mỗi chunk chỉ thuộc đúng 1 trang vật lý**, **không chứa 2 Điều**, và được gắn tiền tố tiêu đề Điều đầy đủ (kể cả khi Điều kéo sang trang sau).
- Giữ nguyên toàn vẹn metadata (`source_file`, `page_number`, `article`, `chunk_index`, `text`, `chunk_id`).
- Bỏ qua PDF scan không có text (ném `PdfNoTextError`), không làm gián đoạn cả thư mục.
- Cung cấp CLI chạy thử nghiệm in tiếng Việt UTF-8 chuẩn trên Windows.

---

## 2. Đề xuất của Builder (Checklist "Xong là gì")

### 2.1. Giao diện (Interfaces & Data Classes)
- `PdfNoTextError(Exception)`: Ngoại lệ khi PDF rỗng/scan ảnh.
- `Chunk(dataclass, frozen=True)`:
  - `chunk_id`: UUIDv5 băm tất định từ `f"{source_file}:{page_number}:{chunk_index}"`.
  - `source_file`: Tên file gốc (basename).
  - `page_number`: Số trang vật lý trong PDF (bắt đầu từ 1).
  - `article`: Chuỗi chuẩn hóa `"Điều X. Tên Điều"` hoặc `None`.
  - `chunk_index`: Thứ tự chunk trong trang (bắt đầu từ 0).
  - `text`: Nội dung chunk (có tiền tố `"Điều X. Tên: "` nếu có `article`).
- Hàm xử lý chính:
  - `normalize_text(text: str) -> str`: Chuẩn hóa NFC và làm sạch khoảng trắng thừa.
  - `parse_article_heading(line: str) -> str | None`: Trích xuất số Điều và tên Điều.
  - `split_by_article(page_text: str, carried: str | None) -> list[tuple[str | None, str]]`: Tách văn bản thành các đoạn theo Điều (và ngắt `carried` về `None` khi gặp tiêu đề cấp trên `Chương` / `Mục` / `Phần` ở đầu dòng).
  - `chunk_pages(page_texts: list[str], source_file: str, chunk_size: int | None = None, chunk_overlap: int | None = None) -> list[Chunk]`: Thực hiện toàn bộ chuẩn hóa NFC, lọc header/footer lặp, tách Điều và cắt chunk từ danh sách chuỗi văn bản từng trang.
  - `chunk_pdf(pdf_path: Path, chunk_size: int | None = None, chunk_overlap: int | None = None) -> list[Chunk]`: Chỉ đọc các trang từ file PDF rồi gọi `chunk_pages`.
  - `chunk_directory(pdf_dir: Path, ...) -> tuple[list[Chunk], list[str]]`: Xử lý cả thư mục, trả về `(chunks, skipped_files)`.

### 2.2. Danh sách Test Case cốt lõi của Builder
1. `test_parse_article_heading`: Nhận diện đúng các dạng `"Điều 5. Học phí"`, `"Điều 5: Học phí"`, và từ chối các câu tham chiếu không có dấu chấm/hai chấm sau số ("Điều 5 của...").
2. `test_split_by_article_separates_two_articles`: Tách 2 Điều trên cùng một trang thành 2 đoạn riêng biệt.
3. `test_metadata_schema_complete`: Mọi chunk có đủ 6 trường dữ liệu.
4. `test_page_number_matches_exact_pdf_page`: `page_number` khớp 100% với số trang vật lý của PDF mẫu.
5. `test_chunk_size_includes_prefix`: `len(chunk.text) <= chunk_size` với các mức 400, 800, 1200 (tính cả tiền tố).
6. `test_deterministic_chunk_id`: 2 lần chạy cho ra cùng một danh sách `chunk_id`.

---

## 3. Ý kiến phản biện & Bổ sung của Critic (Red-Team Checklist)

Critic yêu cầu bổ sung các ca biên nguy hiểm và ràng buộc kỹ thuật sau vào hợp đồng:

### 3.1. Các ca biên bắt buộc phải có test (Edge Cases)
1. **Ca Điều kéo dài qua trang sau (`test_article_prefix_carried_across_pages`):**
   - Phải test trường hợp Điều 5 kéo từ trang 3 sang trang 4: Chunk ở trang 4 phải mang `page_number = 4`, `article = "Điều 5. Học phí"` và `text.startswith("Điều 5. Học phí: ")`.
2. **Ca trang chứa cuối Điều trước và đầu Điều sau (`test_page_with_two_articles_gives_separate_chunks`):**
   - Trên cùng 1 trang, phải sinh ra các chunk riêng cho Điều trước và các chunk riêng cho Điều sau, **tuyệt đối không có chunk nào chứa tiêu đề của Điều khác trong phần thân**.
3. **Ca chống lặp tiền tố (`test_prefix_not_duplicated`):**
   - Đảm bảo không bao giờ sinh ra chuỗi dị dạng `"Điều 5. Học phí: Điều 5. Học phí: ..."`.
4. **Ca `chunk_size` quá nhỏ so với độ dài tiêu đề Điều (`test_chunk_size_too_small_for_title_raises`):**
   - Nếu `chunk_size - len(prefix) < 100` hoặc `chunk_overlap >= body_size`, hệ thống phải ném `ValueError` rõ ràng, không được treo hoặc sinh chunk âm.
5. **Ca Unicode NFD thực tế (`test_unicode_nfc_normalization`):**
   - Chuỗi NFD ("Đ\u0069\u1ec1\u0075" / dấu tổ hợp) phải được chuẩn hóa về NFC ("Điều") để regex và so khớp chuỗi không bị trượt.
6. **Ca lọc Header/Footer lặp (`test_repeated_header_footer_removed`):**
   - Header lặp như `"TRƯỜNG ĐẠI HỌC MẪU"` và footer `"Trang X"` xuất hiện ở `>= 60%` số trang phải bị lọc sạch trước khi cắt chunk.
7. **Ca PDF scan ảnh (`test_scanned_pdf_raises_and_directory_skips`):**
   - `chunk_pdf` phải ném `PdfNoTextError` khi trang không có text; `chunk_directory` phải bỏ qua và ghi nhận vào `skipped_files`, không làm sập tiến trình.

### 3.2. Yêu cầu về kiến trúc & In ấn UTF-8 trên Windows
- **Xử lý UTF-8 dùng chung:** Tạo module tiện ích `src/utils.py` chứa hàm `setup_utf8_console()` để cấu hình bảng mã UTF-8 cho stdout/stderr một lần duy nhất, tránh việc phải copy-paste `sys.stdout.reconfigure` rải rác vào từng file CLI.
- **Tiêu chuẩn kiểm toán (Critic Review Gate):**
  - Critic sẽ chạy trực tiếp lệnh:
    ```powershell
    .venv\Scripts\python -m pytest tests/ingestion/test_chunker.py -v
    .venv\Scripts\python -m src.ingestion.chunker data/pdfs/quy_che_mau.pdf
    ```
  - Báo cáo của Critic bắt buộc phải dán output thực tế của 2 lệnh trên và kiểm tra trực tiếp 3 chunk mẫu in ra. Nếu thiếu bằng chứng chạy thật -> **REJECTED ngay lập tức**.

---

### 3.3. Bổ sung của người dùng (vòng 2 — chốt)
1. **ID cố định:** chạy `chunk_pdf` 2 lần trên cùng file → danh sách `chunk_id` giống hệt (test `test_deterministic_chunk_id`).
2. **Regex Điều:**
   - `"Điều 5 của Quy chế này quy định"` → `None` (không nhận là tiêu đề).
   - `"Điều 5."` (không có tên) → trả `"Điều 5"`, không lỗi, không sinh tiền tố dị dạng; chunk của Điều đó có tiền tố `"Điều 5: "` (test `test_heading_without_title_does_not_break`).
3. **Văn bản trước Điều đầu tiên** (tiêu đề văn bản, căn cứ ban hành, tên Chương I): `article=None`, `text` không có tiền tố (test `test_text_before_first_article_has_no_prefix`).
4. **PdfNoTextError** có tên file trong thông báo; CLI và `chunk_directory` in ra stderr dạng `Bỏ qua <tên file>: không có lớp chữ (PDF scan?) — cần OCR` (test `test_scanned_pdf_message_names_file`).
5. **`setup_utf8_console()`** trong `src/utils.py`: chỉ gọi `reconfigure(encoding="utf-8")` khi `hasattr(stream, "reconfigure")`, áp dụng cho cả stdout và stderr; gọi nhiều lần không lỗi (test `tests/test_utils.py`).

6. **Phân tách `chunk_pages` và `chunk_pdf`:** Toàn bộ xử lý chuẩn hóa NFC, lọc header/footer lặp và cắt chunk nằm trong `chunk_pages`; `chunk_pdf` chỉ đọc PDF rồi gọi `chunk_pages`.
7. **Tiêu đề cấp trên (`Chương`, `Mục`, `Phần`):** Nhận diện ở đầu dòng với số La Mã (`I, V, X, L, C, D, M`) hoặc số thường (`\d+`). Khi gặp dòng này, ngắt `carried_article` về `None` để tiêu đề Chương/Mục/Phần (kể cả khi `"Chương II"` và tên chương nằm ở 2 dòng riêng) tạo thành chunk có `article=None`, không tiền tố, không bị gán nhầm vào Điều liền trước (test `test_chapter_section_part_resets_carried_article` và `test_two_line_chapter_heading_resets_carried_article`).
8. **Kiểm thử đột biến (Mutation Testing) của Critic:** Ngoài chạy `pytest` và CLI thật, Critic phải chủ động thử phá code có chủ đích (ví dụ: làm lệch `page_number`, tắt bước lọc header/footer), chạy `pytest` để xác nhận bộ test chuyển đỏ (FAIL), sau đó hoàn tác code về nguyên trạng và xác nhận `pytest` xanh trở lại.

### 3.4. Ghi chú trung thực về quy trình
- Mục 2 và 3.1–3.2 do **cùng một agent** soạn (chưa có Critic độc lập ở bước hợp đồng).
- Vòng 1: **Subagent Critic độc lập** (`Model='pro'`) thẩm định độc lập, chạy mutation testing và phát hiện 2 lỗi logic biên nghiêm trọng -> Kết luận: **`REJECTED`**.
- Vòng 2: Builder khắc phục toàn diện 2 lỗi và bổ sung các ca kiểm thử theo yêu cầu dưới đây. Subagent Critic MỚI sẽ thẩm định lại độc lập.

### 3.5. Bổ sung sửa đổi Vòng 2 (Khắc phục 2 lỗi do Critic phát hiện)
1. **Lỗi 1 (Mất Điều khi tiêu đề ở cuối trang):**
   - `split_by_article` trả về `(segments, last_seen_article)`.
   - Trong `chunk_pages`, cập nhật `current_article = last_seen_article` để mang Điều cuối trang sang trang sau trọn vẹn kể cả khi trang đó chưa có nội dung thân.
   - Thêm 3 unit test: (a) tiêu đề cuối trang, thân ở trang sau (`test_heading_at_page_bottom_carried_to_next_page`); (b) hai tiêu đề liên tiếp cuối trang không có thân (`test_two_consecutive_headings_at_page_bottom_without_body`); (c) tiêu đề cuối trang rồi dòng Chương ở trang sau (`test_heading_at_page_bottom_then_chapter_on_next_page`).
2. **Lỗi 2 (Bộ lọc Footer xóa nhầm nội dung có chứa số):**
   - Không dùng quy tắc "dòng ngắn".
   - Chỉ chuẩn hóa các mẫu số trang điển hình (`Trang N`, `Trang N/M`, `N/M`, `- N -`, `[N]`, `N`, không phân biệt hoa thường). Mọi dòng khác bắt buộc phải trùng nguyên văn 100% giữa các trang mới bị coi là lặp.
   - Thêm unit test kiểm tra tài liệu 3 trang có dòng cuối là `"Nội dung Điều 4."`, `"Nội dung Điều 5."`, `"Nội dung Điều 6."` không bị xóa nhầm (`test_distinct_numbered_lines_at_page_bottom_not_removed_as_footer`).
   - Sửa `test_chapter_section_part_resets_carried_article` để khẳng định Trang 1 giữ nguyên nội dung `"Nội dung Điều 4."`.
3. **Hiển thị dòng bị lọc trên CLI:**
   - CLI `main()` in ra danh sách đầy đủ các dòng header/footer lặp đã bị loại bỏ kèm số trang và vị trí (header/footer) (`test_removed_lines_collector_and_cli_output`).

### 3.6. Bổ sung sửa đổi Vòng 3 (Gia cố theo thực nghiệm PDF thật 22 trang Thông tư 08/2021)
1. **Xử lý trang không có chữ (`SkippedPage`):**
   - Định nghĩa dataclass `SkippedPage(page_number: int, reason: str, has_images: bool)`.
   - `chunk_pdf` và `chunk_pages` trả về `tuple[list[Chunk], list[SkippedPage]]`.
   - Nếu một trang không có lớp văn bản: in cảnh báo chi tiết ra `sys.stderr` nêu rõ số trang và phân loại (`"trang scan ảnh"` nếu `has_images=True` hoặc `"trang trắng"` nếu `has_images=False`), thu thập vào `skipped_pages`.
   - Nếu toàn bộ file không có trang nào có chữ: ném `PdfNoTextError` như cũ.
   - Test: `test_skipped_pages_returned_and_warned_on_stderr`.
2. **Tiêu đề Điều nhiều dòng (`parse_multiline_article_heading`):**
   - Tự động gộp dòng kế tiếp khi tiêu đề dòng 1 chưa kết thúc bằng dấu câu ngắt (`.`, `:`, `;`, `!`, `?`) VÀ dòng kế tiếp bắt đầu bằng chữ thường (theo chuẩn Unicode Python `char.islower()`, không phải chữ hoa, và không phải các điểm/khoản liệt kê như `1.`, `a)`, `đ)`).
   - Chuẩn hóa tiêu đề: loại bỏ dấu phẩy hoặc dấu chấm ngắt ở đuôi.
   - Giới hạn tối đa 2 dòng và tổng độ dài `<= 250` ký tự.
   - Test: `test_multiline_heading_with_comma_merged` (Điều 16 có dấu phẩy), `test_multiline_heading_without_comma_merged` (không dấu phẩy), `test_multiline_heading_not_merged_when_uppercase` (chữ hoa), `test_multiline_heading_not_merged_when_clause_or_point` (`1.`, `a)`, `đ)`), `test_multiline_heading_not_merged_when_ended_with_period` (kết thúc bằng dấu chấm).

---

## 4. Bảng kiểm tra nghiệm thu chính thức (Acceptance Checklist)

Builder chỉ được tuyên bố hoàn thành Task 2 khi toàn bộ các tiêu chí dưới đây đạt:

- [x] **Tiêu chí 1 (Metadata toàn vẹn):** Mọi chunk đều có `source_file`, `page_number` (1-based), `article`, `chunk_index` (0-based liên tục trên từng trang), `text`, và `chunk_id` (UUIDv5).
- [x] **Tiêu chí 2 (Không lệch số trang):** `page_number` của chunk khớp chính xác số trang chứa đoạn văn đó trong file PDF gốc.
- [x] **Tiêu chí 3 (Tách Điều & Tiền tố):** Không chunk nào chứa 2 Điều. Mọi chunk thuộc Điều đều có tiền tố `"<article>: "`. Điều kéo dài sang trang sau vẫn mang đúng nhãn `article` và tiền tố.
- [x] **Tiêu chí 4 (Tôn trọng `chunk_size`):** `len(chunk.text) <= chunk_size` (tính cả tiền tố) trên toàn bộ các mức cấu hình (400, 800, 1200 ký tự).
- [x] **Tiêu chí 5 (Làm sạch & Chuẩn hóa):** Văn bản đầu ra là Unicode NFC; header/footer lặp của PDF bị loại bỏ mà không xóa nhầm nội dung điều khoản.
- [x] **Tiêu chí 6 (Xử lý PDF Scan):** Ném lỗi `PdfNoTextError` khi toàn bộ file là scan; in cảnh báo và thu thập `SkippedPage` khi có trang scan lẻ tẻ.
- [x] **Tiêu chí 7 (UTF-8 dùng chung):** CLI in tiếng Việt mượt mà qua `src/utils.py`, không lỗi font trên console Windows.
- [x] **Tiêu chí 8 (ID cố định):** `chunk_pdf` chạy 2 lần cho cùng danh sách `chunk_id`.
- [x] **Tiêu chí 9 (Regex biên):** câu tham chiếu "Điều 5 của…" không bị nhận; "Điều 5." không tên không làm hỏng chunk.
- [x] **Tiêu chí 10 (Phần mở đầu & Chương/Mục/Phần):** văn bản trước Điều đầu tiên và các dòng tiêu đề `Chương`/`Mục`/`Phần` (kể cả tên chương ở dòng kế tiếp) có `article=None`, không tiền tố.
- [x] **Tiêu chí 11 (Báo lỗi rõ):** thông báo bỏ qua PDF scan nêu đúng tên file.
- [x] **Tiêu chí 12 (Tiêu đề cuối trang):** Tiêu đề Điều nằm ở cuối trang trước được bảo lưu nguyên vẹn sang trang sau, không bị mất và không bị gán nhầm vào Điều trước.
- [x] **Tiêu chí 13 (Không nuốt nhầm câu văn):** Bộ lọc footer bảo toàn các câu văn có chứa số (như `"Nội dung Điều 4."`).
- [x] **Tiêu chí 14 (CLI báo cáo dòng lọc):** CLI in rõ danh sách dòng header/footer lặp bị loại bỏ kèm số trang.
- [x] **Tiêu chí 15 (Trang không có chữ & SkippedPage):** `chunk_pdf` và `chunk_pages` trả về `(chunks, skipped_pages)`, in cảnh báo `stderr` nêu rõ số trang và loại trang (scan ảnh vs trang trắng).
- [x] **Tiêu chí 16 (Tiêu đề Điều nhiều dòng):** Nhận diện và gộp tiêu đề kéo dài 2 dòng (như Điều 16 có dấu phẩy ở dòng 1), không gộp nhầm các dòng chữ hoa hay điểm/khoản `1.`, `a)`, `đ)`.
- [x] **Tiêu chí 17 (Kiểm thử & Review độc lập):** Toàn bộ unit tests PASS 100%; Subagent Critic độc lập kiểm toán thực tế và duyệt `APPROVED`.

