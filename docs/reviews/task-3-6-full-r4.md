# Critic — review đầy đủ Chunker + Indexer @ a3a3fd5 (Sonnet, ngân sách ~30 lệnh, dùng ~20)

**Kết luận:** Chunker **APPROVED** (3 khuyến nghị); Indexer **REJECTED** (lỗi #1 xóa dữ liệu im lặng, đã xác nhận bằng chạy thật).

Bằng chứng chạy thật: pytest bản sao `119 passed, 1 deselected`; CLI chunker PDF mẫu 6 trang/14 chunk/10 Điều; Thông tư 08 với chunk_size 400/800/1200 → 168/77/58 chunk (max len 399/800/1196), trang 1–2 scan báo đúng, mất chữ 5/3430 từ; Indexer + Qdrant v1.12.5 + BGE-M3 thật: lần 1 91 point (77 + 14), lần 2 giữ nguyên, xóa 0.

## Lỗi mới
| # | Mức | Mô tả | Tái hiện | Mong đợi |
|---|---|---|---|---|
| 1 | **Cao** | File có lớp chữ nhưng 0 chunk (mọi dòng bị lọc như header/footer) không vào `skipped_files`, không cảnh báo → `index_corpus` coi là "file biến mất" và xóa sạch point cũ. Gốc: `chunk_pages` trả `[]` im lặng. | Index `quy_che_mau.pdf` (14 point), thay bằng PDF 3 trang chỉ có "Trang N", chạy lại → `Đã xóa 14 point thừa` | File 0 chunk vào `skipped_files` (được preserve) |
| 2 | Trung bình | `CHUNK_SIZE=abc`: indexer `main()` traceback `ValueError` (`load_settings()` ngoài `try`); chunker CLI cũng vậy | `CHUNK_SIZE=abc python -m src.ingestion.indexer` | Thông báo tiếng Việt, exit 1 |
| 3 | Trung bình | Model lỗi (`EMBEDDING_MODEL=nonexistent/zzz`, offline chưa cache) → traceback `OSError` (`BgeM3Embedder(...)` ngoài `try`) | `EMBEDDING_MODEL=nonexistent/zzz` | Thông báo thân thiện, exit 1 |
| 4 | Thấp | Qdrant 403 (sai API key) báo như "chưa bật Docker" (detail bị bỏ với `QdrantUnavailableError`) | server giả trả 403 | Nêu nguyên nhân thật |
| 5 | Thấp | `BgeM3Embedder.dense_dim = 1024` cứng dù `EMBEDDING_MODEL` đổi được (suy luận, chưa chạy) | — | Lấy dim từ model hoặc ghi rõ ràng buộc |
| 6 | Thấp | `Ðiều 5.` (Ð U+00D0) và `ĐIỀU 5.` không nhận là tiêu đề Điều; TT08 không dính | `parse_article_heading('Ðiều 5. Học phí')` → None | Chuẩn hóa Ð→Đ, không phân biệt hoa thường |
| 7 | Thấp | `chunk_pdf` không bọc lỗi đọc từng trang (`get_text`) — đọc code, chưa kiểm chứng | — | Chuyển thành `PdfUnreadableError` |

## Đầu vào xấu (5 ca)
`.pdf` rỗng / chứa text thường / thư mục tên `x.pdf` → vào `skipped_files` (tốt); collection unnamed → báo + `--recreate` (tốt); 403 → sai hướng (#4); `CHUNK_SIZE=abc` → traceback (#2); model sai → traceback (#3).

## Phép phá (12/12)
C1 page bắt đầu 2, C2 không mang Điều, C3 file hỏng không vào skipped, C4 bỏ NFC, C5 chunk_id bỏ trang, C6 ngưỡng 60→100%, I1 xóa trước upsert, I2 bỏ kiểm distance, I3 bỏ kiểm IDF, I5 bỏ phân trang scroll, I6 `--recreate` không xóa → **KILLED**. I4 bỏ guard `if not chunks` → SURVIVED, gần như tương đương (`average_doc_length` vẫn ném `ValueError` trước khi ghi). Không có lỗ hổng test thật; #1 là lỗ hổng thiết kế.

## Checklist
`page_number`/`source_file` Đạt; kéo trang Đạt; không trùng Đạt (91→91); chống bịa ngoài phạm vi; key không lộ; lỗi thân thiện đạt một phần (#2–#4); có test `page_number` Đạt.

## Chưa kiểm
#7; tokenize với chunk bất thường trên Qdrant thật (fuzz 8 chuỗi không lỗi); local mode qua CLI; header/footer nhiều dòng.

## Điều kiện APPROVED Indexer
Sửa #1 (file 0 chunk vào `skipped_files` hoặc báo lỗi, kèm test) và bọc `load_settings` + `BgeM3Embedder` trong xử lý lỗi (#2, #3).
