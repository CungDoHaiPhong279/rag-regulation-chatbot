# Critic — kiểm chứng bản sửa Chunker (vòng 3) + Indexer (vòng 2) @ 38b190e

> Subagent Critic chạy trên **Sonnet**, phạm vi hẹp: chỉ kiểm chứng các phát hiện đã báo ở
> `task-3-chunker-r2.md` và `task-6-indexer-r1.md`, và chạy lại đúng các phép phá trước đây còn sống.
> Không tìm lỗi mới, không chạy lại CLI với Qdrant/PDF thật. Kết quả: **Chunker APPROVED, Indexer APPROVED**.

**pytest (repo thật, HEAD 38b190e):** `119 passed, 1 deselected, 5 warnings in 17.04s`. Không sửa repo, không commit, không đọc `.env`.

## Phát hiện → trạng thái
| Phát hiện | Trạng thái | Bằng chứng |
|---|---|---|
| Chunker #N1 PDF cắt cụt vẫn bị index | Đã xử lý | `chunker.py:532` `doc.is_repaired` → `PdfUnreadableError`; `test_truncated_pdf_reported_as_unreadable`, `test_repaired_pdf_with_text_is_unreadable`, `test_repaired_pdf_skipped_in_directory` |
| Chunker #N2 test log MuPDF vô hiệu | Đã xử lý | test giờ FAIL khi bỏ `mupdf_display_errors(False)` |
| Chunker #N3 0 trang báo "scan" | Đã xử lý | `chunker.py:532` `len(doc) == 0`; `test_zero_page_pdf_is_unreadable` |
| Chunker #N4 cấu hình riêng | Đã xử lý | `chunker.py:30,402` dùng `load_settings()`; `test_chunk_params_follow_config_rules` |
| Indexer #1 preserve qua CLI | Đã xử lý | `test_cli_preserves_points_of_file_that_became_corrupt` |
| Indexer #2 avgdl toàn kho | Đã xử lý | `test_avgdl_computed_over_whole_corpus` |
| Indexer #3 payload index keyword | Đã xử lý | `test_payload_index_created_for_source_file` |
| Indexer #4 `chunk_index` payload | Đã xử lý | `test_payload_chunk_index_preserved` |
| Indexer #5 `wait=True` | Đã xử lý | `test_upsert_and_delete_wait_for_completion` |
| Indexer #6 distance, thiếu payload index | Đã xử lý | `indexer.py:100` kiểm COSINE, `:114` tạo bù; 2 test `test_existing_collection_*` |
| Indexer #7 traceback | Đã xử lý | `indexer.py:228,279` bắt lỗi, `:239` `finally`; 2 test CLI |
| Indexer #8 `--recreate` xóa trước embed | Chấp nhận bằng ghi chú | theo chỉ định |

## Phép phá (mỗi phép một bản sao riêng, đã diff xác nhận sed đổi file)
| Phép phá | Kết quả | Test FAIL |
|---|---|---|
| Chunker M9 bỏ `last_seen_article = None` | KILLED | `test_chapter_heading_at_page_bottom_resets_next_page` |
| Chunker N3 bỏ `mupdf_display_errors(False)` | KILLED | `test_directory_output_has_no_mupdf_noise_and_honest_label` |
| Chunker bỏ `doc.is_repaired or` | KILLED | `test_truncated_pdf_reported_as_unreadable` |
| Indexer M9 bỏ `create_payload_index` đầu | KILLED | `test_payload_index_created_for_source_file` |
| Indexer M11 `token_lists[:1]` | KILLED | `test_avgdl_computed_over_whole_corpus` |
| Indexer M20 `wait=False` | KILLED | `test_upsert_and_delete_wait_for_completion` |
| Indexer M21 `"chunk_index": 0` | KILLED | `test_payload_chunk_index_preserved` |
| Indexer M22 `preserve_files=frozenset()` | KILLED | `test_cli_preserves_points_of_file_that_became_corrupt` |

## Kết luận
- **Chunker: APPROVED** — #N1–#N4 đã xử lý, có test; 3/3 phép phá KILLED.
- **Indexer: APPROVED** — #1–#7 đã xử lý, có test; #8 chấp nhận bằng ghi chú; 5/5 phép phá KILLED.

**Giới hạn của lượt này:** chỉ khẳng định các lỗi đã biết đã được sửa và có test bắt; không tìm lỗi mới. Lỗi mới (nếu có) do bản sửa gây ra sẽ được xem ở review cuối nhánh.
