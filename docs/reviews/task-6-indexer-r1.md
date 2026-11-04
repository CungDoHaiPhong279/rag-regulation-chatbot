# Báo cáo Critic — Task 4 hợp đồng (Indexer + Sparse BM25 + Embedder + Config) @ b1e7daa — vòng 1

> Lưu nội dung kết luận và phát hiện của subagent Critic (chạy 2026-11-04). Kết quả: **REJECTED**.
> Trong lúc review HEAD chuyển sang 0294972 (commit chunker của luồng khác); `git diff b1e7daa HEAD` trên 9 file phạm vi rỗng. Không sửa repo; mutation trên bản sao.

## Kết luận: REJECTED (vòng 1)
Hành vi chạy thật đúng hợp đồng (dense/sparse IDF, payload, idempotent, dọn point thừa, preserve, fast-fail, local mode), nhưng **5 phép phá trên bất biến quan trọng để test xanh** — 4 trong số đó đúng là các mục hợp đồng chỉ có bằng chứng chạy tay, không có test bảo vệ. Theo Tiêu chí 15 và critic.md ("phép phá không làm test đỏ = lỗ hổng kiểm thử") task chưa đạt. Sửa nhỏ, chủ yếu thêm test.

## 1. Chạy thật
- `pytest -q` toàn repo: `103 passed, 1 deselected, 5 warnings in 12.73s`. 4 file test Task 4: `42 passed, 1 deselected` (19 test trong `test_indexer.py`, khớp đủ 17 tên mục 5.7).
- CLI thật (BGE-M3, Qdrant v1.12.5 container tạm cổng 6399, không `.env`): `Đã index 77 chunks từ 1 file vào 'critic_t4'`, `Đã xóa 0 point thừa`, trang 1–2 scan ảnh, real 1m26s, exit 0. Chạy lại: vẫn 77, 0 xóa.
- Collection: dense 1024 Cosine; sparse `modifier=IDF`; `payload_schema: {'source_file': KEYWORD}`; count 77, 77 id duy nhất; `page_number` kiểu int; 5 khóa payload.
- Truy vấn thật: DENSE "Sinh viên bị cảnh báo học tập khi nào?" → p13 Điều 11 (0.684), p14 Điều 11, p14 Điều 12; DENSE "Điều kiện công nhận tốt nghiệp" → p15 Điều 14 (0.786); SPARSE "cảnh báo học tập" → p14 Điều 11 (5.902), p13 Điều 11; SPARSE "thời gian tối đa hoàn thành khóa học" → p4 Điều 2 (10.903).

## 2. Đầu vào xấu
| Ca | Kết quả | Đánh giá |
|---|---|---|
| A1 a.pdf + b.pdf hợp lệ | `{'a.pdf': 14, 'b.pdf': 14}` | OK |
| A2 b.pdf bị hỏng | rc=0, vẫn `{a:14, b:14}`, "File bị bỏ qua…: ['b.pdf']" | OK (preserve qua CLI đúng) |
| A3 xóa b.pdf khỏi thư mục | `{'a.pdf': 14}` | OK |
| B thư mục rỗng | rc=1, dữ liệu giữ nguyên | OK |
| B2 thư mục không tồn tại | rc=1 | OK |
| E Qdrant tắt | rc=1 sau 2.6s, gợi ý docker/--local | OK |
| F embed lỗi qua CLI | exit 1, **traceback**, dữ liệu giữ | #7 |
| J `--recreate` rồi embed lỗi | **collection trống** | #8 |
| C collection có sẵn `Distance.DOT` dim 8 + IDF | được chấp nhận; `payload_schema: {}` | #6 |
| D local đang bị client khác giữ khóa | exit 1, **traceback** `Storage folder … already accessed` | #7 |
| G chunk chỉ dấu câu lẫn chunk thường | OK | OK |
| G2 toàn kho chỉ dấu câu | `ValueError avgdl phải > 0` | rất hiếm |
| Import indexer | torch False, sentence_transformers False | OK |

## 3. Mutation (24 phép; baseline 42 passed)
KILLED: M1 `page_number+1`; M2 xóa trước upsert; M3 bỏ preserve trong `index_corpus`; M4 scroll không phân trang; M5 uuid4; M6 tạo không IDF; M7 bỏ kiểm dim; M8 bỏ kiểm IDF; M10 DOT khi tạo; M12 tf thô; M13 bỏ chuẩn hóa độ dài; M14 rỗng không raise; M15 load embedder trước check; M16 import torch top-level; M17 config ép str; M18 bỏ recreate; M19 article None; M23 chỉ xóa lô đầu.

SURVIVED: **M9** bỏ `create_payload_index`; **M11** `avgdl` chỉ chunk đầu; **M20** upsert `wait=False`; **M21** payload `chunk_index` luôn 0; **M22** CLI truyền `preserve_files=frozenset()`; M24 bỏ NFC trong `tokenize` — mutant tương đương (pyvi tự trả NFC), không tính.

## 4. Phát hiện
**#1 [Trung bình, M22] Đường preserve file hỏng/scan qua CLI không có test** (`indexer.py:248`). Đổi thành `frozenset()` vẫn 42 xanh → mỗi lần PDF đã index bị hỏng tạm thời, CLI âm thầm xóa sạch dữ liệu cũ của nó (vi phạm 5.3). `bad.pdf` trong `test_cli_end_to_end_local` chưa từng có dữ liệu nên không kiểm gì. Mong đợi: test CLI chạy 2 lần (lần 1 a+b hợp lệ, lần 2 b hỏng) → assert point b còn.

**#2 [Trung bình, M11] Không test `avgdl` toàn kho** (`indexer.py:152`; 5.2, ADR-2). Mong đợi: sparse value của một chunk a.pdf khi index chung với b.pdf (dài hơn) khác khi index riêng; hoặc assert trực tiếp theo công thức với avgdl toàn kho.

**#3 [Trung bình, M9] Tiêu chí 2 (payload index keyword `source_file`) không có test** (`indexer.py:81-85`). Mong đợi: assert `payload_schema["source_file"].data_type == KEYWORD` (kiểm tra `:memory:` có trả payload_schema không, nếu không dùng local path).

**#4 [Thấp–Trung bình, M21] `chunk_index` payload không được kiểm** (`test_indexer.py:22-33, 62-68`): `mk()` luôn `chunk_index=0`, test payload không so `chunk_index`. Mong đợi: dữ liệu nhiều chunk cùng trang và assert `payload["chunk_index"] == src.chunk_index`.

**#5 [Thấp, M20] `wait=True` không được test.** Chấp nhận test spy bọc `client.upsert`/`client.delete` assert `kwargs["wait"] is True`.

**#6 [Thấp, ca C] `ensure_collection` không kiểm `distance` dense trên collection có sẵn** (`indexer.py:88-105`); collection có sẵn thiếu payload index không bị phát hiện. Mong đợi: distance ≠ COSINE → `CollectionSchemaError` + `--recreate`; thiếu payload index → tạo bù (idempotent).

**#7 [Thấp, ca F, D] Lỗi hay gặp vẫn traceback:** embed/upsert lỗi trong `main()` (`indexer.py:243` không bắt; exit 1 và dữ liệu giữ nhưng người dùng thấy stacktrace); local mode khi thư mục bị API/Streamlit giữ khóa (`get_client` `indexer.py:207` ngoài `try`). `client.close()` (dòng 262) không trong `finally`.

**#8 [Thấp, ca J] `--recreate` xóa collection trước khi embed** (`indexer.py:236-243`): embed lỗi sau recreate → collection trống. Người dùng chủ động chọn xóa nên Thấp; nên embed trước rồi mới recreate, hoặc ghi rõ trong `--help`.

Ghi nhận (không phải lỗi): hợp đồng ghi "< 5 giây" — chỉ collect đã 5.45s, 4 file ~9–10s (không tải model, không torch); spec dòng 120 nói scroll "không lấy vector/payload" nhưng code lấy payload `["source_file"]` — cần cho preserve, nên sửa câu chữ spec; không có test `query_points` (mục 5.7 đã bỏ), chạy thật đúng; chạy nhầm `--pdf-dir` sẽ dọn point mọi file khác (đúng thiết kế 5.3) — nên cảnh báo khi `deleted_stale` lớn.

## 5. Checklist
- [x] Không mất/lệch `page_number`/`source_file` (M1, payload int, query thật).
- [ ] Chunk kéo trang: phạm vi chunker.
- [x] Chạy lại không trùng (77→77, M5).
- [ ] Chống bịa: chưa thuộc task.
- [x] Không lộ key (`qdrant_api_key repr=False`).
- [~] Thông báo lỗi: OK với Qdrant tắt/thư mục rỗng/không tồn tại/PDF hỏng; traceback với embed/upsert lỗi và local bị khóa (#7).
- [~] Có test: đủ 17 tên; còn lỗ hổng #1–#5.

## 6. Yêu cầu để APPROVED vòng 2
1. Bắt buộc: test cho #1, #2, #3, #4.
2. Nên: #5, #6, #7.
3. Tùy chọn: #8.
Vòng 2 sẽ chạy lại M9, M11, M21, M22 (và M20 nếu có spy) — phải chuyển KILLED.
