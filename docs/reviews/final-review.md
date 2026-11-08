# Review cuối nhánh feat/rag-pipeline (Critic, Sonnet) — 2026-11-08

Phạm vi: `git diff checkpoint/pre-redesign` (gồm thay đổi chưa commit lúc review). Báo cáo nguyên văn bên dưới;
phần "Xử lý" ở cuối do Claude (agent chính) ghi.

---

KẾT LUẬN: APPROVED có điều kiện. Tôi không tìm thấy lỗi chặn (blocker) nào. Có 3 điểm nhỏ cần xử lý hoặc ghi chú, nêu ở phần phát hiện.

**Pytest thật** (`.venv\Scripts\python -m pytest -q`, từ gốc repo):
`329 passed, 5 deselected, 5 warnings in 51.64s`
Warning chỉ là DeprecationWarning của SwigPyObject và swigvarlink (thư viện bên thứ ba). Không chạy `-m slow` theo yêu cầu.

**Bảng phép phá** (làm trên bản sao trong thư mục tạm, chạy bằng `.venv` gốc với `PYTHONPATH` trỏ vào bản sao, chạy đủ bộ test cho mỗi phép):
| # | Phép phá | Kết quả | Test đỏ đầu tiên |
|---|---|---|---|
| M1 | `pipeline.py`: trích dẫn lấy `sources[i]` thay vì `sources[n-1]` | ĐỎ (1 failed) | `test_pipeline::test_citations_are_renumbered_to_match_list_positions` |
| M2 | `pipeline.py`: bỏ lớp 1 (ngưỡng rerank) | ĐỎ (5 failed) | `test_main::test_chat_refusal_is_200`, `test_evaluate_answers::test_out_of_scope_refused_...` |
| M3 | `pipeline.py`: lớp 2 bỏ kiểm tra số `[n]` ngoài khoảng | ĐỎ (3 failed) | `test_pipeline::test_out_of_range_citation_refused[Sai [9].]` |
| M4 | `indexer.py`: không xóa point thừa | ĐỎ (4 failed) | `test_indexer::test_shrunk_file_stale_points_deleted`, `test_removed_file_points_deleted` |
| M5 | `evaluate_answers.py`: `sweep_threshold` bỏ guard split dev | ĐỎ (1 failed) | `test_sweep_threshold_rejects_test_split` |
| M6 | `chunker.py`: `page_number` thành `page_number + 1` trong chunk | ĐỎ (12 failed) | `test_chunker::test_page_with_two_articles_gives_separate_chunks` |
| M7 | `evaluate_answers.py`: `cited_ok += 1` luôn tính đúng | ĐỎ (1 failed) | `test_citation_not_matching_gold_is_invalid` |
Cả 7 phép phá đều bị bắt, không có phép nào sống sót.

Ghi chú: lần chạy đầu của tôi bị lỗi vì bản sao thiếu `eval/data`. Tôi đã copy bổ sung rồi chạy lại. Bảng trên là kết quả của lần chạy lại.

**Đầu vào xấu đã thử:**
- PDF giả (file text đặt tên .pdf): in "Bỏ qua bad.pdf: file PDF hỏng hoặc bị khóa". Thông báo dễ hiểu, không crash.
- Đường dẫn không tồn tại: in "Không tìm thấy đường dẫn: ...". Đúng.
- `evaluate_answers --split test --sweep 0.1`: bị từ chối với "--sweep chỉ chạy với --split dev".
- `--sweep abc`: báo "Lỗi tham số...", không có traceback.
- Quét secret trong `src`, `eval`, `tests`, `README.md`, `.env.example` (không đọc `.env`): không thấy API key. Chỉ có chuỗi giả `secret-xyz` trong test.
- Git: `git ls-files` không có PDF hay `.env`. Log chỉ ghi `question=` khi `settings.debug` bật (`src/api/main.py:156`).

**Số liệu README so với `eval/reports/*.json`:** khớp hết những gì kiểm được.
- Chọn `chunk_size` (400/800/1200): Recall@5 hybrid và hybrid+rerank, MRR@5 đều khớp. Recall@20 = 0,923 đúng.
- Bảng ngưỡng 0,01 đến 0,6: khớp. Tôi kiểm lại luận điểm "0,02 là ngưỡng lớn nhất có từ chối nhầm ≤ 10%" (0,0769) và "ngưỡng 0,3 từ chối nhầm 15%" (0,1538): đúng.
- Test retrieval (0,850/0,900/0,917; MRR 0,718/0,768/0,787; Recall@20 0,942/0,950): khớp.
- Test answers (10/10 từ chối đúng, 0,0833 = 5/60, 0,9091 = 50/55, `llm_errors` = 0): khớp.

**Phát hiện (xếp theo mức nghiêm trọng):**
1. TRUNG BÌNH, đã xác nhận: README mục "Viết lại câu hỏi nối tiếp" (khoảng dòng 102: 0,900 / 0,967 / mất 0,067 / 15 hội thoại) không có file báo cáo nào trong `eval/reports/` để đối chiếu. Thư mục chỉ có báo cáo retrieval, threshold và answers, không có `rewrite_*.json`. Hướng khắc phục: chạy lại và lưu báo cáo, hoặc ghi rõ số liệu lấy từ đâu. Tôi không chạy `evaluate_rewrite` vì gọi Gemini.
2. THẤP, đã xác nhận: báo cáo answers ghi `git_commit: e767347` trong khi cây làm việc chưa commit (`chunker.py`, `config.py`, `evaluate_answers.py`, `.env.example` đang sửa dở). Kết quả eval không truy ngược chính xác về một commit. Nên commit trước rồi chạy lại hoặc ghi chú. Chưa xác nhận được phần sửa chưa commit có ảnh hưởng số liệu hay không.
3. THẤP, chưa kiểm chứng đầy đủ: stderr của `evaluate_answers` hiện chữ Việt dạng `ỉ...` khi chạy qua pipe trên Windows, kèm một ký tự `�`. Có thể do console encoding của môi trường chạy của tôi, không chắc lỗi của code. Nếu người dùng thấy tương tự ở terminal thật thì nên đặt `PYTHONIOENCODING=utf-8` hoặc `sys.stderr.reconfigure`.
4. THẤP, đã xác nhận: `sweep_threshold` tính từ chối bằng `top < t`, trùng với pipeline. Còn `top=None` (không có ứng viên) tính là từ chối, cũng khớp pipeline. Không có lỗi, chỉ ghi lại là đã kiểm.

**Chưa kiểm:**
- Chạy CLI chunker trên PDF mẫu thật `data/samples/quy_che_mau.pdf` và PDF scan thật.
- Qdrant không chạy hoặc collection sai cấu trúc (không đụng Qdrant theo yêu cầu).
- Hai hợp đồng `docs/contracts/task-2.md`, `task-4.md` chưa đối chiếu từng mục. Tôi chỉ kiểm bằng phép phá và test.
- UI (`src/ui/app.py`) chỉ xem lướt `page_number`.
- Chunk kéo qua nhiều trang: chỉ gián tiếp qua test chunker, phép phá M6 làm 12 test đỏ nên `page_number` có test bao phủ.
- Phép phá bỏ riêng kiểm tra `not ids` (trả lời không trích dẫn) ở lớp 2 chưa thử.
- Số chunk 467/225/169 trong README không có trong JSON nên không đối chiếu được.

---

## Xử lý (Claude)
1. `evaluate_rewrite` giờ ghi `eval/reports/rewrite_*.json`; chạy lại sau commit `fea5fad`: Recall@5 viết lại 0,967 =
   độc lập 0,967 (lần chạy đầu, chưa lưu file, là 0,900). README ghi cả hai lần chạy.
2. Code đã commit (`fea5fad`) trước khi chạy lại phần rewrite. Ba báo cáo còn lại giữ nguyên, README ghi chú rằng thay
   đổi chưa commit lúc đó (giá trị mặc định, `--delay`) không ảnh hưởng số liệu vì cấu hình truyền qua biến môi trường.
3. Không sửa: các lần chạy thật của Claude đều đặt `PYTHONIOENCODING=utf-8`; ghi là rủi ro còn lại.
