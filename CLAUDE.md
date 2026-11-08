# CLAUDE.md — Quy tắc dự án (nguồn duy nhất)

Chatbot RAG hỏi đáp quy chế đào tạo (Thông tư 08/2021 + quy chế trường), trả lời tiếng Việt kèm trích dẫn
**file / trang / Điều**. Dự án CV làm một mình: người dùng phải giải thích được mọi quyết định khi phỏng vấn,
nên luôn giải thích **vì sao**, không chỉ **cái gì**.

- Yêu cầu và kiến trúc: [`docs/superpowers/specs/2026-11-04-rag-chatbot-design.md`](docs/superpowers/specs/2026-11-04-rag-chatbot-design.md)
- Plan đang thực hiện: [`docs/superpowers/plans/2026-11-04-master-plan.md`](docs/superpowers/plans/2026-11-04-master-plan.md)
- Tài liệu cũ (AGENTS.md bản đầy đủ, plan Tuần 1, cấu hình Antigravity): `docs/archive/`

## 1. Quy tắc làm việc

- Làm theo task của plan; không tự mở rộng phạm vi. Mơ hồ hoặc muốn làm khác plan/spec → hỏi trước.
- Sau mỗi task, báo cáo tiếng Việt đơn giản: thay đổi gì, vì sao, file nào, test đã chạy, **kết quả thật**, rủi ro còn lại.
  Giải thích thuật ngữ ở lần đầu xuất hiện. Không bao giờ bịa kết quả kiểm chứng.
- TDD: test đỏ trước, rồi mới code. Task chưa có test hoặc chưa chạy được thì chưa xong.
- Không thêm thư viện ngoài plan; không đổi cấu trúc thư mục ngoài plan khi chưa hỏi.
- Git: commit sau mỗi task đã verify (kèm dòng `Co-Authored-By`); **không push** khi chưa được đồng ý; không viết lại
  lịch sử; không làm trực tiếp trên `main`. Điểm khôi phục: tag `checkpoint/pre-redesign`.
- Phải hỏi trước khi: xóa dữ liệu/tài liệu của người dùng, xóa collection hoặc volume Qdrant, push, cài đặt làm đổi PyTorch.

## 2. Quy ước kỹ thuật

- Câu từ chối duy nhất (hằng `REFUSAL`, `src/rag/constants.py`): `Không tìm thấy thông tin trong tài liệu quy chế.`
  Chỉ mang nghĩa "tài liệu không có thông tin"; lỗi kỹ thuật dùng thông báo khác.
- Payload mỗi chunk/point: `source_file`, `page_number` (int, 1-based), `article` (str | None), `chunk_index`, `text`.
  Không bước nào được làm mất hay tính lại `page_number`; số trang trong câu trả lời lấy từ metadata, không từ LLM.
- Chunker (bất biến có test): mỗi chunk thuộc đúng 1 trang và tối đa 1 Điều; tiền tố `"Điều X. Tên: "`, tính vào
  `chunk_size`; `chunk_size`/`chunk_overlap` đơn vị **ký tự** (mặc định 800/150, chốt bằng Recall@5);
  ID = `uuid5(NAMESPACE_URL, f"{source_file}:{page_number}:{chunk_index}")`; Unicode NFC; PDF scan/hỏng bị bỏ qua và báo rõ.
- Qdrant: named vectors `dense` (1024, Cosine, BGE-M3) + `sparse` (BM25, `Modifier.IDF`). BM25: `k1=1.2`, `b=0.75`,
  tách từ `pyvi`, token ID = 32 bit đầu `blake2b`. Docker `qdrant/qdrant:v1.12.5` là môi trường chính; local mode chỉ
  cho test (`:memory:`) và dự phòng (`--local`).
- Thành phần nặng (embedder, reranker, LLM) nằm sau Protocol; test mặc định dùng bản giả trong `tests/fakes.py`,
  không tải model, không gọi mạng. Test cần model thật đánh dấu `@pytest.mark.slow`.
- Cấu hình chỉ qua `.env` (mẫu: `.env.example`) và `src/config.py`.

## 3. Môi trường Windows (đã kiểm chứng)

Python **3.11** (venv hiện tại 3.11.9; `python` trên PATH là 3.12 — không dùng để chạy dự án). Luôn gọi
`.venv\Scripts\python`, không dùng `pip` trần.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.lock      # requirements.txt = danh sách gói trực tiếp
.venv\Scripts\python -m pytest                                 # test nhanh (bỏ qua slow)
.venv\Scripts\python -m pytest -m slow                         # cần model đã tải / Docker
docker compose up -d                                           # Qdrant tại 127.0.0.1:6333
.venv\Scripts\python -m src.ingestion.chunker data/pdfs/       # xem chunk
```

Chạy lệnh từ thư mục gốc repo. Cổng: API 8000, Streamlit 8502 (8501 đang bị ứng dụng khác dùng).

Bật/tắt cả hệ thống (Qdrant + API + giao diện; GPU chỉ bị dùng khi API chạy, `stop` trả GPU về 0):
`powershell -ExecutionPolicy Bypass -File chatbot.ps1 start|stop|status` (log trong `.run\`).

## 4. Quy trình Superpowers

- Thực thi plan: `superpowers:executing-plans`; mọi code theo `superpowers:test-driven-development`.
- Lỗi / test đỏ bất ngờ: `superpowers:systematic-debugging` trước khi sửa.
- Trước khi tuyên bố xong: `superpowers:verification-before-completion` (chạy lệnh thật, đọc output).
- Việc mới chưa có spec: `superpowers:brainstorming` → spec → `superpowers:writing-plans`.
- Review độc lập: subagent `critic` (`.claude/agents/critic.md`) cho task có hợp đồng trong `docs/contracts/`
  (chunker, indexer) và review cuối nhánh. Chỉ giao đường dẫn hợp đồng + file nguồn; không gợi ý phép phá hay kết quả.
  Báo cáo lưu `docs/reviews/<task>-r<n>.md` và dán nguyên văn cho người dùng; tối đa 3 vòng sửa.

## 5. Bảo mật

- Không đọc `.env` hay bất kỳ secret nào; chỉ sửa `.env.example` (không chứa giá trị thật).
- Không ghi API key vào code, log, commit; log không ghi toàn văn câu hỏi trừ khi `DEBUG`.
- Không commit PDF (`data/pdfs/*.pdf` bị ignore), dữ liệu Qdrant, cache model.
