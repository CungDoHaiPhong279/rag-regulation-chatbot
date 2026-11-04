# Thiết kế: Chatbot RAG hỏi đáp quy chế đào tạo (có trích nguồn)

> **Trạng thái:** đã duyệt từng phần trong hội thoại ngày 2026-11-04; file này chờ người dùng duyệt bản viết.
> **Thay thế:** phần spec sản phẩm trong `AGENTS.md` và plan `docs/superpowers/plans/2026-11-02-week1-ingestion.md`
> (sẽ được lưu vào `docs/archive/` khi thực hiện plan tổng).
> **Điểm khôi phục:** tag git `checkpoint/pre-redesign` (commit `c4101b1`).

## 1. Mục tiêu và phạm vi

| Mục | Quyết định |
|---|---|
| Mục tiêu thành công | Demo phỏng vấn: chạy được, có số liệu đánh giá thật, code giải thích được từng quyết định |
| Triển khai | Chỉ chạy local trên Windows 11 (RAM 16 GB, RTX 3050 Ti 4 GB, Docker) |
| Người dùng | Một người, không đăng nhập |
| Tài liệu | Thông tư 08/2021/TT-BGDĐT + quy chế đào tạo của trường (người dùng cung cấp PDF có lớp chữ) |
| LLM mặc định | Gemini (Google), tên model đặt trong `.env`, đổi được không cần sửa code |
| Hội thoại | Nhiều lượt; câu hỏi nối tiếp được LLM viết lại thành câu độc lập trước khi truy xuất |
| Thời gian | ~2 tuần, gần full-time |

**Bắt buộc:** trích dẫn đúng file / trang / Điều; hybrid search (dense + BM25) + rerank; chống bịa 2 lớp với
câu từ chối cố định `Không tìm thấy thông tin trong tài liệu quy chế.`; bộ đánh giá tự động; FastAPI; Streamlit.

**Ngoài phạm vi:** triển khai công khai, nhiều người dùng, đăng nhập, OCR cho PDF scan, framework RAG trọn gói
(LangChain / LlamaIndex), tài liệu không phải quy chế đào tạo.

## 2. Bằng chứng kiểm chứng (spike ngày 2026-11-04, code thử nghiệm không giữ lại)

| Câu hỏi | Kết quả đo được |
|---|---|
| Qdrant hỗ trợ sparse + `Modifier.IDF`, `prefetch`, `Fusion.RRF`? | Có, trên server `v1.12.5` (Docker), local path và `:memory:` với `qdrant-client==1.12.1`; điểm số giống hệt nhau |
| Công thức IDF của Qdrant | `ln((N − df + 0.5)/(df + 0.5) + 1)` — khớp BM25 (Robertson/Lucene) |
| Thiết kế sparse có phải BM25 đầy đủ? | Có: điểm Qdrant trên 77 chunk TT08 khớp cài đặt BM25 tham chiếu, sai số tối đa 9.2e-7 |
| Token ID bằng `blake2b` 32 bit | 0 va chạm trên 716 từ vựng TT08 |
| pyvi tách thuật ngữ | `học phí`→`học_phí`, `tín chỉ`→`tín_chỉ`, `cảnh báo học tập`→`cảnh_báo học_tập`; `tín_chỉ` nhất quán 38/39 lần trong văn bản; `buộc thôi học` không gộp nhưng nhất quán hai phía |
| "học phí" trong TT08 | **0 lần xuất hiện** → câu hỏi học phí chỉ trả lời được từ quy chế trường |
| Local mode Qdrant | Khóa file, không cho nhiều tiến trình; trên Windows còn giữ khóa `storage.sqlite` sau khi xóa collection |
| BGE-M3 trên CPU | dim 1024; embed 77 chunk 67 s; 1 câu hỏi 0.27 s; chunk dài 14–246 token (trung vị 155) |
| `bge-reranker-v2-m3` trên CPU (torch 2.15 CPU, 8 luồng) | top-20: **14.2 s** (trung vị); top-10: 7.3 s; điểm đã sigmoid trong [0, 1] |

## 3. Quyết định kiến trúc (ADR rút gọn)

### ADR-1. Truy xuất: Qdrant + BM25 tự cài + RRF + cross-encoder
- **Hiện có:** chỉ có chunker; collection chưa tạo.
- **Phương án:** (A) dense BGE-M3 + BM25 (pyvi) trong Qdrant, RRF, rerank; (B) BGE-M3 dense + learned sparse
  (FlagEmbedding); (C) LangChain/LlamaIndex trọn gói.
- **Chọn A.** Đúng yêu cầu "BM25", mỗi bước giải thích được, cùng code chạy được 3 cấu hình để so sánh.
- **Hệ quả:** tự viết `sparse.py` (~100 dòng) và thêm `pyvi`; khi kho tài liệu đổi phải tính lại sparse cho mọi chunk.

### ADR-2. BM25: phần tf/độ dài ở client, IDF ở Qdrant
- Giá trị sparse của chunk = `tf·(k1+1) / (tf + k1·(1 − b + b·|d|/avgdl))`, `k1=1.2`, `b=0.75` (cấu hình được).
- Truy vấn: mỗi từ duy nhất trọng số 1.0; Qdrant nhân IDF lúc tìm.
- Không được gọi là BM25 nếu chỉ dùng tf thô × IDF — test so với công thức tham chiếu bảo vệ điều này.
- `avgdl` tính trên toàn bộ chunk của lượt index; indexer luôn xử lý cả thư mục.
- Token: NFC → `ViTokenizer.tokenize` → tách khoảng trắng → bỏ token toàn dấu câu → chữ thường.
- Token ID: 32 bit đầu của `blake2b(token, digest_size=4)`; không dùng `hash()` (đổi theo tiến trình).

### ADR-3. Qdrant Docker là môi trường chính
- Local mode khóa file (API + indexer + eval không chạy đồng thời) và có lỗi khóa file trên Windows.
- Local mode chỉ dùng cho unit test (`:memory:`) và dự phòng (`--local`).
- `docker-compose.yml` giữ `qdrant/qdrant:v1.12.5`, đổi cổng sang `127.0.0.1:6333` (không có mật khẩu).

### ADR-4. Reranker: CPU làm baseline
- `BAAI/bge-reranker-v2-m3` qua `sentence_transformers.CrossEncoder`, rerank top-20 → top-5.
- 14 s/truy vấn chấp nhận được cho đánh giá, chậm cho demo. **Quyết định hoãn** đến khi có số liệu Recall:
  (a) cài torch CUDA cho GPU 4 GB, hoặc (b) giảm còn top-10. Không đổi PyTorch trước quyết định đó.

### ADR-5. Số trang đến từ metadata, không từ LLM
- Context đánh số `[1]…[5]`; LLM chỉ trích số; code ánh xạ số → `{source_file, page_number, article}`.
- Trích số không tồn tại hoặc không trích gì → coi là trả lời không bám nguồn → trả câu từ chối.

### ADR-6. Mọi thành phần nặng sau một Protocol
- `Embedder`, `Reranker`, `LLMClient` là Protocol; test dùng bản giả, không tải model, không gọi mạng.

### ADR-7. Quy trình: Claude Code + Superpowers, CLAUDE.md là nguồn quy tắc chính
- Xem mục 9.

## 4. Module

```
src/
  config.py              Settings (dataclass) đọc .env: Qdrant, model, chunk, k1/b, top_k, ngưỡng, timeout
  utils.py               (giữ) setup_utf8_console
  ingestion/
    sample_pdf.py        (giữ) PDF mẫu biết trước Điều ↔ trang
    chunker.py           (giữ, sửa nhỏ) PDF → Chunk; thêm xử lý PDF hỏng/khóa
    sparse.py            tokenize, token_id, bm25_doc_vector(tokens, avgdl), query_vector(text)
    embedder.py          Embedder (Protocol), BgeM3Embedder
    indexer.py           ensure_collection, index_corpus, CLI (--pdf-dir, --recreate, --local, --collection)
  retrieval/
    hybrid_search.py     search(query, mode: dense|sparse|hybrid, limit) -> list[Hit]
    reranker.py          Reranker (Protocol), BgeReranker
  rag/
    constants.py         REFUSAL
    prompts.py           prompt trả lời (chống bịa), prompt viết lại câu hỏi
    llm.py               LLMClient (Protocol), GeminiClient (timeout, retry 1 lần với 429/5xx)
    rewriter.py          rewrite(question, history) -> str
    pipeline.py          answer(question, history) -> Answer
  api/main.py            FastAPI: POST /chat, GET /health; tải model một lần trong lifespan
  ui/app.py              Streamlit (cổng 8502), gọi API qua httpx
eval/
  data/questions.jsonl   70 câu: 50 test + 20 dev (người dùng gán nhãn)
  data/dialogues.jsonl   ~10 hội thoại nhiều lượt
  label_helper.py        in chunk ứng viên để người dùng gán nhãn nhanh (không tự sinh đáp án)
  evaluate_retrieval.py  Dense / Hybrid / Hybrid+Rerank
  evaluate_answers.py    từ chối đúng/nhầm, độ hợp lệ trích dẫn
  evaluate_rewrite.py
  reports/               kết quả có ghi model, chunk_size, commit, ngày
```

Payload mỗi point: `source_file`, `page_number` (int, 1-based), `article` (str | None), `chunk_index`, `text`.
Named vectors: `dense` (1024, Cosine), `sparse` (`SparseVectorParams(modifier=IDF)`). Payload index keyword `source_file`.

## 5. Luồng dữ liệu

**Index** (`python -m src.ingestion.indexer`):
1. Kiểm tra Qdrant (`get_collections`) trước khi tải model.
2. `chunk_directory` cả thư mục → chunks, file bị bỏ qua, trang bị bỏ qua.
3. `avgdl` trên toàn bộ chunk; mỗi chunk → dense (BGE-M3) + sparse (BM25).
4. Embed xong toàn bộ → upsert theo lô (`wait=True`).
5. Chỉ khi upsert thành công: xóa point có ID không thuộc tập mới (scroll phân trang, không lấy vector, chỉ lấy payload `source_file` để giữ point của file bị bỏ qua),
   gồm cả point của file không còn trong thư mục.
6. In báo cáo: số file, số chunk, file/trang bị bỏ qua.

**Trả lời** (`POST /chat {question, history}`):
1. Có `history` → `rewriter` viết lại câu hỏi (lỗi → dùng câu gốc, `rewrite_failed=true`).
2. `hybrid_search`: prefetch dense top-20 + sparse top-20 → RRF → 20 ứng viên.
3. `reranker` → top-5.
4. Lớp 1: điểm rerank cao nhất < `RERANK_THRESHOLD` → `REFUSAL`, không gọi LLM.
5. Lớp 2: LLM với prompt chỉ dùng context, trích `[n]`, trả đúng `REFUSAL` nếu không đủ.
6. Kiểm tra trích dẫn (ADR-5); chuẩn hóa câu từ chối của LLM về đúng chuỗi `REFUSAL`.
7. Trả `{answer, citations[{source_file, page_number, article, text}], rewritten_query, debug}`.

## 6. Xử lý lỗi

`REFUSAL` chỉ mang nghĩa "tài liệu không có thông tin"; lỗi kỹ thuật dùng thông báo khác.

| Tình huống | Hành vi |
|---|---|
| Qdrant chưa bật (index) | Báo hướng dẫn `docker compose up -d` / `--local`, thoát 1, không tải model |
| PDF scan | Bỏ qua, ghi vào báo cáo |
| PDF hỏng / có mật khẩu | Bỏ qua với thông báo `Bỏ qua <file>: file PDF hỏng hoặc bị khóa`, tiếp tục file khác |
| Không còn PDF hợp lệ | Thoát 1, không xóa dữ liệu cũ |
| Collection sai cấu trúc (thiếu vector, lệch dim, sparse thiếu IDF) | Báo rõ, gợi ý `--recreate`, thoát 1 |
| Lỗi embed / upsert | Dừng, không xóa point thừa, thoát 1 |
| Câu hỏi rỗng / > 1000 ký tự | HTTP 422 |
| Qdrant mất kết nối / collection rỗng | HTTP 503, thông báo tiếng Việt |
| Không có ứng viên / dưới ngưỡng | HTTP 200, `REFUSAL` |
| Rewriter lỗi | Dùng câu gốc |
| Gemini lỗi / timeout 30 s / hết quota | Thử lại 1 lần (429/5xx); vẫn lỗi → HTTP 503 kèm `sources` đã truy xuất |
| LLM không trích dẫn hoặc trích sai | HTTP 200, `REFUSAL`, `citation_check_failed` trong debug |
| Thiếu `GEMINI_API_KEY` | API vẫn khởi động; `/chat` 503; `/health` báo trạng thái |

Model tải một lần lúc khởi động API. `/health` báo riêng Qdrant / model / LLM key. Streamlit timeout 60 s.
Log mỗi request: id, độ dài câu hỏi, thời gian từng bước, điểm cao nhất; không ghi key; toàn văn câu hỏi chỉ khi `DEBUG`.

## 7. Kiểm thử

| Tầng | Nội dung | Chạy |
|---|---|---|
| Unit | BM25 so với công thức tham chiếu, bão hòa tf, chuẩn hóa độ dài, token ID cố định, tách thuật ngữ tiếng Việt; hai lớp chống bịa; kiểm tra trích dẫn; chuẩn hóa câu từ chối; chunker (test cũ + PDF hỏng) | `pytest` mặc định |
| Tích hợp | Qdrant `:memory:` (named vectors, payload, hybrid/RRF giữ `page_number`, re-index không trùng, xóa point file bị bỏ, lỗi upsert không xóa); FastAPI `TestClient` (422/503/200) | `pytest` mặc định, < 10 s |
| Model thật | BGE-M3 dim 1024, reranker xếp đúng cặp liên quan, 1 truy vấn end-to-end trên Docker | `pytest -m slow` |

TDD: test đỏ trước khi cài đặt. Test "không mất `page_number`" có ở chunk, upsert, search, rerank, response.

## 8. Đánh giá

**Dữ liệu** `eval/data/questions.jsonl`, mỗi dòng:
```json
{"id": "q07", "question": "...", "in_scope": true, "split": "test",
 "gold": [{"source_file": "thong_tu_08_2021.pdf", "article": "Điều 11", "pages": [13, 14]}]}
```
- **70 câu:** 50 `test` (chỉ để báo cáo) + 20 `dev` (chỉnh ngưỡng, chọn `chunk_size`). Mỗi split có câu ngoài phạm vi
  (~10 trong test).
- **Chunk liên quan** ⇔ cùng `source_file` ∧ cùng số Điều ∧ `page_number ∈ pages`.
- Câu `in_scope=false` không tính Recall; chỉ đo khả năng từ chối.
- Người dùng gán nhãn; `label_helper.py` chỉ hiển thị ứng viên.

**Chỉ số:** Recall@20 (ứng viên), Recall@5 (kết quả cuối, chỉ số chính), MRR@5, tỉ lệ từ chối đúng (câu ngoài
phạm vi), tỉ lệ từ chối nhầm (câu trong phạm vi), độ hợp lệ trích dẫn. Không dùng nDCG vì nhãn nhị phân.

**Cấu hình so sánh:** Dense-only (top-5 của dense), Hybrid (top-5 sau RRF), Hybrid + Rerank (top-5 sau rerank 20
ứng viên RRF). Recall@20 đo trên 20 ứng viên của dense và của RRF. Chỉ số từ chối và trích dẫn chỉ đo trên pipeline
đầy đủ (Hybrid + Rerank + LLM). `chunk_size` 400 / 800 / 1200 dùng collection riêng
(`regulations_cs400`…). Báo cáo ghi tên model embedding / reranker / LLM, `chunk_size`, commit, ngày.

**Viết lại câu hỏi** (`dialogues.jsonl`): Recall@5 với câu do LLM viết lại so với câu độc lập người dùng viết tay;
chênh lệch = mất mát do bước viết lại, tách khỏi lỗi truy xuất.

## 9. Môi trường, thư viện và quy trình

**Thư viện thêm:** `pyvi==0.1.1` (+ `python-crfsuite`, `sklearn-crfsuite`, `tabulate`; không đổi numpy/scikit-learn/torch),
`google-genai`, `fastapi`, `uvicorn`, `streamlit`, `httpx`. Phiên bản ghim `==` lúc cài.
Thêm `requirements.lock` (`pip freeze`) để ghim cả gói phụ (torch, numpy hiện chưa được ghim).
Không thêm: LangChain/LlamaIndex, FlagEmbedding, rank-bm25, pydantic-settings.

**Python:** chuẩn hóa 3.11 (venv hiện tại 3.11.9, tạo bằng `py -3.11 -m venv .venv`); sửa tài liệu đang ghi 3.12.

**Chỉ dẫn agent:**

| Thành phần | Hành động |
|---|---|
| Spec sản phẩm | File này |
| `CLAUDE.md` | Nguồn quy tắc chính; bỏ import `@AGENTS.md` |
| `AGENTS.md` | Bản đầy đủ → `docs/archive/`; file gốc thành vài dòng trỏ tới `CLAUDE.md` và spec |
| `.agent/agents/` | → `docs/archive/antigravity/` |
| Critic | `.claude/agents/critic.md` (Read, Grep, Glob, Bash): đánh giá theo hợp đồng, chạy test thật, thử phá code trên bản sao, `APPROVED`/`REJECTED`; báo cáo lưu `docs/reviews/` |
| Builder | Bỏ (agent chính làm) |
| Plan Tuần 1 | → `docs/archive/`, thay bằng plan tổng |
| Hợp đồng | `task-2.md` giữ làm lịch sử; `task-4.md` cập nhật theo ADR-2, mục 5 |

**Git:** tag `checkpoint/pre-redesign` đã tạo. Agent tự commit sau mỗi task đã qua test và verification (một commit
mỗi task); không push khi chưa được đồng ý; không viết lại lịch sử. Thêm `.gitattributes` thống nhất xuống dòng.

**Superpowers:** writing-plans → executing-plans (từng task, dừng giải thích tiếng Việt), test-driven-development,
systematic-debugging, verification-before-completion, requesting-code-review + subagent critic cho task có hợp đồng.

## 10. Rủi ro và quyết định còn mở

| Mục | Xử lý |
|---|---|
| Reranker CPU 14 s | Quyết định CUDA hay top-10 sau khi có Recall (ADR-4) |
| Tên model Gemini | Chọn và kiểm tra khi người dùng có API key; ghi trong `.env` và báo cáo |
| PDF quy chế trường | Người dùng cung cấp; phải có lớp chữ |
| Trang 1–2 TT08 là ảnh scan | Phần thân Thông tư không được index; nếu tìm được bản có lớp chữ sẽ xuất hiện hai "Điều 1" — payload phân biệt bằng `source_file`, cần kiểm tra lại khi đổi tài liệu |
| Tách từ pyvi phụ thuộc ngữ cảnh | Theo dõi qua Recall của cấu hình sparse; nếu thấp, thử thêm token âm tiết (đo trên dev) |
| Ngưỡng `RERANK_THRESHOLD` | Chỉnh trên 20 câu dev, báo cáo trên 50 câu test |
| Cổng 8501 đang bị ứng dụng khác dùng | Streamlit chạy cổng 8502 |
