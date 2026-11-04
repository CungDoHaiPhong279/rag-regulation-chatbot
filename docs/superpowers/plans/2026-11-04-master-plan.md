# Chatbot RAG quy chế đào tạo — Plan tổng

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hoàn thiện chatbot RAG chạy local trả lời quy chế đào tạo bằng tiếng Việt, trích dẫn đúng file/trang/Điều, kèm bộ đánh giá có số liệu thật.

**Architecture:** Chunker có sẵn → BGE-M3 (dense) + BM25 tự cài với pyvi (sparse, IDF do Qdrant tính) → Qdrant Docker (named vectors, prefetch + RRF) → cross-encoder rerank → Gemini với hai lớp chống bịa → FastAPI → Streamlit. Thành phần nặng nằm sau Protocol để test chạy bằng bản giả.

**Tech Stack:** Python 3.11, PyMuPDF, langchain-text-splitters, sentence-transformers (BGE-M3, bge-reranker-v2-m3), pyvi, qdrant-client 1.12.1 / Qdrant 1.12.5, google-genai, FastAPI, uvicorn, Streamlit, httpx, pytest.

**Spec:** [`docs/superpowers/specs/2026-11-04-rag-chatbot-design.md`](../specs/2026-11-04-rag-chatbot-design.md)

## Global Constraints

- Python **3.11** trong `.venv`; mọi lệnh dùng `.venv\Scripts\python` và `.venv\Scripts\python -m pip`.
- Câu từ chối duy nhất, hằng số `REFUSAL` trong `src/rag/constants.py`: `Không tìm thấy thông tin trong tài liệu quy chế.`
- `REFUSAL` chỉ nghĩa "tài liệu không có thông tin"; lỗi kỹ thuật dùng thông báo khác.
- Payload mỗi point đúng 5 trường: `source_file` (str), `page_number` (int, 1-based), `article` (str | None), `chunk_index` (int), `text` (str).
- Named vectors: `dense` (1024, Cosine) và `sparse` (`SparseVectorParams(modifier=Modifier.IDF)`).
- BM25: `k1=1.2`, `b=0.75`; token ID = 32 bit đầu `blake2b(token, digest_size=4)`; không dùng `hash()`.
- `chunk_size`/`chunk_overlap` tính bằng **ký tự**, gồm tiền tố Điều; mặc định 800/150.
- Qdrant `qdrant/qdrant:v1.12.5`, `qdrant-client==1.12.1`; Docker là môi trường chính; cổng chỉ mở `127.0.0.1`.
- Không đổi PyTorch (hiện `2.15.0+cpu`) trước quyết định ở Task 15.
- Thư viện chỉ thêm: `pyvi==0.1.1`, `google-genai`, `fastapi`, `uvicorn`, `streamlit`, `httpx` — ghim `==` đúng bản pip cài, ghi vào `requirements.txt`, rồi tạo lại `requirements.lock`.
- Streamlit cổng **8502**; API `127.0.0.1:8000`.
- Không ghi API key vào code, log, commit; `.env` không được đọc bởi agent ngoài việc code nạp qua `python-dotenv`.
- Test mặc định không tải model, không gọi mạng; test cần model thật đánh dấu `@pytest.mark.slow`.
- Sau mỗi task: một commit (kèm dòng `Co-Authored-By`), không push; báo cáo tiếng Việt: thay đổi gì, vì sao, file, test đã chạy, kết quả thật, rủi ro còn lại.

## Review Focus

1. Câu hỏi chỉ gồm dấu câu/khoảng trắng (`"???"`, `"   "`) → API trả 422, không phải 500; tìm kiếm với câu có 0 token BM25 không được gọi Qdrant bằng sparse vector rỗng. (Task 7, Task 11)
2. PDF bị đổi tên hoặc xóa khỏi `data/pdfs/` giữa hai lần index → point cũ của tên cũ bị xóa; file còn trong thư mục nhưng lần này bị bỏ qua (hỏng) → point cũ giữ nguyên. (Task 6)
3. LLM trả câu từ chối gần đúng (thiếu dấu chấm, có ngoặc kép, có khoảng trắng/xuống dòng) → chuẩn hóa về đúng `REFUSAL`; câu trả lời chứa `REFUSAL` lẫn nội dung khác không bị coi là từ chối. (Task 9)
4. Lịch sử hội thoại rất dài (50 lượt, mỗi lượt 5000 ký tự) → chỉ 3 lượt cuối, mỗi lượt cắt 500 ký tự, đưa vào prompt viết lại. (Task 10)
5. Câu hỏi không dấu (`"hoc phi bao nhieu"`) → hệ thống không lỗi; kết quả truy xuất được ghi nhận trong báo cáo đánh giá (không có ngưỡng bắt buộc). (Task 7 test không-lỗi; Task 13 thêm 2 câu không dấu vào bộ dev)

---

## Bản đồ file

| File | Trách nhiệm | Task |
|---|---|---|
| `CLAUDE.md`, `AGENTS.md`, `.claude/agents/critic.md`, `.gitattributes`, `docs/archive/**`, `docs/reviews/` | Quy trình, lưu trữ tài liệu cũ | 1 |
| `requirements.txt`, `requirements.lock`, `pytest.ini`, `docker-compose.yml` | Môi trường tái lập | 1 |
| `src/config.py`, `.env.example` | Cấu hình tập trung | 2 |
| `src/ingestion/chunker.py` | (sửa) PDF hỏng/khóa | 3 |
| `src/ingestion/sparse.py` | Tách từ + BM25 | 4 |
| `src/ingestion/embedder.py`, `tests/fakes.py` | Dense embedding + bản giả dùng chung | 5 |
| `src/ingestion/indexer.py`, `docs/contracts/task-4.md` | Collection, index toàn kho, CLI | 6 |
| `src/retrieval/hybrid_search.py` | Dense / sparse / hybrid RRF | 7 |
| `src/retrieval/reranker.py` | Cross-encoder | 8 |
| `src/rag/constants.py`, `src/rag/citations.py`, `src/rag/prompts.py`, `src/rag/llm.py` | Câu từ chối, trích dẫn, prompt, Gemini | 9 |
| `src/rag/rewriter.py`, `src/rag/pipeline.py` | Viết lại câu hỏi, điều phối hai lớp chống bịa | 10 |
| `src/api/main.py` | FastAPI | 11 |
| `src/ui/app.py` | Streamlit | 12 |
| `eval/dataset.py`, `eval/metrics.py`, `eval/label_helper.py`, `eval/evaluate_retrieval.py` | Bộ dữ liệu, chỉ số, gán nhãn, đánh giá truy xuất | 13 |
| `eval/evaluate_answers.py`, `eval/evaluate_rewrite.py` | Đánh giá từ chối, trích dẫn, viết lại | 14 |
| `README.md`, `eval/reports/*` | Chạy thật, số liệu, tài liệu | 15 |

Mỗi package mới (`src/retrieval`, `src/rag`, `src/api`, `src/ui`, `eval`, `tests/retrieval`, `tests/rag`, `tests/api`, `tests/eval`) có `__init__.py` rỗng, tạo trong task đầu tiên đụng tới nó.

---

### Task 1: Chuẩn hóa quy trình và môi trường

**Files:**
- Move (`git mv`): `AGENTS.md` → `docs/archive/AGENTS-2026-11-04.md`; `.agent/agents/builder.md`, `.agent/agents/critic.md` → `docs/archive/antigravity/`; `docs/superpowers/plans/2026-11-02-week1-ingestion.md` → `docs/archive/`
- Create: `AGENTS.md` (stub), `.claude/agents/critic.md`, `docs/reviews/.gitkeep`, `.gitattributes`, `requirements.lock`
- Modify: `CLAUDE.md`, `pytest.ini`, `docker-compose.yml`, `requirements.txt`

**Interfaces:**
- Produces: marker pytest `slow`; subagent `critic` dùng cho Task 3, 6 và review cuối; quy ước lưu báo cáo `docs/reviews/<task>-<round>.md`.

- [ ] **Step 1: Ghi nhận baseline**
Run: `.venv\Scripts\python -m pytest -q`
Expected: `56 passed`.

- [ ] **Step 2: Lưu trữ tài liệu cũ bằng `git mv`** (giữ lịch sử, không xóa nội dung). Xóa thư mục `.agent/` khi đã rỗng.

- [ ] **Step 3: Viết lại `CLAUDE.md`** — bỏ dòng `@AGENTS.md`; nguồn quy tắc duy nhất. Các mục bắt buộc:
  1. Mục đích + link spec và plan này.
  2. Quy tắc làm việc (lấy từ `docs/archive/AGENTS-2026-11-04.md` mục 3, sửa): làm theo task của plan; báo cáo tiếng Việt sau mỗi task (thay đổi, lý do, file, test, kết quả thật, rủi ro), giải thích thuật ngữ lần đầu; hỏi khi mơ hồ; không thêm thư viện ngoài plan; commit sau mỗi task đã verify, không push, không viết lại lịch sử; cần hỏi trước khi xóa dữ liệu người dùng, xóa collection/volume, push.
  3. Quy ước kỹ thuật (Global Constraints ở trên, rút gọn) + bất biến chunker (giữ từ `CLAUDE.md` hiện tại mục 3–4).
  4. Lệnh Windows đã kiểm chứng: tạo venv `py -3.11 -m venv .venv`, cài `-r requirements.lock`, `pytest`, `pytest -m slow`, `docker compose up -d`, indexer, API, UI.
  5. Superpowers: executing-plans/subagent-driven-development, TDD, systematic-debugging, verification-before-completion, requesting-code-review; khi nào gọi subagent `critic` (task có hợp đồng trong `docs/contracts/`), báo cáo critic lưu `docs/reviews/` và dán nguyên văn cho người dùng.
  6. Bảo mật: không đọc `.env`, không log key, không commit PDF.
  Giới hạn ~120 dòng.

- [ ] **Step 4: `AGENTS.md` mới** — ≤ 10 dòng: dự án dùng Claude Code; quy tắc ở `CLAUDE.md`; yêu cầu sản phẩm ở spec; bản cũ ở `docs/archive/AGENTS-2026-11-04.md`.

- [ ] **Step 5: `.claude/agents/critic.md`** — frontmatter `name: critic`, `description` (review độc lập theo hợp đồng nghiệm thu), `tools: Read, Grep, Glob, Bash`. Nội dung: đầu vào chỉ là đường dẫn hợp đồng + file nguồn; mặc định nghi ngờ; bắt buộc chạy `pytest` và CLI liên quan, dán output thật; tự chọn phép phá code (mutation) trên **bản sao trong thư mục tạm**, không sửa file thật; checklist Critic (lấy nguyên từ `AGENTS.md` cũ mục 4); kết luận `APPROVED` (bằng chứng cho từng mục hợp đồng) hoặc `REJECTED` (lý do, file:dòng, cách tái hiện); tối đa 3 vòng.

- [ ] **Step 6: Môi trường**
  - `pytest.ini`: thêm `markers = slow: cần model thật hoặc Docker` và `addopts = -m "not slow"`.
  - `docker-compose.yml`: ports `"127.0.0.1:6333:6333"`, `"127.0.0.1:6334:6334"`.
  - `.gitattributes`: `* text=auto eol=lf`, `*.bat text eol=crlf`, `*.ps1 text eol=crlf`, `*.pdf binary`, `*.ttf binary`.
  - `.venv\Scripts\python -m pip install pyvi==0.1.1`; thêm dòng `pyvi==0.1.1` vào `requirements.txt`.
  - `.venv\Scripts\python -m pip freeze > requirements.lock`.

- [ ] **Step 7: Verify**
Run: `.venv\Scripts\python -m pytest -q` → `56 passed`.
Run: `git status --short` → chỉ các file của task này; `git diff --stat --find-renames HEAD` hiện các rename `R`.
Run: `git check-ignore -v .env` → `.gitignore:6:.env`.

- [ ] **Step 8: Commit** `chore: consolidate project rules into CLAUDE.md, archive Antigravity config, pin environment`

---

### Task 2: Cấu hình tập trung

**Files:**
- Create: `src/config.py`, `tests/test_config.py`
- Modify: `.env.example`

**Interfaces:**
- Produces:
```python
@dataclass(frozen=True)
class Settings:
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_path: str | None = None
    collection: str = "regulations"
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    gemini_api_key: str | None = None
    gemini_model: str | None = None
    chunk_size: int = 800
    chunk_overlap: int = 150
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    candidate_k: int = 20
    final_k: int = 5
    rerank_threshold: float = 0.3      # giá trị khởi đầu, chỉnh ở Task 15 trên tập dev
    llm_timeout_s: float = 30.0
    debug: bool = False

def load_settings(env_file: str | Path | None = ".env") -> Settings
```
Biến môi trường: `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_PATH`, `QDRANT_COLLECTION`, `EMBEDDING_MODEL`, `RERANKER_MODEL`, `GEMINI_API_KEY`, `GEMINI_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `BM25_K1`, `BM25_B`, `CANDIDATE_K`, `FINAL_K`, `RERANK_THRESHOLD`, `LLM_TIMEOUT_S`, `DEBUG`.

- [ ] **Step 1: Test đỏ** — `tests/test_config.py`:
  - `test_defaults_when_env_empty(monkeypatch)`: xóa mọi biến trên, `load_settings(env_file=None)` == `Settings()`.
  - `test_reads_and_casts(monkeypatch)`: `CHUNK_SIZE="400"`, `RERANK_THRESHOLD="0.55"`, `DEBUG="true"` → `400`, `0.55`, `True`.
  - `test_empty_string_is_none(monkeypatch)`: `QDRANT_PATH=""`, `GEMINI_API_KEY=""` → `None`.
  - `test_invalid_number_raises_with_name(monkeypatch)`: `CHUNK_SIZE="abc"` → `ValueError` có chữ `CHUNK_SIZE`.
  - `test_repr_hides_api_key()`: `"secret-xyz" not in repr(Settings(gemini_api_key="secret-xyz"))`.
- [ ] **Step 2:** `.venv\Scripts\python -m pytest tests/test_config.py -v` → FAIL (`ModuleNotFoundError`).
- [ ] **Step 3: Cài đặt** `load_settings` trong `src/config.py`: `load_dotenv(env_file, override=False)` khi `env_file` không None; chuỗi rỗng → `None`; `DEBUG` nhận `1/true/yes` (không phân biệt hoa thường). `gemini_api_key` khai báo `field(default=None, repr=False)`. Cập nhật `.env.example` đủ mọi biến, `GEMINI_API_KEY=` để trống, chú thích tiếng Việt.
- [ ] **Step 4:** chạy lại → PASS; `pytest -q` → 61 passed.
- [ ] **Step 5: Commit** `feat(config): centralize settings loaded from .env`

---

### Task 3: Chunker xử lý PDF hỏng / bị khóa + kiểm tra kho thật (Critic review)

**Files:**
- Modify: `src/ingestion/chunker.py` (`chunk_pdf` :516-541, `chunk_directory` :544-579), `tests/ingestion/test_chunker.py`

**Interfaces:**
- Produces: `class PdfUnreadableError(Exception)`; `chunk_pdf` ném `PdfUnreadableError(f"{name}: file PDF hỏng hoặc bị khóa")` khi `fitz.open` lỗi hoặc `doc.needs_pass`; `chunk_directory` bắt cả `PdfNoTextError` và `PdfUnreadableError`, thêm tên file vào `skipped_files`, in stderr `Bỏ qua <tên file>: file PDF hỏng hoặc bị khóa`.

- [ ] **Step 1: Test đỏ**
  - `test_corrupt_pdf_raises_unreadable(tmp_path)`: ghi `b"not a pdf"` vào `bad.pdf` → `pytest.raises(PdfUnreadableError, match="bad.pdf")`.
  - `test_encrypted_pdf_raises_unreadable(tmp_path)`: tạo PDF 1 trang có chữ, `doc.save(p, encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="x", owner_pw="y")` → `PdfUnreadableError`.
  - `test_directory_skips_corrupt_and_continues(tmp_path, capsys)`: `ok.pdf` (sample) + `bad.pdf` → `skipped == ["bad.pdf"]`, mọi chunk `source_file == "ok.pdf"`, stderr chứa `Bỏ qua bad.pdf: file PDF hỏng hoặc bị khóa`.
  - Sửa `_pdf_with_blank_and_text_pages` (test_chunker.py:53) dùng `FONT_PATH` từ `src.ingestion.sample_pdf` thay đường dẫn tương đối.
- [ ] **Step 2:** `pytest tests/ingestion/test_chunker.py -v` → 3 FAIL.
- [ ] **Step 3: Cài đặt** theo Interfaces; CLI `main` với một file hỏng → stderr thông báo, `return 1`.
- [ ] **Step 4:** `pytest -q` → toàn bộ PASS.
- [ ] **Step 5: Kiểm tra kho thật** — `.venv\Scripts\python -m src.ingestion.chunker data/pdfs/` ; nếu người dùng đã đặt PDF quy chế trường vào `data/pdfs/` thì cùng xem: (1) file/trang scan bị báo; (2) ký tự lạ `µ ¸ ®`; (3) danh sách Điều đối chiếu mục lục; (4) header/footer còn sót. Lỗi phát hiện → hỏi người dùng trước khi sửa; mỗi lỗi sửa kèm test dùng chuỗi trích từ PDF thật. Chưa có PDF trường → ghi vào báo cáo, làm lại bước này ở Task 15.
- [ ] **Step 6: Critic** — gọi subagent `critic` với `docs/contracts/task-2.md` + `src/ingestion/chunker.py` + `tests/ingestion/test_chunker.py`; lưu báo cáo `docs/reviews/task-3-chunker-r1.md`; `REJECTED` → sửa, tối đa 3 vòng.
- [ ] **Step 7: Commit** `fix(chunker): skip corrupt or encrypted PDFs with clear message`

---

### Task 4: BM25 sparse với pyvi

**Files:**
- Create: `src/ingestion/sparse.py`, `tests/ingestion/test_sparse.py`

**Interfaces:**
- Produces:
```python
@dataclass(frozen=True)
class SparseVec:
    indices: list[int]
    values: list[float]

def tokenize(text: str) -> list[str]
def token_id(token: str) -> int
def average_doc_length(token_lists: list[list[str]]) -> float
def bm25_doc_vector(tokens: list[str], avgdl: float, k1: float = 1.2, b: float = 0.75) -> SparseVec
def query_vector(text: str) -> SparseVec          # mỗi token duy nhất giá trị 1.0; rỗng nếu 0 token
```

- [ ] **Step 1: Test đỏ** — `tests/ingestion/test_sparse.py`:
```python
@pytest.mark.parametrize("text,expected", [
    ("Học phí", ["học_phí"]),
    ("Số tín chỉ tối thiểu", ["số", "tín_chỉ", "tối_thiểu"]),
    ("Sinh viên bị cảnh báo học tập khi nào?", ["sinh_viên", "bị", "cảnh_báo", "học_tập", "khi", "nào"]),
    ("???  ", []),
])
def test_tokenize_vietnamese(text, expected): assert tokenize(text) == expected

def test_tokenize_normalizes_nfd():
    assert tokenize(unicodedata.normalize("NFD", "Học phí")) == ["học_phí"]

def test_token_id_is_stable():
    assert token_id("học_phí") == 3962211330 and token_id("tín_chỉ") == 1233938498
    assert token_id("cảnh_báo") == 3775047272

def test_bm25_tf_saturation_and_length_norm():
    v = lambda toks, avg: dict(zip(bm25_doc_vector(toks, avg).indices, bm25_doc_vector(toks, avg).values))
    a = token_id("a")
    assert v(["a"] + ["x"] * 9, 10.0)[a] == pytest.approx(1.0)            # tf=1, dl=avgdl
    assert v(["a", "a"] + ["x"] * 8, 10.0)[a] == pytest.approx(1.375)     # tf=2
    assert v(["a"] + ["x"] * 19, 10.0)[a] == pytest.approx(0.7096774)     # dl = 2·avgdl
    assert v(["a"] * 10, 10.0)[a] == pytest.approx(1.9642857)             # bão hòa < k1+1

def test_not_raw_tf():   # chống thoái hóa về tf thô
    vals = dict(zip(*astuple(bm25_doc_vector(["a"] * 10, 10.0))))
    assert vals[token_id("a")] < 2.2

def test_query_vector_unique_terms_weight_one():
    q = query_vector("học phí học phí")
    assert q.indices == [token_id("học_phí")] and q.values == [1.0]

def test_query_vector_empty_for_punctuation(): assert query_vector("???") == SparseVec([], [])

def test_doc_vector_has_unique_indices():
    v = bm25_doc_vector(tokenize("tín chỉ tín chỉ học phí"), 5.0)
    assert len(v.indices) == len(set(v.indices))

def test_average_doc_length(): assert average_doc_length([["a"], ["a", "b", "c"]]) == 2.0
def test_average_doc_length_empty_raises():
    with pytest.raises(ValueError): average_doc_length([])
```
- [ ] **Step 2:** `pytest tests/ingestion/test_sparse.py -v` → FAIL.
- [ ] **Step 3: Cài đặt** — `tokenize`: `unicodedata.normalize("NFC")` → `ViTokenizer.tokenize` → `split()` → bỏ token khớp `^[\W_]+$` → `.lower()`. `token_id`: `int.from_bytes(blake2b(tok.encode("utf-8"), digest_size=4).digest(), "big")`. `bm25_doc_vector`: `tf·(k1+1)/(tf + k1·(1 − b + b·dl/avgdl))`; nếu hai token khác nhau trùng ID thì **cộng** giá trị (Qdrant cấm index trùng); indices sắp tăng dần. Docstring giải thích: IDF do Qdrant `Modifier.IDF` nhân lúc truy vấn nên đây là BM25 đầy đủ, không phải TF-IDF.
- [ ] **Step 4:** PASS; `pytest -q` toàn bộ PASS.
- [ ] **Step 5: Commit** `feat(sparse): BM25 document/query vectors with pyvi word segmentation`

---

### Task 5: Embedder và bản giả dùng chung

**Files:**
- Create: `src/ingestion/embedder.py`, `tests/fakes.py`, `tests/ingestion/test_embedder.py`

**Interfaces:**
- Produces:
```python
class Embedder(Protocol):
    dense_dim: int
    def embed_dense(self, texts: list[str]) -> list[list[float]]: ...   # len == len(texts), chuẩn hóa L2

class BgeM3Embedder:
    dense_dim = 1024
    def __init__(self, model_name: str = "BAAI/bge-m3", batch_size: int = 16, device: str = "cpu") -> None
    # import sentence_transformers BÊN TRONG __init__
    def embed_dense(self, texts: list[str]) -> list[list[float]]

# tests/fakes.py
class FakeEmbedder:   dense_dim = 8; calls: int     # vector tất định từ sha256(text), chuẩn hóa L2
class FailingEmbedder: dense_dim = 8                 # embed_dense luôn raise RuntimeError("embed failed")
```
(`tests/fakes.py` sẽ được bổ sung `FakeReranker` ở Task 8 và `FakeLLM` ở Task 9.)

- [ ] **Step 1: Test đỏ**
  - `test_fake_embedder_deterministic_and_normalized`: cùng text → cùng vector, `abs(norm - 1) < 1e-6`, len 8.
  - `test_importing_embedder_does_not_load_torch`: chạy subprocess `.venv\Scripts\python -c "import src.ingestion.embedder, sys; print('torch' in sys.modules)"` → in `False`.
  - `@pytest.mark.slow test_bge_m3_real_dimension`: `BgeM3Embedder().embed_dense(["Học phí"])` → 1 vector dài 1024, norm ≈ 1.
- [ ] **Step 2:** FAIL.
- [ ] **Step 3: Cài đặt** theo Interfaces (`SentenceTransformer(model_name, device=device)`, `encode(..., batch_size, normalize_embeddings=True)`, `.tolist()`; kiểm tra `len` kết quả, sai → `RuntimeError`).
- [ ] **Step 4:** `pytest -q` PASS; `pytest -m slow tests/ingestion/test_embedder.py -v` PASS (model đã cache).
- [ ] **Step 5: Commit** `feat(embedder): BGE-M3 dense embedder behind a Protocol`

---

### Task 6: Indexer — collection, index toàn kho, CLI (Critic review)

**Files:**
- Create: `tests/ingestion/test_indexer.py`
- Modify: `src/ingestion/indexer.py` (đang 0 byte), `docs/contracts/task-4.md`

**Interfaces:**
- Consumes: `Chunk`, `chunk_directory` (chunker), `tokenize`, `average_doc_length`, `bm25_doc_vector` (Task 4), `Embedder` (Task 5), `Settings`, `load_settings` (Task 2).
- Produces:
```python
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

class QdrantUnavailableError(Exception): ...
class CollectionSchemaError(Exception): ...

@dataclass(frozen=True)
class IndexReport:
    upserted: int
    deleted_stale: int
    files: list[str]

def get_client(settings: Settings) -> QdrantClient            # qdrant_path có giá trị -> local mode
def check_connection(client: QdrantClient, where: str) -> None  # lỗi -> QdrantUnavailableError
def ensure_collection(client: QdrantClient, name: str, dense_dim: int, recreate: bool = False) -> None
def index_corpus(client: QdrantClient, chunks: list[Chunk], embedder: Embedder, collection: str, *,
                 preserve_files: frozenset[str] = frozenset(), k1: float = 1.2, b: float = 0.75,
                 upsert_batch_size: int = 64) -> IndexReport
def main(argv: list[str] | None = None) -> int   # --pdf-dir data/pdfs, --recreate, --local, --collection
```

- [ ] **Step 1: Cập nhật hợp đồng** `docs/contracts/task-4.md`: thêm mục "Bổ sung theo spec 2026-11-04": sparse BM25 với `Modifier.IDF`; index **toàn kho** một lượt (avgdl toàn kho); xóa point thừa trên **cả collection** trừ point thuộc `preserve_files`; `CollectionSchemaError` khi sparse thiếu IDF; đổi tên test cho khớp danh sách Step 2. Giữ nguyên các mục khác.
- [ ] **Step 2: Test đỏ** — fixture `client = QdrantClient(":memory:")` + `ensure_collection(client, "t", 8)`; helper `mk(source, n)` tạo `Chunk` trang 1..n.
  - `test_collection_has_dense_cosine_and_sparse_idf`: `params.vectors["dense"].distance == Distance.COSINE`, `params.sparse_vectors["sparse"].modifier == Modifier.IDF`.
  - `test_payload_exact_schema_and_int_page`: tập khóa payload == 5 trường; `page_number` là `int`; giá trị khớp chunk gốc (không mất `page_number`).
  - `test_point_has_dense_and_sparse_vectors`: `retrieve(..., with_vectors=True)` có cả hai tên vector.
  - `test_reindex_same_data_no_duplicates`: index 2 lần → count == 4.
  - `test_shrunk_file_stale_points_deleted`: `a.pdf` 5 chunk rồi 3 chunk → count `a.pdf` == 3, `report.deleted_stale == 2`.
  - `test_removed_file_points_deleted`: index `a`(2)+`b`(2), rồi chỉ `a` → `b` còn 0.
  - `test_preserved_file_points_kept`: như trên nhưng `preserve_files=frozenset({"b.pdf"})` → `b` còn 2.
  - `test_embed_failure_keeps_old_data`: `FailingEmbedder` → `RuntimeError`, count không đổi.
  - `test_upsert_failure_skips_deletion(monkeypatch)`: `client.upsert` ném lỗi → không gọi `client.delete`, count không đổi.
  - `test_stale_cleanup_paginates(monkeypatch)`: 150 point cũ, scroll `limit=50` (hằng `SCROLL_PAGE = 50` monkeypatch) → còn đúng số mới.
  - `test_dimension_mismatch_raises`: tạo collection dim 4, `ensure_collection(...,8)` → `CollectionSchemaError` có `dim=4` và `--recreate`.
  - `test_unnamed_or_missing_sparse_raises`: collection vector không tên, hoặc sparse không IDF → `CollectionSchemaError`.
  - `test_recreate_resets_dimension`: `recreate=True` → dim mới.
  - `test_empty_chunks_raises_without_touching_data`: `index_corpus(client, [], ...)` → `ValueError`, count không đổi.
  - `test_local_mode_persists(tmp_path)`: settings `qdrant_path=tmp_path/"q"` → ghi 2 point, đóng client, mở lại → 2.
  - `test_cli_qdrant_down_fails_fast(monkeypatch, capsys, tmp_path)`: `QDRANT_URL=http://127.0.0.1:1`, `QDRANT_PATH=""`; patch `BgeM3Embedder` ghi nhận khởi tạo → `main([...]) == 1`, stderr chứa `Không kết nối được Qdrant` và `--local`, embedder chưa khởi tạo.
  - `test_cli_end_to_end_local(monkeypatch, tmp_path, capsys)`: sample PDF + patch `BgeM3Embedder` → `FakeEmbedder`, `--local` trỏ `tmp_path` → `0`, stdout `Đã index`.
- [ ] **Step 3:** FAIL.
- [ ] **Step 4: Cài đặt**
  - `ensure_collection`: chưa có → tạo theo Global Constraints + `create_payload_index("source_file", KEYWORD)`; đã có → kiểm tra `vectors` là dict có `dense` đúng dim, `sparse_vectors["sparse"].modifier == IDF`, sai → `CollectionSchemaError` (nêu dim hiện tại/mới, gợi ý `--recreate`).
  - `index_corpus`: rỗng → `ValueError`; token hóa mọi chunk → `avgdl` → embed toàn bộ (lỗi bay ra trước khi ghi) → upsert lô `wait=True` với `PointStruct(id=chunk_id, vector={"dense": d, "sparse": SparseVector(...)}, payload=...)` → scroll phân trang (`with_vectors=False`, `with_payload=["source_file"]`, `limit=SCROLL_PAGE`) lấy ID không thuộc tập mới và `source_file ∉ preserve_files` → `delete(points=..., wait=True)`; so sánh `str(id)`.
  - `main`: `load_settings` → `--local` ép `qdrant_path="./qdrant_local"` → `get_client` + `check_connection` (lỗi: stderr `Không kết nối được Qdrant tại {where}. Chạy "docker compose up -d" hoặc thêm --local.`, return 1) → `chunk_directory` với `skipped_pages_collector` → không chunk: stderr, return 1 → `preserve_files = set(skipped_files)` → `BgeM3Embedder` → `ensure_collection` (lỗi schema: stderr, return 1) → `index_corpus` → in báo cáo (file, chunk, xóa thừa, file bỏ qua, trang bỏ qua) → return 0.
- [ ] **Step 5:** `pytest -q` PASS (< 10 s, không tải model).
- [ ] **Step 6: Critic** — subagent `critic` với `docs/contracts/task-4.md`, `src/ingestion/indexer.py`, `src/ingestion/sparse.py`, `src/ingestion/embedder.py`, test tương ứng; lưu `docs/reviews/task-6-indexer-r<n>.md`; tối đa 3 vòng; tick checklist hợp đồng khi `APPROVED`.
- [ ] **Step 7: Commit** `feat(indexer): named dense+sparse collection, full-corpus BM25 indexing, safe stale cleanup`

---

### Task 7: Hybrid search

**Files:**
- Create: `src/retrieval/__init__.py`, `tests/retrieval/__init__.py`, `tests/retrieval/test_hybrid_search.py`
- Modify: `src/retrieval/hybrid_search.py` (0 byte)

**Interfaces:**
- Consumes: `query_vector` (Task 4), `Embedder` (Task 5), `DENSE_VECTOR_NAME`, `SPARSE_VECTOR_NAME`, `ensure_collection`, `index_corpus` (Task 6).
- Produces:
```python
SearchMode = Literal["dense", "sparse", "hybrid"]

@dataclass(frozen=True)
class Hit:
    chunk_id: str
    source_file: str
    page_number: int
    article: str | None
    chunk_index: int
    text: str
    score: float

class SearchUnavailableError(Exception): ...

class HybridSearcher:
    def __init__(self, client: QdrantClient, collection: str, embedder: Embedder, candidate_k: int = 20) -> None
    def search(self, query: str, mode: SearchMode = "hybrid", limit: int = 20) -> list[Hit]
    def is_ready(self) -> bool     # collection tồn tại và count > 0
```

- [ ] **Step 1: Test đỏ** — index 6 chunk tiếng Việt viết tay (Điều về tín chỉ, cảnh báo học tập, tốt nghiệp…) bằng `FakeEmbedder` vào `:memory:`.
  - `test_sparse_mode_ranks_keyword_match_first`: `search("cảnh báo học tập", "sparse")[0].article` là Điều cảnh báo.
  - `test_dense_mode_returns_limit_hits`: `len(search(q, "dense", limit=3)) == 3`.
  - `test_hybrid_uses_rrf_prefetch(monkeypatch)`: bắt tham số `client.query_points` → có 2 `Prefetch` (`using="dense"`, `using="sparse"`, `limit=candidate_k`) và `query=FusionQuery(fusion=Fusion.RRF)`.
  - `test_hits_preserve_page_metadata`: mọi `Hit` có `page_number`, `source_file`, `article` khớp payload đã index.
  - `test_query_with_zero_tokens_falls_back_to_dense(monkeypatch)`: `search("???", "hybrid")` không gửi sparse rỗng (chỉ 1 prefetch hoặc truy vấn dense), không lỗi. *(Review Focus 1)*
  - `test_unaccented_query_does_not_crash`: `search("hoc phi bao nhieu", "hybrid")` trả list. *(Review Focus 5)*
  - `test_qdrant_error_wrapped(monkeypatch)`: `query_points` ném `ConnectionError` → `SearchUnavailableError`.
  - `test_is_ready_false_on_empty_collection`.
- [ ] **Step 2:** FAIL.
- [ ] **Step 3: Cài đặt** — dense: `embed_dense([query])[0]`; sparse: `query_vector`; hybrid: `query_points(prefetch=[...], query=FusionQuery(fusion=Fusion.RRF), limit=limit, with_payload=True)`; sparse rỗng → mode thực tế `dense` (ghi chú trong docstring).
- [ ] **Step 4:** `pytest -q` PASS.
- [ ] **Step 5: Commit** `feat(retrieval): dense, BM25 and RRF hybrid search over Qdrant`

---

### Task 8: Reranker

**Files:**
- Create: `src/retrieval/reranker.py`, `tests/retrieval/test_reranker.py`
- Modify: `tests/fakes.py`

**Interfaces:**
- Consumes: `Hit` (Task 7).
- Produces:
```python
class Reranker(Protocol):
    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]: ...
    # trả tối đa top_k Hit, sắp giảm dần, Hit.score = điểm rerank trong [0, 1]

class BgeReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cpu", max_length: int = 512) -> None
    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]

# tests/fakes.py
class FakeReranker:  # score = số token chung giữa query và text / số token query; ghi lại các lần gọi
```

- [ ] **Step 1: Test đỏ**
  - `test_fake_reranker_orders_and_truncates`: 4 hit → `top_k=2` trả 2, giảm dần.
  - `test_rerank_preserves_metadata`: `dataclasses.replace` chỉ đổi `score`; `page_number`, `chunk_id` giữ nguyên.
  - `test_rerank_empty_hits_returns_empty` (cả `BgeReranker` — không tải model khi rỗng: kiểm bằng lazy load).
  - `@pytest.mark.slow test_bge_reranker_prefers_relevant`: câu "Sinh viên bị cảnh báo học tập khi nào?" → đoạn về cảnh báo học tập điểm cao hơn đoạn về tốt nghiệp; điểm trong [0, 1].
- [ ] **Step 2:** FAIL.
- [ ] **Step 3: Cài đặt** — `CrossEncoder(model_name, device=device, max_length=max_length)` tải lười ở lần gọi đầu hoặc `load()`; `predict([(query, h.text) ...])`.
- [ ] **Step 4:** `pytest -q` PASS; `pytest -m slow tests/retrieval/test_reranker.py` PASS.
- [ ] **Step 5: Commit** `feat(retrieval): cross-encoder reranker`

---

### Task 9: Câu từ chối, trích dẫn, prompt, Gemini client

**Files:**
- Create: `src/rag/__init__.py`, `src/rag/constants.py`, `src/rag/models.py`, `src/rag/citations.py`, `src/rag/prompts.py`, `src/rag/llm.py`, `tests/rag/__init__.py`, `tests/rag/test_citations.py`, `tests/rag/test_prompts.py`, `tests/rag/test_llm.py`
- Modify: `tests/fakes.py`, `requirements.txt`, `requirements.lock`

**Interfaces:**
- Consumes: `Hit` (Task 7).
- Produces:
```python
# constants.py
REFUSAL = "Không tìm thấy thông tin trong tài liệu quy chế."

# models.py
@dataclass(frozen=True)
class Turn:
    role: Literal["user", "assistant"]
    content: str

# citations.py
def normalize_refusal(answer: str) -> str        # biến thể gần đúng -> REFUSAL; khác -> answer.strip()
def is_refusal(answer: str) -> bool               # normalize_refusal(answer) == REFUSAL
def extract_citation_ids(answer: str) -> list[int]   # "[1][3] ... [1]" -> [1, 3], theo thứ tự xuất hiện, không trùng

# prompts.py
ANSWER_SYSTEM_PROMPT: str
REWRITE_SYSTEM_PROMPT: str
def build_answer_prompt(question: str, hits: list[Hit]) -> str     # khối "[n] (file, trang p, Điều) text"
def build_rewrite_prompt(question: str, history: list[Turn]) -> str

# llm.py
class LLMError(Exception): ...
class LLMClient(Protocol):
    def generate(self, prompt: str, system: str | None = None) -> str: ...
class GeminiClient:
    def __init__(self, api_key: str, model: str, timeout_s: float = 30.0) -> None
    def generate(self, prompt: str, system: str | None = None) -> str

# tests/fakes.py
class FakeLLM:  # trả lần lượt các chuỗi trong responses; ghi prompts; raise nếu phần tử là Exception
```
`Turn` nằm ở `src/rag/models.py` để `prompts.py`, `rewriter.py`, `pipeline.py` cùng dùng mà không import vòng.

- [ ] **Step 1: Test đỏ**
  - `test_refusal_constant_exact`: `REFUSAL == "Không tìm thấy thông tin trong tài liệu quy chế."`
  - `@pytest.mark.parametrize` biến thể → `REFUSAL`: không dấu chấm cuối; bọc `"..."`/`“...”`; khoảng trắng/xuống dòng hai đầu; dạng NFD. *(Review Focus 3)*
  - `test_answer_containing_refusal_plus_content_is_not_refusal`: `REFUSAL + " Tuy nhiên theo Điều 5 [1]..."` → `is_refusal` False.
  - `test_extract_citation_ids`: `"A [2]. B [1][2]. C [10]"` → `[2, 1, 10]`; không có → `[]`.
  - `test_answer_prompt_numbers_contexts_and_states_rules`: prompt chứa `[1]`, `[2]`, tên file, `trang 13`, `REFUSAL` nguyên văn; `ANSWER_SYSTEM_PROMPT` chứa yêu cầu chỉ dùng context và trích `[n]`.
  - `test_answer_prompt_does_not_ask_llm_for_page_numbers`: system prompt yêu cầu trích bằng số `[n]`, không tự ghi số trang.
  - `test_gemini_client_retries_once_on_429(monkeypatch)`: patch hàm gọi SDK nội bộ ném lỗi mã 429 lần 1, trả text lần 2 → kết quả text, gọi 2 lần.
  - `test_gemini_client_raises_llmerror_after_second_failure` và `..._on_400_no_retry` (gọi 1 lần).
  - `test_gemini_client_never_logs_key(caplog)`: key `"k-secret"` không xuất hiện trong log ở mọi nhánh lỗi.
- [ ] **Step 2:** FAIL.
- [ ] **Step 3: Cài đặt**
  - `.venv\Scripts\python -m pip install google-genai` → ghim bản đã cài vào `requirements.txt`; tạo lại `requirements.lock`; báo cáo các gói phụ được thêm.
  - `GeminiClient`: `genai.Client(api_key=..., http_options=HttpOptions(timeout=int(timeout_s*1000)))`; `models.generate_content(model, contents=prompt, config=GenerateContentConfig(system_instruction=system, temperature=0))`; tách lời gọi SDK vào `_call(prompt, system) -> str` để test patch; retry 1 lần khi mã 429 hoặc ≥ 500 hoặc timeout; mọi lỗi cuối → `LLMError` (thông điệp không chứa key).
  - Prompt trả lời: tiếng Việt; chỉ dùng context; mỗi ý trích `[n]`; không đủ thông tin → trả đúng `REFUSAL` và không thêm gì.
- [ ] **Step 4:** `pytest -q` PASS.
- [ ] **Step 5 (cần key, không bắt buộc để xong task):** nếu người dùng đã đặt `GEMINI_API_KEY` và `GEMINI_MODEL` trong `.env`: `.venv\Scripts\python -c "from src.config import load_settings; from src.rag.llm import GeminiClient; s=load_settings(); print(GeminiClient(s.gemini_api_key, s.gemini_model).generate('Trả lời đúng một chữ: OK'))"` → in `OK`. Chưa có key → ghi trong báo cáo.
- [ ] **Step 6: Commit** `feat(rag): refusal constant, citation parsing, prompts and Gemini client`

---

### Task 10: Viết lại câu hỏi và pipeline hai lớp chống bịa

**Files:**
- Create: `src/rag/rewriter.py`, `src/rag/pipeline.py`, `tests/rag/test_rewriter.py`, `tests/rag/test_pipeline.py`

**Interfaces:**
- Consumes: `HybridSearcher`, `Hit`, `SearchUnavailableError` (Task 7), `Reranker` (Task 8), `REFUSAL`, `Turn`, `normalize_refusal`, `extract_citation_ids`, prompts, `LLMClient`, `LLMError` (Task 9), `Settings` (Task 2).
- Produces:
```python
# rewriter.py  (Turn import từ src.rag.models)
@dataclass(frozen=True)
class RewriteResult:
    query: str
    failed: bool

MAX_HISTORY_TURNS = 3
MAX_TURN_CHARS = 500
def rewrite(question: str, history: Sequence[Turn], llm: LLMClient | None) -> RewriteResult

# pipeline.py
@dataclass(frozen=True)
class Citation:
    source_file: str
    page_number: int
    article: str | None
    text: str

@dataclass(frozen=True)
class Answer:
    answer: str
    citations: list[Citation]
    rewritten_query: str
    refused: bool
    debug: dict[str, Any]     # timings_ms, top_rerank_score, rewrite_failed, citation_check_failed, n_candidates

class LLMUnavailableError(Exception):
    sources: list[Citation]

class RagPipeline:
    def __init__(self, searcher: HybridSearcher, reranker: Reranker, llm: LLMClient | None, settings: Settings) -> None
    def retrieve(self, query: str) -> list[Hit]                  # hybrid candidate_k -> rerank final_k
    def answer(self, question: str, history: Sequence[Turn] = ()) -> Answer
```

- [ ] **Step 1: Test đỏ** (`FakeLLM`, `FakeReranker`, searcher thật trên `:memory:` với `FakeEmbedder`)
  - Rewriter: không history → trả nguyên câu, LLM không bị gọi; có history → trả chuỗi LLM (strip); LLM ném `LLMError` → `RewriteResult(question, failed=True)`; `llm=None` → nguyên câu, `failed=False`.
  - `test_rewrite_prompt_bounded_history`: 50 lượt × 5000 ký tự → prompt chỉ chứa 3 lượt cuối, mỗi lượt ≤ 500 ký tự. *(Review Focus 4)*
  - `test_layer1_refuses_without_calling_llm`: `FakeReranker` cho điểm 0.1 < `rerank_threshold` 0.3 → `answer == REFUSAL`, `refused`, `FakeLLM.prompts == []`.
  - `test_no_candidates_refuses`.
  - `test_valid_answer_maps_citations_from_metadata`: LLM trả `"Sinh viên bị cảnh báo khi ... [1]."` → `citations[0].page_number` == `page_number` của hit đầu (không lấy từ chữ của LLM).
  - `test_llm_refusal_variant_normalized`: LLM trả `"Không tìm thấy thông tin trong tài liệu quy chế"` → `answer == REFUSAL`, `refused`.
  - `test_answer_without_citation_refused`: `debug["citation_check_failed"] is True`, `answer == REFUSAL`.
  - `test_out_of_range_citation_refused`: `[9]` với 5 context → refused.
  - `test_llm_error_raises_unavailable_with_sources`: `LLMError` → `LLMUnavailableError`, `sources` khác rỗng.
  - `test_llm_none_raises_unavailable`.
  - `test_search_unavailable_propagates`: `SearchUnavailableError` bay ra (API xử lý).
  - `test_history_triggers_rewrite_and_reports_query`: `rewritten_query` == chuỗi LLM viết lại.
- [ ] **Step 2:** FAIL.
- [ ] **Step 3: Cài đặt** theo luồng spec mục 5 (bước 1–7); `debug["timings_ms"]` gồm `rewrite`, `search`, `rerank`, `llm`.
- [ ] **Step 4:** `pytest -q` PASS.
- [ ] **Step 5: Commit** `feat(rag): query rewriting and two-layer anti-hallucination pipeline`

---

### Task 11: FastAPI

**Files:**
- Create: `src/api/__init__.py`, `tests/api/__init__.py`, `tests/api/test_main.py`
- Modify: `src/api/main.py` (0 byte), `requirements.txt`, `requirements.lock`

**Interfaces:**
- Consumes: `RagPipeline`, `Answer`, `LLMUnavailableError` (Task 10), `SearchUnavailableError`, `HybridSearcher` (Task 7), `Settings`, `load_settings` (Task 2), `get_client` (Task 6), `BgeM3Embedder`, `BgeReranker`, `GeminiClient`.
- Produces:
```python
def create_app(pipeline_factory: Callable[[], RagPipeline] | None = None) -> FastAPI
app = create_app()        # uvicorn src.api.main:app
# POST /chat  body {"question": str, "history": [{"role": "user"|"assistant", "content": str}]}
#   200 {"answer", "refused", "citations": [{"source_file","page_number","article","text"}], "rewritten_query", "debug"}
#   422 câu hỏi rỗng sau strip / chỉ dấu câu-khoảng trắng / > 1000 ký tự
#   503 {"error": "..."} (+ "sources" khi LLM lỗi)
# GET /health -> 200 {"qdrant": bool, "collection_ready": bool, "models_loaded": bool, "llm_configured": bool}
```
Thông báo 503: `Hệ thống tìm kiếm tạm thời không khả dụng`, `Chưa có dữ liệu, hãy chạy indexer`, `Mô hình ngôn ngữ tạm thời không khả dụng`, `Chưa cấu hình GEMINI_API_KEY`.

- [ ] **Step 1:** `.venv\Scripts\python -m pip install fastapi uvicorn httpx` → ghim vào `requirements.txt`, tạo lại `requirements.lock`.
- [ ] **Step 2: Test đỏ** (`TestClient`, `pipeline_factory` trả pipeline dùng bản giả)
  - `test_chat_ok_returns_citations_with_page_numbers`.
  - `@pytest.mark.parametrize("q", ["", "   ", "???", "x" * 1001])` → 422. *(Review Focus 1)*
  - `test_chat_refusal_is_200`.
  - `test_search_unavailable_503`, `test_empty_collection_503`, `test_llm_error_503_with_sources`, `test_missing_api_key_503`.
  - `test_health_reports_components`.
  - `test_models_loaded_once`: factory đếm số lần gọi qua 3 request → 1.
  - `test_logs_have_no_question_text_unless_debug(caplog)`.
- [ ] **Step 3:** FAIL.
- [ ] **Step 4: Cài đặt** — pipeline tạo trong `lifespan`; pydantic model `ChatRequest` validate (strip, phải có ít nhất 1 chữ/số, ≤ 1000); `history` tối đa 20 phần tử; log một dòng/request (id, độ dài câu hỏi, timings, điểm cao nhất, số chunk).
- [ ] **Step 5:** `pytest -q` PASS.
- [ ] **Step 6: Commit** `feat(api): FastAPI chat and health endpoints`

---

### Task 12: Giao diện Streamlit

**Files:**
- Create: `src/ui/__init__.py`, `tests/ui/__init__.py`, `tests/ui/test_app_helpers.py`
- Modify: `src/ui/app.py` (0 byte), `requirements.txt`, `requirements.lock`

**Interfaces:**
- Consumes: hợp đồng HTTP của Task 11.
- Produces:
```python
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
def format_citation(c: dict) -> str     # "thong_tu_08_2021.pdf — trang 13 — Điều 11. Xử lý kết quả học tập"
def build_history(messages: list[dict]) -> list[dict]   # session_state -> history gửi API (bỏ citations)
def call_chat(question: str, history: list[dict], timeout_s: float = 60.0) -> dict  # lỗi mạng -> {"error": "Chưa bật API: chạy uvicorn src.api.main:app"}
```

- [ ] **Step 1:** `.venv\Scripts\python -m pip install streamlit` → ghim, tạo lại lock.
- [ ] **Step 2: Test đỏ** — `format_citation` (có/không `article`); `build_history` giữ role/content; `call_chat` với `httpx.MockTransport`/monkeypatch: kết nối lỗi → `"error"` chứa `uvicorn`; 503 → trả `error` từ body.
- [ ] **Step 3:** FAIL → **Step 4: Cài đặt** — `st.chat_message` hiển thị lịch sử; mỗi câu trả lời có expander "Nguồn" liệt kê `format_citation` + đoạn trích; câu từ chối hiển thị nhãn riêng; sidebar hiện `/health`; nút "Hội thoại mới" xóa lịch sử; chế độ debug hiện `rewritten_query` và timings.
- [ ] **Step 5:** `pytest -q` PASS.
- [ ] **Step 6: Kiểm tra tay** (khi có Docker + index + key): `.venv\Scripts\python -m uvicorn src.api.main:app --port 8000` và `.venv\Scripts\python -m streamlit run src/ui/app.py --server.port 8502`; hỏi 1 câu trong phạm vi, 1 câu ngoài phạm vi, 1 câu nối tiếp; chụp màn hình vào báo cáo. Chưa đủ điều kiện → để Task 15.
- [ ] **Step 7: Commit** `feat(ui): Streamlit chat with citations`

---

### Task 13: Dữ liệu đánh giá, chỉ số, gán nhãn, đánh giá truy xuất

**Files:**
- Create: `eval/__init__.py`, `eval/dataset.py`, `eval/metrics.py`, `eval/label_helper.py`, `eval/evaluate_retrieval.py`, `eval/data/questions.jsonl` (mẫu 3 dòng), `eval/data/README.md`, `tests/eval/__init__.py`, `tests/eval/test_dataset.py`, `tests/eval/test_metrics.py`, `tests/eval/test_evaluate_retrieval.py`
- Delete: `eval/evaluate_recall.py` (0 byte, thay bằng `evaluate_retrieval.py`)

**Interfaces:**
- Consumes: `HybridSearcher`, `Hit`, `SearchMode` (Task 7), `Reranker` (Task 8). Gọi trực tiếp searcher và reranker (không qua `RagPipeline`) để đo từng tầng riêng.
- Produces:
```python
# dataset.py
@dataclass(frozen=True)
class Gold:
    source_file: str
    article: str          # "Điều 11"
    pages: tuple[int, ...]

@dataclass(frozen=True)
class Question:
    id: str
    question: str
    in_scope: bool
    split: Literal["dev", "test"]
    gold: tuple[Gold, ...]

def load_questions(path: Path, split: str | None = None) -> list[Question]
# ValueError khi: id trùng; split ngoài {dev,test}; in_scope=True mà gold rỗng; in_scope=False mà gold khác rỗng

# metrics.py
def article_number(article: str | None) -> int | None        # "Điều 11. Xử lý..." -> 11
def is_relevant(hit: Hit, gold: Gold) -> bool                 # cùng file ∧ cùng số Điều ∧ page ∈ pages
def recall_at_k(hits: list[Hit], gold: Sequence[Gold], k: int) -> float   # |gold được trúng trong top-k| / |gold|
def mrr_at_k(hits: list[Hit], gold: Sequence[Gold], k: int) -> float      # 1/rank hit liên quan đầu tiên, 0 nếu không

# evaluate_retrieval.py
CONFIGS = ("dense", "hybrid", "hybrid_rerank")
def evaluate(questions: list[Question], searcher: HybridSearcher, reranker: Reranker | None,
             candidate_k: int = 20, final_k: int = 5) -> dict[str, dict[str, float]]
# {"dense": {"recall@5", "recall@20", "mrr@5", "n"}, "hybrid": {...}, "hybrid_rerank": {"recall@5", "mrr@5", "n"}}
def main(argv: list[str] | None = None) -> int   # --split test|dev, --collection, --no-rerank, --out eval/reports
```

- [ ] **Step 1: Test đỏ**
  - Dataset: đọc mẫu hợp lệ; 4 ca `ValueError` ở Interfaces; lọc theo `split`.
  - Metrics: `article_number("Điều 11. Xử lý kết quả học tập") == 11`, `article_number(None) is None`; `is_relevant` sai file / sai Điều / sai trang → False, đúng cả ba → True; Điều kéo 2 trang, hit trang thứ hai → True; `recall_at_k` với gold ở hạng 6 → `k=5` 0.0, `k=20` 1.0; 2 gold trúng 1 → 0.5; `mrr_at_k` hạng 2 → 0.5.
  - `test_evaluate_excludes_out_of_scope`: câu `in_scope=False` không vào `n`.
  - `test_evaluate_runs_three_configs` (bản giả, `:memory:`): đủ 3 khóa; `hybrid_rerank` dùng 20 ứng viên của hybrid.
  - `test_report_records_run_metadata(tmp_path)`: file JSON có `embedding_model`, `reranker_model`, `collection`, `git_commit`, `date`, `split`.
- [ ] **Step 2:** FAIL → **Step 3: Cài đặt**; `label_helper.py`: `python -m eval.label_helper "câu hỏi"` in top-10 hybrid (file, trang, Điều, 200 ký tự) để người dùng tự chọn gold — **không** ghi vào `questions.jsonl`.
- [ ] **Step 4:** `pytest -q` PASS.
- [ ] **Step 5:** `eval/data/README.md`: schema, quy tắc gán nhãn (70 câu: 50 test, 20 dev; mỗi split có câu ngoài phạm vi, ~10 ở test; dev gồm 2 câu không dấu — Review Focus 5; câu học phí ngoài phạm vi nếu chưa có quy chế trường), cách dùng `label_helper`.
- [ ] **Step 6: Commit** `feat(eval): dataset schema, retrieval metrics and evaluation runner`

---

### Task 14: Đánh giá câu trả lời và bước viết lại

**Files:**
- Create: `eval/evaluate_answers.py`, `eval/evaluate_rewrite.py`, `eval/data/dialogues.jsonl` (mẫu 1 hội thoại), `tests/eval/test_evaluate_answers.py`, `tests/eval/test_evaluate_rewrite.py`

**Interfaces:**
- Consumes: `Question`, `load_questions`, `recall_at_k`, `is_relevant` (Task 13), `RagPipeline`, `Answer`, `LLMUnavailableError` (Task 10), `rewrite`, `Turn` (Task 10).
- Produces:
```python
# evaluate_answers.py
def evaluate_answers(questions: list[Question], pipeline: RagPipeline) -> dict[str, float]
# {"correct_refusal_rate" (out-of-scope), "false_refusal_rate" (in-scope), "citation_valid_rate"
#  (in-scope, không từ chối, ≥1 citation trúng gold), "llm_errors": int}
def sweep_threshold(questions: list[Question], pipeline: RagPipeline, thresholds: Sequence[float]) -> list[dict]
# chỉ cho split dev; chạy truy xuất+rerank một lần, áp từng ngưỡng cho lớp 1 (không gọi LLM)
def main(argv: list[str] | None = None) -> int   # --split, --sweep 0.1,0.2,...

# evaluate_rewrite.py
@dataclass(frozen=True)
class DialogueCase:
    id: str
    history: tuple[Turn, ...]
    question: str
    standalone: str
    gold: tuple[Gold, ...]
def load_dialogues(path: Path) -> list[DialogueCase]
def evaluate_rewrite(cases: list[DialogueCase], llm: LLMClient, searcher: HybridSearcher,
                     reranker: Reranker, k: int = 5) -> dict[str, float]
# {"recall@5_rewritten", "recall@5_standalone", "gap", "rewrite_failures"}
```

- [ ] **Step 1: Test đỏ** — pipeline giả trả từ chối cho câu ngoài phạm vi → `correct_refusal_rate == 1.0`; từ chối nhầm 1/2 câu trong phạm vi → `false_refusal_rate == 0.5`; `LLMUnavailableError` đếm vào `llm_errors`, không tính là từ chối; `sweep_threshold` từ chối `--split test` (ValueError: chỉ dev); `evaluate_rewrite` với `FakeLLM` trả đúng câu độc lập → `gap == 0.0`.
- [ ] **Step 2:** FAIL → **Step 3: Cài đặt** → **Step 4:** `pytest -q` PASS.
- [ ] **Step 5: Commit** `feat(eval): refusal, citation and query-rewrite evaluation`

---

### Task 15: Chạy thật, hiệu chỉnh trên dev, báo cáo, README

Task này cần đầu vào của người dùng; dừng và hỏi ở mỗi mốc ⏸.

**Files:**
- Create: `eval/reports/*.json|md`, `docs/reviews/final-review.md`
- Modify: `README.md` (0 byte), `.env.example` (ngưỡng đã chọn), `CLAUDE.md` (trạng thái), `requirements.*` (nếu chọn CUDA)

- [ ] **Step 1: ⏸ Đầu vào người dùng** — PDF quy chế trường vào `data/pdfs/` (tên không dấu); `GEMINI_API_KEY`, `GEMINI_MODEL` trong `.env`. Chạy lại Task 3 Step 5 với PDF trường.
- [ ] **Step 2: Index thật** — `docker compose up -d`; `.venv\Scripts\python -m src.ingestion.indexer --recreate` → `Đã index N chunks`; chạy lại không `--recreate` → số point không đổi; `pytest -m slow` PASS.
- [ ] **Step 3: ⏸ Gán nhãn** — người dùng viết 70 câu + ~10 hội thoại theo `eval/data/README.md`, dùng `label_helper`; agent chạy `load_questions` để kiểm tra hợp lệ.
- [ ] **Step 4: Thí nghiệm `chunk_size` trên dev** — index `regulations_cs400`, `regulations_cs800`, `regulations_cs1200` (`--collection`, `CHUNK_SIZE`/`CHUNK_OVERLAP` = 400/80, 800/150, 1200/200); `python -m eval.evaluate_retrieval --split dev --collection ...` cho từng cái; chọn theo Recall@5 của `hybrid_rerank` (hòa → Recall@20 → nhỏ hơn).
- [ ] **Step 5: Ngưỡng trên dev** — `python -m eval.evaluate_answers --split dev --sweep 0.05,0.1,0.2,0.3,0.4,0.5,0.6`; chọn ngưỡng lớn nhất có `false_refusal_rate` ≤ 0.1, ưu tiên `correct_refusal_rate` cao nhất; ghi vào `.env.example` và lý do vào báo cáo.
- [ ] **Step 6: ⏸ Quyết định reranker** — trình bày thời gian thực đo + Recall@5 có/không rerank; người dùng chọn: giữ CPU top-20, CPU top-10, hoặc cài torch CUDA (đo lại sau khi cài, ghim vào lock).
- [ ] **Step 7: Báo cáo cuối trên test** — `evaluate_retrieval --split test`, `evaluate_answers --split test`, `evaluate_rewrite`; bảng so sánh Dense / Hybrid / Hybrid+Rerank (Recall@5, Recall@20, MRR@5), tỉ lệ từ chối, trích dẫn, khoảng cách viết lại; ghi tên model, commit, ngày.
- [ ] **Step 8: Demo tay** — Task 12 Step 6.
- [ ] **Step 9: README** — bài toán; sơ đồ kiến trúc (Mermaid); quyết định thiết kế (ADR rút gọn từ spec) kèm **số liệu thật** từ Step 4–7; cài đặt (`py -3.11 -m venv .venv`, `pip install -r requirements.lock`, Docker); chạy index/API/UI; chạy test và đánh giá; hạn chế đã biết.
- [ ] **Step 10: Review toàn nhánh** — subagent `critic` (hoặc `superpowers:requesting-code-review`) trên toàn bộ thay đổi từ tag `checkpoint/pre-redesign`; lưu `docs/reviews/final-review.md`; sửa các lỗi được xác nhận.
- [ ] **Step 11: Commit** `docs: evaluation results and README`; hỏi người dùng có push không.
