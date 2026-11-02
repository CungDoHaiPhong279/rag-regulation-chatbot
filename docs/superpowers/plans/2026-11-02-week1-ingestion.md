# Tuần 1 — Ingestion Pipeline (PDF → Chunk → BGE-M3 → Qdrant) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Đọc mọi PDF trong `data/pdfs/`, cắt chunk giữ nguyên `page_number` + `source_file`, embed bằng BGE-M3 và lưu vào Qdrant local, chạy được bằng một lệnh CLI.

**Architecture:** Chunk **theo từng trang** (mỗi chunk thuộc đúng 1 trang → số trang trích dẫn luôn chính xác), dùng `RecursiveCharacterTextSplitter` với separator ưu tiên ranh giới `Điều`. Nhãn `Điều X` hiện hành được mang qua ranh giới trang. Embedder được trừu tượng hoá qua `Protocol` để test dùng `FakeEmbedder` + `QdrantClient(":memory:")` (không cần Docker, không tải model 2 GB). Point ID là UUIDv5 tất định → chạy lại indexer không sinh bản trùng.

**Tech Stack:** Python 3.12 (venv), PyMuPDF, langchain-text-splitters, sentence-transformers (BAAI/bge-m3), qdrant-client, python-dotenv, pytest, Docker (Qdrant).

**Spec:** [`AGENTS.md`](../../../AGENTS.md) — mục 1 (Key Requirements #1, #3) và lộ trình Tuần 1.

## Global Constraints

- Metadata bắt buộc trên mọi chunk/point: `page_number` (int, **1-based**, khớp số trang vật lý trong PDF) và `source_file` (tên file, không kèm đường dẫn).
- Embedding model: `BAAI/bge-m3`, dense dim **1024**, distance **Cosine**, `normalize_embeddings=True`.
- Chunking mặc định: `chunk_size=800` ký tự, `chunk_overlap=150` ký tự.
- Không hardcode secret: `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`, `EMBEDDING_MODEL` đọc từ `.env` qua `python-dotenv`.
- Mọi lệnh Python chạy qua `.venv\Scripts\python` (máy có `python`=3.12 nhưng `pip`=3.11 — không dùng `pip` trần).
- Type hints + docstring cho mọi hàm/class public.
- Text được chuẩn hoá Unicode **NFC** trước khi chunk (PDF tiếng Việt hay trả về dấu tổ hợp NFD).

## Review Focus

1. **PDF scan (không có lớp chữ)** → phải báo lỗi rõ `PdfNoTextError` và CLI bỏ qua file đó kèm cảnh báo, không âm thầm index rỗng. *(test trong Task 2)*
2. **Chạy indexer 2 lần** → số point trong collection không đổi (ID tất định). *(test trong Task 3)*
3. **Một `Điều` kéo dài sang trang sau** → chunk ở trang sau mang `page_number` của trang sau nhưng vẫn giữ nhãn `article` của Điều đó. *(test trong Task 2)*
4. **Dấu tiếng Việt dạng NFD** → sau chunk là NFC, so khớp chuỗi "Điều" vẫn hoạt động. *(test trong Task 2)*
5. **Qdrant không chạy** → CLI in thông báo "Không kết nối được Qdrant tại <url>. Đã chạy `docker compose up -d` chưa?" và exit code 1, không traceback dài. *(test trong Task 3)*

---

## File Structure

| File | Trách nhiệm |
|---|---|
| `requirements.txt` | Dependencies Tuần 1 |
| `docker-compose.yml` | Qdrant local, volume `qdrant_storage/` |
| `.env.example` | Mẫu biến môi trường |
| `.gitignore` | Thêm `data/pdfs/*.pdf` |
| `pytest.ini` | `pythonpath = .`, `testpaths = tests` |
| `src/ingestion/__init__.py`, `src/__init__.py` | Package marker |
| `src/ingestion/sample_pdf.py` | Sinh PDF quy chế mẫu (biết trước Điều nào ở trang nào) |
| `src/ingestion/chunker.py` | PDF → `list[Chunk]` |
| `src/ingestion/indexer.py` | `list[Chunk]` → Qdrant; CLI |
| `tests/ingestion/test_sample_pdf.py`, `test_chunker.py`, `test_indexer.py` | Tests |
| `README.md` | Hướng dẫn chạy + lý do chọn chunk_size/overlap |

---

### Task 1: Môi trường + PDF quy chế mẫu

**Files:**
- Create: `requirements.txt`, `docker-compose.yml`, `.env.example`, `pytest.ini`, `src/ingestion/__init__.py`, `src/ingestion/sample_pdf.py`, `tests/__init__.py`, `tests/ingestion/__init__.py`, `tests/ingestion/test_sample_pdf.py`
- Modify: `.gitignore` (thêm `data/pdfs/*.pdf`)

**Interfaces:**
- Produces: `build_sample_pdf(out_path: Path) -> dict[str, int]` — ghi PDF, trả về mapping `"Điều N" -> page_number (1-based)` của dòng tiêu đề mỗi Điều.
- Produces: hằng `SAMPLE_FILENAME = "quy_che_mau.pdf"`.

- [ ] **Step 1: Tạo venv và cài dependencies**

`requirements.txt`:
```
pymupdf>=1.24
langchain-text-splitters>=0.3
sentence-transformers>=3.0
qdrant-client>=1.12
python-dotenv>=1.0
pytest>=8.0
```
Run:
```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```
Expected: cài xong không lỗi (torch ~200 MB+).

- [ ] **Step 2: Tạo `docker-compose.yml`, `.env.example`, `pytest.ini`**

`docker-compose.yml`: service `qdrant`, image `qdrant/qdrant:v1.13.4`, ports `6333:6333`, `6334:6334`, volume `./qdrant_storage:/qdrant/storage`.

`.env.example`:
```
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=regulations
EMBEDDING_MODEL=BAAI/bge-m3
```
`pytest.ini`: `[pytest]` / `pythonpath = .` / `testpaths = tests`.

- [ ] **Step 3: Viết test thất bại** — `tests/ingestion/test_sample_pdf.py`

```python
import fitz
from src.ingestion.sample_pdf import build_sample_pdf

def test_sample_pdf_has_known_article_pages(tmp_path):
    out = tmp_path / "quy_che_mau.pdf"
    mapping = build_sample_pdf(out)
    doc = fitz.open(out)
    assert doc.page_count >= 5
    assert len(mapping) >= 8
    for article, page in mapping.items():
        assert article in doc[page - 1].get_text()

def test_sample_pdf_has_article_spanning_pages(tmp_path):
    mapping = build_sample_pdf(tmp_path / "x.pdf")
    doc = fitz.open(tmp_path / "x.pdf")
    # Điều 5 bắt đầu ở trang p và phần thân tiếp tục sang trang p+1 (có marker)
    p = mapping["Điều 5"]
    assert "[TIẾP ĐIỀU 5]" in doc[p].get_text()

def test_sample_pdf_vietnamese_diacritics_extractable(tmp_path):
    build_sample_pdf(tmp_path / "x.pdf")
    text = fitz.open(tmp_path / "x.pdf")[0].get_text()
    assert "QUY CHẾ" in text and "sinh viên" in text
```

- [ ] **Step 4: Chạy để xác nhận FAIL**

Run: `.venv\Scripts\python -m pytest tests/ingestion/test_sample_pdf.py -v`
Expected: FAIL — `ModuleNotFoundError: src.ingestion.sample_pdf`

- [ ] **Step 5: Implement `build_sample_pdf` trong `src/ingestion/sample_pdf.py`**

Nội dung: "QUY CHẾ ĐÀO TẠO TRÌNH ĐỘ ĐẠI HỌC (MẪU)", 3 Chương, Điều 1–10 (phạm vi, tín chỉ, đăng ký học phần, điểm học phần thang 10/thang 4, cảnh báo học tập, thôi học, khen thưởng, kỷ luật thi cử, tốt nghiệp, hiệu lực) — mỗi Điều chứa ít nhất 1 dữ kiện số cụ thể (vd. "tối thiểu 14 tín chỉ", "điểm trung bình tích lũy dưới 1,2") để làm câu hỏi đánh giá Tuần 3. Ghi bằng `page.insert_htmlbox(...)` (PyMuPDF tự fallback font Unicode cho tiếng Việt), một trang mới cho mỗi nhóm Điều; Điều 5 cố ý chia thân sang trang kế tiếp, mở đầu trang đó bằng `[TIẾP ĐIỀU 5]`. Thêm `if __name__ == "__main__":` ghi ra `data/pdfs/quy_che_mau.pdf` và in mapping.

- [ ] **Step 6: Chạy test PASS + sinh file thật**

Run: `.venv\Scripts\python -m pytest tests/ingestion/test_sample_pdf.py -v` → Expected: 3 passed
Run: `.venv\Scripts\python -m src.ingestion.sample_pdf` → Expected: in mapping, file `data/pdfs/quy_che_mau.pdf` tồn tại.

- [ ] **Step 7: Commit**

```powershell
git add requirements.txt docker-compose.yml .env.example pytest.ini .gitignore src/ingestion/__init__.py src/ingestion/sample_pdf.py tests/
git commit -m "feat(ingestion): env setup and sample regulation PDF generator"
```

---

### Task 2: Chunker giữ `page_number`

**Files:**
- Modify: `src/ingestion/chunker.py`
- Test: `tests/ingestion/test_chunker.py`

**Interfaces:**
- Consumes: `build_sample_pdf(out_path) -> dict[str, int]` (Task 1, dùng làm fixture).
- Produces:
  ```python
  class PdfNoTextError(Exception): ...

  @dataclass(frozen=True)
  class Chunk:
      chunk_id: str        # str(uuid5(NAMESPACE_URL, f"{source_file}:{page_number}:{chunk_index}"))
      text: str
      source_file: str     # basename
      page_number: int     # 1-based
      chunk_index: int     # thứ tự trong trang, 0-based
      article: str | None  # vd "Điều 5"

  def chunk_pdf(pdf_path: Path, chunk_size: int = 800, chunk_overlap: int = 150) -> list[Chunk]
  def chunk_directory(pdf_dir: Path, chunk_size: int = 800, chunk_overlap: int = 150) -> tuple[list[Chunk], list[str]]
      # trả (chunks, skipped_files) — file scan bị bỏ qua thay vì làm hỏng cả lô
  ```

- [ ] **Step 1: Viết test thất bại** — `tests/ingestion/test_chunker.py`

```python
import unicodedata, fitz, pytest
from src.ingestion.sample_pdf import build_sample_pdf
from src.ingestion.chunker import chunk_pdf, chunk_directory, PdfNoTextError

@pytest.fixture
def sample(tmp_path):
    p = tmp_path / "quy_che_mau.pdf"
    return p, build_sample_pdf(p)

def test_every_chunk_has_metadata(sample):
    path, _ = sample
    chunks = chunk_pdf(path)
    assert chunks
    for c in chunks:
        assert c.source_file == "quy_che_mau.pdf"
        assert c.page_number >= 1
        assert c.text.strip()

def test_page_number_matches_physical_page(sample):
    path, mapping = sample
    chunks = chunk_pdf(path)
    for article, page in mapping.items():
        hits = [c for c in chunks if c.text.lstrip().startswith(article) or f"\n{article}" in c.text]
        assert any(c.page_number == page for c in hits), article

def test_article_label_carried_across_page(sample):
    path, mapping = sample
    cont = [c for c in chunk_pdf(path) if "[TIẾP ĐIỀU 5]" in c.text]
    assert cont and all(c.page_number == mapping["Điều 5"] + 1 for c in cont)
    assert all(c.article == "Điều 5" for c in cont)

def test_chunk_size_respected(sample):
    path, _ = sample
    assert all(len(c.text) <= 800 for c in chunk_pdf(path))

def test_ids_deterministic(sample):
    path, _ = sample
    assert [c.chunk_id for c in chunk_pdf(path)] == [c.chunk_id for c in chunk_pdf(path)]

def test_text_is_nfc(sample):
    path, _ = sample
    for c in chunk_pdf(path):
        assert c.text == unicodedata.normalize("NFC", c.text)

def test_scanned_pdf_raises(tmp_path):
    p = tmp_path / "scan.pdf"
    doc = fitz.open(); doc.new_page(); doc.save(p)   # trang không có chữ
    with pytest.raises(PdfNoTextError):
        chunk_pdf(p)

def test_directory_skips_scanned(tmp_path):
    build_sample_pdf(tmp_path / "ok.pdf")
    doc = fitz.open(); doc.new_page(); doc.save(tmp_path / "scan.pdf")
    chunks, skipped = chunk_directory(tmp_path)
    assert skipped == ["scan.pdf"]
    assert {c.source_file for c in chunks} == {"ok.pdf"}
```

- [ ] **Step 2: Chạy để xác nhận FAIL**

Run: `.venv\Scripts\python -m pytest tests/ingestion/test_chunker.py -v`
Expected: FAIL — `ImportError: cannot import name 'chunk_pdf'`

- [ ] **Step 3: Implement trong `src/ingestion/chunker.py`**

- Đọc từng trang bằng `fitz.open(path)`, `page.get_text("text")`, `unicodedata.normalize("NFC", ...)`; trang trắng bỏ qua nhưng **giữ** số trang vật lý (`page.number + 1`).
- Nếu tổng text toàn file rỗng → `raise PdfNoTextError(f"{name}: không có lớp chữ (PDF scan?) — cần OCR")`.
- Splitter: `RecursiveCharacterTextSplitter(chunk_size, chunk_overlap, separators=["\nĐiều ", "\n\n", "\n", ". ", " ", ""], keep_separator="start")`, áp dụng **riêng cho từng trang**.
- Gán `article`: regex `r"^\s*(Điều\s+\d+)"` (re.MULTILINE). Nếu chunk **bắt đầu** bằng tiêu đề Điều → `article` = tiêu đề đó; ngược lại `article` = `current_article` mang từ chunk/trang trước (chunk mở đầu bằng phần tiếp của Điều 5 luôn là "Điều 5" kể cả khi giữa chunk có "Điều 6"). Sau mỗi chunk, cập nhật `current_article` = match **cuối cùng** trong chunk (nếu có).
- `chunk_directory`: duyệt `sorted(pdf_dir.glob("*.pdf"))`, bắt `PdfNoTextError` → thêm vào `skipped`.
- Docstring module ghi lý do tham số (copy sang README ở Task 4):
  > 800 ký tự ≈ 200–250 token tiếng Việt — vừa 1 Khoản/1–2 Điều ngắn, đủ ngữ cảnh mà không pha loãng vector (BGE-M3 nhận tới 8192 token nhưng chunk ngắn cho precision tốt hơn). Overlap 150 (~19%) giữ câu bị cắt ở ranh giới. Chunk theo trang để `page_number` không bao giờ mơ hồ.

- [ ] **Step 4: Chạy test PASS**

Run: `.venv\Scripts\python -m pytest tests/ingestion/test_chunker.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```powershell
git add src/ingestion/chunker.py tests/ingestion/test_chunker.py
git commit -m "feat(ingestion): page-aware chunker preserving page_number and article"
```

---

### Task 3: Indexer BGE-M3 → Qdrant + CLI

**Files:**
- Modify: `src/ingestion/indexer.py`
- Test: `tests/ingestion/test_indexer.py`

**Interfaces:**
- Consumes: `Chunk`, `chunk_directory(pdf_dir) -> tuple[list[Chunk], list[str]]` (Task 2).
- Produces (Tuần 2 sẽ dùng `Embedder`, `BgeM3Embedder`, `get_client`, `load_settings`, `DENSE_VECTOR_NAME`):
  ```python
  DENSE_VECTOR_NAME = "dense"

  class Embedder(Protocol):
      dim: int
      def embed(self, texts: list[str]) -> list[list[float]]: ...

  class BgeM3Embedder:                       # SentenceTransformer(model_name), normalize_embeddings=True, dim=1024
      def __init__(self, model_name: str = "BAAI/bge-m3") -> None: ...
      def embed(self, texts: list[str]) -> list[list[float]]: ...

  @dataclass(frozen=True)
  class Settings:
      qdrant_url: str; qdrant_api_key: str | None; collection: str; embedding_model: str
  def load_settings() -> Settings            # load_dotenv(); default theo .env.example; api_key "" -> None

  def get_client(settings: Settings) -> QdrantClient
  def ensure_collection(client: QdrantClient, name: str, dim: int, recreate: bool = False) -> None
      # named vector DENSE_VECTOR_NAME, Distance.COSINE
  def index_chunks(client: QdrantClient, chunks: list[Chunk], embedder: Embedder,
                   collection: str, batch_size: int = 32) -> int
      # upsert, id=chunk.chunk_id, payload={text, source_file, page_number, chunk_index, article}; trả số point đã upsert
  def main(argv: list[str] | None = None) -> int   # CLI: --pdf-dir (default data/pdfs) --recreate
  ```

- [ ] **Step 1: Viết test thất bại** — `tests/ingestion/test_indexer.py`

```python
import hashlib
from qdrant_client import QdrantClient
from src.ingestion.sample_pdf import build_sample_pdf
from src.ingestion.chunker import chunk_pdf
from src.ingestion import indexer
from src.ingestion.indexer import ensure_collection, index_chunks, DENSE_VECTOR_NAME

class FakeEmbedder:
    dim = 8
    def embed(self, texts):
        return [[b / 255 for b in hashlib.sha256(t.encode()).digest()[:8]] for t in texts]

def _setup(tmp_path):
    p = tmp_path / "quy_che_mau.pdf"; build_sample_pdf(p)
    client = QdrantClient(":memory:")
    ensure_collection(client, "test", FakeEmbedder.dim)
    return client, chunk_pdf(p)

def test_payload_keeps_page_and_source(tmp_path):
    client, chunks = _setup(tmp_path)
    n = index_chunks(client, chunks, FakeEmbedder(), "test")
    assert n == len(chunks)
    points, _ = client.scroll("test", limit=1000, with_payload=True)
    for pt in points:
        assert pt.payload["source_file"] == "quy_che_mau.pdf"
        assert isinstance(pt.payload["page_number"], int) and pt.payload["page_number"] >= 1
        assert pt.payload["text"]
        assert "article" in pt.payload and "chunk_index" in pt.payload

def test_reindex_is_idempotent(tmp_path):
    client, chunks = _setup(tmp_path)
    index_chunks(client, chunks, FakeEmbedder(), "test")
    index_chunks(client, chunks, FakeEmbedder(), "test")
    assert client.count("test").count == len(chunks)

def test_query_returns_page_metadata(tmp_path):
    client, chunks = _setup(tmp_path)
    emb = FakeEmbedder(); index_chunks(client, chunks, emb, "test")
    res = client.query_points("test", query=emb.embed([chunks[3].text])[0],
                              using=DENSE_VECTOR_NAME, limit=1, with_payload=True).points
    assert res[0].payload["page_number"] == chunks[3].page_number

def test_cli_reports_unreachable_qdrant(tmp_path, monkeypatch, capsys):
    build_sample_pdf(tmp_path / "a.pdf")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:1")      # cổng không có gì
    monkeypatch.setattr(indexer, "BgeM3Embedder", lambda *a, **k: FakeEmbedder())
    code = indexer.main(["--pdf-dir", str(tmp_path)])
    assert code == 1
    assert "Không kết nối được Qdrant" in capsys.readouterr().err
```

- [ ] **Step 2: Chạy để xác nhận FAIL**

Run: `.venv\Scripts\python -m pytest tests/ingestion/test_indexer.py -v`
Expected: FAIL — `ImportError: cannot import name 'ensure_collection'`

- [ ] **Step 3: Implement trong `src/ingestion/indexer.py`**

- `BgeM3Embedder` import `sentence_transformers` **bên trong `__init__`** (để test không phải load torch khi chỉ dùng Fake).
- `main`: parse args → `load_settings()` → `chunk_directory` → in cảnh báo cho từng file trong `skipped` ra stderr → kết nối: bọc `client.get_collections()` trong `try/except Exception` → in `f"Không kết nối được Qdrant tại {url}. Đã chạy `docker compose up -d` chưa?"` ra stderr, `return 1` → **kiểm tra kết nối trước khi load model** (tránh chờ load 2 GB rồi mới báo lỗi) → `BgeM3Embedder(settings.embedding_model)` → `ensure_collection(..., recreate=args.recreate)` → `index_chunks` → in `"Đã index {n} chunks từ {k} file vào '{collection}'"`, `return 0`.
- Không chunk nào (thư mục rỗng/toàn scan) → in `"Không có chunk nào để index trong {pdf_dir}"` ra stderr, `return 1`.
- `if __name__ == "__main__": sys.exit(main())`.

- [ ] **Step 4: Chạy test PASS**

Run: `.venv\Scripts\python -m pytest tests/ingestion -v`
Expected: 15 passed

- [ ] **Step 5: Commit**

```powershell
git add src/ingestion/indexer.py tests/ingestion/test_indexer.py
git commit -m "feat(ingestion): BGE-M3 indexer into Qdrant with idempotent upsert and CLI"
```

---

### Task 4: Chạy end-to-end thật + README

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: CLI `python -m src.ingestion.indexer` (Task 3), `python -m src.ingestion.sample_pdf` (Task 1).

- [ ] **Step 1: Bật Qdrant**

Run: `docker compose up -d` rồi `curl http://localhost:6333/healthz`
Expected: `healthz check passed`

- [ ] **Step 2: Index PDF mẫu (lần đầu sẽ tải BGE-M3 ~2.3 GB)**

Run: `Copy-Item .env.example .env; .venv\Scripts\python -m src.ingestion.indexer --recreate`
Expected: `Đã index <N> chunks từ 1 file vào 'regulations'` với N > 0.

- [ ] **Step 3: Kiểm tra truy vấn thật trả đúng trang**

Run (một dòng):
```powershell
.venv\Scripts\python -c "from src.ingestion.indexer import *; s=load_settings(); c=get_client(s); e=BgeM3Embedder(s.embedding_model); r=c.query_points(s.collection, query=e.embed(['Điều kiện bị cảnh báo học tập là gì?'])[0], using=DENSE_VECTOR_NAME, limit=3, with_payload=True).points; [print(p.score, p.payload['page_number'], p.payload['article']) for p in r]"
```
Expected: kết quả top-1 có `article` = Điều nói về cảnh báo học tập và `page_number` trùng mapping in ra ở Task 1 Step 6.

- [ ] **Step 4: Chạy lại để xác nhận idempotent**

Run: `.venv\Scripts\python -m src.ingestion.indexer` (không `--recreate`)
Expected: cùng số N; dashboard http://localhost:6333/dashboard hiển thị đúng N points.

- [ ] **Step 5: Viết `README.md`**

Mục: Giới thiệu dự án · Kiến trúc (luồng trong AGENTS.md) · Cài đặt (venv, `docker compose up -d`, `.env`) · Chạy ingestion · **Chiến lược chunking** (dán đoạn lý do từ docstring chunker) · Chạy test · Lưu ý PDF scan cần OCR.

- [ ] **Step 6: Commit + push**

```powershell
git add README.md
git commit -m "docs: week 1 ingestion usage and chunking rationale"
git push origin main
```

---

### Task 5 (người dùng làm, song song): Nạp tài liệu thật

Không phải code — checklist cho bạn:

- [ ] Tải **Thông tư 08/2021/TT-BGDĐT** (Quy chế đào tạo trình độ đại học) và **Nghị định 13/2023/NĐ-CP** (Bảo vệ dữ liệu cá nhân) dạng PDF từ vanban.chinhphu.vn hoặc thuvienphapluat.vn.
- [ ] Mở PDF, thử bôi đen chữ — nếu không bôi được là bản scan, tìm bản khác (hoặc tải bản Word rồi "Save as PDF").
- [ ] Đặt tên không dấu, không khoảng trắng: `tt08_2021_quy_che_dao_tao.pdf`, `nd13_2023_bao_ve_du_lieu_ca_nhan.pdf` → bỏ vào `data/pdfs/`.
- [ ] Chạy `.venv\Scripts\python -m src.ingestion.indexer --recreate` — nếu file nào bị báo "không có lớp chữ" thì thay bản khác.
