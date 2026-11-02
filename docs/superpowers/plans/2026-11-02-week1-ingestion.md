# Tuần 1 — Ingestion Pipeline (PDF → Chunk → BGE-M3 → Qdrant) Implementation Plan

> **Quy tắc thực thi (theo yêu cầu người dùng & `AGENTS.md`):**
> - **Làm TỪNG TASK MỘT:** Xong 1 task phải dừng lại, giải thích bằng tiếng Việt đơn giản (mỗi file làm gì, vì sao thiết kế như vậy, giải thích thuật ngữ mới, lệnh tự kiểm tra) và chờ người dùng nói **"tiếp"**.
> - **Review bằng Critic:** Chỉ chạy subagent **Critic** review độc lập cho **Task 2 (Chunker)** và **Task 4 (Indexer)** theo checklist trong `AGENTS.md`. Các task khác làm trực tiếp.
> - **Không tự ý commit/push** khi người dùng chưa đồng ý.
> - **Mơ hồ thì hỏi trước** khi viết code.

**Goal:** Đọc các file PDF quy chế trường trong `data/pdfs/`, làm sạch & chuẩn hóa Unicode NFC, tách theo Điều rồi cắt chunk theo từng trang (mỗi chunk thuộc đúng 1 trang và đúng 1 Điều, có tiền tố tiêu đề Điều), và index vào Qdrant (hỗ trợ cả Docker lẫn Qdrant local mode, khai báo sẵn named vectors `dense` + `sparse` cho Tuần 3).

**Architecture:**
1. **Chunk theo từng trang:** Mỗi chunk chỉ thuộc đúng 1 trang vật lý để `page_number` không bao giờ lệch.
2. **Tách theo Điều trước, cắt chunk sau:** Văn bản mỗi trang được tách thành các *đoạn Điều* tại mỗi dòng tiêu đề Điều; chỉ sau đó mới cắt chunk **bên trong từng đoạn** → không chunk nào chứa 2 Điều.
3. **Tự mang ngữ cảnh:** Mọi chunk thuộc một Điều đều có dạng `"Điều 5. Học phí: <nội dung>"`, kể cả phần Điều kéo sang trang sau.
4. **Embed trước — xóa cũ — upsert sau + ID cố định:** Mỗi chunk có UUIDv5 băm từ `source_file + page_number + chunk_index`. Khi index lại một file: embed xong **toàn bộ** chunk của file đó trước, rồi mới xóa point cũ của file và upsert point mới.
5. **Hỗ trợ Qdrant Docker & Local Mode:** Nếu `QDRANT_PATH` được đặt (hoặc dùng cờ `--local`), `QdrantClient(path=...)` lưu thẳng xuống ổ cứng, không cần Docker.

**Tech Stack:** Python 3.12 (`.venv`), PyMuPDF (`fitz`), `langchain-text-splitters`, `sentence-transformers` (`BAAI/bge-m3`), `qdrant-client`, `python-dotenv`, `pytest`. Font: Noto Sans (giấy phép SIL OFL, được phép phân phối kèm repo).

**Spec:** [`AGENTS.md`](../../../AGENTS.md)

---

## Global Constraints

- **Đơn vị `chunk_size` / `chunk_overlap`:** **Số ký tự (characters)**, đặt trong `.env`: `CHUNK_SIZE=800`, `CHUNK_OVERLAP=150`. Đây chỉ là **giá trị khởi đầu**, sẽ chốt ở Tuần 3 sau khi đo Recall@5 với 400 / 800 / 1200.
- **Tiền tố `"Điều X. Tên: "` CÓ tính vào `chunk_size`:** `chunk_size` = độ dài tối đa của **toàn bộ `text` được embed** (gồm cả tiền tố). Phần thân của mỗi đoạn Điều được cắt với `body_size = chunk_size - len(prefix)`. Bất biến cần test: `len(chunk.text) <= chunk_size` với mọi chunk.
  - *Lý do:* Khi so sánh 400/800/1200 ở Tuần 3, con số `chunk_size` phải phản ánh đúng lượng chữ thực sự đưa vào model; nếu tiền tố không tính thì chunk thực tế dài hơn cấu hình một lượng khác nhau tùy độ dài tên Điều.
  - Nếu `body_size < 100` hoặc `chunk_overlap >= body_size` → `ValueError` với thông báo nêu rõ tên Điều quá dài so với `chunk_size`.
- **Chuẩn hóa văn bản:** Mọi trang PDF phải qua `unicodedata.normalize("NFC", text)` và loại bỏ dòng header/footer lặp trước khi tách Điều.
- **Schema metadata bắt buộc trên mỗi chunk:**
  - `source_file` (`str`): Tên file PDF gốc, không kèm đường dẫn.
  - `page_number` (`int`): Số trang trong PDF (bắt đầu từ 1).
  - `article` (`str | None`): Số + tên Điều dạng chuẩn `"Điều 5. Học phí"`; `None` cho phần trước Điều đầu tiên (phần mở đầu, căn cứ ban hành).
  - `chunk_index` (`int`): Thứ tự chunk trong trang (bắt đầu từ 0, liên tục qua các đoạn Điều trên cùng trang).
  - `text` (`str`): Nội dung chunk, có tiền tố `"<article>: "` khi `article` khác `None`.
- **Collection Qdrant:** Tạo sẵn named vectors `dense` (1024 chiều, Cosine) và `sparse` (`SparseVectorParams`) ngay Tuần 1. Tuần 1 chỉ ghi `dense`; `sparse` để trống tới Tuần 3.
- **Môi trường chạy lệnh:** Luôn dùng `.venv\Scripts\python` và `.venv\Scripts\python -m pip`.
- **Ghim phiên bản:** `==` cho mọi thư viện trong `requirements.txt`; Docker image Qdrant cũng ghim tag cụ thể.
- **Bí mật:** `.env` đã nằm trong `.gitignore` (dòng 6, đã kiểm tra bằng `git check-ignore -v .env`); `.env.example` được commit và không chứa giá trị thật.

## Review Focus (Checklist cho Critic ở Task 2 & Task 4)

1. **Không mất/lệch `page_number` và `source_file`** ở bất kỳ bước nào (chunk, embed, upsert, truy xuất).
2. **Điều kéo dài qua trang sau:** chunk ở trang sau có `page_number` của trang sau, `article` của Điều trước đó, và `text` bắt đầu bằng `"<article>: "`.
3. **Không chunk nào chứa 2 Điều**; `len(text) <= chunk_size` kể cả tiền tố.
4. **Index lại một file:** embed lỗi giữa chừng thì dữ liệu cũ còn nguyên; index thành công thì chỉ còn chunk mới của file đó, file khác không bị ảnh hưởng.
5. **PDF scan / Qdrant chưa chạy:** báo lỗi tiếng Việt dễ hiểu (`PdfNoTextError`; gợi ý `docker compose up -d` hoặc `--local`).

---

## Cấu trúc File Tuần 1

| File | Trách nhiệm |
|---|---|
| `requirements.txt` | Thư viện ghim phiên bản (`==`) |
| `docker-compose.yml` | Chạy Qdrant server bằng Docker (`qdrant/qdrant:v1.12.5`, cùng minor version với `qdrant-client==1.12.1`) |
| `.env.example` | Mẫu cấu hình (`QDRANT_URL`, `QDRANT_PATH`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `EMBEDDING_MODEL`, ...) |
| `pytest.ini` | `pythonpath = .`, `testpaths = tests` |
| `assets/fonts/NotoSans-Regular.ttf`, `assets/fonts/OFL.txt` | Font TTF hỗ trợ đầy đủ tiếng Việt + giấy phép đi kèm |
| `src/ingestion/sample_pdf.py` | Sinh PDF quy chế mẫu tiếng Việt (biết trước Điều nào ở trang nào) |
| `src/ingestion/chunker.py` | Đọc PDF, NFC, lọc header/footer, tách theo Điều, cắt chunk, chèn tiền tố Điều |
| `src/ingestion/embedder.py` | `Embedder` (Protocol) + `BgeM3Embedder` (bản thật) |
| `src/ingestion/indexer.py` | Tạo collection, embed → xóa cũ → upsert, CLI (Docker & local mode) |
| `tests/ingestion/fakes.py` | `FakeEmbedder`, `FailingEmbedder` dùng chung cho test (không tải BGE-M3) |
| `tests/ingestion/test_sample_pdf.py`, `test_chunker.py`, `test_indexer.py` | Tests |
| `README.md` | Hướng dẫn cài đặt, chạy thử, giải thích thiết kế Tuần 1 |

> [!NOTE]
> `src/ingestion/embedder.py` và `tests/ingestion/fakes.py` là **2 file mới** so với cấu trúc thư mục ban đầu (mục 3 của `AGENTS.md`: "không đổi cấu trúc thư mục khi chưa hỏi"). Nằm trong thư mục đã có, chỉ tách trách nhiệm. Người dùng duyệt plan này nghĩa là đồng ý; nếu muốn gộp `Embedder` vào `indexer.py` thì báo trước Task 4.

---

### Task 1: Thiết lập môi trường, font, cấu hình & PDF quy chế mẫu

**Files:**
- Modify: `requirements.txt`, `.env.example`, `.gitignore`
- Create: `docker-compose.yml`, `pytest.ini`, `assets/fonts/NotoSans-Regular.ttf`, `assets/fonts/OFL.txt`, `src/ingestion/__init__.py`, `src/ingestion/sample_pdf.py`, `tests/__init__.py`, `tests/ingestion/__init__.py`, `tests/ingestion/test_sample_pdf.py`

**Interfaces:**
- Produces:
  ```python
  FONT_PATH: Path        # = <repo>/assets/fonts/NotoSans-Regular.ttf
  SAMPLE_FILENAME = "quy_che_mau.pdf"

  def build_sample_pdf(out_path: Path) -> dict[str, int]:
      """Ghi PDF mẫu; trả {"Điều 1. Phạm vi điều chỉnh": 1, ..., "Điều 5. Học phí": p, ...}
      = trang (1-based) chứa dòng tiêu đề của từng Điều."""
  ```

- [ ] **Step 1: Ghim phiên bản và cài vào `.venv`**

```text
pymupdf==1.24.14
langchain-text-splitters==0.3.2
sentence-transformers==3.3.1
qdrant-client==1.12.1
python-dotenv==1.0.1
pytest==8.3.4
```
Run:
```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
```
Expected: cài xong không lỗi. Nếu một phiên bản ghim không còn trên PyPI → dừng lại hỏi người dùng, không tự đổi.

- [ ] **Step 2: Tải font Noto Sans vào `assets/fonts/`**

```powershell
New-Item -ItemType Directory -Force assets/fonts
Invoke-WebRequest "https://github.com/notofonts/notofonts.github.io/raw/main/fonts/NotoSans/hinted/ttf/NotoSans-Regular.ttf" -OutFile assets/fonts/NotoSans-Regular.ttf
Invoke-WebRequest "https://raw.githubusercontent.com/notofonts/latin-greek-cyrillic/main/OFL.txt" -OutFile assets/fonts/OFL.txt
```
Expected: file `.ttf` > 300 KB. Nếu link hỏng → báo người dùng tải thủ công từ fonts.google.com/noto/specimen/Noto+Sans.
*Vì sao không dùng font có sẵn của Windows (Arial, Times New Roman):* giấy phép của chúng không cho phép đưa vào repo công khai, và máy khác (Linux/CI) không có.

- [ ] **Step 3: Tạo `.env.example`, `docker-compose.yml`, `pytest.ini`, cập nhật `.gitignore`**

`.env.example`:
```ini
# Kết nối Qdrant: dùng QDRANT_URL (Docker) HOẶC đặt QDRANT_PATH để chạy local mode không cần Docker
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_PATH=
QDRANT_COLLECTION=regulations

# Model embedding
EMBEDDING_MODEL=BAAI/bge-m3

# Tham số cắt chunk — đơn vị KÝ TỰ, tính cả tiền tố "Điều X. Tên: ".
# Giá trị khởi đầu, sẽ chọn lại bằng Recall@5 ở Tuần 3 (400/800/1200).
CHUNK_SIZE=800
CHUNK_OVERLAP=150
```
`docker-compose.yml`: service `qdrant`, image `qdrant/qdrant:v1.12.5`, ports `6333:6333`, `6334:6334`, volume `./qdrant_storage:/qdrant/storage`.
`.gitignore`: thêm `qdrant_local/` và `data/pdfs/*.pdf` (giữ `data/pdfs/.gitkeep`). Kiểm tra:
```powershell
git check-ignore -v .env              # phải in ra: .gitignore:6:.env  .env
git check-ignore -v .env.example      # không in gì (không bị ignore)
```

- [ ] **Step 4: Viết test thất bại — `tests/ingestion/test_sample_pdf.py`**

```python
import fitz
from src.ingestion.sample_pdf import build_sample_pdf, FONT_PATH

def test_font_file_present():
    assert FONT_PATH.exists() and FONT_PATH.suffix == ".ttf"

def test_article_pages_known(tmp_path):
    out = tmp_path / "quy_che_mau.pdf"
    mapping = build_sample_pdf(out)
    doc = fitz.open(out)
    assert doc.page_count >= 5 and len(mapping) == 10
    for article, page in mapping.items():
        assert article in doc[page - 1].get_text()

def test_dieu5_spans_two_pages(tmp_path):
    out = tmp_path / "x.pdf"
    p = build_sample_pdf(out)["Điều 5. Học phí"]
    assert "[TIẾP ĐIỀU 5]" in fitz.open(out)[p].get_text()      # trang p+1 (0-based index p)

def test_page_has_end_of_one_article_and_start_of_next(tmp_path):
    out = tmp_path / "x.pdf"
    m = build_sample_pdf(out)
    assert m["Điều 6. Đăng ký học phần"] == m["Điều 5. Học phí"] + 1   # trang p+1 chứa cuối Điều 5 VÀ đầu Điều 6

def test_vietnamese_text_extracts_exactly(tmp_path):
    out = tmp_path / "x.pdf"; build_sample_pdf(out)
    text = fitz.open(out)[0].get_text()
    assert "QUY CHẾ ĐÀO TẠO" in text and "sinh viên" in text and "Điều 1. Phạm vi điều chỉnh" in text

def test_repeated_header_footer_on_every_page(tmp_path):
    out = tmp_path / "x.pdf"; build_sample_pdf(out)
    for page in fitz.open(out):
        assert "TRƯỜNG ĐẠI HỌC MẪU" in page.get_text()
```

- [ ] **Step 5: Chạy để xác nhận FAIL**

Run: `.venv\Scripts\python -m pytest tests/ingestion/test_sample_pdf.py -v`
Expected: FAIL — `ModuleNotFoundError: src.ingestion.sample_pdf`

- [ ] **Step 6: Implement `src/ingestion/sample_pdf.py`**

- Font: `page.insert_font(fontname="noto", fontfile=str(FONT_PATH))`, ghi chữ bằng `page.insert_textbox(rect, text, fontname="noto", fontsize=11)`. Thiếu font → `FileNotFoundError` nêu đường dẫn và lệnh tải ở Step 2.
- Mỗi trang: header `"TRƯỜNG ĐẠI HỌC MẪU — QUY CHẾ ĐÀO TẠO"` ở trên, footer `"Trang {n}"` ở dưới.
- Nội dung: tiêu đề "QUY CHẾ ĐÀO TẠO TRÌNH ĐỘ ĐẠI HỌC (MẪU)", Điều 1–10, tên Điều cố định:
  1 Phạm vi điều chỉnh · 2 Tín chỉ · 3 Chương trình đào tạo · 4 Đánh giá học phần · 5 Học phí · 6 Đăng ký học phần · 7 Cảnh báo học tập · 8 Buộc thôi học · 9 Xét tốt nghiệp · 10 Hiệu lực thi hành.
  Mỗi Điều có ít nhất 1 dữ kiện số cụ thể (vd. "tối thiểu 14 tín chỉ", "điểm trung bình tích lũy dưới 1,2") để làm câu hỏi đánh giá Tuần 3. Điều 5 cố ý dài, phần thân tràn sang trang kế tiếp mở đầu bằng `[TIẾP ĐIỀU 5]`, và Điều 6 bắt đầu ngay trên trang đó.
- Bố cục trang được quyết định thủ công (danh sách trang → danh sách khối văn bản), không dựa vào tràn chữ tự động, để mapping trả về là chính xác.
- `if __name__ == "__main__":` ghi `data/pdfs/quy_che_mau.pdf` và in mapping.

- [ ] **Step 7: Chạy test PASS + sinh file thật**

Run:
```powershell
.venv\Scripts\python -m pytest tests/ingestion/test_sample_pdf.py -v
.venv\Scripts\python -m src.ingestion.sample_pdf
```
Expected: 6 passed; file `data/pdfs/quy_che_mau.pdf` mở được bằng trình đọc PDF, hiển thị tiếng Việt đúng dấu.

*(Dừng cuối Task 1: giải thích các file, thuật ngữ venv / embedding / vector DB / TTF / giấy phép OFL, lệnh tự kiểm tra; chờ "tiếp".)*

---

### Task 2: Chunker — NFC, lọc header/footer, tách theo Điều, cắt chunk (Có Critic Review)

**Files:**
- Modify: `src/ingestion/chunker.py`
- Create: `tests/ingestion/test_chunker.py`

**Interfaces:**
- Consumes: `build_sample_pdf` (Task 1).
- Produces:
  ```python
  ARTICLE_HEADING_RE = re.compile(r"^[ \t]*Điều[ \t]+(\d+)[ \t]*[.:][ \t]*(.*?)[ \t]*$", re.MULTILINE)
  # nhóm 1 = số Điều, nhóm 2 = tên Điều (tới hết dòng, có thể rỗng)

  class PdfNoTextError(Exception): ...

  @dataclass(frozen=True)
  class Chunk:
      chunk_id: str        # str(uuid5(NAMESPACE_URL, f"{source_file}:{page_number}:{chunk_index}"))
      source_file: str
      page_number: int     # 1-based
      article: str | None  # "Điều 5. Học phí"
      chunk_index: int     # 0-based, trong trang
      text: str            # "Điều 5. Học phí: <thân>" hoặc "<thân>" nếu article là None

  def normalize_text(text: str) -> str                       # NFC + gộp khoảng trắng thừa trong dòng
  def parse_article_heading(line: str) -> str | None         # "Điều 5: Học phí" -> "Điều 5. Học phí"; "Điều 5" -> "Điều 5"; không khớp -> None
  def split_by_article(page_text: str, carried: str | None) -> list[tuple[str | None, str]]
      # tách 1 trang thành [(article, body), ...]; phần trước tiêu đề đầu tiên thuộc `carried`;
      # dòng tiêu đề bị bỏ khỏi body (vì sẽ được gắn lại làm tiền tố)
  def chunk_pdf(pdf_path: Path, chunk_size: int | None = None, chunk_overlap: int | None = None) -> list[Chunk]
  def chunk_directory(pdf_dir: Path, chunk_size: int | None = None, chunk_overlap: int | None = None) -> tuple[list[Chunk], list[str]]
      # (chunks, skipped_files)
  ```

- [ ] **Step 1: Viết test thất bại — `tests/ingestion/test_chunker.py`**

```python
import unicodedata, fitz, pytest
from src.ingestion.sample_pdf import build_sample_pdf
from src.ingestion.chunker import (chunk_pdf, chunk_directory, parse_article_heading,
                                   split_by_article, normalize_text, ARTICLE_HEADING_RE, PdfNoTextError)

@pytest.fixture
def sample(tmp_path):
    p = tmp_path / "quy_che_mau.pdf"
    return p, build_sample_pdf(p)

# --- Regex tiêu đề Điều ---
@pytest.mark.parametrize("line,expected", [
    ("Điều 5. Học phí", "Điều 5. Học phí"),
    ("Điều 5: Học phí", "Điều 5. Học phí"),
    ("  Điều 12.  Xét tốt nghiệp và công nhận tốt nghiệp ", "Điều 12. Xét tốt nghiệp và công nhận tốt nghiệp"),
    ("Điều 5.", "Điều 5"),
    ("Điều 5 của Quy chế này quy định", None),       # tham chiếu, không có "." hay ":" sau số
    ("theo Điều 5. Học phí", None),                  # không ở đầu dòng
])
def test_parse_article_heading(line, expected):
    assert parse_article_heading(line) == expected

# --- Tách theo Điều trước khi chunk ---
def test_split_by_article_separates_two_articles():
    page = "cuối nội dung Điều 5\nĐiều 6. Đăng ký học phần\nSinh viên đăng ký tối thiểu 14 tín chỉ."
    assert split_by_article(page, carried="Điều 5. Học phí") == [
        ("Điều 5. Học phí", "cuối nội dung Điều 5"),
        ("Điều 6. Đăng ký học phần", "Sinh viên đăng ký tối thiểu 14 tín chỉ."),
    ]

def test_no_chunk_contains_two_articles(sample):
    path, _ = sample
    for c in chunk_pdf(path):
        body = c.text[len(c.article) + 2:] if c.article else c.text   # bỏ tiền tố "Điều X. Tên: "
        assert ARTICLE_HEADING_RE.search(body) is None               # thân chunk không chứa tiêu đề Điều nào khác

def test_page_with_two_articles_gives_separate_chunks(sample):
    path, m = sample
    page = m["Điều 6. Đăng ký học phần"]
    arts = {c.article for c in chunk_pdf(path) if c.page_number == page}
    assert {"Điều 5. Học phí", "Điều 6. Đăng ký học phần"} <= arts

# --- Metadata & số trang ---
def test_metadata_schema_complete(sample):
    path, _ = sample
    for c in chunk_pdf(path):
        assert c.source_file == "quy_che_mau.pdf" and c.page_number >= 1 and c.chunk_index >= 0 and c.text.strip()

def test_page_number_matches_exact_pdf_page(sample):
    path, m = sample
    chunks = chunk_pdf(path)
    for article, page in m.items():
        assert min(c.page_number for c in chunks if c.article == article) == page

def test_article_prefix_carried_across_pages(sample):
    path, m = sample
    cont = [c for c in chunk_pdf(path) if "[TIẾP ĐIỀU 5]" in c.text]
    assert cont
    for c in cont:
        assert c.page_number == m["Điều 5. Học phí"] + 1
        assert c.article == "Điều 5. Học phí" and c.text.startswith("Điều 5. Học phí: ")

def test_prefix_not_duplicated(sample):
    path, _ = sample
    for c in chunk_pdf(path):
        if c.article:
            assert c.text.startswith(f"{c.article}: ") and f"{c.article}: {c.article}" not in c.text

def test_chunk_index_sequential_per_page(sample):
    path, _ = sample
    by_page = {}
    for c in chunk_pdf(path):
        by_page.setdefault(c.page_number, []).append(c.chunk_index)
    assert all(v == list(range(len(v))) for v in by_page.values())

# --- chunk_size tính cả tiền tố ---
@pytest.mark.parametrize("size,overlap", [(400, 80), (800, 150), (1200, 200)])
def test_chunk_size_includes_prefix(sample, size, overlap):
    path, _ = sample
    assert all(len(c.text) <= size for c in chunk_pdf(path, size, overlap))

def test_chunk_size_too_small_for_title_raises(sample):
    path, _ = sample
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_pdf(path, chunk_size=60, chunk_overlap=10)

# --- Làm sạch ---
def test_unicode_nfc_normalization():
    nfd = unicodedata.normalize("NFD", "Điều 5. Học phí")
    assert normalize_text(nfd) == unicodedata.normalize("NFC", "Điều 5. Học phí")

def test_repeated_header_footer_removed(sample):
    path, _ = sample
    for c in chunk_pdf(path):
        assert "TRƯỜNG ĐẠI HỌC MẪU" not in c.text and "Trang " not in c.text

def test_deterministic_chunk_id(sample):
    path, _ = sample
    assert [c.chunk_id for c in chunk_pdf(path)] == [c.chunk_id for c in chunk_pdf(path)]

# --- PDF scan ---
def test_scanned_pdf_raises_and_directory_skips(tmp_path):
    build_sample_pdf(tmp_path / "ok.pdf")
    doc = fitz.open(); doc.new_page(); doc.save(tmp_path / "scan.pdf")
    with pytest.raises(PdfNoTextError):
        chunk_pdf(tmp_path / "scan.pdf")
    chunks, skipped = chunk_directory(tmp_path)
    assert skipped == ["scan.pdf"] and {c.source_file for c in chunks} == {"ok.pdf"}
```

- [ ] **Step 2: Chạy để xác nhận FAIL**

Run: `.venv\Scripts\python -m pytest tests/ingestion/test_chunker.py -v`
Expected: FAIL — `ImportError: cannot import name 'chunk_pdf'`

- [ ] **Step 3: Implement `src/ingestion/chunker.py`**

Pipeline cho mỗi file (thứ tự cố định):
1. `page.get_text("text")` cho từng trang → `normalize_text` (NFC). Toàn file rỗng → `PdfNoTextError(f"{name}: không có lớp chữ (PDF scan?) — cần OCR")`.
2. **Lọc header/footer:** nếu file có ≥ 3 trang có chữ, xét dòng đầu và dòng cuối mỗi trang sau khi thay chữ số bằng `#` (để `Trang 1`, `Trang 2` thành cùng một mẫu); mẫu nào xuất hiện ở ≥ 60% số trang thì loại các dòng đó.
3. **Tách theo Điều** bằng `split_by_article(page_text, carried=current_article)`, cập nhật `current_article` = article của đoạn cuối cùng trên trang (mang sang trang sau).
4. **Cắt chunk trong từng đoạn:** `prefix = f"{article}: "` nếu có article, ngược lại `""`; `body_size = chunk_size - len(prefix)`; kiểm tra `body_size >= 100` và `chunk_overlap < body_size` nếu không → `ValueError`. Dùng `RecursiveCharacterTextSplitter(chunk_size=body_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ". ", " ", ""])` → `text = prefix + body`.
5. `chunk_index` đếm liên tục trên trang qua mọi đoạn; `chunk_id = uuid5(NAMESPACE_URL, f"{source_file}:{page_number}:{chunk_index}")`.
6. `chunk_size`/`chunk_overlap` là `None` → đọc `CHUNK_SIZE`/`CHUNK_OVERLAP` từ env (mặc định 800/150).

CLI: `python -m src.ingestion.chunker <pdf_hoặc_thư_mục>` in tổng số trang, số chunk, danh sách Điều nhận diện được, file bị bỏ qua, và 3 chunk mẫu kèm metadata.

Hạn chế đã biết (ghi vào docstring, không xử lý ở Tuần 1): tên Điều bị xuống dòng trong PDF thật chỉ lấy phần trên dòng đầu; Task 3 sẽ kiểm chứng trên PDF thật.

- [ ] **Step 4: Chạy test PASS**

Run: `.venv\Scripts\python -m pytest tests/ingestion/test_chunker.py -v`
Expected: tất cả PASS (6 ca regex + 3 ca chunk_size + 13 ca khác).

- [ ] **Step 5: Critic review**

Gọi subagent **Critic** đọc `chunker.py` + `test_chunker.py` theo checklist `AGENTS.md` và Review Focus mục 1–3, 5. `REJECTED` → sửa, tối đa 3 vòng.

*(Dừng cuối Task 2: giải thích thuật toán tách Điều → cắt chunk, NFC, UUIDv5, regex, kết quả Critic, lệnh tự chạy; chờ "tiếp".)*

---

### Task 3: Kiểm tra Chunker với PDF quy chế thật

**Mục tiêu:** Chạy `chunker.py` trên **Thông tư 08/2021/TT-BGDĐT** và/hoặc **quy chế đào tạo của trường bạn**, trước khi đưa vào Qdrant, để phát hiện sớm lỗi font tiếng Việt, bảng biểu, header/footer đặc thù, tên Điều xuống dòng.

**Files:**
- Người dùng bỏ PDF thật vào `data/pdfs/` (tên không dấu, không khoảng trắng, vd. `tt08_2021_quy_che_dao_tao.pdf`).
- Modify (nếu cần): `src/ingestion/chunker.py`, `tests/ingestion/test_chunker.py`.

- [ ] **Step 1: Kiểm tra PDF có lớp chữ** — mở file, thử bôi đen một dòng; không bôi được = bản scan → tìm bản khác.

- [ ] **Step 2: Chạy CLI**

Run: `.venv\Scripts\python -m src.ingestion.chunker data/pdfs/`
Cùng xem 4 điểm:
1. Có file nào bị báo PDF scan không?
2. Tiếng Việt có ký tự lạ (`µ`, `¸`, `®` — dấu hiệu font TCVN3/VNI cũ) không?
3. Danh sách Điều có đủ và đúng tên (đối chiếu mục lục văn bản) không?
4. Header/footer lặp đã sạch chưa?

- [ ] **Step 3: Nếu phát hiện vấn đề** → hỏi người dùng hướng xử lý trước, rồi mới sửa `chunker.py` và thêm test tái hiện đúng mẫu lỗi đó (dùng chuỗi trích từ PDF thật).

*(Dừng cuối Task 3: cùng xem kết quả trên PDF thật; chờ "tiếp".)*

---

### Task 4: Embedder + Indexer (Qdrant named vectors, local mode) (Có Critic Review)

**Files:**
- Create: `src/ingestion/embedder.py`, `tests/ingestion/fakes.py`, `tests/ingestion/test_indexer.py`
- Modify: `src/ingestion/indexer.py`

**Interfaces:**
- Consumes: `Chunk`, `chunk_directory` (Task 2).
- Produces:
  ```python
  # src/ingestion/embedder.py
  class Embedder(Protocol):
      """Bất kỳ object nào có thuộc tính dense_dim và hàm embed_dense đều dùng được làm embedder."""
      dense_dim: int
      def embed_dense(self, texts: list[str]) -> list[list[float]]: ...
          # trả đúng len(texts) vector, mỗi vector dài dense_dim, đã chuẩn hóa L2

  class BgeM3Embedder:
      dense_dim = 1024
      def __init__(self, model_name: str = "BAAI/bge-m3", batch_size: int = 16) -> None
          # import sentence_transformers BÊN TRONG __init__ (test không phải load torch)
      def embed_dense(self, texts: list[str]) -> list[list[float]]
          # SentenceTransformer.encode(texts, batch_size=..., normalize_embeddings=True)

  # tests/ingestion/fakes.py
  class FakeEmbedder:            # dense_dim = 8; vector tất định từ sha256(text); ghi lại số lần gọi
  class FailingEmbedder:         # dense_dim = 8; embed_dense luôn raise RuntimeError("embed failed")

  # src/ingestion/indexer.py
  DENSE_VECTOR_NAME = "dense"
  SPARSE_VECTOR_NAME = "sparse"

  @dataclass(frozen=True)
  class Settings:
      qdrant_url: str; qdrant_api_key: str | None; qdrant_path: str | None
      collection: str; embedding_model: str; chunk_size: int; chunk_overlap: int

  def load_settings() -> Settings                      # load_dotenv(); chuỗi rỗng -> None
  def get_client(settings: Settings) -> QdrantClient   # qdrant_path có giá trị -> QdrantClient(path=...), ngược lại url
  def ensure_collection(client: QdrantClient, name: str, dense_dim: int, recreate: bool = False) -> None
      # vectors_config={DENSE_VECTOR_NAME: VectorParams(size=dense_dim, distance=Distance.COSINE)}
      # sparse_vectors_config={SPARSE_VECTOR_NAME: SparseVectorParams()}
      # tạo payload index keyword cho "source_file" (để xóa theo file nhanh)
  def delete_file_chunks(client: QdrantClient, collection: str, source_file: str) -> None
  def index_chunks(client: QdrantClient, chunks: list[Chunk], embedder: Embedder,
                   collection: str, upsert_batch_size: int = 64) -> int
  def main(argv: list[str] | None = None) -> int       # --pdf-dir (mặc định data/pdfs), --recreate, --local
  ```

- [ ] **Step 1: Viết test thất bại — `tests/ingestion/test_indexer.py`**

```python
import dataclasses, pytest
from uuid import uuid5, NAMESPACE_URL
from qdrant_client import QdrantClient
from src.ingestion.chunker import Chunk
from src.ingestion import indexer
from src.ingestion.indexer import ensure_collection, index_chunks, DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from tests.ingestion.fakes import FakeEmbedder, FailingEmbedder

def mk(source, n):
    return [Chunk(chunk_id=str(uuid5(NAMESPACE_URL, f"{source}:{i + 1}:0")),
                  source_file=source, page_number=i + 1, article=f"Điều {i+1}. X",
                  chunk_index=0, text=f"Điều {i+1}. X: nội dung {source} {i}") for i in range(n)]

@pytest.fixture
def client():
    c = QdrantClient(":memory:"); ensure_collection(c, "t", FakeEmbedder.dense_dim); return c

def count(c, source=None):
    pts, _ = c.scroll("t", limit=1000, with_payload=True)
    return sum(1 for p in pts if source is None or p.payload["source_file"] == source)

def test_collection_has_dense_and_sparse(client):
    info = client.get_collection("t").config.params
    assert DENSE_VECTOR_NAME in info.vectors and SPARSE_VECTOR_NAME in info.sparse_vectors

def test_payload_preserves_full_schema(client):
    index_chunks(client, mk("a.pdf", 3), FakeEmbedder(), "t")
    pts, _ = client.scroll("t", limit=10, with_payload=True)
    for p in pts:
        assert set(p.payload) == {"source_file", "page_number", "article", "chunk_index", "text"}
        assert isinstance(p.payload["page_number"], int)

def test_reindex_file_replaces_only_that_file(client):
    index_chunks(client, mk("a.pdf", 5) + mk("b.pdf", 2), FakeEmbedder(), "t")
    index_chunks(client, mk("a.pdf", 3), FakeEmbedder(), "t")
    assert count(client, "a.pdf") == 3 and count(client, "b.pdf") == 2

def test_reindex_same_data_no_duplicates(client):
    ch = mk("a.pdf", 4)
    index_chunks(client, ch, FakeEmbedder(), "t"); index_chunks(client, ch, FakeEmbedder(), "t")
    assert count(client) == 4

def test_embed_failure_keeps_old_data(client):
    index_chunks(client, mk("a.pdf", 5), FakeEmbedder(), "t")
    with pytest.raises(RuntimeError):
        index_chunks(client, mk("a.pdf", 3), FailingEmbedder(), "t")
    assert count(client, "a.pdf") == 5                     # embed lỗi TRƯỚC khi xóa → dữ liệu cũ nguyên vẹn

def test_query_returns_page_metadata(client):
    ch = mk("a.pdf", 5); emb = FakeEmbedder(); index_chunks(client, ch, emb, "t")
    hit = client.query_points("t", query=emb.embed_dense([ch[3].text])[0], using=DENSE_VECTOR_NAME,
                              limit=1, with_payload=True).points[0]
    assert hit.payload["page_number"] == 4 and hit.payload["source_file"] == "a.pdf"

def test_local_mode_persists(tmp_path):
    s = dataclasses.replace(indexer.load_settings(), qdrant_path=str(tmp_path / "q"))
    c = indexer.get_client(s); ensure_collection(c, "t", 8); index_chunks(c, mk("a.pdf", 2), FakeEmbedder(), "t"); c.close()
    assert indexer.get_client(s).count("t").count == 2

def test_cli_unreachable_qdrant_message(tmp_path, monkeypatch, capsys):
    from src.ingestion.sample_pdf import build_sample_pdf
    build_sample_pdf(tmp_path / "a.pdf")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:1"); monkeypatch.setenv("QDRANT_PATH", "")
    loaded = []
    monkeypatch.setattr(indexer, "BgeM3Embedder", lambda *a, **k: loaded.append(1) or FakeEmbedder())
    assert indexer.main(["--pdf-dir", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "Không kết nối được Qdrant" in err and "--local" in err
    assert loaded == []                                    # báo lỗi trước khi load model
```

- [ ] **Step 2: Chạy để xác nhận FAIL**

Run: `.venv\Scripts\python -m pytest tests/ingestion/test_indexer.py -v`
Expected: FAIL — `ModuleNotFoundError: tests.ingestion.fakes` / `ImportError`

- [ ] **Step 3: Implement `embedder.py`, `fakes.py`, `indexer.py`**

`index_chunks` — thứ tự bắt buộc, **theo từng `source_file`**:
1. Gom chunk theo `source_file`.
2. **Embed toàn bộ** chunk của file: `vectors = embedder.embed_dense([c.text for c in file_chunks])`; kiểm tra `len(vectors) == len(file_chunks)`. Lỗi ở bước này → exception bay ra, chưa đụng gì tới Qdrant.
3. `delete_file_chunks(client, collection, source_file)` — `FilterSelector(Filter(must=[FieldCondition(key="source_file", match=MatchValue(value=source_file))]))`, `wait=True`.
4. Upsert theo lô `upsert_batch_size`: `PointStruct(id=c.chunk_id, vector={DENSE_VECTOR_NAME: v}, payload={source_file, page_number, article, chunk_index, text})`, `wait=True`.
5. Trả tổng số point đã upsert.

*Giới hạn còn lại (ghi vào docstring + README):* Qdrant không có transaction, nên nếu upsert lỗi **sau** khi xóa (vd. mất kết nối giữa chừng) thì file đó tạm thiếu dữ liệu — chạy lại lệnh index là khôi phục, vì ID cố định.

`main`: `load_settings()` → `--local` ép `qdrant_path="./qdrant_local"` → `chunk_directory` (cảnh báo file bị bỏ qua ra stderr; không có chunk → stderr + `return 1`) → `get_client` + `client.get_collections()` trong `try/except Exception` → lỗi thì stderr:
`Không kết nối được Qdrant tại {url}. Bật Docker và chạy "docker compose up -d", hoặc chạy lại với --local (hoặc đặt QDRANT_PATH trong .env) để dùng Qdrant local mode.` và `return 1` → **sau đó mới** `BgeM3Embedder(settings.embedding_model)` → `ensure_collection(..., embedder.dense_dim, recreate)` → `index_chunks` → in `Đã index {n} chunks từ {k} file vào '{collection}'` → `return 0`.

- [ ] **Step 4: Chạy test PASS**

Run: `.venv\Scripts\python -m pytest tests/ingestion -v`
Expected: toàn bộ test Task 1, 2, 4 PASS; không có dòng nào tải model (chạy dưới ~30 giây).

- [ ] **Step 5: Critic review**

Gọi subagent **Critic** đọc `embedder.py`, `indexer.py`, `test_indexer.py` theo checklist `AGENTS.md` và Review Focus mục 1, 4, 5. `REJECTED` → sửa, tối đa 3 vòng.

*(Dừng cuối Task 4: giải thích named vectors, Protocol, thứ tự embed → xóa → upsert, Docker vs local mode, kết quả Critic, lệnh tự chạy; chờ "tiếp".)*

---

### Task 5: Chạy tích hợp End-to-End & viết `README.md`

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Index dữ liệu thật** (lần đầu sẽ tải BGE-M3 ~2.3 GB)

```powershell
Copy-Item .env.example .env
# Cách 1 — Docker
docker compose up -d
.venv\Scripts\python -m src.ingestion.indexer --recreate
# Cách 2 — không có Docker
.venv\Scripts\python -m src.ingestion.indexer --local --recreate
```
Expected: `Đã index <N> chunks từ <k> file vào 'regulations'`.

- [ ] **Step 2: Truy vấn thử** — lệnh Python ngắn lấy top-3 cho 2–3 câu hỏi thật, in `score`, `source_file`, `page_number`, `article`, 150 ký tự đầu của `text`; người dùng mở PDF đối chiếu số trang.

- [ ] **Step 3: Chạy lại không `--recreate`** → số point không đổi (xác nhận không trùng).

- [ ] **Step 4: Viết `README.md`**

1. Bài toán & kiến trúc (luồng trong `AGENTS.md`).
2. Quyết định thiết kế Tuần 1:
   - Chunk theo trang + tách theo Điều trước khi cắt + tiền tố `"Điều X. Tên: "`.
   - Chuẩn hóa NFC cho tiếng Việt.
   - Named vectors `dense` + `sparse` khai báo sẵn.
   - Thứ tự embed → xóa → upsert và giới hạn còn lại (không có transaction).
   - **`chunk_size=800`, `chunk_overlap=150` ký tự (tính cả tiền tố) là GIÁ TRỊ KHỞI ĐẦU, chưa có số liệu kiểm chứng.** Tuần 3 sẽ so 400 / 800 / 1200 bằng Recall@5 trên bộ 50 câu đánh giá rồi mới chốt. Không viết như thể đã được chứng minh.
3. Cài đặt & chạy (Docker và local mode).
4. Chạy test (`pytest`).

*(Dừng cuối Task 5: người dùng kiểm tra toàn bộ Tuần 1 và quyết định khi nào commit/push.)*
