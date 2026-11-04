# Hợp đồng nghiệm thu: Task 4 — Module Indexer & Embedder (Named Vectors, Local Mode & Cập Nhật Dữ Liệu An Toàn)

> **File:** `docs/contracts/task-4.md`  
> **Các bên tham gia:** Builder (Thiết kế & Lập trình) & Critic (Phản biện & Kiểm thử độc lập)  
> **Tài liệu căn chiếu:** `AGENTS.md`, `docs/superpowers/plans/2026-11-02-week1-ingestion.md`

---

## 1. Mục tiêu của Task 4
Xây dựng module Embedder và Indexer để đưa các đoạn văn bản (chunks) từ Task 2/3 vào Vector Database (Qdrant), đáp ứng:
1. **Kiến trúc Named Vectors:** Khai báo đồng thời `dense` (kích thước theo embedder, cosine) và `sparse` (sẵn sàng cho BM25 Tuần 3) ngay từ đầu.
2. **Embedder linh hoạt:** Định nghĩa giao diện `Embedder(Protocol)` cho phép hoán đổi giữa `FakeEmbedder` (kiểm thử nhanh không tải model) và `BgeM3Embedder` (`BAAI/bge-m3`, tải lười `sentence_transformers`).
3. **Cập nhật dữ liệu an toàn & chống mất mát (Zero Downtime / Safe Re-index):**
   - Embed theo lô toàn bộ chunk của từng file trước.
   - **Upsert chunk mới trước:** ghi đè ngay các point trùng ID. Nếu upsert lỗi, dữ liệu cũ vẫn còn nguyên.
   - **Xóa các point cũ có ID không còn trong tập mới:** giải quyết triệt để trường hợp file bị thu nhỏ (ví dụ từ 5 trang xuống 3 trang).
   - Point ID là UUIDv5 cố định được lấy từ `Chunk.chunk_id`, chống nhân bản dữ liệu khi chạy lại nhiều lần.
   - File không còn chunk nào: bỏ qua, cảnh báo, **tuyệt đối không xóa dữ liệu cũ**.
4. **Kiểm tra tương thích cấu trúc Collection:**
   - Nếu collection đã tồn tại nhưng kích thước vector `dense` khác với embedder hiện tại: báo lỗi rõ ràng (nêu rõ số chiều hiện tại vs số chiều của embedder) và gợi ý dùng cờ `--recreate`.
5. **Hỗ trợ 2 chế độ Qdrant & Fast-Fail:**
   - Chế độ máy chủ Docker / Cloud qua URL + API Key.
   - Chế độ cục bộ (local mode) lưu ra thư mục đĩa (khi Docker không khả dụng hoặc kiểm thử nhanh).
   - Kiểm tra kết nối Qdrant trước khi nạp model nặng: nếu chưa bật, in hướng dẫn `docker compose up -d` hoặc `--local`, thoát với mã khác 0; đảm bảo `BgeM3Embedder` không bị khởi tạo lãng phí tài nguyên.
6. **Báo cáo toàn diện:** Đưa danh sách trang bị bỏ qua (`SkippedPage`) và file scan bị bỏ qua (`skipped_files`) vào tổng kết cuối cùng của lượt index.
7. **Không tải model thật trong unit test:** 100% unit tests chạy với `FakeEmbedder` và Qdrant in-memory (`:memory:`), thời gian chạy < 5 giây.

---

## 2. Đề xuất của Builder (Checklist "Xong là gì")

### 2.1. Giao diện & Schema dữ liệu

#### A. Module `src/ingestion/embedder.py`
```python
class Embedder(Protocol):
    dense_dim: int
    def embed_dense(self, texts: list[str]) -> list[list[float]]:
        """Trả về đúng len(texts) vectors, mỗi vector có độ dài dense_dim, đã chuẩn hóa L2."""
        ...

class BgeM3Embedder:
    dense_dim: int = 1024
    def __init__(self, model_name: str = "BAAI/bge-m3", batch_size: int = 16) -> None:
        """Tải lười sentence_transformers bên trong __init__ để test không bị kéo torch."""
        ...
    def embed_dense(self, texts: list[str]) -> list[list[float]]: ...
```

#### B. Module kiểm thử `tests/ingestion/fakes.py`
```python
class FakeEmbedder:
    dense_dim: int = 8
    call_count: int = 0
    def embed_dense(self, texts: list[str]) -> list[list[float]]:
        """Vector tất định băm từ sha256(text), chuẩn hóa L2, tăng call_count."""
        ...

class FailingEmbedder:
    dense_dim: int = 8
    def embed_dense(self, texts: list[str]) -> list[list[float]]:
        """Luôn ném RuntimeError('embed failed') để kiểm thử rollback/an toàn dữ liệu."""
        ...
```

#### C. Module `src/ingestion/indexer.py`
- Tên vector hằng số:
  - `DENSE_VECTOR_NAME = "dense"`
  - `SPARSE_VECTOR_NAME = "sparse"`
- Cấu hình hệ thống:
  ```python
  @dataclass(frozen=True)
  class Settings:
      qdrant_url: str
      qdrant_api_key: str | None
      qdrant_path: str | None
      collection: str
      embedding_model: str
      chunk_size: int
      chunk_overlap: int
  ```
- Hàm khởi tạo client:
  - `get_client(settings: Settings) -> QdrantClient`: Ưu tiên `qdrant_path` (local mode) nếu được cấu hình, ngược lại dùng `qdrant_url` và `qdrant_api_key`.
- Khởi tạo Collection với Named Vectors & Kiểm tra tương thích:
  - `ensure_collection(client: QdrantClient, name: str, dense_dim: int, recreate: bool = False) -> None`:
    - Nếu `recreate=True`: xóa collection cũ nếu có và tạo mới.
    - Nếu collection đã tồn tại: kiểm tra `current_dim == dense_dim`. Nếu lệch số chiều: ném `ValueError(f"Collection '{name}' có dense vector dim={current_dim}, khác với embedder dim={dense_dim}. Chạy lại với --recreate để tạo lại collection.")`.
    - Nếu chưa tồn tại: tạo mới với `vectors_config={DENSE_VECTOR_NAME: VectorParams(size=dense_dim, distance=Distance.COSINE)}` và `sparse_vectors_config={SPARSE_VECTOR_NAME: SparseVectorParams()}`.
    - Tạo Payload Index kiểu `keyword` cho trường `source_file` để tối ưu truy vấn theo file.
- Quy trình cập nhật an toàn (`index_chunks`):
  - `index_chunks(client: QdrantClient, chunks: list[Chunk], embedder: Embedder, collection: str, embed_batch_size: int = 16, upsert_batch_size: int = 64) -> int`:
    - Gom chunk theo từng `source_file`.
    - Với mỗi file:
      1. Nếu file không có chunk nào: in cảnh báo ra `sys.stderr`, bỏ qua, **không xóa dữ liệu cũ**.
      2. **Embed toàn bộ chunk theo lô (`embed_batch_size`):** nếu embed ném ngoại lệ -> dừng ngay, dữ liệu cũ trên Qdrant không bị chạm tới.
      3. **Upsert chunk mới trước theo lô (`upsert_batch_size`):** ghi đè các point có cùng chunk_id. Nếu upsert ném ngoại lệ -> dừng ngay, các point cũ chưa bị xóa.
      4. **Xóa các point mồ côi (stale points):** tìm tất cả point của file đó trong collection có `id` không nằm trong tập ID của các chunk mới (ví dụ file bị rút ngắn từ 5 trang xuống 3 trang), và xóa bằng `client.delete(points=stale_ids, wait=True)`.
    - Trả về tổng số chunk đã upsert thành công.
- CLI chạy dòng lệnh:
  - `main(argv: list[str] | None = None) -> int`:
    - Tham số: `--pdf-dir` (mặc định `data/pdfs`), `--recreate`, `--local`.
    - Kiểm tra kết nối Qdrant bằng `client.get_collections()` trước. Nếu lỗi: in ra `sys.stderr` thông báo rõ ràng kèm hướng dẫn bật Docker (`docker compose up -d`) hoặc dùng `--local`, và thoát với mã `1`. **Tuyệt đối không khởi tạo `BgeM3Embedder`**.
    - Cắt chunk từ thư mục, thu thập `skipped_files` và `skipped_pages`.
    - Khởi tạo embedder, `ensure_collection`, `index_chunks`.
    - In báo cáo tổng kết gồm: số file thành công, tổng số chunk index, danh sách file scan bị bỏ qua, và danh sách trang scan/trắng bị bỏ qua.

---

## 3. Ý kiến phản biện & Bổ sung của Critic (Subagent Red-Team)

Sau khi rà soát kỹ thuật độc lập và đối chiếu với `AGENTS.md` cùng `docs/superpowers/plans/2026-11-02-week1-ingestion.md`, Critic đưa ra các phản biện khắt khe và yêu cầu bắt buộc Builder phải tuân thủ các điều kiện sau trong thiết kế và cài đặt:

### 3.1. Cơ chế Scroll phân trang & Chống xóa nhầm khi dọn dẹp point mồ côi (Stale Points)
- **Nguy cơ bỏ sót point mồ côi (Pagination Risk):** `client.scroll()` chỉ trả về tối đa `limit` points kèm `next_page_offset`. Nếu một file PDF quy chế lớn có số chunk vượt quá `limit` (ví dụ 150 chunks), nếu không có vòng lặp phân trang (`while next_page_offset is not None`), các point cũ ở các trang sau sẽ **bị bỏ sót hoàn toàn**, trở thành dữ liệu rác vĩnh viễn trong Qdrant.
  - **Yêu cầu bắt buộc:** Bắt buộc cài đặt hàm thu thập point ID cũ theo vòng lặp scroll phân trang đầy đủ cho đến khi `next_page_offset is None`.
- **Tối ưu tài nguyên mạng & RAM:** Khi scroll để lấy ID xóa point cũ, BẮT BUỘC đặt `with_payload=False` và `with_vectors=False` để không kéo vector (1024 floats) và text về bộ nhớ, tránh nghẽn mạng và OOM.
- **Bảo vệ chống xóa nhầm file khác:** Điều kiện filter khi scroll PHẢI là `models.Filter(must=[models.FieldCondition(key="source_file", match=models.MatchValue(value=source_file))])`. Tuyệt đối không xóa point không thuộc `source_file` đang xử lý.
- **Chuẩn hóa kiểu ID:** Phải ép kiểu `str(record.id)` khi so sánh với `str(chunk.chunk_id)` (UUIDv5) để tránh lỗi so sánh khác kiểu dữ liệu giữa SDK và model.
- **File không có chunk (rỗng / toàn ảnh):** Nếu danh sách chunk của file rỗng (`len(chunks) == 0`), hàm phải cảnh báo ra `sys.stderr` và **return ngay**, tuyệt đối KHÔNG gọi scroll và KHÔNG gọi delete, giữ nguyên 100% dữ liệu cũ.

### 3.2. Kiểm tra tương thích Collection & Xử lý Named Vectors an toàn
- **Lỗi crash khi kiểm tra cấu trúc `vectors`:** Trong `qdrant-client==1.12.1`, khi dùng named vectors, `collection_info.config.params.vectors` là `dict[str, VectorParams]`. Nhưng nếu collection cũ được tạo dạng unnamed vector (single vector), thuộc tính này là một đối tượng `VectorParams` (không phải dict). Truy cập thẳng `vectors["dense"]` sẽ gây `TypeError: 'VectorParams' object is not subscriptable`, hoặc `vectors.size` sẽ gây `AttributeError: 'dict' object has no attribute 'size'`.
  - **Yêu cầu bắt buộc:** `ensure_collection` phải kiểm tra an toàn:
    1. Kiểm tra `client.collection_exists(name)`. Nếu chưa có -> tạo mới với cả `vectors_config={"dense": VectorParams(size=dense_dim, distance=Distance.COSINE)}` và `sparse_vectors_config={"sparse": SparseVectorParams()}`.
    2. Nếu đã tồn tại: kiểm tra `isinstance(vectors, dict)` và `"dense" in vectors`. Nếu không thỏa mãn (unnamed vector hoặc thiếu `"dense"`): ném `ValueError` thông báo collection không đúng chuẩn named vectors, yêu cầu chạy lại với `--recreate`.
    3. Nếu có `"dense"`: so sánh `vectors["dense"].size == dense_dim`. Nếu lệch: ném `ValueError(f"Collection '{name}' có dense vector dim={current_dim}, khác với embedder dim={dense_dim}. Chạy lại với --recreate để tạo lại collection.")`.
    4. Kiểm tra sự tồn tại của named vector `"sparse"` trong `sparse_vectors`. Nếu thiếu `"sparse"`, ném `ValueError` yêu cầu `--recreate` (để tránh Tuần 3 bị lỗi khi tích hợp BM25).

### 3.3. Quy trình Batching & An toàn dữ liệu khi có lỗi
- **Tuyệt đối không vừa embed vừa upsert (No Streaming Upsert):** Phải embed **xong toàn bộ 100% chunks của một file** rồi mới bắt đầu batch upsert đầu tiên. Nếu có lỗi khi đang embed (OOM, timeout API), hàm phải dừng ngay, không có point nào của file đó bị ghi đè hay xóa dở dang.
- **Upsert trước -> Xóa sau có bảo vệ:**
  - Nếu bất kỳ batch upsert nào gặp lỗi mạng hoặc crash: hàm phải raise lỗi ngay, **KHÔNG ĐƯỢC CHẠY BƯỚC XÓA STALE POINTS**.
  - Mọi lệnh `client.upsert()` và `client.delete()` đều PHẢI truyền `wait=True` để đảm bảo tính nhất quán dữ liệu, chống race condition trong kiểm thử và vận hành.

### 3.4. Thứ tự Fast-Fail & Ngăn nạp thư viện nặng
- **Kiểm tra kết nối Qdrant đầu tiên trong `main()`:**
  - `QdrantClient` có cơ chế lazy connection, nên việc gọi `get_client()` không phát hiện được server chết.
  - Hàm `main()` BẮT BUỘC phải gọi `client.get_collections()` ngay sau khi tạo client, TRƯỚC KHI thực hiện chunk PDF và TRƯỚC KHI khởi tạo `BgeM3Embedder`.
  - Nếu `client.get_collections()` ném ngoại lệ: in hướng dẫn ra `sys.stderr` (`docker compose up -d` hoặc `--local`), thoát với mã `1`.
- **Tải lười `sentence_transformers` (Zero-Torch Unit Tests):**
  - Trong `src/ingestion/embedder.py`, lệnh `from sentence_transformers import SentenceTransformer` PHẢI nằm bên trong phương thức `BgeM3Embedder.__init__`. Tuyệt đối không import ở cấp module (top-level).
  - Đảm bảo 100% unit tests chạy dưới 5 giây và không làm xuất hiện `torch` hay `sentence_transformers` trong `sys.modules`.

### 3.5. Bổ sung Danh mục Test Case bắt buộc (`tests/ingestion/test_indexer.py`)
Bộ test của Builder bắt buộc phải có đủ 14 test cases sau:
1. `test_collection_has_dense_and_sparse`: Collection tạo mới có đủ named vector `dense` (Cosine) và `sparse`.
2. `test_payload_preserves_full_schema`: Đủ 5 trường metadata, đúng kiểu dữ liệu (`page_number` là int).
3. `test_reindex_file_replaces_only_that_file`: Re-index `a.pdf` thì chỉ cập nhật `a.pdf`, các point của `b.pdf` giữ nguyên 100%.
4. `test_reindex_same_data_no_duplicates`: Chạy re-index cùng dữ liệu nhiều lần không làm tăng số lượng point (UUIDv5 cố định).
5. `test_embed_failure_keeps_old_data`: Embed bị lỗi giữa chừng thì dữ liệu cũ trên Qdrant không suy suyển.
6. `test_upsert_failure_prevents_stale_deletion`: Giả lập lỗi ở bước upsert, xác nhận không gọi lệnh xóa point cũ.
7. `test_scroll_pagination_cleans_all_stale_points`: Kiểm tra vòng lặp phân trang scroll xóa sạch stale points kể cả khi số stale points lớn hơn scroll limit.
8. `test_dimension_mismatch_raises_informative_error`: Kiểm tra lệch chiều vector ném `ValueError` kèm thông báo rõ ràng và cờ `--recreate`.
9. `test_ensure_collection_recreate_resets_dimension`: Dùng `recreate=True` tái tạo collection thành công khi đổi embedder.
10. `test_invalid_named_vectors_schema_raises_error`: Collection cũ không có named vector `"dense"` / `"sparse"` bị phát hiện và yêu cầu `--recreate`.
11. `test_empty_chunks_preserves_old_data`: File không có chunk nào thì bỏ qua, không xóa dữ liệu cũ.
12. `test_query_returns_page_metadata`: `query_points` trả về đúng metadata `page_number`, `source_file`, `article`.
13. `test_local_mode_persists`: Qdrant local mode ghi dữ liệu bền vững xuống thư mục đĩa.
14. `test_cli_fast_fail_and_no_heavy_imports`: CLI kiểm tra Qdrant trước, báo lỗi dễ hiểu, thoát mã 1 và không khởi tạo `BgeM3Embedder`.

---

## 4. Bảng kiểm tra nghiệm thu chính thức (Acceptance Checklist)

Builder chỉ được tuyên bố hoàn thành Task 4 khi toàn bộ các tiêu chí dưới đây đạt:

- [ ] **Tiêu chí 1 (Named Vectors đầy đủ):** Collection Qdrant được tạo với cả 2 named vectors: `dense` (kích thước theo embedder, cosine) và `sparse` (SparseVectorParams), sẵn sàng cho Hybrid Search Tuần 3.
- [ ] **Tiêu chí 2 (Payload Index):** Tự động tạo payload index kiểu `keyword` cho trường `source_file` để tối ưu truy vấn lọc và dọn dẹp.
- [ ] **Tiêu chí 3 (Payload Schema toàn vẹn):** Mọi point chứa đúng 5 trường metadata: `source_file` (str), `page_number` (int), `article` (str | None), `chunk_index` (int), `text` (str).
- [ ] **Tiêu chí 4 (Thứ tự Upsert trước -> Xóa thừa sau):** Embed toàn bộ chunk của file trước; sau đó upsert chunk mới theo lô (`wait=True`). Nếu có lỗi ở bước embed hoặc upsert thì dừng ngay, giữ nguyên dữ liệu cũ, tuyệt đối không gọi xóa stale points.
- [ ] **Tiêu chí 5 (Dọn dẹp Stale Points có Scroll phân trang an toàn):** Khi file bị rút ngắn (ví dụ 5 trang thành 3 trang), dọn dẹp toàn bộ point mồ côi bằng scroll có phân trang (vòng lặp `offset` cho đến hết), đặt `with_payload=False` và `with_vectors=False`, filter chính xác theo `source_file`, không xóa nhầm point của file khác.
- [ ] **Tiêu chí 6 (Chống nhân bản dữ liệu):** Point ID là UUIDv5 từ `Chunk.chunk_id`; chạy index lại nhiều lần trên cùng dữ liệu không làm tăng số lượng point.
- [ ] **Tiêu chí 7 (File rỗng / không chunk):** File không có chunk nào thì bỏ qua, in cảnh báo, không xóa dữ liệu cũ đã có của file đó trong collection.
- [ ] **Tiêu chí 8 (Kiểm tra lệch số chiều vector & Cấu trúc Named Vectors):** Phát hiện collection cũ không có named vector hoặc lệch kích thước `dense`: báo lỗi rõ ràng nêu số chiều hiện tại vs mới, kèm hướng dẫn `--recreate`.
- [ ] **Tiêu chí 9 (Xử lý theo lô):** Cả bước embed và upsert đều hỗ trợ xử lý theo lô (`embed_batch_size`, `upsert_batch_size`), mọi lệnh upsert/delete đều dùng `wait=True`.
- [ ] **Tiêu chí 10 (Fast-Fail khi Qdrant sập):** Khi Qdrant không bật, CLI kiểm tra kết nối ngay đầu hàm `main()` qua `client.get_collections()`, báo lỗi rõ ràng hướng dẫn `docker compose up -d` hoặc `--local`, thoát mã `1`, và **chứng minh `BgeM3Embedder` không hề được khởi tạo**.
- [ ] **Tiêu chí 11 (Truy vấn trả đúng số trang):** Thực hiện `query_points` bằng vector dense trả về đúng `page_number`, `source_file` và `article`.
- [ ] **Tiêu chí 12 (Hỗ trợ Local Mode):** Cho phép chạy Qdrant local mode qua đường dẫn thư mục (`qdrant_path` hoặc `--local`), dữ liệu được ghi bền vững xuống đĩa.
- [ ] **Tiêu chí 13 (Báo cáo tổng kết toàn diện):** CLI in đầy đủ số file, số chunk, danh sách file scan bị bỏ qua và danh sách trang không có lớp chữ bị bỏ qua (`SkippedPage`).
- [ ] **Tiêu chí 14 (Tải lười & Zero-Torch Test):** `SentenceTransformer` được tải lười bên trong `BgeM3Embedder.__init__`; 100% unit tests chạy với `FakeEmbedder` và Qdrant in-memory (`:memory:`), thời gian chạy < 5 giây, không import `torch` ở top-level.
- [ ] **Tiêu chí 15 (Bao phủ kiểm thử & Mutation Testing):** Đạt tối thiểu 14 test cases trong `tests/ingestion/test_indexer.py`; Critic thực hiện kiểm thử đột biến thành công trên bản sao và duyệt `APPROVED`.
