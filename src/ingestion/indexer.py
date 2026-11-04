"""Đưa chunk vào Qdrant: collection named vectors (dense + sparse BM25), index toàn kho, CLI.

Thứ tự an toàn khi index lại:
1. Cắt chunk + tính vector cho TOÀN BỘ kho (lỗi embed → chưa đụng tới Qdrant).
2. Upsert theo lô (`wait=True`): ghi đè point cùng ID (ID cố định từ chunk_id → chạy lại không sinh trùng).
3. Chỉ khi upsert thành công mới xóa point thừa: ID không còn trong tập mới (file bị thu nhỏ, đổi tên, xóa),
   trừ point của file vẫn còn trong thư mục nhưng lượt này bị bỏ qua (`preserve_files`).
Qdrant không có transaction; nếu lỗi giữa bước 2, dữ liệu cũ vẫn còn và chạy lại lệnh là đủ.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from pathlib import Path
import sys

from qdrant_client import QdrantClient, models

from src.config import Settings, load_settings
from src.ingestion.chunker import Chunk, SkippedPage, chunk_directory
from src.ingestion.embedder import BgeM3Embedder, Embedder
from src.ingestion.sparse import average_doc_length, bm25_doc_vector, tokenize
from src.utils import setup_utf8_console

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
SCROLL_PAGE = 256
LOCAL_QDRANT_PATH = "./qdrant_local"


class QdrantUnavailableError(Exception):
    """Không kết nối được Qdrant."""


class CollectionSchemaError(Exception):
    """Collection đã tồn tại nhưng cấu trúc không khớp thiết kế (cần --recreate)."""


@dataclass(frozen=True)
class IndexReport:
    upserted: int
    deleted_stale: int
    files: list[str]


def get_client(settings: Settings) -> QdrantClient:
    """`qdrant_path` có giá trị → local mode (file), ngược lại kết nối server qua URL."""
    if settings.qdrant_path:
        return QdrantClient(path=settings.qdrant_path)
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


def check_connection(client: QdrantClient, where: str) -> None:
    """QdrantClient kết nối lười, nên phải gọi thử một lệnh để phát hiện server chưa bật."""
    try:
        client.get_collections()
    except Exception as exc:
        raise QdrantUnavailableError(where) from exc


def _recreate_hint(name: str) -> str:
    return f"Chạy lại indexer với --recreate để tạo lại collection '{name}' (dữ liệu cũ sẽ bị xóa)."


def ensure_collection(client: QdrantClient, name: str, dense_dim: int, recreate: bool = False) -> None:
    if recreate and client.collection_exists(name):
        client.delete_collection(name)

    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(size=dense_dim, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
        client.create_payload_index(
            collection_name=name,
            field_name="source_file",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        return

    params = client.get_collection(name).config.params
    vectors = params.vectors
    if not isinstance(vectors, dict) or DENSE_VECTOR_NAME not in vectors:
        raise CollectionSchemaError(
            f"Collection '{name}' không có named vector '{DENSE_VECTOR_NAME}'. {_recreate_hint(name)}"
        )
    current_dim = vectors[DENSE_VECTOR_NAME].size
    if current_dim != dense_dim:
        raise CollectionSchemaError(
            f"Collection '{name}' có dense vector dim={current_dim}, khác với embedder dim={dense_dim}. "
            f"{_recreate_hint(name)}"
        )
    if vectors[DENSE_VECTOR_NAME].distance != models.Distance.COSINE:
        raise CollectionSchemaError(
            f"Collection '{name}' dùng distance={vectors[DENSE_VECTOR_NAME].distance}, cần Cosine. "
            f"{_recreate_hint(name)}"
        )
    sparse = (params.sparse_vectors or {}).get(SPARSE_VECTOR_NAME)
    if sparse is None or sparse.modifier != models.Modifier.IDF:
        raise CollectionSchemaError(
            f"Collection '{name}' thiếu sparse vector '{SPARSE_VECTOR_NAME}' với modifier=IDF (cần cho BM25). "
            f"{_recreate_hint(name)}"
        )
    # Thiếu payload index thì tạo bù (thao tác idempotent; local mode không báo schema nên luôn gọi)
    schema = client.get_collection(name).payload_schema or {}
    if "source_file" not in schema:
        client.create_payload_index(
            collection_name=name,
            field_name="source_file",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )


def _payload(chunk: Chunk) -> dict:
    return {
        "source_file": chunk.source_file,
        "page_number": int(chunk.page_number),
        "article": chunk.article,
        "chunk_index": int(chunk.chunk_index),
        "text": chunk.text,
    }


def _existing_points(client: QdrantClient, collection: str) -> list[tuple[str, str | None]]:
    """(id, source_file) của mọi point, scroll phân trang, không kéo vector."""
    found: list[tuple[str, str | None]] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=SCROLL_PAGE,
            offset=offset,
            with_payload=["source_file"],
            with_vectors=False,
        )
        found.extend((str(p.id), (p.payload or {}).get("source_file")) for p in points)
        if offset is None:
            return found


def index_corpus(
    client: QdrantClient,
    chunks: list[Chunk],
    embedder: Embedder,
    collection: str,
    *,
    preserve_files: frozenset[str] = frozenset(),
    k1: float = 1.2,
    b: float = 0.75,
    upsert_batch_size: int = 64,
) -> IndexReport:
    """Index toàn bộ kho (avgdl tính trên mọi chunk), rồi dọn point thừa. Xem docstring module."""
    if not chunks:
        raise ValueError("Không có chunk nào để index; dữ liệu cũ được giữ nguyên.")

    # 1. Tính toàn bộ vector trước khi ghi gì vào Qdrant
    token_lists = [tokenize(c.text) for c in chunks]
    avgdl = average_doc_length(token_lists)
    sparse_vectors = [bm25_doc_vector(tokens, avgdl, k1=k1, b=b) for tokens in token_lists]
    dense_vectors = embedder.embed_dense([c.text for c in chunks])
    if len(dense_vectors) != len(chunks):
        raise RuntimeError(f"Embedder trả {len(dense_vectors)} vector cho {len(chunks)} chunk.")

    # 2. Upsert theo lô
    points = [
        models.PointStruct(
            id=chunk.chunk_id,
            vector={
                DENSE_VECTOR_NAME: dense,
                SPARSE_VECTOR_NAME: models.SparseVector(indices=sparse.indices, values=sparse.values),
            },
            payload=_payload(chunk),
        )
        for chunk, dense, sparse in zip(chunks, dense_vectors, sparse_vectors)
    ]
    for start in range(0, len(points), upsert_batch_size):
        client.upsert(collection_name=collection, points=points[start : start + upsert_batch_size], wait=True)

    # 3. Dọn point thừa (chỉ chạy khi mọi lô upsert đã thành công)
    new_ids = {str(c.chunk_id) for c in chunks}
    stale = [
        pid
        for pid, source in _existing_points(client, collection)
        if pid not in new_ids and source not in preserve_files
    ]
    for start in range(0, len(stale), SCROLL_PAGE):
        client.delete(
            collection_name=collection,
            points_selector=models.PointIdsList(points=stale[start : start + SCROLL_PAGE]),
            wait=True,
        )

    files = sorted({c.source_file for c in chunks})
    return IndexReport(upserted=len(points), deleted_stale=len(stale), files=files)


def main(argv: list[str] | None = None) -> int:
    setup_utf8_console()
    parser = argparse.ArgumentParser(description="Index PDF quy chế vào Qdrant (dense BGE-M3 + sparse BM25).")
    parser.add_argument("--pdf-dir", type=Path, default=Path("data/pdfs"))
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Xóa và tạo lại collection TRƯỚC khi embed; nếu embed lỗi sau đó, collection sẽ trống (chạy lại là đủ).",
    )
    parser.add_argument("--local", action="store_true", help=f"Dùng Qdrant local mode tại {LOCAL_QDRANT_PATH}.")
    parser.add_argument("--collection", default=None, help="Tên collection (mặc định QDRANT_COLLECTION).")
    args = parser.parse_args(argv)

    try:
        settings = load_settings()
    except ValueError as exc:
        print(f"Lỗi cấu hình (.env / biến môi trường): {exc}", file=sys.stderr)
        return 1
    if args.local:
        settings = dataclasses.replace(settings, qdrant_path=LOCAL_QDRANT_PATH)
    collection = args.collection or settings.collection
    where = settings.qdrant_path or settings.qdrant_url

    # Kiểm tra Qdrant trước khi cắt chunk và tải model nặng.
    # get_client cũng có thể lỗi (local mode: thư mục đang bị API/Streamlit giữ khóa).
    try:
        client = get_client(settings)
        check_connection(client, where)
    except Exception as exc:
        # Nêu nguyên nhân thật (vd. 403 sai API key, thư mục local bị khóa), không chỉ đoán "chưa bật Docker"
        cause = exc.__cause__ if isinstance(exc, QdrantUnavailableError) else exc
        detail = f" (nguyên nhân: {cause})" if cause else ""
        print(
            f'Không kết nối được Qdrant tại {where}{detail}. Chạy "docker compose up -d" hoặc thêm --local '
            "(hoặc đặt QDRANT_PATH trong .env) để dùng Qdrant local mode. Nếu dùng local mode, hãy tắt "
            "tiến trình khác đang mở cùng thư mục (API, Streamlit).",
            file=sys.stderr,
        )
        return 1
    try:
        return _index_with_client(args, settings, client, collection)
    finally:
        client.close()


def _index_with_client(args: argparse.Namespace, settings: Settings, client: QdrantClient, collection: str) -> int:
    if not args.pdf_dir.is_dir():
        print(f"Không tìm thấy thư mục PDF: {args.pdf_dir}", file=sys.stderr)
        return 1

    skipped_pages: list[tuple[str, SkippedPage]] = []
    chunks, skipped_files = chunk_directory(
        args.pdf_dir,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        skipped_pages_collector=skipped_pages,
    )
    if not chunks:
        print(
            f"Không có chunk nào từ {args.pdf_dir} (không có PDF hợp lệ). Dữ liệu cũ được giữ nguyên.",
            file=sys.stderr,
        )
        return 1

    try:
        embedder = BgeM3Embedder(settings.embedding_model)
    except Exception as exc:
        print(
            f"Không tải được model embedding '{settings.embedding_model}': {exc}. "
            "Kiểm tra EMBEDDING_MODEL và kết nối mạng (lần đầu cần tải model).",
            file=sys.stderr,
        )
        return 1
    try:
        ensure_collection(client, collection, embedder.dense_dim, recreate=args.recreate)
    except CollectionSchemaError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        report = index_corpus(
            client,
            chunks,
            embedder,
            collection,
            preserve_files=frozenset(skipped_files),
            k1=settings.bm25_k1,
            b=settings.bm25_b,
        )
    except Exception as exc:
        # index_corpus chỉ xóa point thừa sau khi upsert xong, nên lỗi ở đây không làm mất dữ liệu cũ
        print(f"Lỗi khi index: {exc}. Dữ liệu cũ được giữ nguyên; sửa lỗi rồi chạy lại.", file=sys.stderr)
        return 1

    print(f"Đã index {report.upserted} chunks từ {len(report.files)} file vào '{collection}': {report.files}")
    print(f"Đã xóa {report.deleted_stale} point thừa.")
    if skipped_files:
        print(f"File bị bỏ qua (scan / hỏng / bị khóa), dữ liệu cũ của chúng được giữ: {skipped_files}")
    if skipped_pages:
        print(f"Trang không có lớp chữ bị bỏ qua ({len(skipped_pages)}):")
        for filename, page in skipped_pages:
            kind = "scan ảnh" if page.reason == "scan" else "trang trắng"
            print(f"  - [{filename} | Trang {page.page_number}]: {kind}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
