import dataclasses
from uuid import NAMESPACE_URL, uuid5

import pytest
from qdrant_client import QdrantClient, models

from src.config import Settings
from src.ingestion import indexer
from src.ingestion.chunker import Chunk
from src.ingestion.indexer import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    CollectionSchemaError,
    ensure_collection,
    index_corpus,
)
from tests.fakes import FailingEmbedder, FakeEmbedder

COL = "t"


def mk(source: str, n: int) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=str(uuid5(NAMESPACE_URL, f"{source}:{i + 1}:0")),
            source_file=source,
            page_number=i + 1,
            article=f"Điều {i + 1}. Quy định {i + 1}",
            chunk_index=0,
            text=f"Điều {i + 1}. Quy định {i + 1}: sinh viên đăng ký tín chỉ học phần {source} {i}",
        )
        for i in range(n)
    ]


@pytest.fixture
def client():
    c = QdrantClient(":memory:")
    ensure_collection(c, COL, FakeEmbedder.dense_dim)
    return c


def count(c: QdrantClient, source: str | None = None) -> int:
    pts, _ = c.scroll(COL, limit=10_000, with_payload=True)
    return sum(1 for p in pts if source is None or p.payload["source_file"] == source)


def test_collection_has_dense_cosine_and_sparse_idf(client):
    params = client.get_collection(COL).config.params
    assert params.vectors[DENSE_VECTOR_NAME].distance == models.Distance.COSINE
    assert params.vectors[DENSE_VECTOR_NAME].size == 8
    assert params.sparse_vectors[SPARSE_VECTOR_NAME].modifier == models.Modifier.IDF


def test_payload_exact_schema_and_int_page(client):
    chunks = mk("a.pdf", 3)
    index_corpus(client, chunks, FakeEmbedder(), COL)
    by_id = {c.chunk_id: c for c in chunks}
    pts, _ = client.scroll(COL, limit=10, with_payload=True)
    assert len(pts) == 3
    for p in pts:
        assert set(p.payload) == {"source_file", "page_number", "article", "chunk_index", "text"}
        assert isinstance(p.payload["page_number"], int)
        src = by_id[str(p.id)]
        assert p.payload["page_number"] == src.page_number
        assert p.payload["source_file"] == src.source_file
        assert p.payload["article"] == src.article
        assert p.payload["text"] == src.text


def test_point_has_dense_and_sparse_vectors(client):
    chunks = mk("a.pdf", 1)
    index_corpus(client, chunks, FakeEmbedder(), COL)
    (pt,) = client.retrieve(COL, [chunks[0].chunk_id], with_vectors=True)
    assert set(pt.vector) == {DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME}
    assert len(pt.vector[SPARSE_VECTOR_NAME].indices) > 0


def test_reindex_same_data_no_duplicates(client):
    chunks = mk("a.pdf", 4)
    index_corpus(client, chunks, FakeEmbedder(), COL)
    index_corpus(client, chunks, FakeEmbedder(), COL)
    assert count(client) == 4


def test_shrunk_file_stale_points_deleted(client):
    index_corpus(client, mk("a.pdf", 5), FakeEmbedder(), COL)
    report = index_corpus(client, mk("a.pdf", 3), FakeEmbedder(), COL)
    assert count(client, "a.pdf") == 3
    assert report.deleted_stale == 2 and report.upserted == 3 and report.files == ["a.pdf"]


def test_removed_file_points_deleted(client):
    index_corpus(client, mk("a.pdf", 2) + mk("b.pdf", 2), FakeEmbedder(), COL)
    index_corpus(client, mk("a.pdf", 2), FakeEmbedder(), COL)
    assert count(client, "a.pdf") == 2 and count(client, "b.pdf") == 0


def test_preserved_file_points_kept(client):
    index_corpus(client, mk("a.pdf", 2) + mk("b.pdf", 2), FakeEmbedder(), COL)
    index_corpus(client, mk("a.pdf", 2), FakeEmbedder(), COL, preserve_files=frozenset({"b.pdf"}))
    assert count(client, "a.pdf") == 2 and count(client, "b.pdf") == 2


def test_embed_failure_keeps_old_data(client):
    index_corpus(client, mk("a.pdf", 5), FakeEmbedder(), COL)
    with pytest.raises(RuntimeError, match="embed failed"):
        index_corpus(client, mk("a.pdf", 3), FailingEmbedder(), COL)
    assert count(client, "a.pdf") == 5


def test_upsert_failure_skips_deletion(client, monkeypatch):
    index_corpus(client, mk("a.pdf", 5), FakeEmbedder(), COL)
    deletes = []

    def boom(*a, **k):
        raise ConnectionError("upsert failed")

    monkeypatch.setattr(client, "upsert", boom)
    monkeypatch.setattr(client, "delete", lambda *a, **k: deletes.append(1))
    with pytest.raises(ConnectionError):
        index_corpus(client, mk("a.pdf", 3), FakeEmbedder(), COL)
    assert deletes == []
    monkeypatch.undo()
    assert count(client, "a.pdf") == 5


def test_stale_cleanup_paginates(client, monkeypatch):
    monkeypatch.setattr(indexer, "SCROLL_PAGE", 50)
    index_corpus(client, mk("a.pdf", 150), FakeEmbedder(), COL)
    report = index_corpus(client, mk("a.pdf", 10), FakeEmbedder(), COL)
    assert count(client) == 10 and report.deleted_stale == 140


def test_dimension_mismatch_raises():
    c = QdrantClient(":memory:")
    ensure_collection(c, COL, 4)
    with pytest.raises(CollectionSchemaError, match=r"dim=4.*--recreate|--recreate.*dim=4"):
        ensure_collection(c, COL, 8)


@pytest.mark.parametrize("kind", ["unnamed", "sparse_without_idf", "no_sparse"])
def test_unnamed_or_missing_sparse_raises(kind):
    c = QdrantClient(":memory:")
    dense = models.VectorParams(size=8, distance=models.Distance.COSINE)
    if kind == "unnamed":
        c.create_collection(COL, vectors_config=dense)
    elif kind == "sparse_without_idf":
        c.create_collection(
            COL,
            vectors_config={DENSE_VECTOR_NAME: dense},
            sparse_vectors_config={SPARSE_VECTOR_NAME: models.SparseVectorParams()},
        )
    else:
        c.create_collection(COL, vectors_config={DENSE_VECTOR_NAME: dense})
    with pytest.raises(CollectionSchemaError, match="--recreate"):
        ensure_collection(c, COL, 8)


def test_recreate_resets_dimension():
    c = QdrantClient(":memory:")
    ensure_collection(c, COL, 4)
    ensure_collection(c, COL, 8, recreate=True)
    assert c.get_collection(COL).config.params.vectors[DENSE_VECTOR_NAME].size == 8


def test_empty_chunks_raises_without_touching_data(client):
    index_corpus(client, mk("a.pdf", 2), FakeEmbedder(), COL)
    with pytest.raises(ValueError):
        index_corpus(client, [], FakeEmbedder(), COL)
    assert count(client) == 2


def test_local_mode_persists(tmp_path):
    s = dataclasses.replace(Settings(), qdrant_path=str(tmp_path / "q"))
    c = indexer.get_client(s)
    ensure_collection(c, COL, 8)
    index_corpus(c, mk("a.pdf", 2), FakeEmbedder(), COL)
    c.close()
    c2 = indexer.get_client(s)
    assert c2.count(COL).count == 2
    c2.close()


def _clean_env(monkeypatch):
    for name in ["QDRANT_URL", "QDRANT_PATH", "QDRANT_API_KEY", "QDRANT_COLLECTION", "CHUNK_SIZE", "CHUNK_OVERLAP"]:
        monkeypatch.delenv(name, raising=False)


def test_cli_qdrant_down_fails_fast(monkeypatch, capsys, tmp_path):
    from src.ingestion.sample_pdf import build_sample_pdf

    _clean_env(monkeypatch)
    monkeypatch.chdir(tmp_path)  # không có .env ở đây
    build_sample_pdf(tmp_path / "pdfs" / "a.pdf")
    monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:1")
    loaded = []
    monkeypatch.setattr(indexer, "BgeM3Embedder", lambda *a, **k: loaded.append(1) or FakeEmbedder())
    assert indexer.main(["--pdf-dir", str(tmp_path / "pdfs")]) == 1
    err = capsys.readouterr().err
    assert "Không kết nối được Qdrant" in err and "--local" in err
    assert loaded == []


def test_cli_end_to_end_local(monkeypatch, capsys, tmp_path):
    from src.ingestion.sample_pdf import build_sample_pdf

    _clean_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    build_sample_pdf(tmp_path / "pdfs" / "a.pdf")
    (tmp_path / "pdfs" / "bad.pdf").write_bytes(b"not a pdf")
    monkeypatch.setattr(indexer, "BgeM3Embedder", lambda *a, **k: FakeEmbedder())
    assert indexer.main(["--pdf-dir", str(tmp_path / "pdfs"), "--local"]) == 0
    out = capsys.readouterr()
    assert "Đã index" in out.out and "bad.pdf" in out.out + out.err
    c = QdrantClient(path=str(tmp_path / "qdrant_local"))
    pts, _ = c.scroll("regulations", limit=1000, with_payload=True)
    assert pts and {p.payload["source_file"] for p in pts} == {"a.pdf"}
    c.close()


# --- Critic vòng 1 (docs/reviews/task-6-indexer-r1.md) ---
def _spy(monkeypatch, client, name, calls):
    original = getattr(client, name)

    def wrapper(*args, **kwargs):
        calls.append((name, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(client, name, wrapper)


def test_cli_preserves_points_of_file_that_became_corrupt(monkeypatch, capsys, tmp_path):
    # #1: PDF đã index bị hỏng ở lần chạy sau -> dữ liệu cũ của nó phải được giữ
    from src.ingestion.sample_pdf import build_sample_pdf

    _clean_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    pdfs = tmp_path / "pdfs"
    build_sample_pdf(pdfs / "a.pdf")
    build_sample_pdf(pdfs / "b.pdf")
    monkeypatch.setattr(indexer, "BgeM3Embedder", lambda *a, **k: FakeEmbedder())
    assert indexer.main(["--pdf-dir", str(pdfs), "--local"]) == 0
    (pdfs / "b.pdf").write_bytes(b"not a pdf")
    assert indexer.main(["--pdf-dir", str(pdfs), "--local"]) == 0
    c = QdrantClient(path=str(tmp_path / "qdrant_local"))
    pts, _ = c.scroll("regulations", limit=1000, with_payload=True)
    c.close()
    per_file = {}
    for p in pts:
        per_file[p.payload["source_file"]] = per_file.get(p.payload["source_file"], 0) + 1
    assert per_file["b.pdf"] == per_file["a.pdf"] > 0


def test_avgdl_computed_over_whole_corpus():
    # #2: trọng số BM25 của một chunk phụ thuộc avgdl của TOÀN kho, không chỉ file của nó
    from src.ingestion.sparse import average_doc_length, bm25_doc_vector, tokenize

    short = mk("a.pdf", 1)
    long_text = "Điều 9. Dài: " + "sinh viên đăng ký tín chỉ học phần " * 20
    long = [dataclasses.replace(mk("b.pdf", 1)[0], text=long_text)]
    corpus = short + long
    c = QdrantClient(":memory:")
    ensure_collection(c, COL, 8)
    index_corpus(c, corpus, FakeEmbedder(), COL)
    (pt,) = c.retrieve(COL, [short[0].chunk_id], with_vectors=True)
    avgdl = average_doc_length([tokenize(x.text) for x in corpus])
    expected = bm25_doc_vector(tokenize(short[0].text), avgdl)
    stored = dict(zip(pt.vector[SPARSE_VECTOR_NAME].indices, pt.vector[SPARSE_VECTOR_NAME].values))
    assert stored == pytest.approx(dict(zip(expected.indices, expected.values)))
    alone = bm25_doc_vector(tokenize(short[0].text), average_doc_length([tokenize(short[0].text)]))
    assert stored != pytest.approx(dict(zip(alone.indices, alone.values)))


def test_payload_index_created_for_source_file(monkeypatch):
    # #3: Qdrant local không báo payload_schema -> kiểm bằng spy lời gọi
    c = QdrantClient(":memory:")
    calls = []
    _spy(monkeypatch, c, "create_payload_index", calls)
    ensure_collection(c, COL, 8)
    assert calls and calls[0][1]["field_name"] == "source_file"
    assert calls[0][1]["field_schema"] == models.PayloadSchemaType.KEYWORD


def test_payload_chunk_index_preserved(client):
    # #4: nhiều chunk cùng trang, chunk_index 0..2 phải giữ nguyên trong payload
    base = mk("a.pdf", 1)[0]
    chunks = [
        dataclasses.replace(base, chunk_id=str(uuid5(NAMESPACE_URL, f"a.pdf:1:{i}")), chunk_index=i, text=f"{base.text} phần {i}")
        for i in range(3)
    ]
    index_corpus(client, chunks, FakeEmbedder(), COL)
    by_id = {c.chunk_id: c for c in chunks}
    pts, _ = client.scroll(COL, limit=10, with_payload=True)
    assert sorted(p.payload["chunk_index"] for p in pts) == [0, 1, 2]
    for p in pts:
        assert p.payload["chunk_index"] == by_id[str(p.id)].chunk_index


def test_upsert_and_delete_wait_for_completion(client, monkeypatch):
    # #5: mọi lệnh ghi/xóa phải wait=True
    index_corpus(client, mk("a.pdf", 5), FakeEmbedder(), COL)
    calls = []
    _spy(monkeypatch, client, "upsert", calls)
    _spy(monkeypatch, client, "delete", calls)
    index_corpus(client, mk("a.pdf", 3), FakeEmbedder(), COL)
    assert {name for name, _ in calls} == {"upsert", "delete"}
    assert all(kw.get("wait") is True for _, kw in calls)


def test_existing_collection_wrong_distance_raises():
    # #6
    c = QdrantClient(":memory:")
    c.create_collection(
        COL,
        vectors_config={DENSE_VECTOR_NAME: models.VectorParams(size=8, distance=models.Distance.DOT)},
        sparse_vectors_config={SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    with pytest.raises(CollectionSchemaError, match="Cosine|COSINE|cosine"):
        ensure_collection(c, COL, 8)


def test_existing_collection_missing_payload_index_is_created(monkeypatch):
    # #6: collection đúng cấu trúc nhưng thiếu payload index -> tạo bù
    c = QdrantClient(":memory:")
    c.create_collection(
        COL,
        vectors_config={DENSE_VECTOR_NAME: models.VectorParams(size=8, distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    calls = []
    _spy(monkeypatch, c, "create_payload_index", calls)
    ensure_collection(c, COL, 8)
    assert calls and calls[0][1]["field_name"] == "source_file"


def test_cli_embed_failure_is_friendly_and_keeps_data(monkeypatch, capsys, tmp_path):
    # #7: lỗi khi index -> một dòng tiếng Việt, exit 1, không traceback, dữ liệu cũ còn
    from src.ingestion.sample_pdf import build_sample_pdf

    _clean_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    build_sample_pdf(tmp_path / "pdfs" / "a.pdf")
    monkeypatch.setattr(indexer, "BgeM3Embedder", lambda *a, **k: FakeEmbedder())
    assert indexer.main(["--pdf-dir", str(tmp_path / "pdfs"), "--local"]) == 0
    capsys.readouterr()
    monkeypatch.setattr(indexer, "BgeM3Embedder", lambda *a, **k: FailingEmbedder())
    assert indexer.main(["--pdf-dir", str(tmp_path / "pdfs"), "--local"]) == 1
    err = capsys.readouterr().err
    assert "Lỗi khi index" in err and "embed failed" in err and "Traceback" not in err
    c = QdrantClient(path=str(tmp_path / "qdrant_local"))
    assert c.count("regulations").count > 0
    c.close()


def test_cli_local_storage_locked_is_friendly(monkeypatch, capsys, tmp_path):
    # #7: thư mục local đang bị tiến trình khác (API/Streamlit) giữ khóa
    _clean_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pdfs").mkdir()
    holder = QdrantClient(path=str(tmp_path / "qdrant_local"))
    try:
        assert indexer.main(["--pdf-dir", str(tmp_path / "pdfs"), "--local"]) == 1
    finally:
        holder.close()
    err = capsys.readouterr().err
    assert "Không kết nối được Qdrant" in err and "Traceback" not in err
