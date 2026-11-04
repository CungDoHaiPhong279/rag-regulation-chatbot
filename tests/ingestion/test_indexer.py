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
