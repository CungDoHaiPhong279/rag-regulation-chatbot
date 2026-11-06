from __future__ import annotations

from typing import Any
from uuid import NAMESPACE_URL, uuid5

import pytest
from qdrant_client import QdrantClient, models

from src.ingestion.chunker import Chunk
from src.ingestion.indexer import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME, ensure_collection, index_corpus
from src.retrieval.hybrid_search import Hit, HybridSearcher, SearchUnavailableError
from tests.fakes import FakeEmbedder

COLLECTION = "t"


def _make_chunk(source_file: str, page_number: int, article: str | None, chunk_index: int, text: str) -> Chunk:
    cid = str(uuid5(NAMESPACE_URL, f"{source_file}:{page_number}:{chunk_index}"))
    return Chunk(
        chunk_id=cid,
        source_file=source_file,
        page_number=page_number,
        article=article,
        chunk_index=chunk_index,
        text=text,
    )


CHUNKS: list[Chunk] = [
    _make_chunk(
        "thong_tu_08_2021.pdf",
        3,
        "Điều 3. Chương trình đào tạo và tín chỉ",
        0,
        "Điều 3. Chương trình đào tạo và tín chỉ: Một tín chỉ được tính tương đương 50 giờ học tập định mức của người học.",
    ),
    _make_chunk(
        "thong_tu_08_2021.pdf",
        7,
        "Điều 9. Đánh giá kết quả học tập",
        0,
        "Điều 9. Đánh giá kết quả học tập: Điểm học phần được tính từ điểm đánh giá bộ phận và điểm thi kết thúc học phần.",
    ),
    _make_chunk(
        "thong_tu_08_2021.pdf",
        13,
        "Điều 11. Xử lý kết quả học tập theo tín chỉ",
        0,
        "Điều 11. Xử lý kết quả học tập theo tín chỉ: Sinh viên bị cảnh báo học tập khi khối lượng tín chỉ không đạt trong học kỳ vượt ngưỡng quy định.",
    ),
    _make_chunk(
        "thong_tu_08_2021.pdf",
        16,
        "Điều 14. Công nhận tốt nghiệp và cấp bằng",
        0,
        "Điều 14. Công nhận tốt nghiệp và cấp bằng: Sinh viên tích lũy đủ số học phần và khối lượng quy định được xét công nhận tốt nghiệp.",
    ),
    _make_chunk(
        "quy_che_truong.pdf",
        5,
        "Điều 6. Đăng ký khối lượng học tập",
        0,
        "Điều 6. Đăng ký khối lượng học tập: Đầu mỗi học kỳ sinh viên đăng ký các học phần dự định học trong học kỳ đó.",
    ),
    _make_chunk(
        "quy_che_truong.pdf",
        11,
        "Điều 15. Nghỉ học tạm thời và thôi học",
        0,
        "Điều 15. Nghỉ học tạm thời và thôi học: Sinh viên được quyền viết đơn xin nghỉ học tạm thời và bảo lưu kết quả đã học.",
    ),
]


@pytest.fixture()
def indexed_searcher() -> tuple[HybridSearcher, QdrantClient, dict[str, Chunk]]:
    client = QdrantClient(":memory:")
    embedder = FakeEmbedder()
    ensure_collection(client, COLLECTION, embedder.dense_dim)
    index_corpus(client, CHUNKS, embedder, COLLECTION)
    searcher = HybridSearcher(client=client, collection=COLLECTION, embedder=embedder, candidate_k=20)
    by_id = {c.chunk_id: c for c in CHUNKS}
    return searcher, client, by_id


def test_sparse_mode_ranks_keyword_match_first(
    indexed_searcher: tuple[HybridSearcher, QdrantClient, dict[str, Chunk]],
) -> None:
    searcher, _, _ = indexed_searcher
    hits = searcher.search("cảnh báo học tập", mode="sparse")
    assert len(hits) > 0
    assert hits[0].article == "Điều 11. Xử lý kết quả học tập theo tín chỉ"


def test_dense_mode_returns_limit_hits(
    indexed_searcher: tuple[HybridSearcher, QdrantClient, dict[str, Chunk]],
) -> None:
    searcher, _, _ = indexed_searcher
    hits = searcher.search("quy định về tín chỉ", mode="dense", limit=3)
    assert len(hits) == 3
    assert all(isinstance(h, Hit) for h in hits)


def test_hybrid_uses_rrf_prefetch(
    indexed_searcher: tuple[HybridSearcher, QdrantClient, dict[str, Chunk]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    searcher, client, _ = indexed_searcher
    captured: dict[str, Any] = {}
    real_query_points = client.query_points

    def spy_query_points(*args: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return real_query_points(*args, **kwargs)

    monkeypatch.setattr(client, "query_points", spy_query_points)

    hits = searcher.search("cảnh báo học tập tín chỉ", mode="hybrid", limit=4)
    assert len(hits) == 4
    prefetch = captured.get("prefetch")
    assert isinstance(prefetch, list) and len(prefetch) == 2
    by_using = {p.using: p for p in prefetch}
    assert set(by_using.keys()) == {DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME}
    assert by_using[DENSE_VECTOR_NAME].limit == searcher.candidate_k
    assert by_using[SPARSE_VECTOR_NAME].limit == searcher.candidate_k
    assert captured.get("query") == models.FusionQuery(fusion=models.Fusion.RRF)
    assert captured.get("limit") == 4
    assert captured.get("with_payload") is True


def test_hits_preserve_page_metadata(
    indexed_searcher: tuple[HybridSearcher, QdrantClient, dict[str, Chunk]],
) -> None:
    searcher, _, by_id = indexed_searcher
    for mode in ("dense", "sparse", "hybrid"):
        hits = searcher.search("cảnh báo học tập tín chỉ tốt nghiệp", mode=mode, limit=6)
        assert len(hits) > 0
        for h in hits:
            expected = by_id[h.chunk_id]
            assert h.source_file == expected.source_file
            assert h.page_number == expected.page_number
            assert h.article == expected.article
            assert h.chunk_index == expected.chunk_index
            assert h.text == expected.text
            assert isinstance(h.score, float)


def test_query_with_zero_tokens_falls_back_to_dense(
    indexed_searcher: tuple[HybridSearcher, QdrantClient, dict[str, Chunk]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    searcher, client, _ = indexed_searcher
    calls: list[dict[str, Any]] = []
    real_query_points = client.query_points

    def spy_query_points(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return real_query_points(*args, **kwargs)

    monkeypatch.setattr(client, "query_points", spy_query_points)

    hits = searcher.search("???", mode="hybrid", limit=3)
    assert len(hits) == 3
    assert len(calls) == 1
    call = calls[0]
    assert call.get("prefetch") is None
    assert call.get("using") == DENSE_VECTOR_NAME


def test_sparse_mode_zero_tokens_returns_empty(
    indexed_searcher: tuple[HybridSearcher, QdrantClient, dict[str, Chunk]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    searcher, client, _ = indexed_searcher

    def fail_if_called(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("query_points should not be called when sparse query has zero tokens")

    monkeypatch.setattr(client, "query_points", fail_if_called)
    assert searcher.search("???", mode="sparse") == []


def test_unaccented_query_does_not_crash(
    indexed_searcher: tuple[HybridSearcher, QdrantClient, dict[str, Chunk]],
) -> None:
    searcher, _, _ = indexed_searcher
    hits = searcher.search("hoc phi bao nhieu", mode="hybrid")
    assert isinstance(hits, list)
    assert len(hits) > 0


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_qdrant_error_wrapped(
    indexed_searcher: tuple[HybridSearcher, QdrantClient, dict[str, Chunk]],
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    searcher, client, _ = indexed_searcher

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise ConnectionError("qdrant down")

    monkeypatch.setattr(client, "query_points", boom)
    with pytest.raises(SearchUnavailableError) as exc_info:
        searcher.search("tín chỉ", mode=mode)
    assert isinstance(exc_info.value.__cause__, ConnectionError)


def test_is_ready_false_on_empty_collection() -> None:
    client = QdrantClient(":memory:")
    embedder = FakeEmbedder()
    ensure_collection(client, COLLECTION, embedder.dense_dim)
    searcher = HybridSearcher(client=client, collection=COLLECTION, embedder=embedder)
    assert searcher.is_ready() is False


def test_is_ready_true_after_index(
    indexed_searcher: tuple[HybridSearcher, QdrantClient, dict[str, Chunk]],
) -> None:
    searcher, _, _ = indexed_searcher
    assert searcher.is_ready() is True


def test_is_ready_false_when_collection_missing() -> None:
    client = QdrantClient(":memory:")
    embedder = FakeEmbedder()
    searcher = HybridSearcher(client=client, collection="missing_collection", embedder=embedder)
    assert searcher.is_ready() is False
