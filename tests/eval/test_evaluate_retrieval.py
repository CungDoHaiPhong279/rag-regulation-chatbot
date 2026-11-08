import json
from uuid import NAMESPACE_URL, uuid5

import pytest
from qdrant_client import QdrantClient

from eval.dataset import Gold, Question
from eval.evaluate_retrieval import CONFIGS, build_metadata, evaluate, write_report
from src.config import Settings
from src.ingestion.chunker import Chunk
from src.ingestion.indexer import ensure_collection, index_corpus
from src.retrieval.hybrid_search import HybridSearcher
from tests.fakes import FakeEmbedder, FakeReranker

COL = "test_eval"
# 25 chunk để chắc chắn ứng viên bị cắt đúng ở 20.
CHUNKS = [
    Chunk(str(uuid5(NAMESPACE_URL, f"tt08.pdf:{i}:{i}")), "tt08.pdf", i, f"Điều {i}. Tên", i, f"nội dung riêng số {i} chuyện {i}")
    for i in range(1, 26)
]
CHUNKS[6] = Chunk(CHUNKS[6].chunk_id, "tt08.pdf", 7, "Điều 7. Cảnh báo", 6, "cảnh báo học tập khi điểm thấp")


def _q(id_, question, in_scope=True, gold=(Gold("tt08.pdf", "Điều 7", (7,)),)):
    return Question(id_, question, in_scope, "test", gold if in_scope else ())


@pytest.fixture
def searcher():
    client = QdrantClient(":memory:")
    ensure_collection(client, COL, dense_dim=FakeEmbedder.dense_dim)
    index_corpus(client, CHUNKS, FakeEmbedder(), COL)
    return HybridSearcher(client, COL, FakeEmbedder())


class _SpyReranker(FakeReranker):
    def __init__(self):
        super().__init__()
        self.sizes: list[int] = []

    def rerank(self, query, hits, top_k):
        self.sizes.append(len(hits))
        return super().rerank(query, hits, top_k)


def test_evaluate_runs_three_configs(searcher):
    reranker = _SpyReranker()
    result = evaluate([_q("q1", "cảnh báo học tập")], searcher, reranker)
    assert set(result) == set(CONFIGS)
    assert set(result["dense"]) == {"recall@5", "recall@20", "mrr@5", "n"}
    assert set(result["hybrid"]) == {"recall@5", "recall@20", "mrr@5", "n"}
    assert set(result["hybrid_rerank"]) == {"recall@5", "mrr@5", "n"}
    assert reranker.sizes == [20]  # rerank 20 ứng viên của hybrid
    assert reranker.calls == [("cảnh báo học tập", 5)]
    # BM25 khớp đúng từ khóa nên hybrid và rerank tìm thấy chunk gold ở hạng 1
    assert result["hybrid"]["recall@5"] == 1.0
    assert result["hybrid_rerank"]["mrr@5"] == 1.0


def test_evaluate_excludes_out_of_scope(searcher):
    questions = [_q("q1", "cảnh báo học tập"), _q("q2", "học phí", in_scope=False)]
    result = evaluate(questions, searcher, FakeReranker())
    assert all(metrics["n"] == 1 for metrics in result.values())


def test_evaluate_averages_over_questions(searcher):
    questions = [
        _q("q1", "cảnh báo học tập"),
        _q("q2", "cảnh báo học tập", gold=(Gold("khac.pdf", "Điều 1", (1,)),)),  # không thể trúng
    ]
    result = evaluate(questions, searcher, FakeReranker())
    assert result["hybrid"]["recall@5"] == 0.5
    assert result["hybrid"]["n"] == 2


class _RankedSearcher:
    """Trả 20 ứng viên, chunk gold nằm ở hạng 6 với cả dense và hybrid."""

    def search(self, query, mode="hybrid", limit=20):
        from src.retrieval.hybrid_search import Hit

        hits = [Hit(f"m{i}", "tt08.pdf", 99, "Điều 99", i, "x", 0.5) for i in range(limit)]
        hits[5] = Hit("gold", "tt08.pdf", 7, "Điều 7. Cảnh báo", 5, "x", 0.5)
        return hits


def test_evaluate_separates_recall_at_5_and_20():
    result = evaluate([_q("q1", "bất kỳ")], _RankedSearcher(), None)
    for mode in ("dense", "hybrid"):
        assert result[mode]["recall@5"] == 0.0
        assert result[mode]["recall@20"] == 1.0
        assert result[mode]["mrr@5"] == 0.0


def test_evaluate_without_reranker_skips_rerank_config(searcher):
    result = evaluate([_q("q1", "cảnh báo học tập")], searcher, None)
    assert set(result) == {"dense", "hybrid"}


def test_evaluate_without_in_scope_questions_is_error(searcher):
    with pytest.raises(ValueError):
        evaluate([_q("q2", "học phí", in_scope=False)], searcher, FakeReranker())


def test_report_records_run_metadata(tmp_path):
    settings = Settings(embedding_model="emb-x", reranker_model="rr-y", chunk_size=800)
    meta = build_metadata(settings, collection="regulations", split="test", rerank=True)
    results = {"dense": {"recall@5": 0.5, "recall@20": 1.0, "mrr@5": 0.5, "n": 2}}
    path = write_report(results, meta, tmp_path)
    report = json.loads(path.read_text(encoding="utf-8"))
    for key in ("embedding_model", "reranker_model", "collection", "git_commit", "date", "split", "chunk_size"):
        assert key in report["meta"]
    assert report["meta"]["embedding_model"] == "emb-x"
    assert report["meta"]["reranker_model"] == "rr-y"
    assert report["meta"]["git_commit"] != ""
    assert report["results"] == results
    assert path.parent == tmp_path and path.suffix == ".json"


def test_label_helper_formats_candidates():
    from eval.label_helper import format_candidates

    hits = [
        replace_hit(CHUNKS[6], 0.91),
        replace_hit(Chunk("x", "tt08.pdf", 3, None, 0, "a" * 500), 0.5),
    ]
    text = format_candidates(hits, preview_chars=200)
    lines = text.splitlines()
    assert lines[0].startswith(" 1. tt08.pdf | trang 7 | Điều 7. Cảnh báo | 0.910")
    assert "trang 3 | (không có Điều)" in text
    assert "a" * 200 in text and "a" * 201 not in text


def replace_hit(chunk: Chunk, score: float):
    from src.retrieval.hybrid_search import Hit

    return Hit(chunk.chunk_id, chunk.source_file, chunk.page_number, chunk.article, chunk.chunk_index, chunk.text, score)


def test_metadata_without_rerank_has_no_reranker_model():
    meta = build_metadata(Settings(), collection="c", split="dev", rerank=False)
    assert meta["reranker_model"] is None
