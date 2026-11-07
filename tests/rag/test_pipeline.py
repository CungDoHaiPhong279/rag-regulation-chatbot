from dataclasses import replace
import json
from uuid import NAMESPACE_URL, uuid5

import pytest
from qdrant_client import QdrantClient

from src.config import Settings
from src.ingestion.chunker import Chunk
from src.ingestion.indexer import ensure_collection, index_corpus
from src.rag.constants import REFUSAL
from src.rag.llm import LLMError
from src.rag.models import Turn
from src.rag.pipeline import Citation, LLMUnavailableError, RagPipeline
from src.rag.prompts import ANSWER_SYSTEM_PROMPT, REWRITE_SYSTEM_PROMPT
from src.retrieval.hybrid_search import HybridSearcher, SearchUnavailableError
from tests.fakes import FakeEmbedder, FakeLLM, FakeReranker

COLLECTION = "test_pipeline"
QUESTION = "cảnh báo học tập"
HISTORY = [Turn("user", "Cảnh báo học tập là gì?"), Turn("assistant", "Là ... [1].")]
SETTINGS = Settings(candidate_k=20, final_k=5, rerank_threshold=0.3)


def _chunk(i: int, text: str) -> Chunk:
    page = 100 + i
    return Chunk(
        chunk_id=str(uuid5(NAMESPACE_URL, f"a.pdf:{page}:{i}")),
        source_file="a.pdf",
        page_number=page,
        article=f"Điều {i}",
        chunk_index=i,
        text=text,
    )


# Mỗi chunk chứa cả 3 từ của câu hỏi nên điểm FakeReranker = 1.0; thêm chunk khác để kiểm tra final_k.
CHUNKS = [_chunk(i, f"cảnh báo học tập nội dung {i}") for i in range(7)]


@pytest.fixture
def searcher():
    client = QdrantClient(":memory:")
    embedder = FakeEmbedder()
    ensure_collection(client, COLLECTION, dense_dim=embedder.dense_dim)
    index_corpus(client, CHUNKS, embedder, COLLECTION)
    return HybridSearcher(client, COLLECTION, embedder)


class _FixedScoreReranker:
    """Giữ thứ tự, gán cùng một điểm cho mọi ứng viên."""

    def __init__(self, score: float) -> None:
        self.score = score

    def rerank(self, query, hits, top_k):
        return [replace(hit, score=self.score) for hit in hits][:top_k]


class _BrokenSearcher:
    def search(self, query, mode="hybrid", limit=20):
        raise SearchUnavailableError("Qdrant tắt")


class _EmptySearcher:
    def search(self, query, mode="hybrid", limit=20):
        return []


def _pipeline(searcher, llm, reranker=None, settings=SETTINGS) -> RagPipeline:
    return RagPipeline(searcher, reranker or FakeReranker(), llm, settings)


def _answer_with(searcher, text: str):
    return _pipeline(searcher, FakeLLM([text])).answer(QUESTION)


def test_retrieve_returns_final_k_reranked_hits(searcher):
    hits = _pipeline(searcher, None).retrieve(QUESTION)
    assert len(hits) == 5
    assert all(hit.score == 1.0 for hit in hits)


def test_layer1_refuses_without_calling_llm(searcher):
    llm = FakeLLM([])
    result = _pipeline(searcher, llm, _FixedScoreReranker(0.1)).answer(QUESTION)
    assert result.answer == REFUSAL
    assert result.refused
    assert result.citations == []
    assert llm.prompts == []
    assert result.debug["top_rerank_score"] == 0.1


def test_score_equal_to_threshold_is_not_refused(searcher):
    result = _pipeline(searcher, FakeLLM(["Có [1]."]), _FixedScoreReranker(0.3)).answer(QUESTION)
    assert not result.refused


def test_no_candidates_refuses(searcher):
    llm = FakeLLM([])
    result = _pipeline(_EmptySearcher(), llm).answer(QUESTION)
    assert result.answer == REFUSAL
    assert result.refused
    assert llm.prompts == []
    assert result.debug["n_candidates"] == 0
    assert result.debug["top_rerank_score"] is None


def test_valid_answer_maps_citations_from_metadata(searcher):
    pipeline = _pipeline(searcher, FakeLLM(["Sinh viên bị cảnh báo khi điểm thấp, trang 999 [1]."]))
    first_hit = pipeline.retrieve(QUESTION)[0]
    result = pipeline.answer(QUESTION)
    assert not result.refused
    assert result.citations == [
        Citation(first_hit.source_file, first_hit.page_number, first_hit.article, first_hit.text)
    ]
    assert result.citations[0].page_number != 999  # số trang lấy từ metadata, không từ chữ của LLM
    assert result.debug["citation_check_failed"] is False


def test_answer_prompt_gets_answer_system_and_reranked_context(searcher):
    llm = FakeLLM(["Có [1]."])
    pipeline = _pipeline(searcher, llm)
    first_hit = pipeline.retrieve(QUESTION)[0]
    pipeline.answer(QUESTION)
    assert llm.systems == [ANSWER_SYSTEM_PROMPT]
    assert f"[1] (a.pdf, trang {first_hit.page_number}" in llm.prompts[0]
    assert "[5] (" in llm.prompts[0]
    assert "[6] (" not in llm.prompts[0]


def test_citations_are_renumbered_to_match_list_positions(searcher):
    # Trích [3] rồi [1]: người đọc thấy [k] phải tra đúng citations[k-1].
    pipeline = _pipeline(searcher, FakeLLM(["Ý A [3]. Ý B [1]. Ý C [3, 1]."]))
    hits = pipeline.retrieve(QUESTION)
    result = pipeline.answer(QUESTION)
    assert result.answer == "Ý A [1]. Ý B [2]. Ý C [1, 2]."
    assert [c.page_number for c in result.citations] == [hits[2].page_number, hits[0].page_number]


@pytest.mark.parametrize("variant", [REFUSAL.rstrip("."), f"“{REFUSAL}”", f"\n{REFUSAL}\n"])
def test_llm_refusal_variant_normalized(searcher, variant):
    result = _answer_with(searcher, variant)
    assert result.answer == REFUSAL
    assert result.refused
    assert result.citations == []
    assert result.debug["citation_check_failed"] is False  # LLM từ chối đúng cách, không phải trích sai


def test_answer_without_citation_refused(searcher):
    result = _answer_with(searcher, "Sinh viên bị cảnh báo khi điểm thấp.")
    assert result.answer == REFUSAL
    assert result.refused
    assert result.citations == []
    assert result.debug["citation_check_failed"] is True


@pytest.mark.parametrize("text", ["Sai [9].", "Sai [0].", "Một đúng một sai [1][6]."])
def test_out_of_range_citation_refused(searcher, text):
    result = _answer_with(searcher, text)
    assert result.answer == REFUSAL
    assert result.refused
    assert result.citations == []
    assert result.debug["citation_check_failed"] is True


def test_refusal_plus_content_without_citation_refused(searcher):
    result = _answer_with(searcher, REFUSAL + " Tuy nhiên có thể là 2 kỳ.")
    assert result.answer == REFUSAL
    assert result.debug["citation_check_failed"] is True


def test_llm_error_raises_unavailable_with_sources(searcher):
    pipeline = _pipeline(searcher, FakeLLM([LLMError("x")]))
    hits = pipeline.retrieve(QUESTION)
    with pytest.raises(LLMUnavailableError) as err:
        pipeline.answer(QUESTION)
    assert [s.page_number for s in err.value.sources] == [h.page_number for h in hits]
    assert REFUSAL not in str(err.value)  # lỗi kỹ thuật không được dùng câu từ chối


def test_llm_none_raises_unavailable(searcher):
    with pytest.raises(LLMUnavailableError) as err:
        _pipeline(searcher, None).answer(QUESTION)
    assert err.value.sources


def test_llm_none_below_threshold_still_refuses(searcher):
    # Lớp 1 không cần LLM nên vẫn trả lời được khi thiếu key.
    result = _pipeline(searcher, None, _FixedScoreReranker(0.1)).answer(QUESTION)
    assert result.answer == REFUSAL


def test_search_unavailable_propagates():
    with pytest.raises(SearchUnavailableError):
        _pipeline(_BrokenSearcher(), FakeLLM([])).answer(QUESTION)


def test_history_triggers_rewrite_and_reports_query(searcher):
    llm = FakeLLM([QUESTION, "Có [1]."])
    result = _pipeline(searcher, llm).answer("Khi nào bị?", HISTORY)
    assert result.rewritten_query == QUESTION
    assert llm.systems == [REWRITE_SYSTEM_PROMPT, ANSWER_SYSTEM_PROMPT]
    assert f"Câu hỏi: {QUESTION}" in llm.prompts[1]
    assert result.debug["rewrite_failed"] is False


def test_rewritten_query_is_what_gets_searched_and_reranked(searcher):
    searched: list[str] = []
    real_search = searcher.search

    def spy_search(query, mode="hybrid", limit=20):
        searched.append(query)
        return real_search(query, mode=mode, limit=limit)

    searcher.search = spy_search
    reranker = FakeReranker()
    _pipeline(searcher, FakeLLM([QUESTION, "Có [1]."]), reranker).answer("Khi nào bị?", HISTORY)
    assert searched == [QUESTION]
    assert reranker.calls == [(QUESTION, 5)]


def test_rewrite_failure_uses_original_question(searcher):
    llm = FakeLLM([LLMError("x"), "Có [1]."])
    result = _pipeline(searcher, llm).answer(QUESTION, HISTORY)
    assert result.rewritten_query == QUESTION
    assert result.debug["rewrite_failed"] is True
    assert not result.refused


def test_no_history_does_not_rewrite(searcher):
    llm = FakeLLM(["Có [1]."])
    result = _pipeline(searcher, llm).answer(QUESTION)
    assert llm.systems == [ANSWER_SYSTEM_PROMPT]
    assert result.rewritten_query == QUESTION


def test_debug_has_timings_and_is_json_serializable(searcher):
    result = _answer_with(searcher, "Có [1].")
    assert set(result.debug["timings_ms"]) == {"rewrite", "search", "rerank", "llm"}
    assert result.debug["n_candidates"] == 7
    assert result.debug["top_rerank_score"] == 1.0
    json.dumps(result.debug)


def test_debug_keys_are_the_same_on_refusal(searcher):
    answered = _answer_with(searcher, "Có [1].")
    refused = _pipeline(searcher, FakeLLM([]), _FixedScoreReranker(0.1)).answer(QUESTION)
    assert set(refused.debug) == set(answered.debug)
    assert set(refused.debug["timings_ms"]) == set(answered.debug["timings_ms"])
