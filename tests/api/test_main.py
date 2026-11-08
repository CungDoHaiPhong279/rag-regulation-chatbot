from dataclasses import replace
import logging
from uuid import NAMESPACE_URL, uuid5

from fastapi.testclient import TestClient
import pytest
from qdrant_client import QdrantClient

from src.api.main import (
    MSG_LLM_DOWN,
    MSG_NO_DATA,
    MSG_NO_KEY,
    MSG_SEARCH_DOWN,
    create_app,
)
from src.config import Settings
from src.ingestion.chunker import Chunk
from src.ingestion.indexer import ensure_collection, index_corpus
from src.rag.constants import REFUSAL
from src.rag.llm import LLMError
from src.rag.pipeline import RagPipeline
from src.rag.prompts import REWRITE_SYSTEM_PROMPT
from src.retrieval.hybrid_search import HybridSearcher, SearchUnavailableError
from tests.fakes import FailingEmbedder, FakeEmbedder, FakeLLM, FakeReranker

COL = "test_api"
QUESTION = "cảnh báo học tập"
CHUNKS = [
    Chunk(str(uuid5(NAMESPACE_URL, f"a.pdf:{100 + i}:{i}")), "a.pdf", 100 + i, f"Điều {i}", i, f"cảnh báo học tập {i}")
    for i in range(3)
]


def _searcher(chunks=CHUNKS, embedder=None) -> HybridSearcher:
    client = QdrantClient(":memory:")
    ensure_collection(client, COL, dense_dim=FakeEmbedder.dense_dim)
    if chunks:
        index_corpus(client, chunks, FakeEmbedder(), COL)
    return HybridSearcher(client, COL, embedder or FakeEmbedder())


class _FixedScoreReranker:
    def __init__(self, score: float) -> None:
        self.score = score

    def rerank(self, query, hits, top_k):
        return [replace(hit, score=self.score) for hit in hits][:top_k]


class _DownClient:
    def get_collections(self):
        raise ConnectionError("down")


class _StubSearcher:
    def __init__(self, ready: bool, qdrant_up: bool = True) -> None:
        self.ready = ready
        self.client = QdrantClient(":memory:") if qdrant_up else _DownClient()

    def is_ready(self) -> bool:
        return self.ready

    def search(self, query, mode="hybrid", limit=20):
        raise SearchUnavailableError("mất kết nối")


def _app(searcher=None, llm="default", reranker=None, settings=None) -> TestClient:
    if llm == "default":
        llm = FakeLLM(["Sinh viên bị cảnh báo [1]."] * 10)
    pipeline = RagPipeline(searcher or _searcher(), reranker or FakeReranker(), llm, settings or Settings())
    return TestClient(create_app(lambda: pipeline))


def test_chat_ok_returns_citations_with_page_numbers():
    with _app() as client:
        response = client.post("/chat", json={"question": QUESTION, "history": []})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"answer", "refused", "citations", "rewritten_query", "debug"}
    assert body["refused"] is False
    (citation,) = body["citations"]
    assert set(citation) == {"source_file", "page_number", "article", "text"}
    assert citation["page_number"] in {c.page_number for c in CHUNKS}
    assert isinstance(citation["page_number"], int)


def test_history_is_optional_and_reaches_rewriter():
    llm = FakeLLM([QUESTION, "Có [1]."])
    with _app(llm=llm) as client:
        response = client.post(
            "/chat",
            json={"question": "Khi nào bị?", "history": [{"role": "user", "content": "Cảnh báo là gì?"}]},
        )
    assert response.status_code == 200
    assert response.json()["rewritten_query"] == QUESTION
    assert llm.systems[0] == REWRITE_SYSTEM_PROMPT


@pytest.mark.parametrize("q", ["", "   ", "???", " ... !! ", "x" * 1001])
def test_invalid_question_422(q):
    with _app() as client:
        assert client.post("/chat", json={"question": q, "history": []}).status_code == 422


def test_question_of_1000_chars_after_strip_is_accepted():
    with _app() as client:
        response = client.post("/chat", json={"question": "  " + "a" * 1000 + "  "})
    assert response.status_code == 200


@pytest.mark.parametrize(
    "history",
    [
        [{"role": "user", "content": "x"}] * 21,
        [{"role": "system", "content": "x"}],
        [{"role": "user", "content": "x" * 10_001}],
        [{"role": "user"}],
    ],
)
def test_invalid_history_422(history):
    with _app() as client:
        assert client.post("/chat", json={"question": QUESTION, "history": history}).status_code == 422


def test_chat_refusal_is_200():
    with _app(reranker=_FixedScoreReranker(0.01)) as client:
        response = client.post("/chat", json={"question": QUESTION})
    assert response.status_code == 200
    assert response.json()["answer"] == REFUSAL
    assert response.json()["refused"] is True


def test_search_unavailable_503():
    with _app(searcher=_StubSearcher(ready=True)) as client:
        response = client.post("/chat", json={"question": QUESTION})
    assert response.status_code == 503
    assert response.json() == {"error": MSG_SEARCH_DOWN}


def test_qdrant_down_503():
    with _app(searcher=_StubSearcher(ready=False, qdrant_up=False)) as client:
        response = client.post("/chat", json={"question": QUESTION})
    assert response.status_code == 503
    assert response.json() == {"error": MSG_SEARCH_DOWN}


def test_empty_collection_503():
    with _app(searcher=_searcher(chunks=[])) as client:
        response = client.post("/chat", json={"question": QUESTION})
    assert response.status_code == 503
    assert response.json() == {"error": MSG_NO_DATA}


def test_embedding_failure_503_not_500():
    with _app(searcher=_searcher(embedder=FailingEmbedder())) as client:
        response = client.post("/chat", json={"question": QUESTION})
    assert response.status_code == 503
    assert response.json() == {"error": MSG_SEARCH_DOWN}


def test_llm_error_503_with_sources():
    with _app(llm=FakeLLM([LLMError("Gemini lỗi mã 500")])) as client:
        response = client.post("/chat", json={"question": QUESTION})
    assert response.status_code == 503
    body = response.json()
    assert body["error"] == MSG_LLM_DOWN
    assert {s["page_number"] for s in body["sources"]} == {c.page_number for c in CHUNKS}


def test_missing_api_key_503():
    with _app(llm=None) as client:
        response = client.post("/chat", json={"question": QUESTION})
    assert response.status_code == 503
    assert response.json()["error"] == MSG_NO_KEY
    assert response.json()["sources"]


def test_pipeline_factory_failure_still_starts():
    def broken():
        raise OSError("không tải được model")

    with TestClient(create_app(broken)) as client:
        health = client.get("/health")
        chat = client.post("/chat", json={"question": QUESTION})
    assert health.status_code == 200
    assert health.json()["models_loaded"] is False
    assert chat.status_code == 503


def test_health_reports_components():
    with _app() as client:
        assert client.get("/health").json() == {
            "qdrant": True,
            "collection_ready": True,
            "models_loaded": True,
            "llm_configured": True,
        }
    with _app(searcher=_searcher(chunks=[]), llm=None) as client:
        assert client.get("/health").json() == {
            "qdrant": True,
            "collection_ready": False,
            "models_loaded": True,
            "llm_configured": False,
        }
    with _app(searcher=_StubSearcher(ready=False, qdrant_up=False)) as client:
        body = client.get("/health").json()
        assert body["qdrant"] is False
        assert body["collection_ready"] is False


def test_models_loaded_once():
    calls = []
    pipeline = RagPipeline(_searcher(), FakeReranker(), FakeLLM(["Có [1]."] * 3), Settings())

    def factory():
        calls.append(1)
        return pipeline

    with TestClient(create_app(factory)) as client:
        for _ in range(3):
            assert client.post("/chat", json={"question": QUESTION}).status_code == 200
    assert calls == [1]


@pytest.mark.parametrize("debug", [False, True])
def test_logs_have_no_question_text_unless_debug(caplog, debug):
    secret = "cảnh báo học tập bimat123"
    caplog.set_level(logging.DEBUG)
    with _app(settings=Settings(debug=debug)) as client:
        assert client.post("/chat", json={"question": secret}).status_code == 200
    assert ("bimat123" in caplog.text) is debug
    request_lines = [r.getMessage() for r in caplog.records if r.name == "src.api.main"]
    assert any(f"q_len={len(secret)}" in line and "n_candidates=3" in line for line in request_lines)
