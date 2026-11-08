import json
from uuid import NAMESPACE_URL, uuid5

import pytest
from qdrant_client import QdrantClient

from eval.dataset import Gold
from eval.evaluate_rewrite import DialogueCase, evaluate_rewrite, load_dialogues
from src.ingestion.chunker import Chunk
from src.ingestion.indexer import ensure_collection, index_corpus
from src.rag.llm import LLMError
from src.rag.models import Turn
from src.retrieval.hybrid_search import HybridSearcher
from tests.fakes import FakeEmbedder, FakeLLM, FakeReranker

COL = "test_eval_rewrite"
CHUNKS = [
    Chunk(str(uuid5(NAMESPACE_URL, f"tt08.pdf:{i}:{i}")), "tt08.pdf", i, f"Điều {i}. Tên", i, f"nội dung riêng số {i}")
    for i in range(1, 8)
]
CHUNKS[6] = Chunk(CHUNKS[6].chunk_id, "tt08.pdf", 7, "Điều 7. Cảnh báo", 6, "cảnh báo học tập khi điểm thấp")
STANDALONE = "cảnh báo học tập"
CASE = DialogueCase(
    "d1",
    (Turn("user", "Điểm thấp thì sao?"), Turn("assistant", "Có thể bị cảnh báo [1].")),
    "Vậy nội dung riêng số 3 thì sao?",  # câu gốc khớp Điều 3, không phải gold
    STANDALONE,
    (Gold("tt08.pdf", "Điều 7", (7,)),),
)


@pytest.fixture
def searcher():
    client = QdrantClient(":memory:")
    ensure_collection(client, COL, dense_dim=FakeEmbedder.dense_dim)
    index_corpus(client, CHUNKS, FakeEmbedder(), COL)
    return HybridSearcher(client, COL, FakeEmbedder())


def test_perfect_rewrite_has_zero_gap(searcher):
    result = evaluate_rewrite([CASE], FakeLLM([STANDALONE]), searcher, FakeReranker())
    assert result == {"recall@5_rewritten": 1.0, "recall@5_standalone": 1.0, "gap": 0.0, "rewrite_failures": 0, "n": 1}


def test_bad_rewrite_shows_positive_gap(searcher):
    result = evaluate_rewrite([CASE], FakeLLM(["nội dung riêng số 2"]), searcher, FakeReranker(), k=1)
    assert result["recall@1_rewritten"] == 0.0
    assert result["recall@1_standalone"] == 1.0
    assert result["gap"] == 1.0


def test_rewrite_failure_is_counted_and_falls_back_to_original(searcher):
    llm = FakeLLM([LLMError("timeout")])
    result = evaluate_rewrite([CASE], llm, searcher, FakeReranker(), k=1)
    assert result["rewrite_failures"] == 1
    assert result["recall@1_rewritten"] == 0.0  # dùng câu gốc: top-1 là Điều 3
    assert "riêng số 3" in llm.prompts[0]


def test_load_dialogues_reads_sample_file():
    cases = load_dialogues("eval/data/dialogues.jsonl")
    assert len(cases) >= 1
    case = cases[0]
    assert case.history and all(isinstance(t, Turn) for t in case.history)
    assert case.gold and case.gold[0].article.startswith("Điều")


@pytest.mark.parametrize(
    "patch, msg",
    [
        ({"history": []}, "history"),
        ({"history": [{"role": "bot", "content": "x"}]}, "role"),
        ({"standalone": " "}, "standalone"),
        ({"gold": [{"source_file": "a.pdf", "article": "Khoản 2", "pages": [1]}]}, "Điều"),
        ({"gold": []}, "gold"),
    ],
)
def test_load_dialogues_rejects_bad_rows(tmp_path, patch, msg):
    row = {
        "id": "d1",
        "history": [{"role": "user", "content": "a"}],
        "question": "b",
        "standalone": "c",
        "gold": [{"source_file": "a.pdf", "article": "Điều 1", "pages": [1]}],
    } | patch
    path = tmp_path / "d.jsonl"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=msg):
        load_dialogues(path)
