from uuid import NAMESPACE_URL, uuid5

import pytest
from qdrant_client import QdrantClient

from eval.dataset import Gold, Question
from eval.evaluate_answers import evaluate_answers, main, sweep_threshold
from src.config import Settings
from src.ingestion.chunker import Chunk
from src.ingestion.indexer import ensure_collection, index_corpus
from src.rag.constants import REFUSAL
from src.rag.llm import LLMError
from src.rag.pipeline import RagPipeline
from src.retrieval.hybrid_search import HybridSearcher
from tests.fakes import FakeEmbedder, FakeLLM, FakeReranker

COL = "test_eval_answers"
CHUNKS = [
    Chunk(str(uuid5(NAMESPACE_URL, f"tt08.pdf:{i}:{i}")), "tt08.pdf", i, f"Điều {i}. Tên", i, f"nội dung riêng số {i}")
    for i in range(1, 8)
]
CHUNKS[6] = Chunk(CHUNKS[6].chunk_id, "tt08.pdf", 7, "Điều 7. Cảnh báo", 6, "cảnh báo học tập khi điểm thấp")
GOLD7 = (Gold("tt08.pdf", "Điều 7", (7,)),)
IN = "cảnh báo học tập"  # FakeReranker: điểm 1.0 với chunk Điều 7
OUT = "lệ phí thi"  # không từ nào khớp: điểm 0 -> lớp 1 từ chối, không gọi LLM


def _q(id_, question, in_scope=True, split="dev", gold=GOLD7):
    return Question(id_, question, in_scope, split, gold if in_scope else ())


@pytest.fixture
def searcher():
    client = QdrantClient(":memory:")
    ensure_collection(client, COL, dense_dim=FakeEmbedder.dense_dim)
    index_corpus(client, CHUNKS, FakeEmbedder(), COL)
    return HybridSearcher(client, COL, FakeEmbedder())


def _pipeline(searcher, responses, threshold=0.3):
    return RagPipeline(searcher, FakeReranker(), FakeLLM(responses), Settings(rerank_threshold=threshold))


def test_out_of_scope_refused_counts_as_correct_refusal(searcher):
    result = evaluate_answers([_q("o1", OUT, in_scope=False), _q("o2", OUT, in_scope=False)], _pipeline(searcher, []))
    assert result["correct_refusal_rate"] == 1.0
    assert result["llm_errors"] == 0


def test_out_of_scope_answered_is_not_correct_refusal(searcher):
    # Ngưỡng 0 cho câu ngoài phạm vi qua lớp 1; LLM trả lời có trích dẫn -> chatbot bịa.
    result = evaluate_answers([_q("o1", OUT, in_scope=False)], _pipeline(searcher, ["Học phí là X [1]."], threshold=0.0))
    assert result["correct_refusal_rate"] == 0.0


def test_false_refusal_rate_and_citation_validity(searcher):
    questions = [_q("i1", IN), _q("i2", IN)]
    result = evaluate_answers(questions, _pipeline(searcher, ["Bị cảnh báo khi điểm thấp [1].", REFUSAL]))
    assert result["false_refusal_rate"] == 0.5
    assert result["citation_valid_rate"] == 1.0  # câu không từ chối trích đúng chunk gold


def test_citation_not_matching_gold_is_invalid(searcher):
    wrong_gold = (Gold("tt08.pdf", "Điều 3", (3,)),)
    result = evaluate_answers([_q("i1", IN, gold=wrong_gold)], _pipeline(searcher, ["Trả lời [1]."]))
    assert result["false_refusal_rate"] == 0.0
    assert result["citation_valid_rate"] == 0.0


def test_llm_error_counted_separately_not_as_refusal(searcher):
    questions = [_q("i1", IN), _q("i2", IN)]
    result = evaluate_answers(questions, _pipeline(searcher, [LLMError("503"), "Trả lời [1]."]))
    assert result["llm_errors"] == 1
    assert result["false_refusal_rate"] == 0.0  # câu lỗi bị loại khỏi mẫu số, không tính là từ chối
    assert result["citation_valid_rate"] == 1.0


def test_rates_without_samples_are_none(searcher):
    result = evaluate_answers([_q("i1", IN)], _pipeline(searcher, [REFUSAL]))
    assert result["correct_refusal_rate"] is None
    assert result["citation_valid_rate"] is None  # không có câu nào được trả lời


def test_sweep_threshold_rejects_test_split(searcher):
    with pytest.raises(ValueError, match="dev"):
        sweep_threshold([_q("i1", IN, split="test")], _pipeline(searcher, []), [0.1])


def test_sweep_threshold_applies_layer1_without_llm(searcher):
    pipeline = _pipeline(searcher, [])  # FakeLLM rỗng: gọi LLM sẽ lỗi IndexError
    questions = [_q("i1", IN), _q("o1", OUT, in_scope=False)]
    rows = sweep_threshold(questions, pipeline, [0.0, 0.5, 1.5])
    assert [r["threshold"] for r in rows] == [0.0, 0.5, 1.5]
    # điểm cao nhất: câu trong phạm vi 1.0, câu ngoài phạm vi 0.0; từ chối khi điểm < ngưỡng
    assert [(r["correct_refusal_rate"], r["false_refusal_rate"]) for r in rows] == [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    assert pipeline.llm.prompts == []
    assert len(pipeline.reranker.calls) == 2  # truy xuất + rerank một lần mỗi câu, không lặp theo ngưỡng


def test_main_sweep_on_test_split_fails_without_loading_models(tmp_path, capsys):
    assert main(["--split", "test", "--sweep", "0.1,0.2", "--out", str(tmp_path)]) == 1
    assert "dev" in capsys.readouterr().err
