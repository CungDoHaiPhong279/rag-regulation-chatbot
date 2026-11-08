import dataclasses

import pytest

from src.retrieval.hybrid_search import Hit
from src.retrieval.reranker import BgeReranker
from tests.fakes import FakeReranker


def _hit(i: int, text: str, score: float = 0.5) -> Hit:
    return Hit(
        chunk_id=f"id-{i}",
        source_file="a.pdf",
        page_number=i + 10,
        article=f"Điều {i}",
        chunk_index=i,
        text=text,
        score=score,
    )


class _StubCrossEncoder:
    """Thay CrossEncoder thật: điểm lấy theo bảng text -> score."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.pairs: list[tuple[str, str]] = []

    def predict(self, pairs, **kwargs):
        self.pairs = list(pairs)
        return [self.scores[text] for _, text in pairs]


def _reranker_with(scores: dict[str, float]) -> BgeReranker:
    reranker = BgeReranker()
    reranker._model = _StubCrossEncoder(scores)
    return reranker


def test_fake_reranker_orders_and_truncates():
    hits = [_hit(0, "khác hẳn"), _hit(1, "học phí"), _hit(2, "học"), _hit(3, "không")]
    out = FakeReranker().rerank("học phí", hits, top_k=2)
    assert [h.chunk_id for h in out] == ["id-1", "id-2"]
    assert out[0].score >= out[1].score


def test_rerank_orders_by_score_and_truncates():
    hits = [_hit(0, "a"), _hit(1, "b"), _hit(2, "c")]
    out = _reranker_with({"a": 0.1, "b": 0.9, "c": 0.5}).rerank("q", hits, top_k=2)
    assert [h.chunk_id for h in out] == ["id-1", "id-2"]
    assert [h.score for h in out] == [0.9, 0.5]


def test_rerank_scores_query_against_each_hit_text():
    reranker = _reranker_with({"a": 0.1, "b": 0.9})
    reranker.rerank("câu hỏi", [_hit(0, "a"), _hit(1, "b")], top_k=2)
    assert reranker._model.pairs == [("câu hỏi", "a"), ("câu hỏi", "b")]


def test_rerank_preserves_metadata():
    original = _hit(3, "a", score=0.016)
    (out,) = _reranker_with({"a": 0.7}).rerank("q", [original], top_k=5)
    assert out.score == 0.7
    assert dataclasses.replace(out, score=original.score) == original  # chỉ score đổi, page_number giữ nguyên


def test_rerank_score_is_plain_float():
    import numpy as np

    (out,) = _reranker_with({"a": np.float32(0.25)}).rerank("q", [_hit(0, "a")], top_k=1)
    assert type(out.score) is float


def test_rerank_empty_hits_does_not_load_model(monkeypatch):
    reranker = BgeReranker()
    monkeypatch.setattr(reranker, "_load", lambda: pytest.fail("không được tải model khi không có ứng viên"))
    assert reranker.rerank("q", [], top_k=5) == []


def test_constructor_does_not_load_model():
    assert BgeReranker()._model is None


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


@pytest.mark.slow
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_bge_reranker_prefers_relevant(device):
    if device == "cuda" and not _cuda_available():
        pytest.skip("Máy không có GPU CUDA")
    hits = [
        _hit(0, "Điều 14. Công nhận tốt nghiệp: Sinh viên tích lũy đủ tín chỉ được xét công nhận tốt nghiệp."),
        _hit(1, "Điều 11. Xử lý kết quả học tập: Sinh viên bị cảnh báo học tập khi điểm trung bình dưới mức quy định."),
    ]
    out = BgeReranker(device=device).rerank("Sinh viên bị cảnh báo học tập khi nào?", hits, top_k=2)
    assert out[0].chunk_id == "id-1"
    assert all(0.0 <= h.score <= 1.0 for h in out)
    assert all(type(h.score) is float for h in out)
