"""Chấm lại ứng viên bằng cross-encoder: đọc cả câu hỏi và đoạn văn cùng lúc nên chính xác hơn so vector.

`sentence_transformers` chỉ được import khi thật sự cần model, để import module này và chạy test luôn nhẹ.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from src.retrieval.hybrid_search import Hit


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        """Trả tối đa top_k Hit, giảm dần theo điểm; chỉ `score` đổi (trong [0, 1]), metadata giữ nguyên."""
        ...


class BgeReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cpu",
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device, max_length=self.max_length)
            if self.device.startswith("cuda"):
                self._model.model.half()  # float16: vừa GPU 4 GB cùng BGE-M3, nhanh hơn
        return self._model

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        if not hits:
            return []
        scores = self._load().predict([(query, hit.text) for hit in hits], show_progress_bar=False)
        rescored = [replace(hit, score=float(score)) for hit, score in zip(hits, scores)]
        return sorted(rescored, key=lambda hit: hit.score, reverse=True)[:top_k]
