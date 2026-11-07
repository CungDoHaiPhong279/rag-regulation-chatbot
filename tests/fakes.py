"""Bản giả dùng chung cho test: không tải model, không gọi mạng."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import math


class FakeEmbedder:
    """Vector tất định 8 chiều băm từ sha256(text), đã chuẩn hóa L2."""

    dense_dim = 8

    def __init__(self) -> None:
        self.calls = 0

    def embed_dense(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        out = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            raw = [b - 127.5 for b in digest[: self.dense_dim]]
            norm = math.sqrt(sum(x * x for x in raw)) or 1.0
            out.append([x / norm for x in raw])
        return out


class FailingEmbedder:
    dense_dim = 8

    def embed_dense(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embed failed")


class FakeReranker:
    """Điểm = tỉ lệ từ của câu hỏi có trong text; giữ thứ tự cũ khi hòa."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def rerank(self, query, hits, top_k):
        self.calls.append((query, top_k))
        query_tokens = set(query.lower().split())

        def score(hit) -> float:
            return len(query_tokens & set(hit.text.lower().split())) / (len(query_tokens) or 1)

        scored = [replace(hit, score=score(hit)) for hit in hits]
        return sorted(scored, key=lambda h: h.score, reverse=True)[:top_k]
