"""Bản giả dùng chung cho test: không tải model, không gọi mạng."""

from __future__ import annotations

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
