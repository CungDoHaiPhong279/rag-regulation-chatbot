"""Dense embedding bằng BAAI/bge-m3.

`Embedder` là Protocol: bất kỳ object nào có `dense_dim` và `embed_dense` đều dùng được, nhờ vậy test
dùng `FakeEmbedder` (tests/fakes.py) mà không tải model ~2.3 GB. `sentence_transformers` (kéo theo torch)
chỉ được import bên trong `BgeM3Embedder.__init__` để `import src.ingestion.embedder` luôn nhẹ.
"""

from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    dense_dim: int

    def embed_dense(self, texts: list[str]) -> list[list[float]]:
        """Trả đúng len(texts) vector, mỗi vector dài dense_dim, đã chuẩn hóa L2."""
        ...


def default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class BgeM3Embedder:
    dense_dim = 1024

    def __init__(self, model_name: str = "BAAI/bge-m3", batch_size: int = 16, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)
        if device.startswith("cuda"):
            self._model.half()  # GPU 4 GB không chứa nổi BGE-M3 + reranker ở float32

    def embed_dense(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        result = vectors.tolist()
        if len(result) != len(texts) or any(len(v) != self.dense_dim for v in result):
            raise RuntimeError(
                f"Embedding sai kích thước: {len(result)} vector cho {len(texts)} văn bản, cần dim={self.dense_dim}"
            )
        return result
