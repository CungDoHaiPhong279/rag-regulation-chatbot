"""Truy xuất trên Qdrant theo 3 chế độ: dense (BGE-M3), sparse (BM25) và hybrid (RRF)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from qdrant_client import QdrantClient, models

from src.ingestion.embedder import Embedder
from src.ingestion.indexer import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from src.ingestion.sparse import query_vector

SearchMode = Literal["dense", "sparse", "hybrid"]


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    source_file: str
    page_number: int
    article: str | None
    chunk_index: int
    text: str
    score: float


class SearchUnavailableError(Exception):
    """Không truy vấn được Qdrant."""


def _to_hit(point: models.ScoredPoint) -> Hit:
    payload = point.payload or {}
    return Hit(
        chunk_id=str(point.id),
        source_file=str(payload["source_file"]),
        page_number=int(payload["page_number"]),
        article=payload.get("article"),
        chunk_index=int(payload["chunk_index"]),
        text=str(payload["text"]),
        score=float(point.score),
    )


class HybridSearcher:
    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        embedder: Embedder,
        candidate_k: int = 20,
    ) -> None:
        self.client = client
        self.collection = collection
        self.embedder = embedder
        self.candidate_k = candidate_k

    def search(self, query: str, mode: SearchMode = "hybrid", limit: int = 20) -> list[Hit]:
        """Tìm top-`limit` chunk theo `mode` ('dense' | 'sparse' | 'hybrid').

        Khi câu hỏi không có token nào sau tách từ (vd. chỉ có dấu câu), vector sparse rỗng:
        - `sparse`: trả `[]` ngay, không gọi Qdrant.
        - `hybrid`: không gửi sparse vector rỗng mà chuyển sang tìm `dense` trực tiếp.
        """
        if mode not in ("dense", "sparse", "hybrid"):
            raise ValueError(f"Chế độ tìm kiếm không hợp lệ: {mode!r}")

        if mode == "sparse":
            sparse = query_vector(query)
            if not sparse.indices:
                return []
            try:
                response = self.client.query_points(
                    collection_name=self.collection,
                    query=models.SparseVector(indices=sparse.indices, values=sparse.values),
                    using=SPARSE_VECTOR_NAME,
                    limit=limit,
                    with_payload=True,
                )
            except Exception as exc:
                raise SearchUnavailableError(f"Lỗi truy vấn Qdrant collection '{self.collection}'") from exc
            return [_to_hit(p) for p in response.points]

        dense_vec = self.embedder.embed_dense([query])[0]

        if mode == "dense":
            return self._query_dense(dense_vec, limit)

        sparse = query_vector(query)
        if not sparse.indices:
            return self._query_dense(dense_vec, limit)

        try:
            response = self.client.query_points(
                collection_name=self.collection,
                prefetch=[
                    models.Prefetch(
                        query=dense_vec,
                        using=DENSE_VECTOR_NAME,
                        limit=self.candidate_k,
                    ),
                    models.Prefetch(
                        query=models.SparseVector(indices=sparse.indices, values=sparse.values),
                        using=SPARSE_VECTOR_NAME,
                        limit=self.candidate_k,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
        except Exception as exc:
            raise SearchUnavailableError(f"Lỗi truy vấn Qdrant collection '{self.collection}'") from exc
        return [_to_hit(p) for p in response.points]

    def _query_dense(self, dense_vec: list[float], limit: int) -> list[Hit]:
        try:
            response = self.client.query_points(
                collection_name=self.collection,
                query=dense_vec,
                using=DENSE_VECTOR_NAME,
                limit=limit,
                with_payload=True,
            )
        except Exception as exc:
            raise SearchUnavailableError(f"Lỗi truy vấn Qdrant collection '{self.collection}'") from exc
        return [_to_hit(p) for p in response.points]

    def is_ready(self) -> bool:
        try:
            if not self.client.collection_exists(self.collection):
                return False
            return self.client.count(collection_name=self.collection).count > 0
        except Exception:
            return False
