"""Cấu hình tập trung của dự án, đọc từ biến môi trường / file .env.

Mọi tham số có thể đổi mà không sửa code (model, ngưỡng, chunk_size...) đều nằm ở đây.
Chuỗi rỗng trong .env được hiểu là "không đặt" (None) để `.env.example` có thể để trống key.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import os
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = field(default=None, repr=False)
    qdrant_path: str | None = None
    collection: str = "regulations"
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    gemini_api_key: str | None = field(default=None, repr=False)
    gemini_model: str | None = None
    chunk_size: int = 800
    chunk_overlap: int = 150
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    candidate_k: int = 20
    final_k: int = 5
    rerank_threshold: float = 0.3  # giá trị khởi đầu, chỉnh trên tập dev (Task 15)
    llm_timeout_s: float = 30.0
    debug: bool = False


# Tên trường -> tên biến môi trường
_ENV_NAMES = {
    "qdrant_url": "QDRANT_URL",
    "qdrant_api_key": "QDRANT_API_KEY",
    "qdrant_path": "QDRANT_PATH",
    "collection": "QDRANT_COLLECTION",
    "embedding_model": "EMBEDDING_MODEL",
    "reranker_model": "RERANKER_MODEL",
    "gemini_api_key": "GEMINI_API_KEY",
    "gemini_model": "GEMINI_MODEL",
    "chunk_size": "CHUNK_SIZE",
    "chunk_overlap": "CHUNK_OVERLAP",
    "bm25_k1": "BM25_K1",
    "bm25_b": "BM25_B",
    "candidate_k": "CANDIDATE_K",
    "final_k": "FINAL_K",
    "rerank_threshold": "RERANK_THRESHOLD",
    "llm_timeout_s": "LLM_TIMEOUT_S",
    "debug": "DEBUG",
}


def _parse_bool(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes"}


def load_settings(env_file: str | Path | None = ".env") -> Settings:
    """Nạp Settings. Biến môi trường đã có sẵn được ưu tiên hơn giá trị trong file .env."""
    if env_file is not None:
        load_dotenv(env_file, override=False)

    defaults = Settings()
    values: dict[str, Any] = {}
    for f in fields(Settings):
        env_name = _ENV_NAMES[f.name]
        raw = os.getenv(env_name)
        if raw is None:
            continue
        raw = raw.strip()
        default = getattr(defaults, f.name)
        if raw == "":
            # Chuỗi rỗng: trường tùy chọn (mặc định None) -> None; trường khác -> giữ mặc định
            if default is None:
                values[f.name] = None
            continue
        caster: Callable[[str], Any]
        if isinstance(default, bool):
            caster = _parse_bool
        elif isinstance(default, int):
            caster = int
        elif isinstance(default, float):
            caster = float
        else:
            caster = str
        try:
            values[f.name] = caster(raw)
        except ValueError as exc:
            raise ValueError(f"Giá trị không hợp lệ cho {env_name}={raw!r}: {exc}") from exc
    return Settings(**values)
