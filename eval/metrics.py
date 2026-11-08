"""Chỉ số truy xuất với nhãn nhị phân: chunk liên quan ⇔ cùng file ∧ cùng số Điều ∧ trang thuộc gold."""

from __future__ import annotations

import re
from typing import Sequence
import unicodedata

from eval.dataset import Gold
from src.retrieval.hybrid_search import Hit

_ARTICLE = re.compile(r"^Điều\s+(\d+)")


def article_number(article: str | None) -> int | None:
    """So bằng số Điều để "Điều 11. Xử lý..." của chunk khớp với nhãn "Điều 11"."""
    if not article:
        return None
    match = _ARTICLE.match(unicodedata.normalize("NFC", article).strip())
    return int(match.group(1)) if match else None


def is_relevant(hit: Hit, gold: Gold) -> bool:
    number = article_number(hit.article)
    return (
        hit.source_file == gold.source_file
        and number is not None
        and number == article_number(gold.article)
        and hit.page_number in gold.pages
    )


def recall_at_k(hits: list[Hit], gold: Sequence[Gold], k: int) -> float:
    """Tỉ lệ gold có ít nhất một chunk liên quan trong top-k."""
    if not gold:
        raise ValueError("recall cần ít nhất một gold")
    top = hits[:k]
    return sum(any(is_relevant(hit, g) for hit in top) for g in gold) / len(gold)


def mrr_at_k(hits: list[Hit], gold: Sequence[Gold], k: int) -> float:
    for rank, hit in enumerate(hits[:k], start=1):
        if any(is_relevant(hit, g) for g in gold):
            return 1.0 / rank
    return 0.0
