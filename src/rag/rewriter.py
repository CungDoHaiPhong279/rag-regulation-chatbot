"""Viết lại câu hỏi nối tiếp thành câu độc lập để truy xuất; lỗi thì dùng câu gốc, không làm hỏng cả lượt hỏi."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from src.rag.llm import LLMClient, LLMError
from src.rag.models import Turn
from src.rag.prompts import REWRITE_SYSTEM_PROMPT, build_rewrite_prompt

# Giới hạn để hội thoại dài không làm prompt, chi phí và thời gian tăng theo.
MAX_HISTORY_TURNS = 3
MAX_TURN_CHARS = 500


@dataclass(frozen=True)
class RewriteResult:
    query: str
    failed: bool


def rewrite(question: str, history: Sequence[Turn], llm: LLMClient | None) -> RewriteResult:
    if not history or llm is None:
        return RewriteResult(question, failed=False)

    recent = [replace(turn, content=turn.content[:MAX_TURN_CHARS]) for turn in history[-MAX_HISTORY_TURNS:]]
    try:
        query = llm.generate(build_rewrite_prompt(question, recent), system=REWRITE_SYSTEM_PROMPT).strip()
    except LLMError:
        return RewriteResult(question, failed=True)
    return RewriteResult(query, failed=False) if query else RewriteResult(question, failed=True)
