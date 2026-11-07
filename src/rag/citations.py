"""Chuẩn hóa câu từ chối và đọc số trích dẫn [n] trong câu trả lời của LLM."""

from __future__ import annotations

import re
import unicodedata

from src.rag.constants import REFUSAL

_DECORATION = " \t\r\n\"'“”‘’*`."
_CITATION = re.compile(r"\[([0-9]+(?:\s*,\s*[0-9]+)*)\]")


def normalize_refusal(answer: str) -> str:
    """Biến thể gần đúng của câu từ chối -> REFUSAL; câu khác -> answer.strip()."""
    core = " ".join(unicodedata.normalize("NFC", answer).split()).strip(_DECORATION)
    if core.casefold() == REFUSAL.rstrip(".").casefold():
        return REFUSAL
    return answer.strip()


def is_refusal(answer: str) -> bool:
    return normalize_refusal(answer) == REFUSAL


def extract_citation_ids(answer: str) -> list[int]:
    """"[1][3] ... [1]" -> [1, 3]: theo thứ tự xuất hiện, không trùng. Nhận cả dạng gộp "[1, 3]"."""
    ids = (int(n) for group in _CITATION.findall(answer) for n in group.split(","))
    return list(dict.fromkeys(ids))
