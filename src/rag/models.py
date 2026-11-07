"""Kiểu dữ liệu dùng chung, tách riêng để prompts, rewriter và pipeline không import vòng."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Turn:
    role: Literal["user", "assistant"]
    content: str
