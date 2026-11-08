"""Đọc bộ câu hỏi đánh giá (JSONL), kiểm tra chặt để lỗi gán nhãn không lặng lẽ làm sai chỉ số."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

SPLITS = ("dev", "test")


@dataclass(frozen=True)
class Gold:
    source_file: str
    article: str  # "Điều 11"
    pages: tuple[int, ...]


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    in_scope: bool
    split: Literal["dev", "test"]
    gold: tuple[Gold, ...]


def parse_gold(rows: list[dict]) -> tuple[Gold, ...]:
    from eval.metrics import article_number  # import trong hàm: metrics cũng import Gold từ module này

    gold = []
    for g in rows:
        pages = tuple(int(p) for p in g["pages"])
        if not pages:
            raise ValueError("gold.pages rỗng")
        if article_number(g["article"]) is None:
            # Nhãn kiểu "Khoản 2, Điều 11" không bao giờ khớp chunk nào: Recall sẽ âm thầm bằng 0.
            raise ValueError(f"gold.article phải bắt đầu bằng 'Điều <số>', nhận {g['article']!r}")
        gold.append(Gold(g["source_file"], g["article"], pages))
    return tuple(gold)


def _parse(row: dict) -> Question:
    question = row["question"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question rỗng")
    if row["split"] not in SPLITS:
        raise ValueError(f"split phải là dev hoặc test, nhận {row['split']!r}")
    gold = parse_gold(row["gold"])
    in_scope = bool(row["in_scope"])
    if in_scope and not gold:
        raise ValueError("câu in_scope=true phải có gold")
    if not in_scope and gold:
        raise ValueError("câu in_scope=false không được có gold")
    return Question(str(row["id"]), question.strip(), in_scope, row["split"], gold)


def load_questions(path: Path, split: str | None = None) -> list[Question]:
    if split is not None and split not in SPLITS:
        raise ValueError(f"split phải là dev hoặc test, nhận {split!r}")
    questions: list[Question] = []
    seen: set[str] = set()
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            question = _parse(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} dòng {line_no}: JSON hỏng ({exc.msg})") from exc
        except KeyError as exc:
            raise ValueError(f"{path} dòng {line_no}: thiếu trường {exc.args[0]}") from exc
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path} dòng {line_no}: {exc}") from exc
        if question.id in seen:
            raise ValueError(f"{path} dòng {line_no}: id {question.id!r} bị trùng")
        seen.add(question.id)
        questions.append(question)
    return [q for q in questions if split is None or q.split == split]
