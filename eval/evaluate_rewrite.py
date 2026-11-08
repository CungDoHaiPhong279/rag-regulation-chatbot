"""Đo mất mát do bước viết lại câu hỏi: Recall@k với câu LLM viết lại so với câu độc lập viết tay (cùng Hybrid + Rerank).

    .venv\\Scripts\\python -m eval.evaluate_rewrite
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

from eval.dataset import Gold, parse_gold
from eval.metrics import recall_at_k
from src.rag.llm import LLMClient
from src.rag.models import Turn
from src.rag.rewriter import rewrite
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.reranker import Reranker


@dataclass(frozen=True)
class DialogueCase:
    id: str
    history: tuple[Turn, ...]
    question: str
    standalone: str
    gold: tuple[Gold, ...]


def _text(row: dict, key: str) -> str:
    value = row[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} rỗng")
    return value.strip()


def _parse(row: dict) -> DialogueCase:
    history = []
    for turn in row["history"]:
        if turn["role"] not in ("user", "assistant"):
            raise ValueError(f"history.role phải là user hoặc assistant, nhận {turn['role']!r}")
        history.append(Turn(turn["role"], _text(turn, "content")))
    if not history:
        raise ValueError("history rỗng: câu không có ngữ cảnh thì không cần viết lại")
    gold = parse_gold(row["gold"])
    if not gold:
        raise ValueError("gold rỗng")
    return DialogueCase(str(row["id"]), tuple(history), _text(row, "question"), _text(row, "standalone"), gold)


def load_dialogues(path: Path) -> list[DialogueCase]:
    cases = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cases.append(_parse(json.loads(line)))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} dòng {line_no}: JSON hỏng ({exc.msg})") from exc
        except KeyError as exc:
            raise ValueError(f"{path} dòng {line_no}: thiếu trường {exc.args[0]}") from exc
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path} dòng {line_no}: {exc}") from exc
    return cases


def evaluate_rewrite(
    cases: list[DialogueCase], llm: LLMClient, searcher: HybridSearcher, reranker: Reranker, k: int = 5
) -> dict[str, float]:
    if not cases:
        raise ValueError("Không có hội thoại nào để đo")

    def recall(query: str, gold: tuple[Gold, ...]) -> float:
        return recall_at_k(reranker.rerank(query, searcher.search(query, mode="hybrid"), top_k=k), gold, k)

    rewritten = standalone = 0.0
    failures = 0
    for case in cases:
        result = rewrite(case.question, case.history, llm)  # lỗi thì rewriter dùng câu gốc, giống pipeline thật
        failures += result.failed
        rewritten += recall(result.query, case.gold)
        standalone += recall(case.standalone, case.gold)
    r_rw, r_sa = round(rewritten / len(cases), 4), round(standalone / len(cases), 4)
    return {
        f"recall@{k}_rewritten": r_rw,
        f"recall@{k}_standalone": r_sa,
        "gap": round(r_sa - r_rw, 4),  # > 0: bước viết lại làm mất Recall
        "rewrite_failures": failures,
        "n": len(cases),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Đánh giá bước viết lại câu hỏi nối tiếp")
    parser.add_argument("--dialogues", type=Path, default=Path("eval/data/dialogues.jsonl"))
    args = parser.parse_args(argv)
    try:
        cases = load_dialogues(args.dialogues)
    except (OSError, ValueError) as exc:
        print(f"Lỗi bộ hội thoại: {exc}", file=sys.stderr)
        return 1

    from src.api.main import default_pipeline_factory  # import muộn: tránh nạp torch khi chỉ báo lỗi dữ liệu

    pipeline = default_pipeline_factory()
    if pipeline.llm is None or not pipeline.searcher.is_ready():
        print("Cần GEMINI_API_KEY/GEMINI_MODEL và collection Qdrant đã index.", file=sys.stderr)
        return 1
    try:
        result = evaluate_rewrite(cases, pipeline.llm, pipeline.searcher, pipeline.reranker, pipeline.settings.final_k)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    print("  " + "  ".join(f"{name}={value}" for name, value in result.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
