"""Đo câu trả lời của pipeline đầy đủ (Hybrid + Rerank + LLM): từ chối đúng / nhầm, trích dẫn hợp lệ; và quét ngưỡng lớp 1.

    .venv\\Scripts\\python -m eval.evaluate_answers --split test
    .venv\\Scripts\\python -m eval.evaluate_answers --split dev --sweep 0.1,0.2,0.3,0.4,0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Sequence

from eval.dataset import Question, load_questions
from eval.evaluate_retrieval import build_metadata, write_report
from eval.metrics import is_relevant
from src.rag.pipeline import LLMUnavailableError, RagPipeline


def _rate(hits: int, total: int) -> float | None:
    # None thay cho 0.0 khi không có mẫu: 0.0 sẽ bị đọc nhầm là "luôn sai".
    return round(hits / total, 4) if total else None


def evaluate_answers(questions: list[Question], pipeline: RagPipeline, delay_s: float = 0.0) -> dict[str, float | None]:
    """Câu bị lỗi LLM không biết kết quả nên bị loại khỏi mọi mẫu số và chỉ đếm vào llm_errors.

    delay_s: nghỉ giữa các câu để không vượt giới hạn số lần gọi/phút của Gemini (lỗi 429).
    """
    refused_out = n_out = refused_in = n_in = cited_ok = n_answered = llm_errors = 0
    for i, q in enumerate(questions):
        if i and delay_s:
            time.sleep(delay_s)
        try:
            answer = pipeline.answer(q.question)
        except LLMUnavailableError:
            llm_errors += 1
            continue
        if not q.in_scope:
            n_out += 1
            refused_out += answer.refused
            continue
        n_in += 1
        if answer.refused:
            refused_in += 1
            continue
        n_answered += 1
        # Citation có đủ source_file/page_number/article như Hit nên dùng chung tiêu chí "trúng" với Recall.
        cited_ok += any(is_relevant(c, g) for c in answer.citations for g in q.gold)
    return {
        "correct_refusal_rate": _rate(refused_out, n_out),
        "false_refusal_rate": _rate(refused_in, n_in),
        "citation_valid_rate": _rate(cited_ok, n_answered),
        "llm_errors": llm_errors,
        "n_out_of_scope": n_out,
        "n_in_scope": n_in,
    }


def sweep_threshold(questions: list[Question], pipeline: RagPipeline, thresholds: Sequence[float]) -> list[dict]:
    """Chỉ lớp 1 (điểm rerank cao nhất < ngưỡng -> từ chối), không gọi LLM; lớp 2 có thể từ chối thêm."""
    if any(q.split != "dev" for q in questions):
        raise ValueError("sweep_threshold chỉ chạy trên split dev (chỉnh ngưỡng trên test là rò rỉ dữ liệu)")
    # Truy xuất + rerank một lần mỗi câu; điểm cao nhất None khi không có ứng viên (luôn bị từ chối).
    tops = [(q.in_scope, hits[0].score if hits else None) for q in questions for hits in [pipeline.retrieve(q.question)]]
    rows = []
    for t in thresholds:
        refused = [(in_scope, top is None or top < t) for in_scope, top in tops]
        out = [r for in_scope, r in refused if not in_scope]
        ins = [r for in_scope, r in refused if in_scope]
        rows.append({
            "threshold": t,
            "correct_refusal_rate": _rate(sum(out), len(out)),
            "false_refusal_rate": _rate(sum(ins), len(ins)),
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Đánh giá câu trả lời: từ chối, trích dẫn; quét ngưỡng rerank")
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--questions", type=Path, default=Path("eval/data/questions.jsonl"))
    parser.add_argument("--sweep", default=None, help="Danh sách ngưỡng, vd. 0.1,0.2,0.3 (chỉ split dev)")
    parser.add_argument("--delay", type=float, default=0.0, help="Số giây nghỉ giữa các câu (tránh lỗi 429)")
    parser.add_argument("--out", type=Path, default=Path("eval/reports"))
    args = parser.parse_args(argv)

    if args.sweep is not None and args.split != "dev":
        print("--sweep chỉ chạy với --split dev", file=sys.stderr)
        return 1
    try:
        thresholds = [float(x) for x in args.sweep.split(",")] if args.sweep is not None else None
        questions = load_questions(args.questions, split=args.split)
    except (OSError, ValueError) as exc:
        print(f"Lỗi tham số hoặc bộ câu hỏi: {exc}", file=sys.stderr)
        return 1

    from src.api.main import default_pipeline_factory  # import muộn: tránh nạp torch khi chỉ báo lỗi tham số

    pipeline = default_pipeline_factory()
    if not pipeline.searcher.is_ready():
        print(f"Không đọc được collection '{pipeline.settings.collection}': bật Qdrant và chạy indexer.", file=sys.stderr)
        return 1
    if thresholds is None and pipeline.llm is None:
        print("Chưa cấu hình GEMINI_API_KEY và GEMINI_MODEL", file=sys.stderr)
        return 1

    settings = pipeline.settings
    meta = build_metadata(settings, settings.collection, args.split, rerank=True)
    if thresholds is not None:
        results: object = sweep_threshold(questions, pipeline, thresholds)
        path = write_report(results, meta, args.out, kind="threshold")
        for row in results:
            print(f"  ngưỡng {row['threshold']:<5} từ chối đúng={row['correct_refusal_rate']}  "
                  f"từ chối nhầm={row['false_refusal_rate']}")
    else:
        meta |= {"llm_model": settings.gemini_model, "rerank_threshold": settings.rerank_threshold}
        results = evaluate_answers(questions, pipeline, args.delay)
        path = write_report(results, meta, args.out, kind="answers")
        print("  " + "  ".join(f"{name}={value}" for name, value in results.items()))
    print(f"Split {args.split}, {len(questions)} câu. Báo cáo: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
