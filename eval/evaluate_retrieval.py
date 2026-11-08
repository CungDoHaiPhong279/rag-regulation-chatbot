"""Đo truy xuất theo 3 cấu hình: dense, hybrid (RRF), hybrid + rerank. Gọi thẳng searcher/reranker để tách từng tầng.

    .venv\\Scripts\\python -m eval.evaluate_retrieval --split test
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from eval.dataset import Question, load_questions
from eval.metrics import mrr_at_k, recall_at_k
from src.config import Settings, load_settings
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.reranker import Reranker

CONFIGS = ("dense", "hybrid", "hybrid_rerank")


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4)


def evaluate(
    questions: list[Question],
    searcher: HybridSearcher,
    reranker: Reranker | None,
    candidate_k: int = 20,
    final_k: int = 5,
) -> dict[str, dict[str, float]]:
    in_scope = [q for q in questions if q.in_scope]  # câu ngoài phạm vi không có gold, chỉ đo ở bước từ chối
    if not in_scope:
        raise ValueError("Không có câu in_scope nào để đo truy xuất")

    scores: dict[str, dict[str, list[float]]] = {}

    def add(config: str, name: str, value: float) -> None:
        scores.setdefault(config, {}).setdefault(name, []).append(value)

    for q in in_scope:
        for mode in ("dense", "hybrid"):
            candidates = searcher.search(q.question, mode=mode, limit=candidate_k)
            add(mode, f"recall@{final_k}", recall_at_k(candidates, q.gold, final_k))
            add(mode, f"recall@{candidate_k}", recall_at_k(candidates, q.gold, candidate_k))
            add(mode, f"mrr@{final_k}", mrr_at_k(candidates, q.gold, final_k))
            if mode == "hybrid" and reranker is not None:
                reranked = reranker.rerank(q.question, candidates, top_k=final_k)
                add("hybrid_rerank", f"recall@{final_k}", recall_at_k(reranked, q.gold, final_k))
                add("hybrid_rerank", f"mrr@{final_k}", mrr_at_k(reranked, q.gold, final_k))

    return {
        config: {**{name: _mean(values) for name, values in metrics.items()}, "n": len(in_scope)}
        for config, metrics in scores.items()
    }


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_metadata(settings: Settings, collection: str, split: str, rerank: bool) -> dict[str, Any]:
    return {
        "date": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "split": split,
        "collection": collection,
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model if rerank else None,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "candidate_k": settings.candidate_k,
        "final_k": settings.final_k,
    }


def write_report(results: dict, meta: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"retrieval_{meta['split']}_{meta['collection']}_{stamp}.json"
    path.write_text(json.dumps({"meta": meta, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Đánh giá truy xuất: dense / hybrid / hybrid + rerank")
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--questions", type=Path, default=Path("eval/data/questions.jsonl"))
    parser.add_argument("--collection", default=None, help="Mặc định lấy QDRANT_COLLECTION trong .env")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("eval/reports"))
    args = parser.parse_args(argv)

    settings = load_settings()
    collection = args.collection or settings.collection
    try:
        questions = load_questions(args.questions, split=args.split)
    except (OSError, ValueError) as exc:
        print(f"Lỗi bộ câu hỏi: {exc}", file=sys.stderr)
        return 1

    from src.ingestion.embedder import BgeM3Embedder, default_device
    from src.ingestion.indexer import check_connection, get_client
    from src.retrieval.reranker import BgeReranker

    client = get_client(settings)
    try:
        check_connection(client, settings.qdrant_path or settings.qdrant_url)
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 1
    device = default_device()
    searcher = HybridSearcher(client, collection, BgeM3Embedder(settings.embedding_model, device=device), settings.candidate_k)
    if not searcher.is_ready():
        print(f"Collection '{collection}' chưa có dữ liệu: hãy chạy indexer.", file=sys.stderr)
        return 1
    reranker = None if args.no_rerank else BgeReranker(settings.reranker_model, device=device)

    try:
        results = evaluate(questions, searcher, reranker, settings.candidate_k, settings.final_k)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    path = write_report(results, build_metadata(settings, collection, args.split, reranker is not None), args.out)

    print(f"Split {args.split}, collection {collection}, {len(questions)} câu ({results['dense']['n']} trong phạm vi)")
    for config, metrics in results.items():
        print(f"  {config:<14} " + "  ".join(f"{name}={value}" for name, value in metrics.items()))
    print(f"Báo cáo: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
