"""Hỗ trợ gán nhãn: in top-k ứng viên hybrid cho một câu hỏi để người dùng tự chọn gold.

Chỉ hiển thị, không ghi vào questions.jsonl: nhãn phải do người đọc quy chế quyết định.

    .venv\\Scripts\\python -m eval.label_helper "Sinh viên bị cảnh báo học tập khi nào?"
"""

from __future__ import annotations

import argparse
import sys

from src.retrieval.hybrid_search import Hit


def format_candidates(hits: list[Hit], preview_chars: int = 200) -> str:
    lines = []
    for rank, hit in enumerate(hits, start=1):
        article = hit.article or "(không có Điều)"
        lines.append(f"{rank:>2}. {hit.source_file} | trang {hit.page_number} | {article} | {hit.score:.3f}")
        lines.append(f"    {' '.join(hit.text.split())[:preview_chars]}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="In top-k ứng viên hybrid để gán nhãn gold")
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--collection", default=None)
    args = parser.parse_args(argv)

    from src.config import load_settings
    from src.ingestion.embedder import BgeM3Embedder, default_device
    from src.ingestion.indexer import get_client
    from src.retrieval.hybrid_search import HybridSearcher, SearchUnavailableError

    settings = load_settings()
    embedder = BgeM3Embedder(settings.embedding_model, device=default_device())
    searcher = HybridSearcher(get_client(settings), args.collection or settings.collection, embedder)
    try:
        hits = searcher.search(args.question, mode="hybrid", limit=args.k)
    except SearchUnavailableError as exc:
        print(f"{exc}. Qdrant đã bật và đã index chưa?", file=sys.stderr)
        return 1
    print(format_candidates(hits))
    return 0


if __name__ == "__main__":
    sys.exit(main())
