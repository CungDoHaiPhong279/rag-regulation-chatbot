"""Luồng trả lời: viết lại câu hỏi -> hybrid search -> rerank -> 2 lớp chống bịa -> trích dẫn từ metadata.

Lớp 1: điểm rerank cao nhất dưới ngưỡng -> từ chối, không gọi LLM.
Lớp 2: prompt chỉ cho dùng context; câu trả lời không trích hoặc trích sai số [n] -> từ chối.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import time
from typing import Any, Iterator, Sequence

from src.config import Settings
from src.rag.citations import extract_citation_ids, is_refusal, renumber_citations
from src.rag.constants import REFUSAL
from src.rag.llm import LLMClient, LLMError
from src.rag.models import Turn
from src.rag.prompts import ANSWER_SYSTEM_PROMPT, build_answer_prompt
from src.rag.rewriter import rewrite
from src.retrieval.hybrid_search import Hit, HybridSearcher
from src.retrieval.reranker import Reranker


@dataclass(frozen=True)
class Citation:
    source_file: str
    page_number: int
    article: str | None
    text: str


@dataclass(frozen=True)
class Answer:
    answer: str
    citations: list[Citation]
    rewritten_query: str
    refused: bool
    debug: dict[str, Any]


class LLMUnavailableError(Exception):
    """LLM lỗi hoặc chưa cấu hình; kèm các nguồn đã truy xuất để người dùng vẫn tự tra được."""

    def __init__(self, message: str, sources: list[Citation]) -> None:
        super().__init__(message)
        self.sources = sources


def _citation(hit: Hit) -> Citation:
    return Citation(hit.source_file, hit.page_number, hit.article, hit.text)


@contextmanager
def _timed(timings: dict[str, float], step: str) -> Iterator[None]:
    started = time.perf_counter()
    yield
    timings[step] = round((time.perf_counter() - started) * 1000, 1)


class RagPipeline:
    def __init__(
        self,
        searcher: HybridSearcher,
        reranker: Reranker,
        llm: LLMClient | None,
        settings: Settings,
    ) -> None:
        self.searcher = searcher
        self.reranker = reranker
        self.llm = llm
        self.settings = settings

    def _search(self, query: str) -> list[Hit]:
        return self.searcher.search(query, mode="hybrid", limit=self.settings.candidate_k)

    def _rerank(self, query: str, candidates: list[Hit]) -> list[Hit]:
        return self.reranker.rerank(query, candidates, top_k=self.settings.final_k)

    def retrieve(self, query: str) -> list[Hit]:
        return self._rerank(query, self._search(query))

    def answer(self, question: str, history: Sequence[Turn] = ()) -> Answer:
        timings = dict.fromkeys(("rewrite", "search", "rerank", "llm"), 0.0)
        debug: dict[str, Any] = {
            "timings_ms": timings,
            "top_rerank_score": None,
            "rewrite_failed": False,
            "citation_check_failed": False,
            "n_candidates": 0,
        }

        with _timed(timings, "rewrite"):
            rewritten = rewrite(question, history, self.llm)
        query = rewritten.query
        debug["rewrite_failed"] = rewritten.failed

        def refuse() -> Answer:
            return Answer(REFUSAL, [], query, True, debug)

        with _timed(timings, "search"):
            candidates = self._search(query)
        debug["n_candidates"] = len(candidates)
        with _timed(timings, "rerank"):
            hits = self._rerank(query, candidates)

        if not hits:
            return refuse()
        debug["top_rerank_score"] = hits[0].score
        if hits[0].score < self.settings.rerank_threshold:
            return refuse()

        sources = [_citation(hit) for hit in hits]
        if self.llm is None:
            raise LLMUnavailableError("Chưa cấu hình GEMINI_API_KEY và GEMINI_MODEL", sources)
        try:
            with _timed(timings, "llm"):
                text = self.llm.generate(build_answer_prompt(query, hits), system=ANSWER_SYSTEM_PROMPT)
        except LLMError as exc:
            raise LLMUnavailableError(f"Không gọi được LLM: {exc}", sources) from exc

        if is_refusal(text):
            return refuse()

        ids = extract_citation_ids(text)
        if not ids or any(not 1 <= n <= len(hits) for n in ids):
            debug["citation_check_failed"] = True
            return refuse()

        # Số trang chỉ lấy từ metadata của hit, không lấy từ chữ của LLM (ADR-5).
        return Answer(renumber_citations(text.strip()), [sources[n - 1] for n in ids], query, False, debug)
