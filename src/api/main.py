"""FastAPI: POST /chat và GET /health. Chạy: uvicorn src.api.main:app --port 8000

Model tải một lần lúc khởi động (lifespan). Tải lỗi thì API vẫn chạy, /health báo models_loaded=false
và /chat trả 503, để người dùng thấy rõ thiếu gì thay vì server không lên.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict
import logging
from typing import Callable, Literal
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.rag.models import Turn
from src.rag.pipeline import LLMUnavailableError, RagPipeline
from src.retrieval.hybrid_search import SearchUnavailableError

logger = logging.getLogger(__name__)

MSG_SEARCH_DOWN = "Hệ thống tìm kiếm tạm thời không khả dụng"
MSG_NO_DATA = "Chưa có dữ liệu, hãy chạy indexer"
MSG_LLM_DOWN = "Mô hình ngôn ngữ tạm thời không khả dụng"
MSG_NO_KEY = "Chưa cấu hình GEMINI_API_KEY"

MAX_QUESTION_CHARS = 1000
MAX_HISTORY = 20
MAX_TURN_CHARS = 10_000  # chặn body khổng lồ; rewriter chỉ dùng 500 ký tự mỗi lượt


class TurnIn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=MAX_TURN_CHARS)


class ChatRequest(BaseModel):
    question: str
    history: list[TurnIn] = Field(default_factory=list, max_length=MAX_HISTORY)

    @field_validator("question")
    @classmethod
    def _check_question(cls, value: str) -> str:
        value = value.strip()
        if not any(ch.isalnum() for ch in value):
            raise ValueError("Câu hỏi phải có ít nhất một chữ hoặc số")
        if len(value) > MAX_QUESTION_CHARS:
            raise ValueError(f"Câu hỏi tối đa {MAX_QUESTION_CHARS} ký tự")
        return value


def default_pipeline_factory() -> RagPipeline:
    # Import ở đây để test (truyền factory giả) không phải nạp torch và model.
    from src.config import load_settings
    from src.ingestion.embedder import BgeM3Embedder
    from src.ingestion.indexer import get_client
    from src.rag.llm import GeminiClient
    from src.retrieval.hybrid_search import HybridSearcher
    from src.retrieval.reranker import BgeReranker

    settings = load_settings()
    searcher = HybridSearcher(
        get_client(settings), settings.collection, BgeM3Embedder(settings.embedding_model), settings.candidate_k
    )
    reranker = BgeReranker(settings.reranker_model)
    reranker._load()  # tải ngay để câu hỏi đầu tiên không phải chờ tải model
    llm = (
        GeminiClient(settings.gemini_api_key, settings.gemini_model, settings.llm_timeout_s)
        if settings.gemini_api_key and settings.gemini_model
        else None
    )
    return RagPipeline(searcher, reranker, llm, settings)


def _qdrant_up(pipeline: RagPipeline) -> bool:
    try:
        pipeline.searcher.client.get_collections()
        return True
    except Exception:
        return False


def _error(message: str, sources: list | None = None) -> JSONResponse:
    body: dict = {"error": message}
    if sources is not None:
        body["sources"] = [asdict(s) for s in sources]
    return JSONResponse(status_code=503, content=body)


def create_app(pipeline_factory: Callable[[], RagPipeline] | None = None) -> FastAPI:
    if pipeline_factory is None:
        # Chạy thật bằng uvicorn: uvicorn chỉ cấu hình logger của nó, không bật thì dòng log mỗi request bị nuốt.
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    factory = pipeline_factory or default_pipeline_factory

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.pipeline = factory()
        except Exception as exc:
            logger.error("Không khởi tạo được pipeline: %s: %s", type(exc).__name__, exc)
            app.state.pipeline = None
        yield

    app = FastAPI(title="Chatbot quy chế đào tạo", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request) -> dict:
        pipeline: RagPipeline | None = request.app.state.pipeline
        if pipeline is None:
            return {"qdrant": False, "collection_ready": False, "models_loaded": False, "llm_configured": False}
        return {
            "qdrant": _qdrant_up(pipeline),
            "collection_ready": pipeline.searcher.is_ready(),
            "models_loaded": True,
            "llm_configured": pipeline.llm is not None,
        }

    @app.post("/chat")
    def chat(body: ChatRequest, request: Request):
        # def thường (không async): pipeline chạy đồng bộ, FastAPI đưa vào threadpool nên không chặn server.
        pipeline: RagPipeline | None = request.app.state.pipeline
        request_id = uuid.uuid4().hex[:8]
        if pipeline is None:
            return _error(MSG_SEARCH_DOWN)
        if not pipeline.searcher.is_ready():
            return _error(MSG_NO_DATA if _qdrant_up(pipeline) else MSG_SEARCH_DOWN)

        history = [Turn(turn.role, turn.content) for turn in body.history]
        try:
            answer = pipeline.answer(body.question, history)
        except SearchUnavailableError as exc:
            logger.warning("id=%s search lỗi: %s", request_id, exc)
            return _error(MSG_SEARCH_DOWN)
        except LLMUnavailableError as exc:
            logger.warning("id=%s llm lỗi: %s", request_id, exc)
            return _error(MSG_NO_KEY if pipeline.llm is None else MSG_LLM_DOWN, exc.sources)

        debug = answer.debug
        logger.info(
            "id=%s q_len=%d refused=%s top=%s n_candidates=%d timings_ms=%s%s",
            request_id,
            len(body.question),
            answer.refused,
            debug["top_rerank_score"],
            debug["n_candidates"],
            debug["timings_ms"],
            f" question={body.question!r}" if pipeline.settings.debug else "",
        )
        return asdict(answer)

    return app


app = create_app()
