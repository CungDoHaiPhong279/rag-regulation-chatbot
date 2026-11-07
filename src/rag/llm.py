"""LLM nằm sau Protocol; GeminiClient thêm timeout, retry 1 lần và không để lộ API key.

SDK chỉ được import khi gọi thật, để import module này (và chạy test bằng FakeLLM) vẫn nhẹ.
"""

from __future__ import annotations

import logging
import time
from typing import Protocol

logger = logging.getLogger(__name__)

_RETRY_DELAY_S = 1.0


class LLMError(Exception):
    """LLM không trả lời được. Là lỗi kỹ thuật, không mang nghĩa REFUSAL."""


class LLMClient(Protocol):
    def generate(self, prompt: str, system: str | None = None) -> str: ...


def _is_timeout(exc: Exception) -> bool:
    # So theo tên vì SDK có thể chạy trên httpx hoặc httpx2, mỗi bản có lớp Timeout riêng.
    return isinstance(exc, TimeoutError) or "Timeout" in type(exc).__name__


def _is_retryable(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    return _is_timeout(exc) or (isinstance(code, int) and (code == 429 or code >= 500))


def _describe(exc: Exception) -> str:
    # Chỉ dùng mã lỗi và tên lớp: nội dung exception của SDK có thể chứa request/response.
    code = getattr(exc, "code", None)
    return f"Gemini lỗi mã {code}" if isinstance(code, int) else f"Gemini lỗi: {type(exc).__name__}"


class GeminiClient:
    def __init__(self, api_key: str, model: str, timeout_s: float = 30.0) -> None:
        if not api_key:
            raise LLMError("Thiếu GEMINI_API_KEY trong .env")
        if not model:
            raise LLMError("Thiếu GEMINI_MODEL trong .env")
        self._api_key = api_key
        self._model = model
        self._timeout_s = timeout_s
        self._client = None

    def _call(self, prompt: str, system: str | None) -> str:
        from google import genai
        from google.genai import types

        if self._client is None:
            self._client = genai.Client(
                api_key=self._api_key,
                http_options=types.HttpOptions(timeout=int(self._timeout_s * 1000)),
            )
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system, temperature=0),
        )
        return response.text or ""

    def generate(self, prompt: str, system: str | None = None) -> str:
        for attempt in (1, 2):
            try:
                text = self._call(prompt, system)
                break
            except Exception as exc:
                reason = _describe(exc)
                if attempt == 2 or not _is_retryable(exc):
                    raise LLMError(reason) from None
                logger.warning("%s, thử lại 1 lần", reason)
                time.sleep(_RETRY_DELAY_S)
        if not text.strip():
            # Thường do bộ lọc an toàn; không được hiểu nhầm thành "tài liệu không có thông tin".
            raise LLMError("Gemini trả về nội dung rỗng")
        return text
