"""Giao diện chat Streamlit, chỉ nói chuyện với API qua HTTP.

Chạy: .venv\\Scripts\\python -m streamlit run src/ui/app.py --server.port 8502 (cần API ở cổng 8000).
"""

from __future__ import annotations

import os

import httpx

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
MSG_NO_API = "Chưa bật API: chạy uvicorn src.api.main:app"
MSG_INVALID = "Câu hỏi không hợp lệ: cần có chữ hoặc số và tối đa 1000 ký tự."
# Rewriter chỉ dùng 3 lượt cuối; gửi 6 là đủ và luôn dưới giới hạn 20 của API.
MAX_HISTORY_SENT = 6

HEALTH_LABELS = {
    "qdrant": "Qdrant",
    "collection_ready": "Dữ liệu đã index",
    "models_loaded": "Model đã tải",
    "llm_configured": "Gemini đã cấu hình",
}


def format_citation(c: dict) -> str:
    parts = [c["source_file"], f"trang {c['page_number']}"]
    if c.get("article"):
        parts.append(c["article"])
    return " — ".join(parts)


def build_history(messages: list[dict]) -> list[dict]:
    """Chỉ giữ role/content; bỏ thông báo lỗi vì đó không phải câu trả lời của trợ lý."""
    turns = [{"role": m["role"], "content": m["content"]} for m in messages if not m.get("error")]
    return turns[-MAX_HISTORY_SENT:]


def call_chat(question: str, history: list[dict], timeout_s: float = 60.0) -> dict:
    try:
        response = httpx.post(f"{API_URL}/chat", json={"question": question, "history": history}, timeout=timeout_s)
    except httpx.ConnectError:
        return {"error": MSG_NO_API}
    except httpx.TimeoutException:
        return {"error": f"Hệ thống phản hồi quá {timeout_s:.0f} giây, hãy thử lại sau."}
    except httpx.HTTPError as exc:
        return {"error": f"Lỗi kết nối tới API: {type(exc).__name__}"}

    if response.status_code == 200:
        return response.json()
    if response.status_code == 422:
        return {"error": MSG_INVALID}
    try:
        body = response.json()
    except ValueError:
        body = {}
    if isinstance(body, dict) and "error" in body:
        return body
    return {"error": f"API lỗi mã {response.status_code}"}


def get_health() -> dict | None:
    try:
        return httpx.get(f"{API_URL}/health", timeout=5).json()
    except (httpx.HTTPError, ValueError):
        return None


def to_message(result: dict) -> dict:
    if "error" in result:
        return {"role": "assistant", "content": result["error"], "error": True, "sources": result.get("sources", [])}
    return {
        "role": "assistant",
        "content": result["answer"],
        "refused": result["refused"],
        "citations": result["citations"],
        "rewritten_query": result["rewritten_query"],
        "debug": result["debug"],
    }


def _render_sources(st, title: str, citations: list[dict]) -> None:
    with st.expander(title):
        for number, c in enumerate(citations, start=1):
            st.markdown(f"**[{number}] {format_citation(c)}**")
            st.caption(c["text"])


def _render_body(st, message: dict, debug: bool) -> None:
    if message["role"] == "user":
        st.markdown(message["content"])
        return
    if message.get("error"):
        st.error(message["content"])
        if message.get("sources"):
            _render_sources(st, "Các đoạn tài liệu đã tìm được (chưa có câu trả lời)", message["sources"])
        return
    if message["refused"]:
        st.warning(message["content"], icon="📭")
    else:
        st.markdown(message["content"])
        _render_sources(st, "Nguồn", message["citations"])
    if debug:
        st.caption(f"Câu hỏi dùng để tìm: {message['rewritten_query']}")
        st.json(message["debug"], expanded=False)


def _render(st, message: dict, debug: bool) -> None:
    with st.chat_message(message["role"]):
        _render_body(st, message, debug)


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Hỏi đáp quy chế đào tạo", page_icon="📘")
    st.title("Hỏi đáp quy chế đào tạo")
    st.caption("Trả lời dựa trên Thông tư 08/2021 và quy chế trường, kèm file, trang và Điều.")
    messages: list[dict] = st.session_state.setdefault("messages", [])

    with st.sidebar:
        if st.button("Hội thoại mới"):
            messages.clear()
        debug = st.toggle("Chế độ debug")
        st.subheader("Trạng thái hệ thống")
        health = get_health()
        if health is None:
            st.error(MSG_NO_API)
        else:
            for key, label in HEALTH_LABELS.items():
                st.markdown(f"{'✅' if health.get(key) else '❌'} {label}")

    for message in messages:
        _render(st, message, debug)

    question = st.chat_input("Nhập câu hỏi về quy chế...")
    if question:
        history = build_history(messages)
        messages.append({"role": "user", "content": question})
        _render(st, messages[-1], debug)
        # Spinner nằm trong khung trợ lý: lần chạy sau tin nhắn ở đúng vị trí cũ, không để lại bản trùng trên trang.
        with st.chat_message("assistant"):
            with st.spinner("Đang tìm trong tài liệu và soạn câu trả lời..."):
                reply = to_message(call_chat(question, history))
            _render_body(st, reply, debug)
        messages.append(reply)


if __name__ == "__main__":
    main()
