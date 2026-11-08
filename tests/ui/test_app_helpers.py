from pathlib import Path

import httpx
import pytest

from src.ui import app

CITATION = {"source_file": "thong_tu_08_2021.pdf", "page_number": 13, "article": "Điều 11. Xử lý kết quả học tập", "text": "..."}


def test_format_citation_with_article():
    assert app.format_citation(CITATION) == "thong_tu_08_2021.pdf — trang 13 — Điều 11. Xử lý kết quả học tập"


def test_format_citation_without_article():
    assert app.format_citation({**CITATION, "article": None}) == "thong_tu_08_2021.pdf — trang 13"


def test_build_history_keeps_role_content_and_drops_errors():
    messages = [
        {"role": "user", "content": "Hỏi 1"},
        {"role": "assistant", "content": "Đáp 1 [1]", "citations": [CITATION], "refused": False},
        {"role": "user", "content": "Hỏi 2"},
        {"role": "assistant", "content": "Mô hình ngôn ngữ tạm thời không khả dụng", "error": True},
    ]
    assert app.build_history(messages) == [
        {"role": "user", "content": "Hỏi 1"},
        {"role": "assistant", "content": "Đáp 1 [1]"},
        {"role": "user", "content": "Hỏi 2"},
    ]


def test_build_history_sends_only_recent_turns():
    # API từ chối > 20 lượt; hội thoại dài vẫn phải gửi được.
    messages = [{"role": "user", "content": str(i)} for i in range(50)]
    history = app.build_history(messages)
    assert len(history) == app.MAX_HISTORY_SENT <= 20
    assert history[-1]["content"] == "49"


def _post_returning(response=None, error=None, seen=None):
    def fake_post(url, json, timeout):
        if seen is not None:
            seen.update(url=url, json=json, timeout=timeout)
        if error is not None:
            raise error
        return response

    return fake_post


def _response(status: int, body: dict) -> httpx.Response:
    return httpx.Response(status, json=body, request=httpx.Request("POST", "http://test/chat"))


def test_call_chat_ok_returns_body_and_sends_request(monkeypatch):
    seen: dict = {}
    body = {"answer": "Có [1].", "refused": False, "citations": [CITATION]}
    monkeypatch.setattr(app.httpx, "post", _post_returning(_response(200, body), seen=seen))
    assert app.call_chat("Hỏi?", [{"role": "user", "content": "x"}], timeout_s=5) == body
    assert seen == {
        "url": f"{app.API_URL}/chat",
        "json": {"question": "Hỏi?", "history": [{"role": "user", "content": "x"}]},
        "timeout": 5,
    }


def test_call_chat_connection_error_mentions_uvicorn(monkeypatch):
    monkeypatch.setattr(app.httpx, "post", _post_returning(error=httpx.ConnectError("refused")))
    assert "uvicorn" in app.call_chat("Hỏi?", [])["error"]


def test_call_chat_timeout_is_clear_message(monkeypatch):
    monkeypatch.setattr(app.httpx, "post", _post_returning(error=httpx.ReadTimeout("slow")))
    error = app.call_chat("Hỏi?", [], timeout_s=60)["error"]
    assert "60 giây" in error
    assert "uvicorn" not in error  # API có chạy, chỉ là chậm


def test_call_chat_503_returns_error_and_sources_from_body(monkeypatch):
    body = {"error": "Mô hình ngôn ngữ tạm thời không khả dụng", "sources": [CITATION]}
    monkeypatch.setattr(app.httpx, "post", _post_returning(_response(503, body)))
    assert app.call_chat("Hỏi?", []) == body


def test_call_chat_422_is_friendly(monkeypatch):
    monkeypatch.setattr(app.httpx, "post", _post_returning(_response(422, {"detail": []})))
    assert "1000 ký tự" in app.call_chat("???", [])["error"]


def test_call_chat_unexpected_status_has_error(monkeypatch):
    response = httpx.Response(500, text="Internal Server Error", request=httpx.Request("POST", "http://test/chat"))
    monkeypatch.setattr(app.httpx, "post", _post_returning(response))
    assert "500" in app.call_chat("Hỏi?", [])["error"]


class _FakeApi:
    """API giả chạy cục bộ: /health luôn ổn, /chat trả lần lượt các body trong `answers`."""

    def __init__(self, answers: list[dict]) -> None:
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        import json
        import threading

        self.answers = answers
        self.requests: list[dict] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, body: dict) -> None:
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._send(dict.fromkeys(app.HEALTH_LABELS, True))

            def do_POST(self):
                fake.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self._send(fake.answers.pop(0))

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"


def _answer(text: str, refused: bool = False) -> dict:
    return {
        "answer": text,
        "refused": refused,
        "citations": [] if refused else [CITATION],
        "rewritten_query": text,
        "debug": {},
    }


def test_two_questions_render_each_message_once_and_send_history(monkeypatch):
    from streamlit.testing.v1 import AppTest

    api = _FakeApi([_answer("Đáp 1 [1]."), _answer("Không tìm thấy thông tin trong tài liệu quy chế.", refused=True)])
    monkeypatch.setenv("API_URL", api.url)
    try:
        at = AppTest.from_file(str(Path(app.__file__)), default_timeout=30).run()
        at.chat_input[0].set_value("Hỏi 1").run()
        at.chat_input[0].set_value("Hỏi 2").run()
    finally:
        api.server.shutdown()
        api.server.server_close()

    assert not at.exception
    assert [m.name for m in at.chat_message] == ["user", "assistant", "user", "assistant"]
    assert at.session_state["messages"][1]["content"] == "Đáp 1 [1]."
    assert len(at.session_state["messages"]) == 4
    assert api.requests[1]["history"] == [
        {"role": "user", "content": "Hỏi 1"},
        {"role": "assistant", "content": "Đáp 1 [1]."},
    ]
    assert len(at.warning) == 1  # câu từ chối có nhãn riêng
    assert all("✅" in m.value for m in at.sidebar.markdown)


@pytest.fixture
def app_test(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")  # cổng không có gì chạy
    return AppTest.from_file(str(Path(app.__file__)), default_timeout=30)


def test_app_renders_and_reports_missing_api(app_test):
    app_test.run()
    assert not app_test.exception
    assert any("uvicorn" in e.value for e in app_test.sidebar.error)

    app_test.chat_input[0].set_value("Cảnh báo học tập là gì?").run()
    assert not app_test.exception
    assert [m.name for m in app_test.chat_message] == ["user", "assistant"]
    assert any("uvicorn" in e.value for e in app_test.error)

    app_test.sidebar.button[0].click().run()  # "Hội thoại mới"
    assert len(app_test.chat_message) == 0
