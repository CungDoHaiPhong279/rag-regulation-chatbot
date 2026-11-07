"""GeminiClient: test logic retry bằng lỗi thật của SDK, và test đầu-cuối với SDK thật gọi vào server cục bộ."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import threading
import time

from google.genai import errors
import pytest

from src.rag.llm import GeminiClient, LLMError
from tests.fakes import FakeLLM

SECRET = "k-secret"
QUESTION = "cau-hoi-rieng-tu"


@pytest.fixture(autouse=True)
def _no_retry_delay(monkeypatch):
    monkeypatch.setattr("src.rag.llm._RETRY_DELAY_S", 0)


def _client(monkeypatch, *outcomes) -> tuple[GeminiClient, list]:
    client = GeminiClient(SECRET, "gemini-test")
    calls: list = []
    queue = list(outcomes)

    def fake_call(prompt, system):
        calls.append((prompt, system))
        outcome = queue.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(client, "_call", fake_call)
    return client, calls


def _api_error(code: int):
    cls = errors.ClientError if code < 500 else errors.ServerError
    return cls(code, {"error": {"code": code, "message": "boom", "status": "X"}})


@pytest.mark.parametrize("code", [429, 500, 503])
def test_gemini_client_retries_once_on_retryable(monkeypatch, code):
    client, calls = _client(monkeypatch, _api_error(code), "ok")
    assert client.generate("p", system="s") == "ok"
    assert calls == [("p", "s"), ("p", "s")]


def test_gemini_client_raises_llmerror_after_second_failure(monkeypatch):
    client, calls = _client(monkeypatch, _api_error(429), _api_error(503))
    with pytest.raises(LLMError, match="503"):
        client.generate("p")
    assert len(calls) == 2


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_gemini_client_on_4xx_no_retry(monkeypatch, code):
    client, calls = _client(monkeypatch, _api_error(code))
    with pytest.raises(LLMError, match=str(code)):
        client.generate("p")
    assert len(calls) == 1


def test_gemini_client_unknown_error_becomes_llmerror_without_retry(monkeypatch):
    client, calls = _client(monkeypatch, ValueError("hỏng"))
    with pytest.raises(LLMError):
        client.generate("p")
    assert len(calls) == 1


@pytest.mark.parametrize("empty", ["", "   \n"])
def test_gemini_client_empty_response_is_error_not_refusal(monkeypatch, empty):
    # Bộ lọc an toàn trả rỗng: đây là lỗi kỹ thuật, không được hiểu là "tài liệu không có thông tin".
    client, calls = _client(monkeypatch, empty)
    with pytest.raises(LLMError):
        client.generate("p")
    assert len(calls) == 1


@pytest.mark.parametrize(("key", "model"), [(None, "m"), ("", "m"), ("k", None), ("k", "")])
def test_gemini_client_requires_key_and_model(key, model):
    with pytest.raises(LLMError, match="GEMINI_"):
        GeminiClient(key, model)


def test_missing_key_message_has_no_value():
    with pytest.raises(LLMError) as err:
        GeminiClient(SECRET, None)
    assert SECRET not in str(err.value)


def test_fake_llm_returns_in_order_records_and_raises():
    llm = FakeLLM(["a", LLMError("x"), "b"])
    assert llm.generate("p1", system="s1") == "a"
    with pytest.raises(LLMError):
        llm.generate("p2")
    assert llm.generate("p3") == "b"
    assert llm.prompts == ["p1", "p2", "p3"]
    assert llm.systems == ["s1", None, None]


# --- Đầu-cuối: SDK thật + server HTTP cục bộ, không cần key thật, không ra mạng ---


class _Stub:
    """Mỗi request lấy một phần tử script: mã HTTP (int) hoặc số giây treo (float)."""

    def __init__(self, script: list) -> None:
        self.script = script
        self.requests: list[dict] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                stub.requests.append({"path": self.path, "key": self.headers.get("x-goog-api-key"), "body": body})
                step = stub.script.pop(0)
                if isinstance(step, float):
                    time.sleep(step)
                    return
                payload = (
                    {"candidates": [{"content": {"role": "model", "parts": [{"text": "OK"}]}}]}
                    if step == 200
                    else {"error": {"code": step, "message": "boom", "status": "X"}}
                )
                data = json.dumps(payload).encode()
                self.send_response(step)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"


@pytest.fixture
def stub(monkeypatch):
    created = []

    def make(script):
        server = _Stub(script)
        created.append(server)
        monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", server.url)
        return server

    yield make
    for server in created:
        server.server.shutdown()
        server.server.server_close()


def test_e2e_success_sends_system_temperature_and_key_in_header(stub):
    server = stub([200])
    assert GeminiClient(SECRET, "gemini-test").generate("xin chào", system="hệ thống") == "OK"
    (request,) = server.requests
    assert "gemini-test:generateContent" in request["path"]
    assert SECRET not in request["path"]  # key đi trong header, không nằm trên URL
    assert request["key"] == SECRET
    assert request["body"]["generationConfig"]["temperature"] == 0
    assert request["body"]["systemInstruction"]["parts"][0]["text"] == "hệ thống"
    assert request["body"]["contents"][0]["parts"][0]["text"] == "xin chào"


def test_e2e_429_then_ok_calls_server_exactly_twice(stub):
    server = stub([429, 200])
    assert GeminiClient(SECRET, "gemini-test").generate("p") == "OK"
    assert len(server.requests) == 2  # SDK không tự retry thêm


def test_e2e_500_twice_raises_after_exactly_two_calls(stub):
    server = stub([500, 500])
    with pytest.raises(LLMError, match="500"):
        GeminiClient(SECRET, "gemini-test").generate("p")
    assert len(server.requests) == 2


def test_e2e_400_no_retry(stub):
    server = stub([400])
    with pytest.raises(LLMError, match="400"):
        GeminiClient(SECRET, "gemini-test").generate("p")
    assert len(server.requests) == 1


def test_e2e_timeout_retries_once_then_llmerror(stub):
    server = stub([1.5, 1.5])
    started = time.monotonic()
    with pytest.raises(LLMError):
        GeminiClient(SECRET, "gemini-test", timeout_s=0.3).generate("p")
    assert len(server.requests) == 2
    assert time.monotonic() - started < 2.5  # không có timeout thì mất ít nhất 3 giây (2 x 1.5)


def test_e2e_timeout_then_ok(stub):
    server = stub([1.5, 200])
    assert GeminiClient(SECRET, "gemini-test", timeout_s=0.3).generate("p") == "OK"
    assert len(server.requests) == 2


def test_e2e_connection_refused_becomes_llmerror(monkeypatch):
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", "http://127.0.0.1:9")
    with pytest.raises(LLMError):
        GeminiClient(SECRET, "gemini-test", timeout_s=1).generate("p")


@pytest.mark.parametrize("script", [[429, 200], [500, 500], [400], [1.5, 1.5]])
def test_gemini_client_never_logs_key_or_question(stub, caplog, script):
    stub(script)
    caplog.set_level(logging.DEBUG)
    message = ""
    try:
        GeminiClient(SECRET, "gemini-test", timeout_s=0.3).generate(QUESTION)
    except LLMError as exc:
        message = str(exc)
    assert SECRET not in caplog.text
    assert SECRET not in message
    assert QUESTION not in caplog.text
    assert QUESTION not in message


def test_llmerror_does_not_chain_sdk_exception(stub):
    # Exception gốc của SDK có thể mang request/response; không để lọt ra traceback.
    stub([400])
    with pytest.raises(LLMError) as err:
        GeminiClient(SECRET, "gemini-test").generate("p")
    assert err.value.__cause__ is None
    assert err.value.__suppress_context__
