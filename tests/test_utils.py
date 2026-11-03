import io
import sys

from src.utils import setup_utf8_console


def test_switches_stream_to_utf8(monkeypatch):
    fake_out = io.TextIOWrapper(io.BytesIO(), encoding="cp1258")
    fake_err = io.TextIOWrapper(io.BytesIO(), encoding="cp1258")
    monkeypatch.setattr(sys, "stdout", fake_out)
    monkeypatch.setattr(sys, "stderr", fake_err)
    setup_utf8_console()
    assert sys.stdout.encoding == "utf-8" and sys.stderr.encoding == "utf-8"
    print("Điều 5. Học phí")  # không còn UnicodeEncodeError


def test_stream_without_reconfigure_is_ignored(monkeypatch):
    # io.StringIO không có .reconfigure → hàm phải bỏ qua, không ném lỗi
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    setup_utf8_console()


def test_idempotent(monkeypatch):
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(io.BytesIO(), encoding="cp1258"))
    setup_utf8_console()
    setup_utf8_console()
    assert sys.stdout.encoding == "utf-8"

