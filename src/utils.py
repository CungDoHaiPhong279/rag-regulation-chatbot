"""Các tiện ích dùng chung cho toàn dự án."""

from __future__ import annotations

import sys


def setup_utf8_console() -> None:
    """Cấu hình luồng xuất chuẩn (stdout, stderr) dùng bảng mã UTF-8.

    Trên Windows, console mặc định thường dùng bảng mã cp1258 hoặc cp1252,
    gây lỗi UnicodeEncodeError khi in tiếng Việt chuẩn Unicode NFC.
    Hàm kiểm tra hasattr(..., "reconfigure") trước khi gọi để tương thích
    an toàn với các luồng giả lập trong kiểm thử (như io.StringIO).
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

