"""Module đọc PDF quy chế, làm sạch văn bản và cắt chunk theo từng trang.

Chiến lược thiết kế:
1. Chuẩn hóa Unicode NFC cho toàn bộ văn bản tiếng Việt trước khi xử lý.
2. Loại bỏ dòng header/footer lặp lại ở đầu/cuối trang (khi xuất hiện ở >= 60% số trang).
3. Tách văn bản trên mỗi trang theo tiêu đề Điều TRƯỚC, rồi mới cắt chunk bên trong
   từng đoạn Điều -> đảm bảo không có chunk nào chứa lẫn lộn 2 Điều.
4. Gắn tiền tố "<article>: " vào đầu mọi chunk thuộc Điều đó (kể cả khi Điều kéo sang
   trang sau). Khi gặp tiêu đề cấp trên (Chương, Mục, Phần) ở đầu dòng thì ngắt Điều
   đang mang sang để dòng Chương/Mục/Phần không bị gán nhầm vào Điều trước.
5. Tham số chunk_size (mặc định 800 ký tự) và chunk_overlap (mặc định 150 ký tự) là
   GIÁ TRỊ KHỞI ĐẦU tính theo ký tự (bao gồm cả tiền tố Điều), sẽ được thực nghiệm
   so sánh 400 / 800 / 1200 bằng Recall@5 ở Tuần 3.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re
import sys
import unicodedata
from uuid import NAMESPACE_URL, uuid5

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import load_settings
from src.utils import setup_utf8_console

# Nhóm 1: số Điều, Nhóm 2: tên Điều (bắt buộc có dấu "." hoặc ":" ngay sau số Điều)
ARTICLE_HEADING_RE = re.compile(
    r"^[ \t]*Điều[ \t]+(\d+)[ \t]*[.:][ \t]*(.*?)[ \t]*$",
    re.MULTILINE,
)

# Tiêu đề cấp trên: Chương / Mục / Phần + số La Mã hoặc số thường ở đầu dòng
SECTION_HEADING_RE = re.compile(
    r"^[ \t]*(?:Chương|Mục|Phần)[ \t]+(?:[IVXLCDM]+|\d+)(?:[ \t]*[.:–-].*|[ \t]*)$",
    re.IGNORECASE,
)

# Mẫu số trang điển hình: "Trang 1", "Trang 1/10", "1/10", "- 1 -", "[1]", "1"
PAGE_NUMBER_RE = re.compile(
    r"^(?:"
    r"trang\s*\d+(?:\s*/\s*\d+)?"
    r"|\d+\s*/\s*\d+"
    r"|[-–—]\s*\d+\s*[-–—]"
    r"|\[\s*\d+\s*\]"
    r"|\d+"
    r")$",
    re.IGNORECASE,
)

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200
MIN_BODY_SIZE = 100
MAX_ARTICLE_HEADING_LEN = 250
TERMINAL_PUNCTUATION = (".", ":", ";", "!", "?")

# Điểm / khoản liệt kê: "1.", "1)", "a)", "a.", "đ)", "đ.", "-", "–" v.v.
CLAUSE_OR_POINT_RE = re.compile(
    r"^(?:(?:\d+|[a-zA-Zà-ỹđÀ-ỸĐ])[\.:\)]|[-–—•\*\+])(?:\s+|$)"
)
SOFT_LINE_END_PUNCTUATION_RE = re.compile(r"[\.:;?!][\"'\)\]]?$")


class PdfNoTextError(Exception):
    """Ném ra khi file PDF không có lớp văn bản (PDF bản scan ảnh)."""


class PdfUnreadableError(Exception):
    """Ném ra khi file PDF hỏng (không mở được) hoặc bị khóa bằng mật khẩu."""


@dataclass(frozen=True)
class SkippedPage:
    """Thông tin về một trang không có lớp chữ bị bỏ qua."""

    page_number: int  # 1-based
    reason: str       # "scan" (có ảnh) | "blank" (trang trắng)
    has_images: bool


@dataclass(frozen=True)
class RemovedLine:
    """Thông tin về một dòng header hoặc footer lặp bị bộ lọc loại bỏ."""

    page_number: int  # 1-based
    position: str     # "header" | "footer"
    text: str


@dataclass(frozen=True)
class Chunk:
    """Một đoạn văn bản (chunk) đã cắt từ 1 trang PDF kèm đầy đủ metadata trích dẫn."""

    chunk_id: str
    source_file: str
    page_number: int
    article: str | None
    chunk_index: int
    text: str


def normalize_text(text: str) -> str:
    """Chuẩn hóa Unicode NFC và dọn khoảng trắng thừa trên từng dòng."""
    nfc = unicodedata.normalize("NFC", text)
    nfc = nfc.replace("\r\n", "\n").replace("\r", "\n")
    cleaned_lines = [re.sub(r"[ \t]+", " ", line).strip() for line in nfc.split("\n")]
    return "\n".join(cleaned_lines)


def parse_article_heading(line: str) -> str | None:
    """Nhận diện dòng tiêu đề Điều và chuẩn hóa về dạng 'Điều X. Tên Điều' hoặc 'Điều X'."""
    normalized = unicodedata.normalize("NFC", line)
    match = ARTICLE_HEADING_RE.match(normalized)
    if not match or "\n" in normalized.strip():
        return None
    number = match.group(1)
    title = re.sub(r"[ \t]+", " ", match.group(2)).strip().rstrip(".:;")
    if title:
        return f"Điều {number}. {title}"
    return f"Điều {number}"


def is_section_heading(line: str) -> bool:
    """Kiểm tra dòng có phải là tiêu đề Chương, Mục, Phần ở đầu dòng hay không."""
    normalized = unicodedata.normalize("NFC", line)
    if "\n" in normalized.strip():
        return False
    return SECTION_HEADING_RE.match(normalized) is not None


def is_clause_or_point_start(text: str) -> bool:
    """Kiểm tra xem dòng có phải là điểm/khoản liệt kê như '1.', 'a)', 'đ)' hay không."""
    return CLAUSE_OR_POINT_RE.match(text.strip()) is not None


def parse_multiline_article_heading(
    line: str,
    next_line: str | None = None,
) -> tuple[str | None, bool]:
    """Nhận diện dòng tiêu đề Điều, hỗ trợ gộp dòng kế tiếp (tối đa 2 dòng).

    Quy tắc gộp:
    - Tiêu đề ở dòng 1 chưa kết thúc bằng dấu chấm, hai chấm, chấm phẩy (chưa kết thúc
      bằng dấu câu ngắt, kể cả khi kết thúc bằng dấu phẩy ',' hoặc từ nối).
    - Dòng sau bắt đầu bằng chữ thường (theo chuẩn Unicode Python char.islower(),
      không phải chữ hoa, và không phải là '1.', 'a)', 'đ)' v.v.).
    - Tổng độ dài tiêu đề sau khi gộp không vượt quá MAX_ARTICLE_HEADING_LEN (250 ký tự).

    Trả về (tiêu_đề_chuẩn_hóa, đã_gộp_dòng_sau).
    """
    heading = parse_article_heading(line)
    if heading is None:
        return None, False

    if next_line is None:
        return heading, False

    clean_next = next_line.strip()
    if not clean_next:
        return heading, False

    clean_line = line.strip()
    # Nếu tiêu đề dòng 1 đã kết thúc bằng dấu câu ngắt (. : ; ! ?) thì không gộp
    if clean_line.endswith(TERMINAL_PUNCTUATION):
        return heading, False

    # Dòng sau bắt buộc bắt đầu bằng ký tự thường
    first_char = clean_next[0]
    if not first_char.islower():
        return heading, False

    # Không gộp nếu dòng sau là điểm/khoản liệt kê (vd: 'a)', 'đ)', '1.')
    if is_clause_or_point_start(clean_next):
        return heading, False

    combined_title = f"{clean_line} {clean_next}"
    if len(combined_title) > MAX_ARTICLE_HEADING_LEN:
        return heading, False

    match = ARTICLE_HEADING_RE.match(combined_title)
    if match:
        number = match.group(1)
        title = re.sub(r"[ \t]+", " ", match.group(2)).strip()
        title = title.rstrip(",")
        return f"Điều {number}. {title}", True

    return heading, False


def is_soft_line_terminal(line: str) -> bool:
    """Kiểm tra xem dòng có kết thúc bằng dấu ngắt câu (. ; : ? !) hoặc là tiêu đề hay không."""
    stripped = line.strip()
    if not stripped:
        return True
    if is_section_heading(stripped) or parse_article_heading(stripped) is not None:
        return True
    return bool(SOFT_LINE_END_PUNCTUATION_RE.search(stripped))


def is_new_item_start(line: str) -> bool:
    """Kiểm tra dòng có mở đầu một mục mới (1., 1), a), đ), Điều, Chương/Mục/Phần, dòng trống)."""
    stripped = line.strip()
    if not stripped:
        return True
    if is_clause_or_point_start(stripped):
        return True
    if parse_article_heading(stripped) is not None:
        return True
    if is_section_heading(stripped):
        return True
    return False


def join_soft_lines(text: str) -> str:
    """Nối các dòng mềm (soft-wrap) bị ngắt dòng giữa câu trong văn bản.

    Quy tắc:
    - Chỉ nối khi dòng trước KHÔNG kết thúc bằng dấu ngắt câu (. ; : ? !)
      VÀ dòng sau KHÔNG mở đầu một mục mới (1., 1), a), đ), gạch đầu dòng,
      tiêu đề Điều, tiêu đề Chương/Mục/Phần, hoặc dòng trống).
    - Giữ nguyên ngắt dòng sau dấu câu (ví dụ danh sách 'A: ...;' / 'B: ...;').
    """
    lines = text.split("\n")
    if not lines:
        return ""

    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            result.append("")
            continue

        if not result or result[-1] == "":
            result.append(stripped)
            continue

        prev = result[-1]
        if not is_soft_line_terminal(prev) and not is_new_item_start(stripped):
            result[-1] = f"{prev} {stripped}"
        else:
            result.append(stripped)

    return "\n".join(result)


def split_by_article(
    page_text: str,
    carried: str | None = None,
) -> tuple[list[tuple[str | None, str]], str | None]:
    """Tách văn bản của 1 trang thành (segments, last_seen_article).

    - segments: danh sách [(article, body), ...] các đoạn trên trang có nội dung chữ.
    - last_seen_article: Điều cuối cùng xuất hiện trên trang (kể cả khi chưa có body),
      hoặc None nếu tiêu đề cấp trên (Chương/Mục/Phần) xuất hiện sau Điều đó,
      hoặc carried nếu trang không có tiêu đề mới nào.
    - Phần văn bản trước tiêu đề Điều đầu tiên trên trang được gán cho `carried`
      (Điều đang kéo sang từ trang trước), trừ khi gặp dòng tiêu đề Chương/Mục/Phần
      ở đầu dòng thì ngắt `carried` về `None`.
    - Bản thân dòng tiêu đề Điều được lược khỏi `body` vì sẽ được gắn lại làm tiền tố.
    """
    normalized_page = normalize_text(page_text)
    current_article: str | None = carried
    last_seen_article: str | None = carried
    current_lines: list[str] = []
    segments: list[tuple[str | None, str]] = []

    def _flush() -> None:
        body = "\n".join(current_lines).strip()
        if body:
            segments.append((current_article, body))
        current_lines.clear()

    raw_lines = normalized_page.split("\n")
    i = 0
    while i < len(raw_lines):
        raw_line = raw_lines[i]
        line = raw_line.strip()
        if not line:
            if current_lines and current_lines[-1] != "":
                current_lines.append("")
            i += 1
            continue

        # Tìm dòng không rỗng kế tiếp để kiểm tra gộp tiêu đề nhiều dòng
        next_non_empty = None
        next_idx = None
        for j in range(i + 1, len(raw_lines)):
            if raw_lines[j].strip():
                next_non_empty = raw_lines[j].strip()
                next_idx = j
                break

        article_heading, was_consumed = parse_multiline_article_heading(
            line, next_non_empty
        )
        if article_heading is not None:
            _flush()
            current_article = article_heading
            last_seen_article = article_heading
            if was_consumed and next_idx is not None:
                i = next_idx + 1
            else:
                i += 1
            continue

        if is_section_heading(line):
            if current_article is not None:
                _flush()
                current_article = None
            last_seen_article = None
            current_lines.append(line)
            i += 1
            continue

        current_lines.append(line)
        i += 1

    _flush()
    return segments, last_seen_article


def _line_pattern(line: str) -> str:
    """Chuẩn hóa mẫu dòng.

    Chỉ gom nhóm các dòng là số trang (vd: 'Trang 1', '1/10', '- 1 -').
    Mọi dòng khác phải trùng nguyên văn giữa các trang mới bị coi là lặp.
    """
    stripped = line.strip()
    if PAGE_NUMBER_RE.match(stripped):
        return "<PAGE_NUMBER>"
    return stripped


def _remove_repeated_headers_footers(
    pages_lines: list[list[str]],
) -> tuple[list[list[str]], list[RemovedLine]]:
    """Loại bỏ dòng đầu/cuối trang lặp lại ở >= 60% số trang (khi tài liệu có >= 3 trang có chữ).

    Trả về (cleaned_pages_lines, removed_lines).
    """
    non_empty_pages = [lines for lines in pages_lines if lines]
    if len(non_empty_pages) < 3:
        return pages_lines, []

    total = len(non_empty_pages)
    first_counts = Counter(_line_pattern(lines[0]) for lines in non_empty_pages)
    last_counts = Counter(_line_pattern(lines[-1]) for lines in non_empty_pages)

    repeated_headers = {
        pat for pat, cnt in first_counts.items() if pat and (cnt / total) >= 0.6
    }
    repeated_footers = {
        pat for pat, cnt in last_counts.items() if pat and (cnt / total) >= 0.6
    }

    cleaned: list[list[str]] = []
    removed_lines: list[RemovedLine] = []

    for page_idx, lines in enumerate(pages_lines, start=1):
        if not lines:
            cleaned.append([])
            continue
        start = 0
        end = len(lines)
        if (
            start < end
            and _line_pattern(lines[start]) in repeated_headers
            and parse_article_heading(lines[start]) is None
        ):
            removed_lines.append(
                RemovedLine(page_number=page_idx, position="header", text=lines[start])
            )
            start += 1
        if (
            end > start
            and _line_pattern(lines[end - 1]) in repeated_footers
            and parse_article_heading(lines[end - 1]) is None
        ):
            removed_lines.append(
                RemovedLine(page_number=page_idx, position="footer", text=lines[end - 1])
            )
            end -= 1
        cleaned.append(lines[start:end])

    return cleaned, removed_lines


def _resolve_chunk_params(
    chunk_size: int | None,
    chunk_overlap: int | None,
) -> tuple[int, int]:
    """Tham số truyền vào được ưu tiên; thiếu thì lấy từ cấu hình chung (src/config.py, đọc .env)."""
    if chunk_size is not None and chunk_overlap is not None:
        return chunk_size, chunk_overlap
    settings = load_settings()
    return (
        chunk_size if chunk_size is not None else settings.chunk_size,
        chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
    )


def chunk_pages(
    page_texts: list[str],
    source_file: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    removed_lines_collector: list[RemovedLine] | None = None,
    page_has_images: list[bool] | None = None,
) -> tuple[list[Chunk], list[SkippedPage]]:
    """Làm sạch, tách Điều và cắt chunk từ danh sách văn bản từng trang của một file.

    Trả về:
    - chunks: Danh sách các Chunk được cắt ra kèm đầy đủ metadata.
    - skipped_pages: Danh sách các trang không có lớp chữ bị bỏ qua.
    """
    resolved_size, resolved_overlap = _resolve_chunk_params(chunk_size, chunk_overlap)

    normalized_pages = [normalize_text(p) for p in page_texts]
    pages_non_empty_lines = [
        [line for line in p.split("\n") if line.strip()] for p in normalized_pages
    ]

    if not any(pages_non_empty_lines):
        raise PdfNoTextError(
            f"{source_file}: không có lớp chữ (PDF scan?) — cần OCR"
        )

    # Ghi nhận và cảnh báo các trang không có lớp chữ
    skipped_pages: list[SkippedPage] = []
    for page_idx, lines in enumerate(pages_non_empty_lines, start=1):
        if not lines:
            has_img = (
                page_has_images[page_idx - 1]
                if page_has_images and page_idx - 1 < len(page_has_images)
                else False
            )
            reason = "scan" if has_img else "blank"
            skipped_pages.append(
                SkippedPage(page_number=page_idx, reason=reason, has_images=has_img)
            )
            desc = "trang scan ảnh" if reason == "scan" else "trang trắng"
            print(
                f"Cảnh báo [{source_file} | Trang {page_idx}]: không có lớp chữ ({desc}) — bỏ qua.",
                file=sys.stderr,
            )

    cleaned_pages_lines, removed_lines = _remove_repeated_headers_footers(
        pages_non_empty_lines
    )
    if removed_lines_collector is not None:
        removed_lines_collector.extend(removed_lines)

    chunks: list[Chunk] = []
    current_article: str | None = None

    for page_number, lines in enumerate(cleaned_pages_lines, start=1):
        if not lines:
            continue
        page_text = "\n".join(lines)
        segments, current_article = split_by_article(page_text, carried=current_article)

        chunk_index = 0
        for article, body in segments:
            if not body.strip():
                continue

            cleaned_body = join_soft_lines(body)
            if not cleaned_body.strip():
                continue

            prefix = f"{article}: " if article else ""
            body_size = resolved_size - len(prefix)
            if body_size < MIN_BODY_SIZE or resolved_overlap >= body_size:
                raise ValueError(
                    f"chunk_size={resolved_size} quá nhỏ so với độ dài tiêu đề "
                    f"'{article}' (body_size={body_size}, chunk_overlap={resolved_overlap})."
                )

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=body_size,
                chunk_overlap=resolved_overlap,
                separators=["\n\n", "\n", ". ", "; ", " ", ""],
            )
            for piece in splitter.split_text(cleaned_body):
                piece_clean = piece.strip()
                if not piece_clean:
                    continue
                full_text = f"{prefix}{piece_clean}"
                chunk_id = str(
                    uuid5(NAMESPACE_URL, f"{source_file}:{page_number}:{chunk_index}")
                )
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        source_file=source_file,
                        page_number=page_number,
                        article=article,
                        chunk_index=chunk_index,
                        text=full_text,
                    )
                )
                chunk_index += 1

    return chunks, skipped_pages


def chunk_pdf(
    pdf_path: Path,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    removed_lines_collector: list[RemovedLine] | None = None,
) -> tuple[list[Chunk], list[SkippedPage]]:
    """Đọc các trang văn bản và hình ảnh từ một file PDF rồi chuyển cho chunk_pages xử lý."""
    pdf_path = Path(pdf_path)
    unreadable_msg = f"{pdf_path.name}: file PDF hỏng hoặc bị khóa"
    # Lỗi cấu trúc PDF được báo bằng ngoại lệ của mình; tắt log thô của MuPDF in ra stdout
    fitz.TOOLS.mupdf_display_errors(False)
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:  # PyMuPDF ném nhiều loại lỗi khác nhau cho file hỏng
        raise PdfUnreadableError(unreadable_msg) from exc
    # needs_pass: bị khóa mật khẩu. is_repaired: MuPDF phải tự sửa cấu trúc (thường do file tải về bị
    # cắt cụt) — chữ đọc ra có thể là rác (đã gặp: PDF thật cắt 50% -> 129 chunk rác, 0 Điều), nên không
    # index mà báo người dùng tải lại. 0 trang: cây trang hỏng.
    if doc.needs_pass or doc.is_repaired or len(doc) == 0:
        doc.close()
        raise PdfUnreadableError(unreadable_msg)
    try:
        page_texts: list[str] = []
        page_has_images: list[bool] = []
        for page in doc:
            page_texts.append(page.get_text("text"))
            page_has_images.append(len(page.get_images()) > 0)
    finally:
        doc.close()

    return chunk_pages(
        page_texts=page_texts,
        source_file=pdf_path.name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        removed_lines_collector=removed_lines_collector,
        page_has_images=page_has_images,
    )


def chunk_directory(
    pdf_dir: Path,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    removed_lines_collector: list[tuple[str, RemovedLine]] | None = None,
    skipped_pages_collector: list[tuple[str, SkippedPage]] | None = None,
) -> tuple[list[Chunk], list[str]]:
    """Đọc mọi file .pdf trong thư mục, bỏ qua file scan và trả về (chunks, skipped_files)."""
    pdf_dir = Path(pdf_dir)
    all_chunks: list[Chunk] = []
    skipped_files: list[str] = []

    for pdf_file in sorted(pdf_dir.glob("*.pdf")):
        try:
            file_removed: list[RemovedLine] = []
            file_chunks, file_skipped_pages = chunk_pdf(
                pdf_file,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                removed_lines_collector=file_removed,
            )
            if not file_chunks:
                # Có lớp chữ nhưng mọi dòng bị lọc (vd. chỉ có header/footer): báo bỏ qua để indexer
                # giữ dữ liệu cũ của file thay vì coi là file đã biến mất và xóa sạch.
                skipped_files.append(pdf_file.name)
                print(
                    f"Bỏ qua {pdf_file.name}: không còn nội dung sau khi làm sạch (chỉ có header/footer?)",
                    file=sys.stderr,
                )
                continue
            all_chunks.extend(file_chunks)
            if removed_lines_collector is not None:
                for item in file_removed:
                    removed_lines_collector.append((pdf_file.name, item))
            if skipped_pages_collector is not None:
                for sp in file_skipped_pages:
                    skipped_pages_collector.append((pdf_file.name, sp))
        except PdfNoTextError:
            skipped_files.append(pdf_file.name)
            print(
                f"Bỏ qua {pdf_file.name}: không có lớp chữ (PDF scan?) — cần OCR",
                file=sys.stderr,
            )
        except PdfUnreadableError:
            skipped_files.append(pdf_file.name)
            print(
                f"Bỏ qua {pdf_file.name}: file PDF hỏng hoặc bị khóa",
                file=sys.stderr,
            )

    return all_chunks, skipped_files


def main(argv: list[str] | None = None) -> int:
    """CLI chạy thử nghiệm bộ cắt chunk trên 1 file PDF hoặc cả thư mục."""
    setup_utf8_console()
    parser = argparse.ArgumentParser(
        description="Cắt chunk từ file PDF hoặc thư mục chứa PDF quy chế."
    )
    parser.add_argument(
        "target",
        type=Path,
        nargs="?",
        default=Path("data/pdfs"),
        help="Đường dẫn tới file PDF hoặc thư mục chứa PDF (mặc định: data/pdfs).",
    )
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    args = parser.parse_args(argv)

    target: Path = args.target
    if not target.exists():
        print(f"Không tìm thấy đường dẫn: {target}", file=sys.stderr)
        return 1

    try:
        args.chunk_size, args.chunk_overlap = _resolve_chunk_params(args.chunk_size, args.chunk_overlap)
    except ValueError as exc:
        print(f"Lỗi cấu hình: {exc}", file=sys.stderr)
        return 1

    all_removed: list[tuple[str, RemovedLine]] = []
    all_skipped_pages: list[tuple[str, SkippedPage]] = []
    if target.is_file():
        try:
            file_removed: list[RemovedLine] = []
            chunks, skipped_pages = chunk_pdf(
                target,
                args.chunk_size,
                args.chunk_overlap,
                removed_lines_collector=file_removed,
            )
            skipped: list[str] = []
            for item in file_removed:
                all_removed.append((target.name, item))
            for sp in skipped_pages:
                all_skipped_pages.append((target.name, sp))
        except (PdfNoTextError, PdfUnreadableError) as exc:
            print(f"Bỏ qua {exc}", file=sys.stderr)
            return 1
    else:
        chunks, skipped = chunk_directory(
            target,
            args.chunk_size,
            args.chunk_overlap,
            removed_lines_collector=all_removed,
            skipped_pages_collector=all_skipped_pages,
        )

    articles = list(dict.fromkeys(c.article for c in chunks if c.article is not None))
    pages = sorted({(c.source_file, c.page_number) for c in chunks})

    print(f"Tổng số trang có chunk: {len(pages)}")
    print(f"Tổng số chunk sinh ra : {len(chunks)}")
    print(f"Số Điều nhận diện được: {len(articles)} -> {articles}")
    if skipped:
        print(f"File bị bỏ qua (scan / hỏng / bị khóa): {skipped}")

    if all_skipped_pages:
        print(f"\nDanh sách trang không có lớp chữ bị bỏ qua ({len(all_skipped_pages)} trang):")
        for filename, sp in all_skipped_pages:
            desc = "scan ảnh" if sp.reason == "scan" else "trang trắng"
            print(f"  - [{filename} | Trang {sp.page_number}]: {desc}")

    if all_removed:
        print(f"\nDanh sách dòng header/footer lặp bị lọc ({len(all_removed)} dòng):")
        for filename, item in all_removed:
            print(f"  - [{filename} | Trang {item.page_number} | {item.position}]: {item.text}")
    else:
        print("\nKhông có dòng header/footer lặp nào bị lọc.")

    print("\n--- 3 CHUNK MẪU ĐẦU TIÊN ---")
    for sample_chunk in chunks[:3]:
        print(
            f"[{sample_chunk.source_file} | Trang {sample_chunk.page_number} | "
            f"Chunk #{sample_chunk.chunk_index} | article={sample_chunk.article} | "
            f"len={len(sample_chunk.text)} | id={sample_chunk.chunk_id}]"
        )
        print(f"  {sample_chunk.text}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
