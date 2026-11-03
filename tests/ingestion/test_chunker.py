from pathlib import Path
import unicodedata

import fitz
import pytest

from src.ingestion.sample_pdf import build_sample_pdf
from src.ingestion.chunker import (
    ARTICLE_HEADING_RE,
    PdfNoTextError,
    SkippedPage,
    chunk_directory,
    chunk_pages,
    chunk_pdf,
    normalize_text,
    parse_article_heading,
    parse_multiline_article_heading,
    split_by_article,
)


@pytest.fixture
def sample(tmp_path):
    p = tmp_path / "quy_che_mau.pdf"
    return p, build_sample_pdf(p)


def _blank_pdf(path):
    doc = fitz.open()
    doc.new_page()
    doc.save(path)
    doc.close()


def _pdf_with_blank_and_text_pages(path):
    """Tạo file PDF với:
    - Trang 1: trang trắng (không chữ, không ảnh)
    - Trang 2: scan ảnh giả lập bằng cách chèn 1 pixmap nhỏ (không chữ)
    - Trang 3: trang có text tiếng Việt với font Noto Sans
    """
    doc = fitz.open()
    # Trang 1: trắng
    doc.new_page()

    # Trang 2: scan ảnh giả lập
    p2 = doc.new_page()
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 50, 50), 1)
    p2.insert_image(fitz.Rect(100, 100, 200, 200), pixmap=pix)

    # Trang 3: văn bản tiếng Việt
    p3 = doc.new_page()
    font_path = Path("assets/fonts/NotoSans-Regular.ttf")
    font = fitz.Font(fontfile=str(font_path))
    p3.insert_font(fontname="noto", fontbuffer=font.buffer)
    p3.insert_text(
        fitz.Point(72, 100),
        "Điều 1. Phạm vi áp dụng\nNội dung văn bản quy chế.",
        fontname="noto",
        fontsize=12,
    )

    doc.save(path)
    doc.close()


# --- Regex tiêu đề Điều ---
@pytest.mark.parametrize(
    "line,expected",
    [
        ("Điều 5. Học phí", "Điều 5. Học phí"),
        ("Điều 5: Học phí", "Điều 5. Học phí"),
        (
            "  Điều 12.  Xét tốt nghiệp và công nhận tốt nghiệp ",
            "Điều 12. Xét tốt nghiệp và công nhận tốt nghiệp",
        ),
        ("Điều 5.", "Điều 5"),
        ("Điều 5 của Quy chế này quy định", None),  # tham chiếu: không có "." hay ":" sau số
        ("theo Điều 5. Học phí", None),  # không ở đầu dòng
    ],
)
def test_parse_article_heading(line, expected):
    assert parse_article_heading(line) == expected


def test_heading_without_title_does_not_break():
    chunks, _ = chunk_pages(
        ["Điều 5.\nSinh viên đóng học phí theo tín chỉ."], "x.pdf", 800, 150
    )
    assert len(chunks) == 1
    c = chunks[0]
    assert c.article == "Điều 5"
    assert c.text == "Điều 5: Sinh viên đóng học phí theo tín chỉ."


def test_reference_line_is_not_a_heading():
    pages = ["Điều 4. Đánh giá\nNội dung.\nĐiều 5 của Quy chế này quy định học phí."]
    chunks, _ = chunk_pages(pages, "x.pdf", 800, 150)
    assert {c.article for c in chunks} == {"Điều 4. Đánh giá"}


# --- Tiêu đề Điều nhiều dòng ---
def test_multiline_heading_with_comma_merged():
    line1 = "Điều 16. Chuyển ngành, chuyển nơi học, chuyển cơ sở đào tạo,"
    line2 = "chuyển hình thức học"
    heading, consumed = parse_multiline_article_heading(line1, line2)
    assert consumed is True
    assert (
        heading
        == "Điều 16. Chuyển ngành, chuyển nơi học, chuyển cơ sở đào tạo, chuyển hình thức học"
    )


def test_multiline_heading_without_comma_merged():
    line1 = "Điều 12. Xét tốt"
    line2 = "nghiệp và công nhận tốt nghiệp"
    heading, consumed = parse_multiline_article_heading(line1, line2)
    assert consumed is True
    assert heading == "Điều 12. Xét tốt nghiệp và công nhận tốt nghiệp"


def test_multiline_heading_not_merged_when_uppercase():
    line1 = "Điều 2. Khen thưởng"
    line2 = "Điều 3. Kỷ luật"
    heading, consumed = parse_multiline_article_heading(line1, line2)
    assert consumed is False
    assert heading == "Điều 2. Khen thưởng"


@pytest.mark.parametrize(
    "next_line",
    [
        "1. Sinh viên vi phạm quy chế sẽ bị kỷ luật.",
        "a) Khiển trách đối với sinh viên vi phạm lần đầu.",
        "đ) Buộc thôi học đối với sinh viên vi phạm nghiêm trọng.",
    ],
)
def test_multiline_heading_not_merged_when_clause_or_point(next_line):
    line1 = "Điều 20. Kỷ luật sinh viên"
    heading, consumed = parse_multiline_article_heading(line1, next_line)
    assert consumed is False
    assert heading == "Điều 20. Kỷ luật sinh viên"


def test_multiline_heading_not_merged_when_ended_with_period():
    line1 = "Điều 5. Học phí."
    line2 = "mức học phí được quy định hàng năm"
    heading, consumed = parse_multiline_article_heading(line1, line2)
    assert consumed is False
    assert heading == "Điều 5. Học phí"


# --- Tách theo Điều trước khi chunk ---
def test_split_by_article_separates_two_articles():
    page = (
        "cuối nội dung Điều 5\n"
        "Điều 6. Đăng ký học phần\n"
        "Sinh viên đăng ký tối thiểu 14 tín chỉ."
    )
    segments, last_seen = split_by_article(page, carried="Điều 5. Học phí")
    assert segments == [
        ("Điều 5. Học phí", "cuối nội dung Điều 5"),
        ("Điều 6. Đăng ký học phần", "Sinh viên đăng ký tối thiểu 14 tín chỉ."),
    ]
    assert last_seen == "Điều 6. Đăng ký học phần"


def test_no_chunk_contains_two_articles(sample):
    path, _ = sample
    chunks, _ = chunk_pdf(path)
    for c in chunks:
        body = c.text[len(c.article) + 2 :] if c.article else c.text  # bỏ tiền tố "Điều X. Tên: "
        assert ARTICLE_HEADING_RE.search(body) is None


def test_page_with_two_articles_gives_separate_chunks(sample):
    path, m = sample
    chunks, _ = chunk_pdf(path)
    page = m["Điều 6. Đăng ký học phần"]
    arts = {c.article for c in chunks if c.page_number == page}
    assert {"Điều 5. Học phí", "Điều 6. Đăng ký học phần"} <= arts


def test_text_before_first_article_has_no_prefix(sample):
    path, m = sample
    chunks, _ = chunk_pdf(path)
    first_page = m["Điều 1. Phạm vi điều chỉnh"]
    pre = [c for c in chunks if c.article is None and c.page_number == first_page]
    assert pre, "phần mở đầu văn bản phải thành chunk riêng"
    for c in pre:
        assert "QUY CHẾ ĐÀO TẠO" in c.text or "Ban hành" in c.text or "CHƯƠNG" in c.text
        assert not c.text.startswith("Điều")


def test_heading_at_page_bottom_carried_to_next_page():
    pages = [
        "TRƯỜNG ĐẠI HỌC MẪU\nĐiều 1. Khái niệm\nNội dung điều 1.\nĐiều 2. Khen thưởng\nTrang 1",
        "TRƯỜNG ĐẠI HỌC MẪU\nSinh viên có thành tích xuất sắc được khen thưởng theo quy chế.\nTrang 2",
        "TRƯỜNG ĐẠI HỌC MẪU\nĐiều 3. Kỷ luật\nNội dung điều 3.\nTrang 3",
    ]
    chunks, _ = chunk_pages(pages, "test.pdf", 800, 150)
    page2_chunks = [c for c in chunks if c.page_number == 2]
    assert len(page2_chunks) == 1
    assert page2_chunks[0].article == "Điều 2. Khen thưởng"
    assert page2_chunks[0].text.startswith("Điều 2. Khen thưởng: ")
    assert "khen thưởng theo quy chế" in page2_chunks[0].text


def test_two_consecutive_headings_at_page_bottom_without_body():
    pages = [
        "Điều 1. Khái niệm\nNội dung điều 1.\nĐiều 2. Khen thưởng\nĐiều 3. Kỷ luật",
        "Sinh viên vi phạm kỷ luật bị xử lý theo quy định hiện hành.",
    ]
    chunks, _ = chunk_pages(pages, "test.pdf", 800, 150)
    page2_chunks = [c for c in chunks if c.page_number == 2]
    assert len(page2_chunks) == 1
    assert page2_chunks[0].article == "Điều 3. Kỷ luật"
    assert page2_chunks[0].text.startswith("Điều 3. Kỷ luật: ")


def test_heading_at_page_bottom_then_chapter_on_next_page():
    pages = [
        "Điều 1. Khái niệm\nNội dung điều 1.\nĐiều 2. Khen thưởng",
        "CHƯƠNG II. HỌC PHÍ\nNội dung của chương về tài chính.",
    ]
    chunks, _ = chunk_pages(pages, "test.pdf", 800, 150)
    page2_none = [c for c in chunks if c.page_number == 2 and c.article is None]
    assert len(page2_none) == 1
    assert "CHƯƠNG II. HỌC PHÍ" in page2_none[0].text
    assert not page2_none[0].text.startswith("Điều")


def test_distinct_numbered_lines_at_page_bottom_not_removed_as_footer():
    pages = [
        "Điều 1. A\nNội dung Điều 4.",
        "Điều 2. B\nNội dung Điều 5.",
        "Điều 3. C\nNội dung Điều 6.",
    ]
    chunks, _ = chunk_pages(pages, "test.pdf", 800, 150)
    assert len(chunks) == 3
    p1 = [c for c in chunks if c.page_number == 1][0]
    p2 = [c for c in chunks if c.page_number == 2][0]
    p3 = [c for c in chunks if c.page_number == 3][0]
    assert "Nội dung Điều 4." in p1.text
    assert "Nội dung Điều 5." in p2.text
    assert "Nội dung Điều 6." in p3.text


def test_chapter_section_part_resets_carried_article():
    pages = [
        "Điều 4. Đánh giá học phần\nNội dung Điều 4.",
        "CHƯƠNG II. HỌC PHÍ\nMục 1. Quy định chung\nPhần I: Nguyên tắc\nĐiều 5. Học phí\nNội dung Điều 5.",
        "Điều 6. Đăng ký\nTheo Chương II và Mục 1 đã nêu ở trên, sinh viên đăng ký học phần.",
    ]
    chunks, _ = chunk_pages(pages, "x.pdf", 800, 150)
    page1 = [c for c in chunks if c.page_number == 1]
    assert len(page1) == 1
    assert page1[0].article == "Điều 4. Đánh giá học phần"
    assert "Nội dung Điều 4." in page1[0].text

    page2_none = [c for c in chunks if c.page_number == 2 and c.article is None]
    assert len(page2_none) == 1
    assert "CHƯƠNG II. HỌC PHÍ" in page2_none[0].text
    assert "Mục 1. Quy định chung" in page2_none[0].text
    assert "Phần I: Nguyên tắc" in page2_none[0].text
    assert not page2_none[0].text.startswith("Điều")
    page3 = [c for c in chunks if c.page_number == 3]
    assert len(page3) == 1 and page3[0].article == "Điều 6. Đăng ký"


def test_two_line_chapter_heading_resets_carried_article():
    pages = [
        "Điều 4. Đánh giá học phần\nNội dung Điều 4.",
        "Chương II\nHỌC PHÍ VÀ ĐĂNG KÝ HỌC PHẦN\nĐiều 5. Học phí\nMức học phí là 650.000 đồng.",
    ]
    chunks, _ = chunk_pages(pages, "x.pdf", 800, 150)
    page2_none = [c for c in chunks if c.page_number == 2 and c.article is None]
    assert len(page2_none) == 1
    assert page2_none[0].text == "Chương II\nHỌC PHÍ VÀ ĐĂNG KÝ HỌC PHẦN"
    page2_d5 = [c for c in chunks if c.page_number == 2 and c.article == "Điều 5. Học phí"]
    assert len(page2_d5) == 1
    assert page2_d5[0].text == "Điều 5. Học phí: Mức học phí là 650.000 đồng."


# --- Metadata & số trang ---
def test_metadata_schema_complete(sample):
    path, _ = sample
    chunks, _ = chunk_pdf(path)
    assert chunks
    for c in chunks:
        assert c.source_file == "quy_che_mau.pdf"
        assert c.page_number >= 1 and c.chunk_index >= 0 and c.text.strip() and c.chunk_id


def test_page_number_matches_exact_pdf_page(sample):
    path, m = sample
    chunks, _ = chunk_pdf(path)
    for article, page in m.items():
        assert min(c.page_number for c in chunks if c.article == article) == page, article


def test_article_prefix_carried_across_pages(sample):
    path, m = sample
    chunks, _ = chunk_pdf(path)
    cont = [c for c in chunks if "[TIẾP ĐIỀU 5]" in c.text]
    assert cont
    for c in cont:
        assert c.page_number == m["Điều 5. Học phí"] + 1
        assert c.article == "Điều 5. Học phí" and c.text.startswith("Điều 5. Học phí: ")


def test_prefix_not_duplicated(sample):
    path, _ = sample
    chunks, _ = chunk_pdf(path)
    for c in chunks:
        if c.article:
            assert c.text.startswith(f"{c.article}: ")
            assert f"{c.article}: {c.article}" not in c.text


def test_chunk_index_sequential_per_page(sample):
    path, _ = sample
    chunks, _ = chunk_pdf(path)
    by_page: dict[int, list[int]] = {}
    for c in chunks:
        by_page.setdefault(c.page_number, []).append(c.chunk_index)
    assert all(v == list(range(len(v))) for v in by_page.values())


# --- chunk_size tính cả tiền tố ---
@pytest.mark.parametrize("size,overlap", [(400, 80), (800, 150), (1200, 200)])
def test_chunk_size_includes_prefix(sample, size, overlap):
    path, _ = sample
    chunks, _ = chunk_pdf(path, size, overlap)
    assert all(len(c.text) <= size for c in chunks)


def test_chunk_size_too_small_for_title_raises(sample):
    path, _ = sample
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_pdf(path, chunk_size=60, chunk_overlap=10)


# --- Làm sạch ---
def test_unicode_nfc_normalization():
    nfd = unicodedata.normalize("NFD", "Điều 5. Học phí")
    assert nfd != unicodedata.normalize("NFC", nfd)  # đảm bảo đầu vào thật sự là NFD
    assert normalize_text(nfd) == unicodedata.normalize("NFC", "Điều 5. Học phí")


def test_nfd_heading_still_detected():
    nfd_page = unicodedata.normalize("NFD", "Điều 7. Cảnh báo học tập\nNội dung.")
    chunks, _ = chunk_pages([nfd_page], "x.pdf", 800, 150)
    assert chunks[0].article == "Điều 7. Cảnh báo học tập"


def test_repeated_header_footer_removed(sample):
    path, _ = sample
    chunks, _ = chunk_pdf(path)
    for c in chunks:
        assert "TRƯỜNG ĐẠI HỌC MẪU" not in c.text and "Trang " not in c.text


def test_deterministic_chunk_id(sample):
    path, _ = sample
    chunks_1, _ = chunk_pdf(path)
    chunks_2, _ = chunk_pdf(path)
    first = [c.chunk_id for c in chunks_1]
    second = [c.chunk_id for c in chunks_2]
    assert first == second and len(set(first)) == len(first)


# --- PDF scan & Trang không có chữ (SkippedPage) ---
def test_scanned_pdf_raises_and_directory_skips(tmp_path):
    build_sample_pdf(tmp_path / "ok.pdf")
    _blank_pdf(tmp_path / "scan.pdf")
    with pytest.raises(PdfNoTextError):
        chunk_pdf(tmp_path / "scan.pdf")
    chunks, skipped = chunk_directory(tmp_path)
    assert skipped == ["scan.pdf"]
    assert {c.source_file for c in chunks} == {"ok.pdf"}


def test_scanned_pdf_message_names_file(tmp_path, capsys):
    _blank_pdf(tmp_path / "quy_che_scan.pdf")
    with pytest.raises(PdfNoTextError, match="quy_che_scan.pdf"):
        chunk_pdf(tmp_path / "quy_che_scan.pdf")
    chunk_directory(tmp_path)
    assert "Bỏ qua quy_che_scan.pdf" in capsys.readouterr().err


def test_skipped_pages_returned_and_warned_on_stderr(tmp_path, capsys):
    pdf_path = tmp_path / "mixed.pdf"
    _pdf_with_blank_and_text_pages(pdf_path)

    chunks, skipped_pages = chunk_pdf(pdf_path)
    err = capsys.readouterr().err

    assert len(skipped_pages) == 2
    # Trang 1: trắng
    assert skipped_pages[0] == SkippedPage(page_number=1, reason="blank", has_images=False)
    assert "Trang 1" in err and "trang trắng" in err

    # Trang 2: scan ảnh
    assert skipped_pages[1] == SkippedPage(page_number=2, reason="scan", has_images=True)
    assert "Trang 2" in err and "trang scan ảnh" in err

    # Trang 3: sinh chunk bình thường
    assert len(chunks) == 1
    assert chunks[0].page_number == 3
    assert chunks[0].article == "Điều 1. Phạm vi áp dụng"


def test_removed_lines_collector_and_cli_output(sample, capsys):
    path, _ = sample
    from src.ingestion.chunker import RemovedLine, main

    removed: list[RemovedLine] = []
    chunk_pdf(path, removed_lines_collector=removed)
    assert len(removed) >= 6
    assert any(r.position == "header" and "TRƯỜNG ĐẠI HỌC MẪU" in r.text for r in removed)
    assert any(r.position == "footer" and "Trang " in r.text for r in removed)

    # Chạy CLI kiểm tra có in danh sách dòng bị lọc
    exit_code = main([str(path)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Danh sách dòng header/footer lặp bị lọc" in out
    assert "TRƯỜNG ĐẠI HỌC MẪU" in out
