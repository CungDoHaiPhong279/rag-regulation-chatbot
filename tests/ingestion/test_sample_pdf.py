import fitz
from src.ingestion.sample_pdf import build_sample_pdf, FONT_PATH


def test_font_file_present():
    assert FONT_PATH.exists() and FONT_PATH.suffix == ".ttf"


def test_article_pages_known(tmp_path):
    out = tmp_path / "quy_che_mau.pdf"
    mapping = build_sample_pdf(out)
    doc = fitz.open(out)
    assert doc.page_count >= 5 and len(mapping) == 10
    for article, page in mapping.items():
        assert article in doc[page - 1].get_text()


def test_dieu5_spans_two_pages(tmp_path):
    out = tmp_path / "x.pdf"
    p = build_sample_pdf(out)["Điều 5. Học phí"]
    assert "[TIẾP ĐIỀU 5]" in fitz.open(out)[p].get_text()  # trang p+1 (0-based index p)


def test_page_has_end_of_one_article_and_start_of_next(tmp_path):
    out = tmp_path / "x.pdf"
    m = build_sample_pdf(out)
    assert m["Điều 6. Đăng ký học phần"] == m["Điều 5. Học phí"] + 1  # trang p+1 chứa cuối Điều 5 VÀ đầu Điều 6


def test_vietnamese_text_extracts_exactly(tmp_path):
    out = tmp_path / "x.pdf"
    build_sample_pdf(out)
    text = fitz.open(out)[0].get_text()
    assert "QUY CHẾ ĐÀO TẠO" in text and "sinh viên" in text and "Điều 1. Phạm vi điều chỉnh" in text


def test_repeated_header_footer_on_every_page(tmp_path):
    out = tmp_path / "x.pdf"
    build_sample_pdf(out)
    for page in fitz.open(out):
        assert "TRƯỜNG ĐẠI HỌC MẪU" in page.get_text()



def test_sample_output_is_outside_corpus_dir():
    # PDF mẫu là dữ liệu hư cấu: không được nằm trong data/pdfs/ (thư mục indexer đọc)
    from src.ingestion.sample_pdf import REPO_ROOT, SAMPLE_OUTPUT_PATH

    corpus_dir = REPO_ROOT / "data" / "pdfs"
    assert corpus_dir not in SAMPLE_OUTPUT_PATH.parents
    assert SAMPLE_OUTPUT_PATH.name == "quy_che_mau.pdf"
