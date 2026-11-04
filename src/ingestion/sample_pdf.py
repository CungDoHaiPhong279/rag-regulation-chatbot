"""Bộ sinh file PDF quy chế đào tạo mẫu bằng tiếng Việt để kiểm thử tự động.

File PDF sinh ra có cấu trúc Chương - Điều - Khoản rõ ràng, biết trước chính xác
mỗi Điều nằm ở trang nào, có dòng header/footer lặp lại ở mọi trang, và có Điều 5
kéo dài qua hai trang (trang 3 sang trang 4, nơi Điều 6 bắt đầu ngay bên dưới)
để kiểm tra toàn diện bộ cắt chunk (chunker) và bộ lập chỉ mục (indexer).
"""

from __future__ import annotations

from pathlib import Path
import fitz

from src.utils import setup_utf8_console

REPO_ROOT = Path(__file__).resolve().parents[2]
FONT_PATH: Path = REPO_ROOT / "assets" / "fonts" / "NotoSans-Regular.ttf"
SAMPLE_FILENAME: str = "quy_che_mau.pdf"
# PDF mẫu là dữ liệu HƯ CẤU (chỉ để kiểm thử): đặt ngoài data/pdfs/ để indexer không đưa vào kho thật.
SAMPLE_OUTPUT_PATH: Path = REPO_ROOT / "data" / "samples" / SAMPLE_FILENAME

HEADER_TEXT = "TRƯỜNG ĐẠI HỌC MẪU — QUY CHẾ ĐÀO TẠO"

# Bố cục cố định từng trang: mỗi phần tử là danh sách các đoạn văn trên trang đó
# và danh sách các tiêu đề Điều bắt đầu trên trang đó.
PAGES_SPEC: list[tuple[list[str], str]] = [
    # Trang 1: Chương I + Điều 1, Điều 2
    (
        ["Điều 1. Phạm vi điều chỉnh", "Điều 2. Tín chỉ"],
        (
            "QUY CHẾ ĐÀO TẠO TRÌNH ĐỘ ĐẠI HỌC (MẪU)\n"
            "(Ban hành kèm theo Quyết định số 123/QĐ-ĐHM ngày 01 tháng 09 năm 2026)\n\n"
            "CHƯƠNG I. QUY ĐỊNH CHUNG\n\n"
            "Điều 1. Phạm vi điều chỉnh\n"
            "1. Quy chế này quy định về tổ chức và quản lý đào tạo trình độ đại học hình thức "
            "chính quy theo hệ thống tín chỉ tại Trường Đại học Mẫu.\n"
            "2. Quy chế áp dụng đối với sinh viên các khóa đào tạo đại học chính quy và các "
            "khoa, phòng ban có liên quan.\n\n"
            "Điều 2. Tín chỉ\n"
            "1. Tín chỉ là đơn vị dùng để tính khối lượng học tập của sinh viên. Một tín chỉ "
            "được tính tương đương 15 tiết học lý thuyết hoặc 30 tiết thực hành, thí nghiệm.\n"
            "2. Đối với mỗi tiết học lý thuyết trên lớp, sinh viên phải dành ít nhất 2 giờ "
            "tự học và chuẩn bị bài ở nhà."
        ),
    ),
    # Trang 2: Điều 3, Điều 4
    (
        ["Điều 3. Chương trình đào tạo", "Điều 4. Đánh giá học phần"],
        (
            "Điều 3. Chương trình đào tạo\n"
            "1. Chương trình đào tạo đại học có tổng khối lượng tối thiểu 130 tín chỉ (không "
            "bao gồm học phần Giáo dục thể chất và Giáo dục quốc phòng - an ninh).\n"
            "2. Thời gian thiết kế chuẩn cho một khóa học đại học là 4 năm (tương đương 8 học "
            "kỳ chính); thời gian tối đa để sinh viên hoàn thành khóa học không quá 6 năm.\n\n"
            "Điều 4. Đánh giá học phần\n"
            "1. Điểm tổng kết học phần được tính theo thang điểm 10 (làm tròn đến một chữ số "
            "thập phân), gồm điểm quá trình chiếm trọng số 40% và điểm thi kết thúc học phần "
            "chiếm trọng số 60%.\n"
            "2. Học phần đạt yêu cầu tích lũy khi có điểm tổng kết từ 4,0 trở lên theo thang "
            "điểm 10 (tương đương điểm D trở lên theo thang điểm chữ và 1,0 theo thang điểm 4)."
        ),
    ),
    # Trang 3: Chương II + phần đầu Điều 5 (sẽ tràn sang trang 4)
    (
        ["Điều 5. Học phí"],
        (
            "CHƯƠNG II. HỌC PHÍ VÀ ĐĂNG KÝ HỌC PHẦN\n\n"
            "Điều 5. Học phí\n"
            "1. Sinh viên có nghĩa vụ đóng học phí đầy đủ theo số tín chỉ đã đăng ký trong "
            "từng học kỳ. Mức học phí cơ sở là 650.000 đồng cho mỗi tín chỉ học lần đầu.\n"
            "2. Đối với các học phần học lại hoặc học cải thiện điểm, mức học phí áp dụng "
            "bằng 100% mức học phí của học phần học lần đầu theo biểu phí hiện hành.\n"
            "3. Thời hạn hoàn thành nghĩa vụ học phí chậm nhất là 30 ngày kể từ ngày bắt đầu "
            "học kỳ chính thức theo thông báo của Phòng Kế hoạch - Tài chính."
        ),
    ),
    # Trang 4: Phần tiếp của Điều 5 + Điều 6 bắt đầu ngay trên cùng trang
    (
        ["Điều 6. Đăng ký học phần"],
        (
            "[TIẾP ĐIỀU 5] 4. Sinh viên thuộc diện chính sách hoặc có hoàn cảnh khó khăn được "
            "xét gia hạn đóng học phí tối đa 45 ngày nếu nộp đơn xin gia hạn trước hạn chót "
            "ít nhất 7 ngày làm việc.\n"
            "5. Sinh viên không hoàn thành học phí đúng hạn và không có quyết định gia hạn sẽ "
            "bị hủy kết quả đăng ký học phần của học kỳ đó.\n\n"
            "Điều 6. Đăng ký học phần\n"
            "1. Trong mỗi học kỳ chính (trừ học kỳ cuối khóa), sinh viên phải đăng ký tối "
            "thiểu 14 tín chỉ và tối đa 24 tín chỉ.\n"
            "2. Sinh viên có điểm trung bình tích lũy đạt từ 3,2 trở lên theo thang điểm 4 "
            "được phép đăng ký tối đa 28 tín chỉ trong một học kỳ chính."
        ),
    ),
    # Trang 5: Chương III + Điều 7, Điều 8
    (
        ["Điều 7. Cảnh báo học tập", "Điều 8. Buộc thôi học"],
        (
            "CHƯƠNG III. XÉT HỌC VỤ VÀ TỐT NGHIỆP\n\n"
            "Điều 7. Cảnh báo học tập\n"
            "1. Sinh viên bị cảnh báo học tập vào cuối mỗi học kỳ chính nếu rơi vào một trong "
            "các trường hợp: điểm trung bình học kỳ dưới 0,8 đối với học kỳ đầu tiên của khóa "
            "học, hoặc dưới 1,0 đối với các học kỳ tiếp theo.\n"
            "2. Sinh viên có điểm trung bình tích lũy dưới 1,2 ở năm thứ nhất, dưới 1,4 ở năm "
            "thứ hai, dưới 1,6 ở năm thứ ba hoặc dưới 1,8 ở các năm tiếp theo cũng bị cảnh "
            "báo học tập.\n\n"
            "Điều 8. Buộc thôi học\n"
            "1. Sinh viên bị buộc thôi học nếu có số lần cảnh báo học tập vượt quá 2 lần liên "
            "tiếp hoặc tổng cộng vượt quá 3 lần trong toàn khóa học.\n"
            "2. Sinh viên tự ý bỏ học không lý do từ 1 học kỳ chính trở lên hoặc hết thời gian "
            "tối đa được phép học tại trường cũng thuộc diện buộc thôi học."
        ),
    ),
    # Trang 6: Điều 9, Điều 10
    (
        ["Điều 9. Xét tốt nghiệp", "Điều 10. Hiệu lực thi hành"],
        (
            "Điều 9. Xét tốt nghiệp\n"
            "1. Sinh viên được công nhận tốt nghiệp khi tích lũy đủ tối thiểu 130 tín chỉ của "
            "chương trình đào tạo, có điểm trung bình tích lũy toàn khóa đạt từ 2,0 trở lên "
            "theo thang điểm 4.\n"
            "2. Tại thời điểm xét tốt nghiệp, sinh viên phải có chứng chỉ ngoại ngữ đạt chuẩn "
            "đầu ra tương đương bậc 3/6 theo Khung năng lực ngoại ngữ 6 bậc dùng cho Việt Nam "
            "và không đang trong thời gian bị kỷ luật từ mức đình chỉ học tập trở lên.\n\n"
            "Điều 10. Hiệu lực thi hành\n"
            "1. Quy chế này có hiệu lực thi hành kể từ năm học 2026 - 2027.\n"
            "2. Trong quá trình thực hiện, nếu có vướng mắc phát sinh, các đơn vị báo cáo "
            "Hiệu trưởng thông qua Phòng Đào tạo để xem xét, sửa đổi và bổ sung kịp thời."
        ),
    ),
]


def build_sample_pdf(out_path: Path) -> dict[str, int]:
    """Tạo file PDF quy chế mẫu và trả về bảng ánh xạ {tên Điều: số trang (1-based)}.

    Args:
        out_path: Đường dẫn file PDF đầu ra.

    Returns:
        Từ điển ánh xạ tiêu đề chuẩn của từng Điều sang số trang vật lý (bắt đầu từ 1).

    Raises:
        FileNotFoundError: Khi thiếu file font NotoSans-Regular.ttf trong assets/fonts/.
    """
    if not FONT_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy font tiếng Việt tại '{FONT_PATH}'. "
            "Hãy tải NotoSans-Regular.ttf vào thư mục assets/fonts/."
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open()
    article_to_page: dict[str, int] = {}

    for page_idx, (articles_on_page, body_text) in enumerate(PAGES_SPEC, start=1):
        page = doc.new_page(width=595, height=842)  # Khổ giấy A4 chuẩn (points)
        page.insert_font(fontname="noto", fontfile=str(FONT_PATH))

        full_page_text = f"{HEADER_TEXT}\n\n{body_text}\n\nTrang {page_idx}"
        content_rect = fitz.Rect(50, 40, 545, 800)
        rc = page.insert_textbox(
            content_rect,
            full_page_text,
            fontname="noto",
            fontsize=11,
            lineheight=1.35,
        )
        if rc < 0:
            doc.close()
            raise RuntimeError(f"Nội dung trang {page_idx} bị tràn khỏi khung A4 (rc={rc}).")

        for article_title in articles_on_page:
            article_to_page[article_title] = page_idx

    doc.save(out_path)
    doc.close()
    return article_to_page


def main() -> None:
    """Sinh file PDF quy chế mẫu vào data/samples/quy_che_mau.pdf (ngoài kho tài liệu thật)."""
    setup_utf8_console()
    target = SAMPLE_OUTPUT_PATH
    mapping = build_sample_pdf(target)
    print(f"Đã tạo PDF quy chế mẫu tại: {target}")
    for article, page_num in mapping.items():
        print(f"  - {article} -> Trang {page_num}")


if __name__ == "__main__":
    main()
