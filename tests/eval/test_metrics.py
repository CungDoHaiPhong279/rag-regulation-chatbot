import pytest

from eval.dataset import Gold
from eval.metrics import article_number, is_relevant, mrr_at_k, recall_at_k
from src.retrieval.hybrid_search import Hit

GOLD = Gold("tt08.pdf", "Điều 11", (13, 14))


def _hit(page=13, article="Điều 11. Xử lý kết quả học tập", source="tt08.pdf", i=0) -> Hit:
    return Hit(f"id{i}", source, page, article, i, "text", 0.5)


def _miss(i: int) -> Hit:
    return _hit(page=99, article="Điều 99", i=i)


@pytest.mark.parametrize(
    ("article", "expected"),
    [
        ("Điều 11. Xử lý kết quả học tập", 11),
        ("Điều 11", 11),
        ("Điều  7:", 7),
        ("Ðiều 3", None),  # Ð (chữ Đ khác mã) không phải "Điều"
        ("Chương 2", None),
        ("Khoản 2, Điều 11", None),
        (None, None),
        ("", None),
    ],
)
def test_article_number(article, expected):
    assert article_number(article) == expected


def test_article_number_handles_nfd():
    import unicodedata

    assert article_number(unicodedata.normalize("NFD", "Điều 12. Tên")) == 12


@pytest.mark.parametrize(
    ("hit", "expected"),
    [
        (_hit(), True),
        (_hit(page=14), True),  # Điều kéo 2 trang, trúng trang thứ hai
        (_hit(source="khac.pdf"), False),
        (_hit(article="Điều 12. Khác"), False),
        (_hit(article="Điều 110. Khác"), False),  # không khớp kiểu tiền tố chuỗi
        (_hit(page=15), False),
        (_hit(article=None), False),
    ],
)
def test_is_relevant(hit, expected):
    assert is_relevant(hit, GOLD) is expected


def test_missing_article_numbers_never_match_each_other():
    assert not is_relevant(_hit(article=None), Gold("tt08.pdf", "Phụ lục", (13,)))


def test_recall_depends_on_k():
    hits = [_miss(i) for i in range(5)] + [_hit(i=5)]  # gold ở hạng 6
    assert recall_at_k(hits, [GOLD], k=5) == 0.0
    assert recall_at_k(hits, [GOLD], k=20) == 1.0


def test_recall_partial_and_duplicate_hits():
    other = Gold("tt08.pdf", "Điều 20", (20,))
    hits = [_hit(i=0), _hit(page=14, i=1)]  # 2 chunk cùng trúng 1 gold vẫn chỉ tính 1 gold
    assert recall_at_k(hits, [GOLD, other], k=5) == 0.5


def test_recall_empty_hits():
    assert recall_at_k([], [GOLD], k=5) == 0.0


def test_recall_without_gold_is_error():
    with pytest.raises(ValueError):
        recall_at_k([_hit()], [], k=5)


def test_mrr():
    assert mrr_at_k([_miss(0), _hit(i=1)], [GOLD], k=5) == 0.5
    assert mrr_at_k([_hit(i=0)], [GOLD], k=5) == 1.0
    assert mrr_at_k([_miss(i) for i in range(5)] + [_hit(i=5)], [GOLD], k=5) == 0.0
    assert mrr_at_k([], [GOLD], k=5) == 0.0
