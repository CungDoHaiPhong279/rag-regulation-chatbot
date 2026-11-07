import unicodedata

import pytest

from src.rag.citations import extract_citation_ids, is_refusal, normalize_refusal
from src.rag.constants import REFUSAL


def test_refusal_constant_exact():
    assert REFUSAL == "Không tìm thấy thông tin trong tài liệu quy chế."


@pytest.mark.parametrize(
    "variant",
    [
        REFUSAL,
        REFUSAL.rstrip("."),
        f'"{REFUSAL}"',
        f"“{REFUSAL}”",
        f"“{REFUSAL.rstrip('.')}”.",
        f"\n  {REFUSAL}  \n",
        unicodedata.normalize("NFD", REFUSAL),
        f"**{REFUSAL}**",
        REFUSAL.replace("trong tài", "trong\ntài"),
    ],
)
def test_refusal_variants_normalize_to_constant(variant):
    assert normalize_refusal(variant) == REFUSAL
    assert is_refusal(variant)


def test_answer_containing_refusal_plus_content_is_not_refusal():
    answer = REFUSAL + " Tuy nhiên theo Điều 5 [1]..."
    assert not is_refusal(answer)
    assert normalize_refusal(answer) == answer


def test_normal_answer_is_only_stripped():
    assert normalize_refusal('  Sinh viên được nghỉ "tối đa" 2 kỳ [1].\n') == 'Sinh viên được nghỉ "tối đa" 2 kỳ [1].'


def test_empty_answer_is_not_refusal():
    assert not is_refusal("")


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("A [2]. B [1][2]. C [10]", [2, 1, 10]),
        ("Không có trích dẫn", []),
        ("", []),
        ("A [1, 3]. B [2,1]", [1, 3, 2]),  # Mô hình hay gộp số; bỏ sót sẽ thành từ chối nhầm
        ("Xem [a], [1a], [], [-1], [1.5]", []),
    ],
)
def test_extract_citation_ids(answer, expected):
    assert extract_citation_ids(answer) == expected
