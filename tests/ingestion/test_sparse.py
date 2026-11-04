import unicodedata

import pytest

from src.ingestion.sparse import (
    SparseVec,
    average_doc_length,
    bm25_doc_vector,
    query_vector,
    token_id,
    tokenize,
)


def _weights(tokens, avgdl):
    v = bm25_doc_vector(tokens, avgdl)
    return dict(zip(v.indices, v.values))


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Học phí", ["học_phí"]),
        ("Số tín chỉ tối thiểu", ["số", "tín_chỉ", "tối_thiểu"]),
        (
            "Sinh viên bị cảnh báo học tập khi nào?",
            ["sinh_viên", "bị", "cảnh_báo", "học_tập", "khi", "nào"],
        ),
        ("???  ", []),
    ],
)
def test_tokenize_vietnamese(text, expected):
    assert tokenize(text) == expected


def test_tokenize_normalizes_nfd():
    assert tokenize(unicodedata.normalize("NFD", "Học phí")) == ["học_phí"]


def test_token_id_is_stable():
    assert token_id("học_phí") == 3962211330
    assert token_id("tín_chỉ") == 1233938498
    assert token_id("cảnh_báo") == 3775047272


def test_bm25_tf_saturation_and_length_norm():
    a = token_id("a")
    assert _weights(["a"] + ["x"] * 9, 10.0)[a] == pytest.approx(1.0)  # tf=1, dl=avgdl
    assert _weights(["a", "a"] + ["x"] * 8, 10.0)[a] == pytest.approx(1.375)  # tf=2
    assert _weights(["a"] + ["x"] * 19, 10.0)[a] == pytest.approx(0.7096774)  # dl = 2·avgdl
    assert _weights(["a"] * 10, 10.0)[a] == pytest.approx(1.9642857)  # bão hòa < k1+1


def test_not_raw_tf():
    # tf thô sẽ cho 10; BM25 bão hòa dưới k1+1 = 2.2
    assert _weights(["a"] * 10, 10.0)[token_id("a")] < 2.2


def test_custom_k1_b():
    # b=0 tắt chuẩn hóa độ dài: dl gấp đôi vẫn cho cùng trọng số
    v1 = bm25_doc_vector(["a"] + ["x"] * 9, 10.0, k1=1.2, b=0.0)
    v2 = bm25_doc_vector(["a"] + ["x"] * 19, 10.0, k1=1.2, b=0.0)
    i = v1.indices.index(token_id("a"))
    j = v2.indices.index(token_id("a"))
    assert v1.values[i] == pytest.approx(v2.values[j])


def test_query_vector_unique_terms_weight_one():
    q = query_vector("học phí học phí")
    assert q.indices == [token_id("học_phí")] and q.values == [1.0]


def test_query_vector_empty_for_punctuation():
    assert query_vector("???") == SparseVec([], [])


def test_doc_vector_has_unique_sorted_indices():
    v = bm25_doc_vector(tokenize("tín chỉ tín chỉ học phí"), 5.0)
    assert len(v.indices) == len(set(v.indices))
    assert v.indices == sorted(v.indices)


def test_doc_vector_merges_colliding_ids(monkeypatch):
    import src.ingestion.sparse as sparse

    monkeypatch.setattr(sparse, "token_id", lambda tok: 7)
    v = sparse.bm25_doc_vector(["a", "b"], 2.0)
    assert v.indices == [7] and len(v.values) == 1
    assert v.values[0] == pytest.approx(2.0)  # hai token tf=1, dl=avgdl -> 1.0 + 1.0


def test_average_doc_length():
    assert average_doc_length([["a"], ["a", "b", "c"]]) == 2.0


def test_average_doc_length_empty_raises():
    with pytest.raises(ValueError):
        average_doc_length([])
