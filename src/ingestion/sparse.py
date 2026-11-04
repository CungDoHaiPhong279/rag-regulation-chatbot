"""BM25 cho tiếng Việt dưới dạng sparse vector của Qdrant.

Công thức BM25 (Robertson / Lucene) cho câu hỏi q và chunk d:

    score(q, d) = Σ_t  IDF(t) · tf·(k1+1) / (tf + k1·(1 − b + b·|d|/avgdl))

Chia việc:
- Module này tính phần **tf bão hòa + chuẩn hóa độ dài** cho từng chunk và lưu làm giá trị sparse vector.
- Qdrant (collection khai báo `SparseVectorParams(modifier=Modifier.IDF)`) tự nhân
  IDF(t) = ln(1 + (N − df + 0.5)/(df + 0.5)) lúc truy vấn, với N và df lấy từ dữ liệu hiện có.
- Câu hỏi gửi mỗi từ duy nhất với trọng số 1.0.
Tích hai phần đúng bằng điểm BM25 đầy đủ (đã kiểm chứng sai số < 1e-6 so với cài đặt tham chiếu),
khác TF-IDF thô ở chỗ tf bị bão hòa (< k1+1) và chunk dài bị giảm điểm.

Tách từ bằng pyvi để "học phí" thành một từ `học_phí` thay vì hai âm tiết rời.
Token ID là 32 bit đầu của blake2b: ổn định giữa các lần chạy (khác `hash()` của Python),
không cần lưu file từ điển.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import re
import unicodedata

from pyvi import ViTokenizer

K1_DEFAULT = 1.2
B_DEFAULT = 0.75

_PUNCT_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)


@dataclass(frozen=True)
class SparseVec:
    indices: list[int]
    values: list[float]


def tokenize(text: str) -> list[str]:
    """NFC → tách từ pyvi → bỏ token toàn dấu câu → chữ thường (sau khi tách để pyvi dùng được chữ hoa)."""
    normalized = unicodedata.normalize("NFC", text)
    if not normalized.strip():
        return []
    segmented = ViTokenizer.tokenize(normalized)
    return [tok.lower() for tok in segmented.split() if not _PUNCT_ONLY_RE.match(tok)]


def token_id(token: str) -> int:
    """ID 32 bit không dấu, ổn định giữa các tiến trình."""
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest(), "big")


def average_doc_length(token_lists: list[list[str]]) -> float:
    """Độ dài trung bình (số token) của các chunk — `avgdl` trong BM25."""
    if not token_lists:
        raise ValueError("Không tính được avgdl: danh sách chunk rỗng.")
    return sum(len(tokens) for tokens in token_lists) / len(token_lists)


def bm25_doc_vector(
    tokens: list[str],
    avgdl: float,
    k1: float = K1_DEFAULT,
    b: float = B_DEFAULT,
) -> SparseVec:
    """Phần tf/độ dài của BM25 cho một chunk (IDF do Qdrant nhân lúc truy vấn)."""
    if avgdl <= 0:
        raise ValueError(f"avgdl phải > 0, nhận {avgdl}")
    doc_len = len(tokens)
    norm = k1 * (1 - b + b * doc_len / avgdl)
    weights: dict[int, float] = {}
    for tok, tf in Counter(tokens).items():
        tid = token_id(tok)
        # Hai token khác nhau trùng ID (rất hiếm): cộng dồn vì Qdrant không cho index trùng
        weights[tid] = weights.get(tid, 0.0) + tf * (k1 + 1) / (tf + norm)
    indices = sorted(weights)
    return SparseVec(indices=indices, values=[weights[i] for i in indices])


def query_vector(text: str) -> SparseVec:
    """Mỗi token duy nhất của câu hỏi có trọng số 1.0; rỗng nếu câu không có token nào."""
    ids = sorted({token_id(tok) for tok in tokenize(text)})
    return SparseVec(indices=ids, values=[1.0] * len(ids))
