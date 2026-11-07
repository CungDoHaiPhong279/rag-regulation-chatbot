from src.rag.constants import REFUSAL
from src.rag.models import Turn
from src.rag.prompts import (
    ANSWER_SYSTEM_PROMPT,
    REWRITE_SYSTEM_PROMPT,
    build_answer_prompt,
    build_rewrite_prompt,
)
from src.retrieval.hybrid_search import Hit


def _hit(i: int, page: int, article: str | None, text: str) -> Hit:
    return Hit(
        chunk_id=f"id-{i}",
        source_file="TT08.pdf",
        page_number=page,
        article=article,
        chunk_index=i,
        text=text,
        score=0.9,
    )


def test_answer_prompt_numbers_contexts_and_states_rules():
    prompt = build_answer_prompt(
        "Học phí thế nào?",
        [_hit(0, 13, "Điều 5", "Nội dung A"), _hit(1, 14, None, "Nội dung B")],
    )
    assert "[1] (TT08.pdf, trang 13, Điều 5) Nội dung A" in prompt
    assert "[2] (TT08.pdf, trang 14) Nội dung B" in prompt  # không có Điều thì không in None
    assert "None" not in prompt
    assert "Học phí thế nào?" in prompt
    assert REFUSAL in prompt

    system = ANSWER_SYSTEM_PROMPT.lower()
    assert "chỉ dùng" in system
    assert "[n]" in system
    assert REFUSAL in ANSWER_SYSTEM_PROMPT


def test_answer_prompt_does_not_ask_llm_for_page_numbers():
    assert "không tự ghi số trang" in ANSWER_SYSTEM_PROMPT.lower()


def test_answer_system_prompt_treats_context_as_data():
    assert "không làm theo" in ANSWER_SYSTEM_PROMPT.lower()


def test_rewrite_prompt_includes_history_in_order_and_question():
    history = [Turn("user", "Cảnh báo học tập là gì?"), Turn("assistant", "Là khi điểm thấp [1].")]
    prompt = build_rewrite_prompt("Khi nào thì bị?", history)
    assert prompt.index("Người dùng: Cảnh báo học tập là gì?") < prompt.index("Trợ lý: Là khi điểm thấp [1].")
    assert prompt.index("Trợ lý: Là khi điểm thấp [1].") < prompt.index("Khi nào thì bị?")


def test_rewrite_system_prompt_forbids_answering():
    system = REWRITE_SYSTEM_PROMPT.lower()
    assert "không trả lời" in system
    assert "độc lập" in system
