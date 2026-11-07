from src.rag.llm import LLMError
from src.rag.models import Turn
from src.rag.prompts import REWRITE_SYSTEM_PROMPT
from src.rag.rewriter import MAX_HISTORY_TURNS, MAX_TURN_CHARS, RewriteResult, rewrite
from tests.fakes import FakeLLM

HISTORY = [Turn("user", "Cảnh báo học tập là gì?"), Turn("assistant", "Là ... [1].")]


def test_no_history_returns_question_without_calling_llm():
    llm = FakeLLM([])
    assert rewrite("Câu hỏi?", [], llm) == RewriteResult("Câu hỏi?", failed=False)
    assert llm.prompts == []


def test_history_returns_stripped_llm_output_with_rewrite_system():
    llm = FakeLLM(["  Khi nào bị cảnh báo học tập?\n"])
    assert rewrite("Khi nào bị?", HISTORY, llm) == RewriteResult("Khi nào bị cảnh báo học tập?", failed=False)
    assert llm.systems == [REWRITE_SYSTEM_PROMPT]
    assert "Khi nào bị?" in llm.prompts[0]


def test_llm_error_falls_back_to_question():
    assert rewrite("Khi nào bị?", HISTORY, FakeLLM([LLMError("x")])) == RewriteResult("Khi nào bị?", failed=True)


def test_blank_llm_output_falls_back_to_question():
    assert rewrite("Khi nào bị?", HISTORY, FakeLLM(["  \n"])) == RewriteResult("Khi nào bị?", failed=True)


def test_no_llm_returns_question_not_failed():
    assert rewrite("Khi nào bị?", HISTORY, None) == RewriteResult("Khi nào bị?", failed=False)


def test_rewrite_prompt_bounded_history():
    history = [Turn("user", f"<{i:02}>" + "x" * 5000) for i in range(50)]
    llm = FakeLLM(["q"])
    rewrite("Câu hỏi?", history, llm)
    prompt = llm.prompts[0]
    assert MAX_HISTORY_TURNS == 3
    assert MAX_TURN_CHARS == 500
    assert [f"<{i:02}>" in prompt for i in (46, 47, 48, 49)] == [False, True, True, True]
    assert prompt.count("x") <= 3 * 500
    assert prompt.count("x") == 3 * (500 - len("<00>"))
