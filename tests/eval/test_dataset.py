import json

import pytest

from eval.dataset import Gold, Question, load_questions

GOLD = {"source_file": "tt08.pdf", "article": "Điều 11", "pages": [13, 14]}


def _row(id_="q1", in_scope=True, split="test", gold=None, question="Hỏi?"):
    return {"id": id_, "question": question, "in_scope": in_scope, "split": split,
            "gold": [GOLD] if gold is None and in_scope else (gold or [])}


def _write(tmp_path, rows):
    path = tmp_path / "questions.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n\n", encoding="utf-8")
    return path


def test_load_valid_file(tmp_path):
    path = _write(tmp_path, [_row("q1"), _row("q2", in_scope=False, split="dev")])
    assert load_questions(path) == [
        Question("q1", "Hỏi?", True, "test", (Gold("tt08.pdf", "Điều 11", (13, 14)),)),
        Question("q2", "Hỏi?", False, "dev", ()),
    ]


def test_filter_by_split(tmp_path):
    path = _write(tmp_path, [_row("q1", split="test"), _row("q2", split="dev"), _row("q3", split="dev")])
    assert [q.id for q in load_questions(path, split="dev")] == ["q2", "q3"]
    assert [q.id for q in load_questions(path, split="test")] == ["q1"]


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([_row("q1"), _row("q1")], "trùng"),
        ([_row(split="train")], "split"),
        ([_row(in_scope=True, gold=[])], "gold"),
        ([{**_row(in_scope=False), "gold": [GOLD]}], "gold"),
        ([{k: v for k, v in _row().items() if k != "question"}], "question"),
        ([_row(gold=[{**GOLD, "pages": []}])], "pages"),
        ([_row(question="   ")], "question"),
        ([_row(gold=[{**GOLD, "article": "Khoản 2, Điều 11"}])], "Điều <số>"),
        ([_row(gold=[{**GOLD, "article": "11"}])], "Điều <số>"),
    ],
)
def test_invalid_rows_raise_with_line_number(tmp_path, rows, message):
    with pytest.raises(ValueError, match=message) as err:
        load_questions(_write(tmp_path, rows))
    assert "dòng" in str(err.value)


def test_invalid_json_reports_line(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps(_row()) + "\n{hỏng\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dòng 2"):
        load_questions(path)


def test_invalid_split_argument(tmp_path):
    with pytest.raises(ValueError):
        load_questions(_write(tmp_path, [_row()]), split="train")


def test_sample_file_in_repo_is_valid():
    questions = load_questions(__import__("pathlib").Path("eval/data/questions.jsonl"))
    assert len(questions) >= 3
    assert any(not q.in_scope for q in questions)
