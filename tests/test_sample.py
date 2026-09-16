from auditkit.sample import Sample
from auditkit.types import TaskKind


# ---- Sample --------------------------------------------------------------

def test_minimal_sample_defaults():
    s = Sample(input="2+2?", target="4")
    assert s.input == "2+2?"
    assert s.target == "4"
    assert s.kind == TaskKind.GENERATIVE
    assert s.id is None
    assert s.task == ""
    assert s.metadata == {}
    assert s.tags == []
    assert s.actual_output is None


def test_is_golden_true_when_target_present():
    assert Sample(input="q", target="a").is_golden is True


def test_is_golden_false_without_target():
    assert Sample(input="q").is_golden is False
    assert Sample(input="q", target=None).is_golden is False


def test_mutable_defaults_are_not_shared():
    a = Sample(input="x")
    b = Sample(input="y")
    a.metadata["k"] = 1
    a.tags.append("t")
    assert b.metadata == {}
    assert b.tags == []


def test_mcq_sample_carries_choices():
    s = Sample(input="Capital of France?", target="Paris",
               kind=TaskKind.MCQ, choices=["London", "Paris", "Rome"])
    assert s.kind == TaskKind.MCQ
    assert s.choices == ["London", "Paris", "Rome"]


# ---- input_text (what text backends see) ---------------------------------

def test_input_text_from_plain_string():
    assert Sample(input="hello").input_text == "hello"
