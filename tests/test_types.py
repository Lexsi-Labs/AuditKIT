from auditkit.types import (
    TaskKind, ScoreKind, DataType, Direction, Capability,
)


def test_str_enums_compare_as_strings():
    # str-backed enums so `Direction.MAXIMIZE == "maximize"` and JSON-serialize cleanly
    assert Direction.MAXIMIZE == "maximize"
    assert ScoreKind.JUDGE == "judge"


def test_capability_and_datatype_members():
    assert Capability.LOGLIKELIHOOD.value == "loglikelihood"
    assert {d.value for d in DataType} == {"numeric", "categorical", "boolean"}


def test_taskkind_has_the_techniques_we_evaluate():
    for name in ("MCQ", "GENERATIVE", "RAG", "LANGUAGE_MODELING"):
        assert hasattr(TaskKind, name)
