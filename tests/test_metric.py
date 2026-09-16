import asyncio

import pytest

from auditkit.metric import Metric, ExactMatch, QuasiExactMatch
from auditkit.sample import Sample
from auditkit.score import Score
from auditkit.types import ScoreKind


# ---- Metric ABC / applicability ------------------------------------------

def test_metric_is_abstract():
    with pytest.raises(TypeError):
        Metric()  # score() is abstract


def test_applicable_false_when_required_field_missing():
    m = ExactMatch()
    assert m.required_fields == frozenset({"target"})
    assert m.applicable(Sample(input="q", target="a")) is True
    assert m.applicable(Sample(input="q")) is False  # no target


def test_metric_defaults():
    m = ExactMatch()
    assert m.kind == ScoreKind.BENCHMARK
    assert m.is_deterministic is True


# ---- ExactMatch ----------------------------------------------------------

def test_exact_match_hit_and_miss():
    m = ExactMatch()
    s = Sample(input="2+2?", target="4")
    assert m.score(s, "4").value == 1.0
    assert m.score(s, "5").value == 0.0


def test_exact_match_trims_surrounding_whitespace():
    m = ExactMatch()
    s = Sample(input="q", target="Paris")
    assert m.score(s, "  Paris\n").value == 1.0


def test_exact_match_is_case_sensitive():
    m = ExactMatch()
    s = Sample(input="q", target="Paris")
    assert m.score(s, "paris").value == 0.0


def test_exact_match_returns_named_score():
    score = ExactMatch().score(Sample(input="q", target="a"), "a")
    assert isinstance(score, Score)
    assert score.name == "exact_match"


# ---- QuasiExactMatch (normalized) ----------------------------------------

def test_quasi_exact_match_ignores_case_and_articles():
    m = QuasiExactMatch()
    s = Sample(input="q", target="The Paris")
    assert m.score(s, "paris").value == 1.0
    assert m.score(s, "London").value == 0.0


# ---- async wrapper -------------------------------------------------------

def test_ascore_defaults_to_score():
    m = ExactMatch()
    s = Sample(input="q", target="a")
    result = asyncio.run(m.ascore(s, "a"))
    assert result.value == 1.0
