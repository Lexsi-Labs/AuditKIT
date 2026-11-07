"""A TypeError raised inside a scorer must not be retried without ``context``, and never turn into a score."""
from __future__ import annotations

import functools

import pytest

from auditkit.sample import Sample
from auditkit.scorers import FunctionScorer
from auditkit.types import Direction

S = Sample(input="q", target="t")


def scorer(fn):
    return FunctionScorer(fn, Direction.MAXIMIZE)


def counted(fn):
    calls = []

    @functools.wraps(fn)
    def inner(*a, **k):
        calls.append(k.get("context", "no-context"))
        return fn(*a, **k)

    return inner, calls


def test_two_argument_scorer_still_works():
    assert scorer(lambda sample, output: 0.5)(S, "o").value == 0.5


def test_scorer_naming_context_gets_it():
    seen = []

    def s(sample, output, context=None):
        seen.append(context)
        return 1.0

    scorer(s)(S, "o", {"k": 1})
    assert seen == [{"k": 1}]


def test_typeerror_inside_a_context_scorer_is_raised_once():
    runs = []

    def s(sample, output, context=None):
        runs.append(context)
        raise TypeError("boom")

    with pytest.raises(TypeError, match="boom"):
        scorer(s)(S, "o", "ctx")
    assert runs == ["ctx"]


def test_typeerror_inside_a_kwargs_scorer_is_raised_once():
    runs = []

    def s(sample, output, **kw):
        runs.append(kw)
        raise TypeError("boom")

    with pytest.raises(TypeError, match="boom"):
        scorer(s)(S, "o")
    assert len(runs) == 1


def test_kwargs_scorer_gets_context():
    got = {}

    def s(sample, output, **kw):
        got.update(kw)
        return 1.0

    scorer(s)(S, "o", "ctx")
    assert got == {"context": "ctx"}


def test_typeerror_inside_a_decorated_context_scorer_is_raised_once():
    def raises(sample, output, context=None):
        raise TypeError("boom")

    inner, calls = counted(raises)
    with pytest.raises(TypeError, match="boom"):
        scorer(inner)(S, "o")
    assert len(calls) == 1


def test_decorated_two_argument_scorer_falls_back():
    def two(sample, output):
        return 1.0

    inner, calls = counted(two)
    # the wrapper forwards context= to a function that does not take it: one failed call, then the retry
    assert scorer(inner)(S, "o").value == 1.0


def test_legacy_positional_context_name_keeps_working():
    # def s(sample, context): `output` arrives as context, as before; not made an error by this change
    assert scorer(lambda sample, context: 0.25)(S, "o").value == 0.25
