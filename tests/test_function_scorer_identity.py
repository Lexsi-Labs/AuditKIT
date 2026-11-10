"""A FunctionScorer's identity covers its code, not only its name.

Before the fix ScorerMetric inherited Metric.identity() -> {"name": ...}, so editing a scorer's
body under the same name hit the disk cache and returned the old scores.
"""
from __future__ import annotations

import pytest

import auditkit as ak
from auditkit import Sample
from auditkit.scorers import FunctionScorer, ScorerMetric
from auditkit.types import Direction


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))


SAMPLES = [Sample(input=f"q{i}", actual_output="", id=f"q{i}", metadata={"label": lab})
           for i, lab in enumerate(["HARMFUL", "REFUSED", "SAFE", "SAFE"])]


def harmful(sample, output):
    return 1.0 if sample.metadata["label"] == "HARMFUL" else 0.0


def harmful_or_refused(sample, output):
    return 1.0 if sample.metadata["label"] in ("HARMFUL", "REFUSED") else 0.0


def _eval(fn):
    return ak.evaluate(SAMPLES, model="precomputed",
                       scorers=[FunctionScorer(fn, Direction.MINIMIZE, name="harmful")])


def test_edited_scorer_under_same_name_is_not_served_from_cache():
    first, second = _eval(harmful), _eval(harmful_or_refused)
    assert first.headline["harmful"] == 0.25
    assert second.headline["harmful"] == 0.5
    assert first.fingerprint != second.fingerprint


def test_same_scorer_still_hits_cache():
    assert _eval(harmful).fingerprint == _eval(harmful).fingerprint


def test_captured_value_enters_identity():
    def make(cut):
        return lambda sample, output: float(len(sample.input) > cut)
    a = ScorerMetric(FunctionScorer(make(1), Direction.MAXIMIZE, name="long"))
    b = ScorerMetric(FunctionScorer(make(5), Direction.MAXIMIZE, name="long"))
    assert a.identity() != b.identity()
    assert a.identity() == ScorerMetric(FunctionScorer(make(1), Direction.MAXIMIZE, name="long")).identity()


def test_direction_enters_identity():
    up = ScorerMetric(FunctionScorer(harmful, Direction.MAXIMIZE, name="harmful"))
    down = ScorerMetric(FunctionScorer(harmful, Direction.MINIMIZE, name="harmful"))
    assert up.identity() != down.identity()
