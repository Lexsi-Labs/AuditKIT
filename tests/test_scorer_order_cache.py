"""The scorers' order is part of the run's identity: the first one decides Prediction.correct and score.

Before, the cache key sorted the scorers, so the same samples with the scorers in another order were served the
first run's cached ``correct`` even though a fresh run gives another answer.
"""

from __future__ import annotations

import auditkit as ak
import pytest


@pytest.fixture
def cache(tmp_path, monkeypatch):
    def use(name: str):
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / name))
    return use


def run(order):
    samples = [ak.Sample(id="1", input="q", target="paris", actual_output="Paris!")]
    r = ak.evaluate(samples, "precomputed", order)
    return r.predictions[0].correct, r.fingerprint


A = ["exact_match", "f1_score"]   # exact_match first: "Paris!" is not exactly "paris"
B = ["f1_score", "exact_match"]   # f1_score first: the normalised tokens match


def test_the_scorers_decide_differently_in_each_order(cache):
    cache("a"); first = run(A)
    cache("b"); second = run(B)
    assert (first[0], second[0]) == (False, True)


def test_another_order_is_not_served_the_first_orders_cached_result(cache):
    cache("fresh"); fresh_b = run(B)
    cache("shared"); first_a = run(A); then_b = run(B)
    assert then_b[0] == fresh_b[0] is True          # was False: the first run's answer, from the cache
    assert first_a[1] != then_b[1]


def test_the_same_order_still_hits_the_cache(cache):
    cache("same"); first = run(A); again = run(A)
    assert first == again
