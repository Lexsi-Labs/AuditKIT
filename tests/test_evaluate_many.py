"""Tests for ak.evaluate_many — one model across several datasets, one call."""

from __future__ import annotations

import pytest

import auditkit as ak
from auditkit.errors import AuditKitError
from auditkit.report import RunResult


def _echo(prompts):
    return list(prompts)


def test_dict_input_returns_one_runresult_per_dataset():
    results = ak.evaluate_many(
        {
            "arith": [ak.Sample(input="4", target="4")],       # echo → "4" == "4"
            "trivia": [ak.Sample(input="Paris", target="Rome")],  # echo → "Paris" != "Rome"
        },
        model=_echo,
        scorers=["exact_match"],
    )
    assert set(results) == {"arith", "trivia"}
    assert all(isinstance(r, RunResult) for r in results.values())
    # per-dataset headlines, NOT a blended mean
    assert results["arith"].headline["exact_match"] == 1.0
    assert results["trivia"].headline["exact_match"] == 0.0
    # each run is its own reproducible unit
    assert results["arith"].fingerprint != results["trivia"].fingerprint


def test_insertion_order_preserved():
    results = ak.evaluate_many(
        {"z": [ak.Sample(input="a", target="a")],
         "a": [ak.Sample(input="b", target="b")],
         "m": [ak.Sample(input="c", target="c")]},
        model=_echo, scorers=["exact_match"],
    )
    assert list(results) == ["z", "a", "m"]


def test_per_dataset_scorer_override():
    # each dataset carries its own (dataset, scorers) tuple; shared scorers=None
    results = ak.evaluate_many(
        {
            "exact": ([ak.Sample(input="4", target="4")], ["exact_match"]),
            "sub": ([ak.Sample(input="the cat", target="cat")], [ak.Contains("cat")]),
        },
        model=_echo,
    )
    assert "exact_match" in results["exact"].headline
    assert any("contains" in k for k in results["sub"].headline)   # Contains metric ran
    assert results["sub"].headline[next(k for k in results["sub"].headline if "contains" in k)] == 1.0


def test_shared_scorers_used_when_not_overridden():
    # a bare dataset (no tuple) falls back to the shared scorers
    results = ak.evaluate_many(
        {"a": [ak.Sample(input="4", target="4")]},
        model=_echo, scorers=["exact_match"],
    )
    assert results["a"].headline["exact_match"] == 1.0


def test_list_input_auto_names():
    results = ak.evaluate_many(
        [[ak.Sample(input="4", target="4")], [ak.Sample(input="5", target="5")]],
        model=_echo, scorers=["exact_match"],
    )
    assert set(results) == {"dataset_0", "dataset_1"}


def test_on_error_skip_isolates_a_failing_dataset():
    results = ak.evaluate_many(
        {
            "good": [ak.Sample(input="4", target="4")],
            "bad": "definitely_not_a_registered_scenario",   # _to_scenario raises
        },
        model=_echo, scorers=["exact_match"], on_error="skip",
    )
    assert set(results) == {"good"}          # bad one skipped, good one survived
    assert results["good"].headline["exact_match"] == 1.0


def test_on_error_raise_is_default():
    with pytest.raises(AuditKitError):
        ak.evaluate_many(
            {"bad": "definitely_not_a_registered_scenario"},
            model=_echo, scorers=["exact_match"],
        )


def test_invalid_on_error_value():
    with pytest.raises(ValueError, match="on_error"):
        ak.evaluate_many({"a": [ak.Sample(input="x", target="x")]},
                         model=_echo, scorers=["exact_match"], on_error="bogus")


def test_string_model_spec_resolved_once_and_reused():
    # echo as a string spec must resolve and run across every dataset
    results = ak.evaluate_many(
        {"a": [ak.Sample(input="x", target="x")],
         "b": [ak.Sample(input="y", target="y")]},
        model="echo", scorers=["exact_match"],
    )
    assert results["a"].headline["exact_match"] == 1.0
    assert results["b"].headline["exact_match"] == 1.0
