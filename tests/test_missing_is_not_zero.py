"""A score that could not be measured is not 0 (or a neutral value) in an aggregate."""
from __future__ import annotations

import types

import pytest

import auditkit as ak
from auditkit.experiment import Experiment
from auditkit.metrics.hallucination import FactualConsistency
from auditkit.metrics.pairwise import PreferenceAccuracy, WinRate
from auditkit.report import RunResult
from auditkit.score import Score
from auditkit.scoring import WeightedSum

S = ak.Sample(input="q", target="a")


def run(headline, rid="r"):
    return RunResult(run_id=rid, fingerprint="f", stats={}, headline=headline, predictions=[])


class TestExperimentAggregate:
    def test_run_without_the_metric_is_left_out(self):
        e = Experiment("e", [run({"acc": 1.0}, "a"), run({"f1": 0.4}, "b"), run({"acc": 0.5}, "c")])
        assert e.aggregate("acc") == {"acc": 0.75}
        assert e.aggregate() == {"acc": 0.75, "f1": 0.4}

    def test_no_run_has_the_metric_is_none_not_zero(self):
        assert Experiment("e", [run({"acc": 1.0})]).aggregate("f1") == {"f1": None}

    def test_empty_headline_runs_do_not_make_zeros(self):
        e = Experiment("e", [run({}), run({"acc": 0.8})])
        assert e.aggregate() == {"acc": 0.8}

    def test_a_real_zero_still_counts(self):
        e = Experiment("e", [run({"acc": 0.0}), run({"acc": 1.0})])
        assert e.aggregate("acc") == {"acc": 0.5}

    def test_no_runs(self):
        assert Experiment("e").aggregate() == {} and Experiment("e").aggregate("acc") == {}


class TestPairwise:
    def test_win_rate_without_candidates_is_unknown(self):
        sc = WinRate().score(S, "a", context={})
        assert sc.metadata["unknown"] is True and "candidates" in sc.reason

    def test_win_rate_with_candidates_is_measured(self):
        sc = WinRate().score(S, "a", context={"candidates": ["b", "c"]})
        assert "unknown" not in sc.metadata and sc.value == 1.0

    def test_preference_without_data_is_unknown(self):
        sc = PreferenceAccuracy().score(S, "a", context={})
        assert sc.metadata["unknown"] is True

    def test_preference_with_data(self):
        sc = PreferenceAccuracy().score(S, "good", context={"preference_data": [{"chosen": "good", "rejected": "bad"}]})
        assert sc.value == 1.0 and "unknown" not in sc.metadata


class TestFactualConsistency:
    def _metric(self, label):
        m = FactualConsistency()
        m._pipe = lambda text: [{"label": label}]
        # the transformers import happens before the pipe is used; skip when it is missing
        pytest.importorskip("transformers")
        return m

    @pytest.mark.parametrize("label,value", [("ENTAILMENT", 1.0), ("CONTRADICTION", 0.0), ("NEUTRAL", 0.5)])
    def test_known_labels(self, label, value):
        sc = self._metric(label).score(S, "x")
        assert sc.value == value and "unknown" not in sc.metadata

    @pytest.mark.parametrize("label", ["LABEL_0", "weird", ""])
    def test_unrecognised_label_is_unknown_not_neutral(self, label):
        sc = self._metric(label).score(S, "x")
        assert sc.metadata["unknown"] is True and "unrecognised" in sc.reason


class TestWeightedSum:
    def test_no_scores_is_unknown(self):
        sc = WeightedSum().compute([])
        assert sc.metadata["unknown"] is True

    def test_zero_total_weight_is_unknown(self):
        sc = WeightedSum().compute([Score(name="a", value=0.9, weight=0.0)])
        assert sc.metadata["unknown"] is True and "zero" in sc.reason

    def test_normal_sum_unchanged(self):
        sc = WeightedSum().compute([Score(name="a", value=1.0, weight=1.0), Score(name="b", value=0.0, weight=3.0)])
        assert sc.value == pytest.approx(0.25) and "unknown" not in sc.metadata
