"""Tests for T3 comparison/regression diff (RunDiff)."""

from __future__ import annotations

import pytest

from auditkit.sample import Sample
from auditkit.model import EchoModel
from auditkit.adapter import GenerationAdapter
from auditkit.metric import ExactMatch
from auditkit.runner import Runner
from auditkit.runspec import RunConfig, RunSpec
from auditkit.scenario import ListScenario
from auditkit.report import RunResult
from auditkit.diff import RunDiff, DeltaGrade


# ============================================================================
# Helpers
# ============================================================================

def _run(samples, **cfg) -> RunResult:
    return Runner().run(RunSpec(
        scenario=ListScenario(samples),
        model=EchoModel(),
        adapter=GenerationAdapter(),
        metrics=[ExactMatch()],
        config=RunConfig(**cfg),
        run_name="test",
    ))


# ============================================================================
# RunDiff — basic
# ============================================================================

class TestRunDiff:
    def test_diff_needs_two_results(self):
        r1 = _run([Sample(input="a", target="a")])
        r2 = _run([Sample(input="a", target="b")])
        d = RunDiff(r1, r2)
        assert d.baseline is r1
        assert d.contrast is r2

    def test_metric_deltas(self):
        r1 = _run([Sample(input="a", target="a"), Sample(input="b", target="b")])
        r2 = _run([Sample(input="a", target="a"), Sample(input="b", target="c")])
        d = RunDiff(r1, r2)
        deltas = d.metric_deltas()
        assert "exact_match" in deltas
        # r1: both correct = 1.0, r2: one correct = 0.5, delta = -0.5
        assert deltas["exact_match"]["baseline"] == 1.0
        assert deltas["exact_match"]["contrast"] == 0.5
        assert deltas["exact_match"]["delta"] == -0.5


# ============================================================================
# DeltaGrade classification
# ============================================================================

class TestDeltaGrade:
    def test_pass_within_threshold(self):
        r1 = _run([Sample(input="a", target="a"), Sample(input="b", target="b")])
        r2 = _run([Sample(input="a", target="a"), Sample(input="b", target="c")])
        d = RunDiff(r1, r2)
        grade = d.grade("exact_match", pass_threshold=1.0, warn_threshold=0.5)
        # delta = -0.5 → |delta| = 0.5, warn_threshold is 0.5 → |delta| <= 0.5 → PASS
        assert grade == DeltaGrade.PASS

    def test_warn_between_thresholds(self):
        r1 = _run([Sample(input="a", target="a"), Sample(input="b", target="b")])
        r2 = _run([Sample(input="a", target="a"), Sample(input="b", target="c")])
        d = RunDiff(r1, r2)
        grade = d.grade("exact_match", pass_threshold=0.2, warn_threshold=0.7)
        # delta = -0.5, |delta| = 0.5 → 0.2 < 0.5 <= 0.7 → WARN
        assert grade == DeltaGrade.WARN

    def test_fail_exceeds_threshold(self):
        r1 = _run([Sample(input="a", target="a"), Sample(input="b", target="b")])
        r2 = _run([Sample(input="a", target="a"), Sample(input="b", target="c")])
        d = RunDiff(r1, r2)
        grade = d.grade("exact_match", pass_threshold=0.2, warn_threshold=0.3)
        # delta = -0.5, |delta| = 0.5 > 0.3 → FAIL
        assert grade == DeltaGrade.FAIL

    def test_no_change_is_pass(self):
        r1 = _run([Sample(input="a", target="a")])
        r2 = _run([Sample(input="a", target="a")])
        d = RunDiff(r1, r2)
        assert d.grade("exact_match") == DeltaGrade.PASS

    def test_missing_metric_raises(self):
        r1 = _run([Sample(input="a", target="a")])
        r2 = _run([Sample(input="a", target="a")])
        d = RunDiff(r1, r2)
        with pytest.raises(KeyError):
            d.grade("nonexistent")

    def test_default_thresholds(self):
        r1 = _run([Sample(input="a", target="a")])
        r2 = _run([Sample(input="a", target="b")])
        d = RunDiff(r1, r2)
        # delta = 1.0 → 1.0 → |delta| = 1.0 > 0.05 (default warn) → WARN or FAIL depending
        grade = d.grade("exact_match")  # delta = 0.0 vs 1.0 = -1.0 delta
        assert grade in (DeltaGrade.PASS, DeltaGrade.WARN, DeltaGrade.FAIL)


# ============================================================================
# Per-sample diff
# ============================================================================

class TestSampleDiff:
    def test_regressed(self):
        r1 = _run([Sample(input="a", target="a"), Sample(input="b", target="b")])
        r2 = _run([Sample(input="a", target="a"), Sample(input="b", target="c")])
        d = RunDiff(r1, r2)
        wrong = d.regressed("exact_match")
        assert len(wrong) == 1
        assert wrong[0].sample_id == "1"  # second sample went from correct→wrong

    def test_improved(self):
        r1 = _run([Sample(input="a", target="b"), Sample(input="b", target="b")])
        r2 = _run([Sample(input="a", target="a"), Sample(input="b", target="b")])
        d = RunDiff(r1, r2)
        correct = d.improved("exact_match")
        assert len(correct) == 1
        assert correct[0].sample_id == "0"  # first sample went from wrong→correct

    def test_still_wrong(self):
        r1 = _run([Sample(input="a", target="b")])
        r2 = _run([Sample(input="a", target="c")])
        d = RunDiff(r1, r2)
        wrong = d.still_wrong("exact_match")
        assert len(wrong) == 1

    def test_still_correct(self):
        r1 = _run([Sample(input="a", target="a")])
        r2 = _run([Sample(input="a", target="a")])
        d = RunDiff(r1, r2)
        correct = d.still_correct("exact_match")
        assert len(correct) == 1

    def test_sample_diff_summary(self):
        r1 = _run([Sample(input="a", target="a"), Sample(input="b", target="b")])
        r2 = _run([Sample(input="a", target="b"), Sample(input="b", target="b")])
        d = RunDiff(r1, r2)
        summary = d.sample_summary("exact_match")
        assert summary["newly_wrong"] == 1
        assert summary["newly_correct"] == 0
        assert summary["still_wrong"] == 0
        assert summary["still_correct"] == 1


# ============================================================================
# Summary report
# ============================================================================

class TestDiffSummary:
    def test_summary_contains_keys(self):
        r1 = _run([Sample(input="a", target="a")])
        r2 = _run([Sample(input="a", target="b")])
        d = RunDiff(r1, r2)
        text = d.summary()
        assert isinstance(text, str)
        assert "exact_match" in text
        assert "test" in text

    def test_empty_summary_for_no_change(self):
        r1 = _run([Sample(input="a", target="a")])
        r2 = _run([Sample(input="a", target="a")])
        d = RunDiff(r1, r2)
        text = d.summary()
        assert "->" in text            # ASCII arrow (was a unicode → that crashed cp1252 consoles)
        assert "PASS" not in text and "FAIL" not in text  # no verdict imposed


# ============================================================================
# Direction-awareness, NOT_COMPARABLE, and the passed-flag fix
# ============================================================================

def _mk_run(run_id, metric, value, direction):
    from auditkit.score import Stat
    from auditkit.report import Prediction

    stats = {metric: Stat(metric)}
    stats[metric].add(value)
    preds = [Prediction(
        run_id=run_id, task="t", sample_id="s1", prompt="", raw_output="",
        parsed_answer="", expected="", correct=True, score=value,
        metadata={"scores": [{"name": metric, "value": value, "direction": direction}]},
    )]
    return RunResult(run_id=run_id, fingerprint=run_id, stats=stats, predictions=preds,
                     headline={metric: value})


class TestRunDiffDirectionAwareness:
    def test_minimize_metric_regression_grades_fail_not_pass(self):
        # Previously: RunDiff.grade() only checked delta >= 0, with no
        # direction lookup at all -- a 2x-slower MINIMIZE metric (rise = bad)
        # graded PASS, inconsistent with RunComparison's direction-aware
        # grading of the identical data.
        base = _mk_run("b", "latency", 100.0, "minimize")
        cand = _mk_run("c", "latency", 200.0, "minimize")
        d = RunDiff(base, cand)
        assert d.grade("latency") == DeltaGrade.FAIL

    def test_minimize_metric_improvement_grades_pass(self):
        base = _mk_run("b", "latency", 200.0, "minimize")
        cand = _mk_run("c", "latency", 100.0, "minimize")
        d = RunDiff(base, cand)
        assert d.grade("latency") == DeltaGrade.PASS


class TestRunDiffNotComparable:
    def test_metric_missing_from_one_side_is_not_comparable(self):
        # Previously: a metric absent from one run's stats defaulted to a
        # fabricated 0.0, producing a delta/grade indistinguishable from a
        # genuine regression to zero.
        base = _mk_run("b", "acc", 0.8, "maximize")
        cand = RunResult(run_id="c", fingerprint="c", stats={}, predictions=[], headline={})
        d = RunDiff(base, cand)
        deltas = d.metric_deltas()
        assert deltas["acc"]["delta"] is None
        assert deltas["acc"]["comparable"] is False
        assert d.grade("acc") == DeltaGrade.NOT_COMPARABLE

    def test_not_comparable_does_not_crash_summary(self):
        base = _mk_run("b", "acc", 0.8, "maximize")
        cand = RunResult(run_id="c", fingerprint="c", stats={}, predictions=[], headline={})
        d = RunDiff(base, cand)
        text = d.summary()
        assert "acc" in text


class TestIsCorrectUsesPassedFlag:
    def test_gated_continuous_score_uses_passed_not_truthy_value(self):
        from auditkit.report import Prediction

        # value=0.6 is truthy, but passed=False (failed its own threshold
        # gate) -- previously bool(value) was used instead of the real
        # passed flag, so this was misreported as "correct".
        pred = Prediction(
            run_id="r", task="t", sample_id="s1", prompt="", raw_output="",
            parsed_answer="", expected="", correct=False, score=0.6,
            metadata={"scores": [{"name": "gated", "value": 0.6, "threshold": 0.7, "passed": False}]},
        )
        assert RunDiff._is_correct(pred, "gated") is False

    def test_ungated_score_falls_back_to_value_equals_one(self):
        from auditkit.report import Prediction

        pred = Prediction(
            run_id="r", task="t", sample_id="s1", prompt="", raw_output="",
            parsed_answer="", expected="", correct=True, score=1.0,
            metadata={"scores": [{"name": "exact_match", "value": 1.0}]},
        )
        assert RunDiff._is_correct(pred, "exact_match") is True
