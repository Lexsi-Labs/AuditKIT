"""RunComparison: base-vs-candidate (pruned/quantized) comparison.

MAXIMIZE-path tests run real evaluate() with callable models (offline). The
MINIMIZE/direction path uses hand-built RunResults so we can stamp a
`minimize` direction without depending on scorer-direction plumbing.
"""

from __future__ import annotations

import auditkit as ak
from auditkit.comparison import RunComparison, grade_delta
from auditkit.diff import DeltaGrade
from auditkit.metric import ExactMatch
from auditkit.report import Prediction, RunResult
from auditkit.runspec import RunConfig
from auditkit.sample import Sample
from auditkit.score import Stat
from auditkit.types import Direction
from auditkit._bootstrap import score_pairs


# --- shared fixtures ------------------------------------------------------

_ANSWERS = {
    "2+2": "4", "3+3": "6", "4+4": "8",
    "capital of France": "Paris", "capital of Italy": "Rome", "capital of Germany": "Berlin",
}


def _dataset():
    arith = [Sample(input=q, target=a, task="arith") for q, a in list(_ANSWERS.items())[:3]]
    trivia = [Sample(input=q, target=a, task="trivia") for q, a in list(_ANSWERS.items())[3:]]
    return arith + trivia


def _baseline_model(prompts):
    return [_ANSWERS[p] for p in prompts]              # all correct


def _candidate_model(prompts):
    # arith stays correct; trivia all wrong -> a per-task regression on trivia only
    return [_ANSWERS[p] if p in ("2+2", "3+3", "4+4") else "WRONG" for p in prompts]


def _runs(track_performance=None):
    # track_performance=None means "don't care, use the library default"
    # (config=None) -- True/False are threaded through explicitly so a
    # caller can force either state regardless of what that default is.
    data = _dataset()
    config = RunConfig(track_performance=track_performance) if track_performance is not None else None
    base = ak.evaluate(data, model=_baseline_model, scorers=["exact_match"], config=config)
    cand = ak.evaluate(data, model=_candidate_model, scorers=["exact_match"], config=config)
    return base, cand


# --- per-task deltas + grading (MAXIMIZE, real runs) ----------------------

def test_per_task_deltas_isolate_the_regressed_task():
    base, cand = _runs()
    cmp = RunComparison(base, cand)
    by_task = {(td.task, td.metric): td for td in cmp.per_task_deltas()}
    arith = by_task[("arith", "exact_match")]
    trivia = by_task[("trivia", "exact_match")]
    assert arith.baseline == 1.0 and arith.candidate == 1.0
    assert arith.grade == DeltaGrade.PASS           # no change
    assert trivia.baseline == 1.0 and trivia.candidate == 0.0
    assert trivia.grade == DeltaGrade.FAIL          # -100pp
    # the blended overall grade is the worst per-task grade
    assert cmp.grade() == DeltaGrade.FAIL


def test_retention_is_candidate_over_baseline_for_maximize():
    base, cand = _runs()
    cmp = RunComparison(base, cand)
    # blended exact_match: baseline 6/6=1.0, candidate 3/6=0.5 -> retention 0.5
    assert abs(cmp.retention("exact_match") - 0.5) < 1e-9


def test_retention_is_nan_not_misleading_for_negative_valued_metrics():
    import math
    # baseline=-2.0 (worse), candidate=-1.0 (objectively better, closer to
    # zero) -- a plain ratio (-1.0/-2.0 = 0.5) would misleadingly read as
    # "lost half the quality" for what is actually an improvement. Retention
    # is only meaningful as a ratio for non-negative-range metrics.
    base = _mk_run("b", [("s1", "t", "score", -2.0, "maximize")])
    cand = _mk_run("c", [("s1", "t", "score", -1.0, "maximize")])
    cmp = RunComparison(base, cand)
    assert math.isnan(cmp.retention("score"))


def test_regressed_aligned_by_sample_id():
    base, cand = _runs()
    cmp = RunComparison(base, cand)
    nw = cmp.regressed()
    assert len(nw) == 3                              # exactly the 3 trivia samples
    assert {p.expected for p in nw} == {"Paris", "Rome", "Berlin"}


def test_tradeoff_reports_quality_and_size():
    # Both sides local -- size_ratio/quality_per_mb come from each run's own
    # introspected model_size (RunResult.model_size), never a caller kwarg.
    base, cand = _runs()
    base.model_size = {"is_local": True, "model_name": "base", "size_mb": 1000.0}
    cand.model_size = {"is_local": True, "model_name": "cand", "size_mb": 250.0}
    cmp = RunComparison(base, cand)
    t = cmp.tradeoff(metric="exact_match")
    assert abs(t["retention"] - 0.5) < 1e-9
    assert abs(t["size_ratio"] - 0.25) < 1e-9        # candidate 1/4 the size
    # quality_per_mb is retention-based (direction-aware), not the raw score:
    # retention=0.5 over candidate_size_mb=250 -> 0.002.
    assert abs(t["quality_per_mb"] - (0.5 / 250.0)) < 1e-9


def test_tradeoff_marks_api_based_models_instead_of_a_fake_size_ratio():
    # _runs() uses plain callables -- Runner still records model_size for
    # them (is_local=False, an identity name), so no size ratio is possible
    # but both models are named rather than the field being silently absent.
    # (Needs track_performance=True: model_size is None otherwise.)
    base, cand = _runs(track_performance=True)
    cmp = RunComparison(base, cand)
    t = cmp.tradeoff(metric="exact_match")
    assert "size_ratio" not in t
    assert "quality_per_mb" not in t
    assert t["api_based"]["baseline"] == base.model_size["model_name"]
    assert t["api_based"]["candidate"] == cand.model_size["model_name"]

    # One side API-based, the other local -- still no size ratio, but the
    # API-based side is explicitly named instead of silently omitted.
    cand.model_size = {"is_local": True, "model_name": "local-cand", "size_mb": 100.0}
    base.model_size = {"is_local": False, "model_name": "gpt-4o-mini"}
    cmp2 = RunComparison(base, cand)
    t2 = cmp2.tradeoff(metric="exact_match")
    assert "size_ratio" not in t2
    assert t2["api_based"]["baseline"] == "gpt-4o-mini"
    assert t2["api_based"]["candidate"] == "local-cand"


def test_tradeoff_reports_something_when_both_sides_are_local_but_uninstrospected():
    # Previously: two "is_local": True runs with no size_mb (e.g. two
    # VLLMModel instances, which are marked local but don't actually
    # introspect size yet) fell through both branches silently -- no
    # size_ratio, no api_based marker, nothing reported at all.
    base, cand = _runs()
    base.model_size = {"is_local": True, "model_name": "vllm-base"}
    cand.model_size = {"is_local": True, "model_name": "vllm-cand"}
    cmp = RunComparison(base, cand)
    t = cmp.tradeoff(metric="exact_match")
    assert "size_ratio" not in t
    assert t["api_based"]["baseline"] == "vllm-base"
    assert t["api_based"]["candidate"] == "vllm-cand"
    assert "vllm-base" in cmp.summary()
    assert "0.0 MB" not in cmp.summary()  # no fake placeholder size


def test_tradeoff_latency_always_comes_from_measured_perf():
    import time as time_mod

    def fast_model(prompts):
        time_mod.sleep(0.005)
        return [_ANSWERS[p] for p in prompts]

    def slow_model(prompts):
        time_mod.sleep(0.03)
        return [_ANSWERS[p] for p in prompts]

    data = _dataset()
    perf_config = RunConfig(track_performance=True)
    base = ak.evaluate(data, model=fast_model, scorers=["exact_match"], config=perf_config)
    cand = ak.evaluate(data, model=slow_model, scorers=["exact_match"], config=perf_config)
    cmp = RunComparison(base, cand)
    # tradeoff() takes no latency kwargs at all -- always the run's own measured mean.
    t = cmp.tradeoff(metric="exact_match")
    assert t["baseline_latency_ms"] == base.perf["latency_ms"]["mean"]
    assert t["candidate_latency_ms"] == cand.perf["latency_ms"]["mean"]
    assert t["speedup"] < 1.0  # candidate is slower


def test_performance_reports_measured_latency_and_throughput():
    base, cand = _runs(track_performance=True)
    cmp = RunComparison(base, cand)
    perf = cmp.performance()
    assert perf["baseline_latency_ms"]["count"] == 1
    assert perf["candidate_latency_ms"]["count"] == 1
    assert perf["baseline_throughput_rps"] >= 0.0
    assert "speedup" in perf
    assert "performance:" in cmp.summary()


def test_performance_populates_by_default():
    # Neither side opted out -- both are populated under the current
    # default, and nothing crashes.
    base, cand = _runs()
    cmp = RunComparison(base, cand)
    perf = cmp.performance()
    assert perf["baseline_latency_ms"] is not None
    assert perf["candidate_latency_ms"] is not None
    assert perf["baseline_throughput_rps"] is not None
    assert perf["candidate_throughput_rps"] is not None
    assert "speedup" in perf
    assert "performance:" in cmp.summary()
    # tradeoff()/model_size-based comparisons must not crash either.
    cmp.tradeoff(metric="exact_match")


def test_performance_is_none_when_both_sides_opt_out_and_none_for_a_side_that_never_tracked_it():
    # Both sides explicitly opt out -- both are None, not {}/0.0, and nothing crashes.
    base, cand = _runs(track_performance=False)
    cmp = RunComparison(base, cand)
    perf = cmp.performance()
    assert perf["baseline_latency_ms"] is None
    assert perf["candidate_latency_ms"] is None
    assert perf["baseline_throughput_rps"] is None
    assert perf["candidate_throughput_rps"] is None
    assert "speedup" not in perf
    assert "performance:" not in cmp.summary()  # no crash rendering the summary either
    # tradeoff()/model_size-based comparisons must not crash either.
    cmp.tradeoff(metric="exact_match")


def test_performance_reports_real_values_on_the_tracked_side_and_none_on_the_other():
    # The exact scenario the toggle exists for: one run tracked performance,
    # the other never did. The tracked side keeps its real numbers; the
    # untracked side is explicitly None, not a fake zero; speedup (needs
    # both sides) is omitted rather than computed against a missing value.
    data = _dataset()
    tracked = ak.evaluate(data, model=_baseline_model, scorers=["exact_match"],
                          config=RunConfig(track_performance=True))
    untracked = ak.evaluate(data, model=_candidate_model, scorers=["exact_match"],
                            config=RunConfig(track_performance=False))
    cmp = RunComparison(tracked, untracked)
    perf = cmp.performance()
    assert perf["baseline_latency_ms"] is not None
    assert perf["baseline_latency_ms"]["count"] == 1
    assert perf["baseline_throughput_rps"] is not None
    assert perf["candidate_latency_ms"] is None
    assert perf["candidate_throughput_rps"] is None
    assert "speedup" not in perf
    # Doesn't crash building the summary with one side missing perf.
    cmp.summary()

    # And the reverse direction -- untracked as baseline this time.
    cmp2 = RunComparison(untracked, tracked)
    perf2 = cmp2.performance()
    assert perf2["baseline_latency_ms"] is None
    assert perf2["candidate_latency_ms"] is not None
    assert "speedup" not in perf2
    cmp2.summary()


def test_tradeoff_quality_per_mb_is_direction_aware_for_minimize_metrics():
    # A MINIMIZE metric (latency-style): candidate's raw value DOUBLED
    # (worse), so retention is 0.5, not the raw candidate/size ratio the old
    # (buggy) implementation would have computed -- which would nonsensically
    # treat a bigger "badness" number as more "quality per mb".
    base = _mk_run("b", [("s1", "t", "latency", 0.10, "minimize"), ("s2", "t", "latency", 0.10, "minimize")],
                   model_size={"is_local": True, "model_name": "b", "size_mb": 100.0})
    cand = _mk_run("c", [("s1", "t", "latency", 0.20, "minimize"), ("s2", "t", "latency", 0.20, "minimize")],
                   model_size={"is_local": True, "model_name": "c", "size_mb": 100.0})
    cmp = RunComparison(base, cand)
    t = cmp.tradeoff(metric="latency")
    assert abs(t["retention"] - 0.5) < 1e-9
    # Correct: retention (0.5) / size, NOT raw candidate value (0.20) / size.
    assert abs(t["quality_per_mb"] - (0.5 / 100.0)) < 1e-9
    assert t["quality_per_mb"] != 0.20 / 100.0


# --- direction-aware grading (MINIMIZE, hand-built runs) ------------------

def _mk_run(run_id, rows, model_size=None):
    """rows: list of (sample_id, task, metric, value, direction)."""
    stats: dict[str, Stat] = {}
    by_sample: dict[tuple[str, str], list[tuple]] = {}
    for sid, task, metric, value, direction in rows:
        stats.setdefault(metric, Stat(metric)).add(value)
        by_sample.setdefault((sid, task), []).append((metric, value, direction))
    preds = []
    for (sid, task), scores in by_sample.items():
        preds.append(Prediction(
            run_id=run_id, task=task, sample_id=sid, prompt="", raw_output="",
            parsed_answer="", expected="", correct=(scores[0][1] == 1.0), score=scores[0][1],
            metadata={"scores": [{"name": m, "value": v, "direction": d} for m, v, d in scores]},
        ))
    return RunResult(run_id=run_id, fingerprint=run_id, stats=stats, predictions=preds,
                     headline={m: s.mean for m, s in stats.items()},
                     model_size=model_size or {})


def test_minimize_metric_rise_grades_fail():
    base = _mk_run("b", [("s1", "t", "latency", 0.10, "minimize"), ("s2", "t", "latency", 0.10, "minimize")])
    cand = _mk_run("c", [("s1", "t", "latency", 0.20, "minimize"), ("s2", "t", "latency", 0.20, "minimize")])
    cmp = RunComparison(base, cand)
    md = {m.metric: m for m in cmp.metric_deltas()}["latency"]
    assert md.delta == 0.10 and md.grade == DeltaGrade.FAIL   # a RISE in a lower-is-better metric is a regression
    assert abs(cmp.retention("latency") - 0.5) < 1e-9         # base/cand for MINIMIZE


# --- failure visibility -----------------------------------------------------

def test_no_failures_by_default():
    base, cand = _runs()
    cmp = RunComparison(base, cand)
    assert cmp.has_failures is False
    assert cmp.failure_summary() == {
        "baseline_failed": 0, "candidate_failed": 0,
        "baseline_errors": [], "candidate_errors": [],
    }
    assert "WARNING" not in cmp.summary()


def test_failure_summary_surfaces_failed_samples():
    base, cand = _runs()
    cand.failed_count = 2
    cand.errors = [{"sample_id": "x", "error": "rate limited"}]
    cmp = RunComparison(base, cand)
    assert cmp.has_failures is True
    fs = cmp.failure_summary()
    assert fs["baseline_failed"] == 0
    assert fs["candidate_failed"] == 2
    assert fs["candidate_errors"] == [{"sample_id": "x", "error": "rate limited"}]
    assert "WARNING: baseline had 0 failed sample(s), candidate had 2" in cmp.summary()


def test_grade_and_retention_log_a_warning_when_there_are_failures(caplog):
    # Calling grade()/retention()/tradeoff() DIRECTLY (never touching
    # .summary()/.coverage_warnings()) still surfaces the issue via logging --
    # previously only .summary() showed anything, so a caller that just reads
    # cmp.grade() in a script/CI gate got no signal at all.
    base, cand = _runs()
    cand.failed_count = 3
    cmp = RunComparison(base, cand)
    import logging
    with caplog.at_level(logging.WARNING):
        cmp.grade()
    assert any("failed sample" in r.message for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        cmp.retention("exact_match")
    assert any("failed sample" in r.message for r in caplog.records)


def test_no_warning_logged_when_nothing_is_wrong(caplog):
    base, cand = _runs()
    cmp = RunComparison(base, cand)
    import logging
    with caplog.at_level(logging.WARNING):
        cmp.grade()
        cmp.retention()
    assert caplog.records == []


# --- count/std visibility + coverage warnings -------------------------------

def test_metric_deltas_report_count_and_std():
    base, cand = _runs()
    cmp = RunComparison(base, cand)
    md = {m.metric: m for m in cmp.metric_deltas()}["exact_match"]
    assert md.baseline_count == 6 and md.candidate_count == 6   # all 6 samples scored on both sides
    assert md.baseline_std == 0.0                                 # baseline: all correct, zero variance
    assert md.candidate_std > 0.0                                 # candidate: mixed correct/wrong


def test_per_task_deltas_report_count_and_std():
    base, cand = _runs()
    cmp = RunComparison(base, cand)
    by_task = {(td.task, td.metric): td for td in cmp.per_task_deltas()}
    assert by_task[("arith", "exact_match")].baseline_count == 3
    assert by_task[("trivia", "exact_match")].candidate_count == 3


def test_no_coverage_warning_when_counts_match():
    base, cand = _runs()
    cmp = RunComparison(base, cand)
    assert cmp.coverage_warnings() == []
    assert "WARNING" not in cmp.summary()


def test_coverage_warning_on_sample_count_mismatch():
    # Same metric, same direction, but candidate only has 1 real data point
    # vs baseline's 2 -- simulates a metric that failed on some samples
    # without ever showing up in has_failures/failure_summary (a different
    # run-level signal; this is metric-level, per RunComparison.summary()'s
    # own "not a like-for-like comparison" warning). Both the blended AND the
    # per-task check fire here since there's only the one task "t" involved.
    base = _mk_run("b", [("s1", "t", "m", 1.0, "maximize"), ("s2", "t", "m", 1.0, "maximize")])
    cand = _mk_run("c", [("s1", "t", "m", 1.0, "maximize")])
    cmp = RunComparison(base, cand)
    warnings = cmp.coverage_warnings()
    assert len(warnings) == 2
    assert any("baseline n=2, candidate n=1" in w for w in warnings if w.startswith("m:"))
    assert any("t/m: baseline n=2, candidate n=1" in w for w in warnings)
    assert "sample coverage differs per metric" in cmp.summary()


def test_coverage_warning_per_task_only_when_blended_totals_coincide():
    # Two tasks, each individually mismatched (2 vs 1 samples), but the
    # BLENDED totals coincidentally agree (4 vs 4 combined) -- the blended
    # check alone would miss this entirely; the per-task check must catch it.
    base = _mk_run("b", [
        ("s1", "t1", "m", 1.0, "maximize"), ("s2", "t1", "m", 1.0, "maximize"),
        ("s3", "t2", "m", 1.0, "maximize"), ("s4", "t2", "m", 1.0, "maximize"),
    ])
    cand = _mk_run("c", [
        ("s1", "t1", "m", 1.0, "maximize"),
        ("s3", "t2", "m", 1.0, "maximize"), ("s4", "t2", "m", 1.0, "maximize"), ("s5", "t2", "m", 1.0, "maximize"),
    ])
    cmp = RunComparison(base, cand)
    blended = {md.metric: md for md in cmp.metric_deltas()}["m"]
    assert blended.baseline_count == blended.candidate_count == 4  # totals coincide
    warnings = cmp.coverage_warnings()
    assert any("t1/m" in w for w in warnings)  # per-task catches it anyway


class _FlakyOnSomeSamples(ExactMatch):
    """Real metric that crashes on specific inputs -- not a hand-built
    RunResult, so this exercises runner.py's "skip, don't fake a 0.0" fix
    (test_runner.py) all the way through into RunComparison's own coverage
    visibility, proving the two fixes actually compose correctly together."""

    name = "flaky_exact_match"

    def score(self, sample, output, context=None):
        if sample.input in ("4+4", "capital of Germany"):
            raise RuntimeError("simulated real metric crash")
        return super().score(sample, output, context)


def test_real_metric_crash_produces_a_coverage_warning_not_a_fake_zero():
    data = _dataset()  # 3 arith + 3 trivia, from _ANSWERS
    flaky = _FlakyOnSomeSamples()
    base = ak.evaluate(data, model=_baseline_model, scorers=[flaky])
    cand = ak.evaluate(data, model=_candidate_model, scorers=[flaky])

    # Both runs hit the same 2 crashing samples -- confirm the crashing
    # metric really did lose exactly those 2 data points, not silently
    # scored 0.0 for them (runner.py's fix).
    assert base.stats["flaky_exact_match"].count == 4    # 6 samples - 2 crashes
    assert cand.stats["flaky_exact_match"].count == 4

    cmp = RunComparison(base, cand)
    # Same crash on both sides -> counts match -> no coverage warning here;
    # this is the "both runs affected equally" case, distinct from the
    # hand-built mismatched-N test above.
    assert cmp.coverage_warnings() == []
    md = {m.metric: m for m in cmp.metric_deltas()}["flaky_exact_match"]
    assert md.baseline_count == 4 and md.candidate_count == 4


def test_summary_shows_per_task_even_when_row_counts_coincide_with_metric_count():
    # 2 metrics, each scored on a DIFFERENT, non-overlapping single task --
    # per_task_deltas() and metric_deltas() both have exactly 2 rows, so the
    # OLD heuristic (len(tasks) > len(metric_deltas())) would hide the
    # per-task section even though there are genuinely 2 distinct tasks with
    # real information the blended view can't show.
    base = _mk_run("b", [("s1", "t1", "m1", 1.0, "maximize"), ("s2", "t2", "m2", 1.0, "maximize")])
    cand = _mk_run("c", [("s1", "t1", "m1", 0.0, "maximize"), ("s2", "t2", "m2", 1.0, "maximize")])
    cmp = RunComparison(base, cand)
    assert len(cmp.per_task_deltas()) == len(cmp.metric_deltas()) == 2  # counts coincide
    assert "per-task:" in cmp.summary()  # but the section still shows


def test_summary_reports_percentage_diff_only():
    # The default summary shows just the percentage difference per metric --
    # no pass/warn/fail verdict AND no better/worse framing. The reader
    # interprets the numbers for their own metrics.
    base, cand = _runs()
    text = RunComparison(base, cand).summary()
    assert "% rel" in text                 # relative percentage difference shown
    assert "delta=" in text                # absolute delta shown
    # no imposed verdict
    for token in ("[PASS]", "[WARN]", "[FAIL]", "overall grade", "grade:"):
        assert token not in text
    # no better/worse framing either
    for token in ("higher is better", "lower is better", "improved", "regressed",
                  "direction unspecified"):
        assert token not in text


def test_pct_change_populated_on_metric_and_task_deltas():
    base, cand = _runs()
    cmp = RunComparison(base, cand)
    md = {m.metric: m for m in cmp.metric_deltas()}["exact_match"]
    # overall exact_match went 1.0 -> 0.5 (arith stays, trivia breaks) = -50% rel
    assert md.delta == -0.5
    assert md.pct_change == -50.0
    trivia = {(t.task, t.metric): t for t in cmp.per_task_deltas()}[("trivia", "exact_match")]
    assert trivia.delta == -1.0 and trivia.pct_change == -100.0


def test_pct_change_none_when_baseline_zero():
    # baseline scored 0.0 -> percent-of-zero is undefined, reported as None
    from auditkit.diff import relative_pct
    assert relative_pct(0.0, 0.5) is None
    assert relative_pct(None, 0.5) is None
    assert relative_pct(0.5, 0.1) == 20.0


def test_grade_is_still_available_opt_in():
    # grading isn't gone -- it's just no longer the default display. Callers who
    # want a gate can still ask for one (with their own thresholds).
    base, cand = _runs()
    cmp = RunComparison(base, cand)
    assert cmp.grade() == DeltaGrade.FAIL
    # and the caller picks the thresholds that define pass/fail FOR THEM
    lenient = RunComparison(base, cand, pass_threshold=0.6, warn_threshold=0.9)
    assert lenient.grades()["exact_match"] == DeltaGrade.PASS


def test_grade_delta_boundaries():
    # MAXIMIZE
    assert grade_delta(0.10, Direction.MAXIMIZE) == DeltaGrade.PASS   # improvement
    assert grade_delta(-0.01, Direction.MAXIMIZE) == DeltaGrade.PASS  # within 2pp
    assert grade_delta(-0.03, Direction.MAXIMIZE) == DeltaGrade.WARN  # 2-5pp
    assert grade_delta(-0.10, Direction.MAXIMIZE) == DeltaGrade.FAIL  # >5pp
    # MINIMIZE (sign flips)
    assert grade_delta(-0.10, Direction.MINIMIZE) == DeltaGrade.PASS  # dropped -> better
    assert grade_delta(0.03, Direction.MINIMIZE) == DeltaGrade.WARN
    assert grade_delta(0.10, Direction.MINIMIZE) == DeltaGrade.FAIL


# --- bootstrap / alignment ------------------------------------------------

def test_score_pairs_aligns_by_sample_id_not_position():
    base = _mk_run("b", [("s1", "t", "m", 1.0, "maximize"), ("s2", "t", "m", 0.0, "maximize")])
    # candidate predictions deliberately in the OPPOSITE order
    cand = _mk_run("c", [("s2", "t", "m", 1.0, "maximize"), ("s1", "t", "m", 1.0, "maximize")])
    pairs = score_pairs(base, cand, "m")
    assert pairs == [(1.0, 1.0), (0.0, 1.0)]          # paired by id: s1->(1,1), s2->(0,1)


def test_significance_identical_runs_not_significant():
    base, _ = _runs()
    same = ak.evaluate(_dataset(), model=_baseline_model, scorers=["exact_match"])
    sig = RunComparison(base, same).significance("exact_match")
    assert sig["significant"] is False               # no difference -> not significant


# --- polymorphic ak.compare + compare_models.pairwise ---------------------

def test_ak_compare_polymorphic():
    base, cand = _runs()
    assert isinstance(ak.compare(base, cand), RunComparison)   # two runs -> comparison
    lb = ak.compare([base, cand])                              # list -> leaderboard (unchanged)
    assert isinstance(lb, list) and all("run_id" in row for row in lb)


def test_compare_models_pairwise_and_per_model_config():
    from auditkit.model_compare import compare_models
    data = _dataset()
    result = compare_models(
        [_baseline_model, _candidate_model], data, scorers=["exact_match"],
        model_names=["base", "pruned"],
        configs={"base": RunConfig(limit=6), "pruned": RunConfig(limit=2)},  # per-model override
    )
    assert len(result.runs["base"].predictions) == 6
    assert len(result.runs["pruned"].predictions) == 2                        # limit override took effect
    cmp = result.pairwise("base", "pruned")
    assert isinstance(cmp, RunComparison)
