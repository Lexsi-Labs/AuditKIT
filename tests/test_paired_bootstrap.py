"""paired_bootstrap: the p-value must separate a real effect from noise."""

from __future__ import annotations

import random

from auditkit._bootstrap import paired_bootstrap


def test_large_effect_is_significant():
    rng = random.Random(1)
    pairs = [(rng.random() * 0.2, 0.7 + rng.random() * 0.2) for _ in range(50)]
    out = paired_bootstrap(pairs)
    assert out["delta"] > 0.5 and out["p_value"] < 0.01 and out["significant"] is True


def test_noise_is_not_significant():
    rng = random.Random(2)
    pairs = [(0.5, 0.5 + (rng.random() - 0.5) * 0.1) for _ in range(50)]
    out = paired_bootstrap(pairs)
    assert out["p_value"] > 0.05 and out["significant"] is False


def test_identical_runs_have_p_one():
    assert paired_bootstrap([(0.4, 0.4)] * 10)["p_value"] == 1.0


def test_too_few_pairs_report_an_error():
    assert "error" in paired_bootstrap([(1.0, 0.0)])


def test_fewer_than_one_resample_is_refused():
    import pytest

    pairs = [(0.0, 1.0)] * 5 + [(1.0, 1.0)] * 5
    for bad in (0, -5):
        with pytest.raises(ValueError, match="n_resamples"):
            paired_bootstrap(pairs, n_resamples=bad)
    assert 0.0 <= paired_bootstrap(pairs, n_resamples=1)["p_value"] <= 1.0


# -- confidence interval on delta (#43) ---------------------------------------
def _synthetic(rng, n, true_delta):
    """Paired scores whose per-pair difference is true_delta plus N(0, 0.3) noise."""
    out = []
    for _ in range(n):
        b = rng.gauss(0.5, 0.2)
        out.append((b, b + true_delta + rng.gauss(0.0, 0.3)))
    return out


def test_ci_brackets_delta_and_has_the_requested_level():
    rng = random.Random(3)
    out = paired_bootstrap(_synthetic(rng, 40, 0.2))
    lo, hi = out["ci"]
    assert out["ci_level"] == 0.95
    assert lo <= out["delta"] <= hi
    narrow = paired_bootstrap(_synthetic(random.Random(3), 40, 0.2), confidence=0.8)["ci"]
    assert lo <= narrow[0] <= narrow[1] <= hi  # same resamples, a lower level is nested inside


def test_ci_covers_the_true_delta_at_about_the_stated_rate():
    # 200 seeded experiments, n=30 pairs each. The percentile bootstrap covers slightly
    # under nominal at small n, so the band is [0.88, 0.99] for 95% and [0.72, 0.88] for 80%.
    rng = random.Random(11)
    true_delta = 0.15
    hits95 = hits80 = 0
    trials = 200
    for t in range(trials):
        pairs = _synthetic(rng, 30, true_delta)
        lo, hi = paired_bootstrap(pairs, n_resamples=400, seed=t)["ci"]
        hits95 += lo <= true_delta <= hi
        lo, hi = paired_bootstrap(pairs, n_resamples=400, seed=t, confidence=0.8)["ci"]
        hits80 += lo <= true_delta <= hi
    assert 0.88 <= hits95 / trials <= 0.99, hits95
    assert 0.72 <= hits80 / trials <= 0.88, hits80


def test_ci_excludes_zero_for_a_large_effect_and_brackets_zero_for_identical_runs():
    rng = random.Random(1)
    big = paired_bootstrap([(rng.random() * 0.2, 0.7 + rng.random() * 0.2) for _ in range(50)])
    assert big["ci"][0] > 0
    assert paired_bootstrap([(0.4, 0.4)] * 10)["ci"] == [0.0, 0.0]


def test_confidence_outside_zero_one_is_refused():
    import pytest

    for bad in (0.0, 1.0, 95, -0.5):
        with pytest.raises(ValueError, match="confidence"):
            paired_bootstrap([(0.0, 1.0)] * 5, confidence=bad)


def _run(run_id, values, metric="acc"):
    from auditkit.report import Prediction, RunResult

    preds = [Prediction(run_id=run_id, task="t", sample_id=str(i), prompt="", raw_output="",
                        parsed_answer="", expected="", correct=None, score=v,
                        metadata={"scores": [{"name": metric, "value": v}]})
             for i, v in enumerate(values)]
    return RunResult(run_id=run_id, fingerprint=run_id, stats={}, predictions=preds,
                     headline={metric: sum(values) / len(values)})


def test_callers_report_the_ci_on_their_own_delta_direction():
    from auditkit.comparison import RunComparison
    from auditkit.experiment import Experiment
    from auditkit.model_compare import CompareResult

    rng = random.Random(5)
    low = [rng.random() * 0.3 for _ in range(30)]
    high = [0.6 + rng.random() * 0.3 for _ in range(30)]
    a, b = _run("a", high), _run("b", low)

    # RunComparison: candidate - baseline, as paired_bootstrap returns it
    rc = RunComparison(b, a).significance("acc")
    assert rc["delta"] > 0 and 0 < rc["ci"][0] <= rc["delta"] <= rc["ci"][1]

    # CompareResult: delta = mean_a - mean_b; ci on that same difference
    cr = CompareResult(runs={"a": a, "b": b}, dataset_size=30, scorers=["acc"])
    for x, y in (("a", "b"), ("b", "a")):
        sig = cr.significance(x, y, "acc")
        lo, hi = sig["ci"]
        assert lo <= sig["delta"] <= hi
        assert (lo > 0) == (sig["delta"] > 0) and (hi < 0) == (sig["delta"] < 0)
    assert cr.significance("a", "b", "acc")["ci"] == [0.0 - v for v in reversed(cr.significance("b", "a", "acc")["ci"])]

    # Experiment: best run minus runner-up, positive
    exp = Experiment(name="ci")
    exp.add(b)
    exp.add(a)
    sig = exp.significance("acc", confidence=0.9)
    assert sig["baseline"] == "a" and sig["delta"] > 0
    assert 0 < sig["ci"][0] <= sig["delta"] <= sig["ci"][1] and sig["ci_level"] == 0.9
