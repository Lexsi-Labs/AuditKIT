"""RunConfig.failed_as_zero: failed samples count as 0.0 instead of silently shrinking n."""

from __future__ import annotations

import pytest

import auditkit as ak
from auditkit.metric import Metric
from auditkit.model import Generated, Model, Result_
from auditkit.score import Direction, Score


class Flaky(Model):
    """Answers "ok" every time, except the requests whose prompt number is in *fail* (an HTTP 500)."""

    name = "flaky"
    per_request = True

    def __init__(self, fail: set[int]) -> None:
        self.fail = fail

    def generate(self, requests):
        return [
            Result_([Generated(text="", finish_reason="error", error="HTTP 500")])
            if int(r.prompt.removeprefix("q")) in self.fail
            else Result_([Generated(text="ok")])
            for r in requests
        ]


class CrashOn(Metric):
    """1.0 for every sample, but raises on the sample ids in *bad*."""

    name = "crashy"
    direction = Direction.MAXIMIZE

    def __init__(self, bad: set[str]) -> None:
        self.bad = bad

    def identity(self) -> dict:
        return {"name": self.name, "bad": sorted(self.bad)}

    def score(self, sample, output, context=None):
        if sample.id in self.bad:
            raise ValueError("boom")
        return Score(name=self.name, value=1.0, direction=self.direction)


class Judge(Metric):
    """1.0, except an "unknown" verdict on samples 0 and 1 (kept at 0.5 when *keep* opts in)."""

    name = "judge"
    direction = Direction.MAXIMIZE

    def __init__(self, keep: bool) -> None:
        self.keep = keep

    def identity(self) -> dict:
        return {"name": self.name, "keep": self.keep}

    def score(self, sample, output, context=None):
        if sample.id not in ("0", "1"):
            return Score(name=self.name, value=1.0, direction=self.direction)
        meta = {"unknown": True, **({"count_in_aggregate": True} if self.keep else {})}
        return Score(name=self.name, value=0.5 if self.keep else 0.0, direction=self.direction, metadata=meta)


class Multi(Metric):
    """Two scores, ``multi_a`` and ``multi_b``, both 1.0; raises on the sample ids in *bad*. Its own name is not a score name."""

    name = "multi"
    direction = Direction.MAXIMIZE

    def __init__(self, bad: set[str]) -> None:
        self.bad = bad

    def identity(self) -> dict:
        return {"name": self.name, "bad": sorted(self.bad)}

    def score(self, sample, output, context=None):
        if sample.id in self.bad:
            raise ValueError("boom")
        return [Score(name="multi_a", value=1.0, direction=self.direction),
                Score(name="multi_b", value=1.0, direction=self.direction)]


class Rate(Metric):
    """A lower-is-better metric: always 0.2."""

    name = "rate"
    direction = Direction.MINIMIZE

    def score(self, sample, output, context=None):
        return Score(name=self.name, value=0.2, direction=self.direction)


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))


def samples(n: int = 10) -> list[ak.Sample]:
    return [ak.Sample(id=str(i), input=f"q{i}", target="ok") for i in range(n)]


def run(model, metrics, **cfg):
    return ak.evaluate(samples(), model=model, scorers=metrics, config=ak.RunConfig(**cfg))


def test_default_leaves_failed_samples_out_of_the_mean():
    r = run(Flaky({0, 1, 5, 6}), [ak.Contains("ok")])
    assert r.headline["contains(ok)"] == 1.0 and r.stats["contains(ok)"].count == 6
    assert r.failed_count == 4


def test_failed_as_zero_counts_model_failures_as_wrong():
    r = run(Flaky({0, 1, 5, 6}), [ak.Contains("ok")], failed_as_zero=True)
    assert r.stats["contains(ok)"].count == 10
    assert r.headline["contains(ok)"] == pytest.approx(0.6)
    assert r.failed_count == 4


def test_failed_as_zero_counts_a_crashing_metric_as_zero():
    r = ak.evaluate(samples(), model=lambda ps: ["ok"] * len(ps), scorers=[CrashOn({"2", "3"})],
                    config=ak.RunConfig(failed_as_zero=True))
    assert r.stats["crashy"].count == 10 and r.headline["crashy"] == pytest.approx(0.8)


@pytest.mark.parametrize("keep, mean", [(False, 0.8), (True, 0.9)])
def test_failed_as_zero_zeroes_unknown_results_but_keeps_opted_in_scores(keep, mean):
    r = ak.evaluate(samples(), model=lambda ps: ["ok"] * len(ps), scorers=[Judge(keep)],
                    config=ak.RunConfig(failed_as_zero=True))
    assert r.stats["judge"].count == 10 and r.headline["judge"] == pytest.approx(mean)


def test_failed_as_zero_leaves_lower_is_better_metrics_alone():
    r = run(Flaky({0, 1}), [ak.Contains("ok"), Rate()], failed_as_zero=True)
    assert r.stats["contains(ok)"].count == 10
    assert r.stats["rate"].count == 8 and r.headline["rate"] == pytest.approx(0.2)


def test_failed_as_zero_changes_nothing_when_no_sample_fails():
    r = run(Flaky(set()), [ak.Contains("ok")], failed_as_zero=True)
    assert r.headline["contains(ok)"] == 1.0 and r.stats["contains(ok)"].count == 10


def test_summary_reports_failed_samples_either_way():
    left_out = run(Flaky({0, 1}), [ak.Contains("ok")]).summary()
    assert "2 failed sample(s)" in left_out and "left out of the means" in left_out
    as_zero = run(Flaky({0, 1}), [ak.Contains("ok")], failed_as_zero=True).summary()
    assert "2 failed sample(s)" in as_zero and "counted as 0.0" in as_zero


def test_summary_is_unchanged_when_nothing_failed():
    assert "failed" not in run(Flaky(set()), [ak.Contains("ok")]).summary()


def test_a_crashing_multi_score_metric_is_zero_filled_by_its_score_names():
    r = ak.evaluate(samples(), model=lambda ps: ["ok"] * len(ps), scorers=[Multi({"2", "3"})],
                    config=ak.RunConfig(failed_as_zero=True))
    assert r.headline["multi_a"] == pytest.approx(0.8) and r.stats["multi_a"].count == 10
    assert r.headline["multi_b"] == pytest.approx(0.8) and r.stats["multi_b"].count == 10
    assert "multi" not in r.headline                 # its own name is not a score


def test_a_failed_model_call_does_not_add_a_row_named_after_a_multi_score_metric():
    r = run(Flaky({0, 1}), [Multi(set())], failed_as_zero=True)
    assert "multi" not in r.headline and "multi:" not in r.summary()
    assert r.stats["multi_a"].count == 10 and r.headline["multi_a"] == pytest.approx(0.8)


def test_a_metric_that_never_produced_a_score_is_still_zero_filled_by_its_own_name():
    r = ak.evaluate(samples(2), model=lambda ps: ["ok"] * len(ps), scorers=[CrashOn({"0", "1"})],
                    config=ak.RunConfig(failed_as_zero=True))
    assert r.headline["crashy"] == 0.0 and r.stats["crashy"].count == 2


class Mixed(Metric):
    """``mixed_main`` (higher is better) and ``mixed_rate`` (lower is better) on every sample; raises on the ids in *bad*."""

    name = "mixed"
    direction = Direction.MAXIMIZE

    def __init__(self, bad: set[str]) -> None:
        self.bad = bad

    def identity(self) -> dict:
        return {"name": self.name, "bad": sorted(self.bad)}

    def score(self, sample, output, context=None):
        if sample.id in self.bad:
            raise ValueError("boom")
        return [Score(name="mixed_main", value=1.0, direction=Direction.MAXIMIZE),
                Score(name="mixed_rate", value=1.0, direction=Direction.MINIMIZE)]


class Sometimes(Metric):
    """``sometimes_all`` on every sample, ``sometimes_even`` only on even ids; raises on the ids in *bad*."""

    name = "sometimes"
    direction = Direction.MAXIMIZE

    def __init__(self, bad: set[str]) -> None:
        self.bad = bad

    def identity(self) -> dict:
        return {"name": self.name, "bad": sorted(self.bad)}

    def score(self, sample, output, context=None):
        if sample.id in self.bad:
            raise ValueError("boom")
        out = [Score(name="sometimes_all", value=1.0)]
        if int(sample.id) % 2 == 0:
            out.append(Score(name="sometimes_even", value=1.0))
        return out


def test_a_lower_is_better_score_of_a_crashing_metric_is_not_filled_with_the_best_value():
    r = ak.evaluate(samples(), model=lambda ps: ["ok"] * len(ps), scorers=[Mixed({"2", "3"})],
                    config=ak.RunConfig(failed_as_zero=True))
    assert r.stats["mixed_main"].count == 10 and r.headline["mixed_main"] == pytest.approx(0.8)
    assert r.stats["mixed_rate"].count == 8 and r.headline["mixed_rate"] == 1.0   # a 0.0 here would read as "no stale hits"


def test_a_score_a_metric_leaves_out_on_some_samples_is_not_filled_for_a_failed_one():
    r = ak.evaluate(samples(), model=lambda ps: ["ok"] * len(ps), scorers=[Sometimes({"3"})],
                    config=ak.RunConfig(failed_as_zero=True))
    assert r.stats["sometimes_all"].count == 10 and r.headline["sometimes_all"] == pytest.approx(0.9)
    assert r.stats["sometimes_even"].count == 5 and r.headline["sometimes_even"] == 1.0   # sample 3 never had one


class OnlyBig(Metric):
    """``only_big`` applies to the samples whose id is 5 or more; 1.0 there."""

    name = "only_big"
    direction = Direction.MAXIMIZE

    def identity(self) -> dict:
        return {"name": self.name}

    def applicable(self, sample) -> bool:
        return int(sample.id) >= 5

    def score(self, sample, output, context=None):
        return Score(name="only_big", value=1.0, direction=self.direction)


def test_a_failed_model_call_is_not_zero_filled_for_a_metric_that_does_not_apply_to_it():
    r = run(Flaky({0, 1}), [OnlyBig()], failed_as_zero=True)
    assert r.stats["only_big"].count == 5 and r.headline["only_big"] == 1.0   # samples 0 and 1 were never going to count


def test_a_crashing_metric_is_zero_filled_on_precomputed_outputs_too():
    samples_ = [ak.Sample(id=str(i), input=f"q{i}", target="ok", actual_output="ok") for i in range(10)]
    r = ak.evaluate(samples_, model="precomputed", scorers=[CrashOn({"0", "1"})],
                    config=ak.RunConfig(failed_as_zero=True))
    assert r.stats["crashy"].count == 10 and r.headline["crashy"] == pytest.approx(0.8)


def _named(model, name):
    model.name = name   # a distinct model identity, so the two runs do not share a cache entry
    return model


def test_comparison_warning_says_failed_samples_were_counted_as_zero_when_they_were():
    base = run(_named(Flaky(set()), "flaky-a"), [ak.Contains("ok")], failed_as_zero=True)
    cand = run(_named(Flaky({0, 1, 5, 6}), "flaky-b"), [ak.Contains("ok")], failed_as_zero=True)
    comp = ak.RunComparison(base, cand)
    assert comp.failure_summary()["candidate_failed"] == 4
    text = comp.summary()
    assert "counts them as 0.0" in text and "EXCLUDED" not in text


def test_comparison_warning_still_says_excluded_without_failed_as_zero():
    base = run(_named(Flaky(set()), "flaky-c"), [ak.Contains("ok")])
    cand = run(_named(Flaky({0, 1, 5, 6}), "flaky-d"), [ak.Contains("ok")])
    assert "EXCLUDED from the means" in ak.RunComparison(base, cand).summary()


class NeedsExtra(Metric):
    """Raises the error for a missing optional dependency on every sample."""

    name = "needs_extra"
    direction = Direction.MAXIMIZE

    def identity(self) -> dict:
        return {"name": self.name}

    def score(self, sample, output, context=None):
        from auditkit.errors import ExtraNotInstalled
        raise ExtraNotInstalled("nothing", "pip install auditkit[nothing]")


def test_a_missing_extra_is_not_turned_into_a_zero_score():
    r = ak.evaluate(samples(), model=lambda ps: ["ok"] * len(ps), scorers=[NeedsExtra()],
                    config=ak.RunConfig(failed_as_zero=True))
    assert r.headline == {}       # as without failed_as_zero: a setup problem, not a measured 0.0
