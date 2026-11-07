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
