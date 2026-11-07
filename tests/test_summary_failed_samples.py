"""summary() says when samples failed: they are left out of the means, which otherwise read as a clean run."""

from __future__ import annotations

import pytest

import auditkit as ak
from auditkit.model import Generated, Model, Result_


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


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))


def run(fail: set[int]):
    samples = [ak.Sample(id=str(i), input=f"q{i}", target="ok") for i in range(10)]
    return ak.evaluate(samples, model=Flaky(fail), scorers=[ak.Contains("ok")])


def test_failed_samples_are_reported_next_to_a_headline_computed_without_them():
    r = run({0, 1, 5, 6})
    assert r.headline["contains(ok)"] == 1.0 and r.stats["contains(ok)"].count == 6   # 4 of 10 failed
    assert "4 failed sample(s) left out of the means above" in r.summary()


def test_summary_is_unchanged_when_nothing_failed():
    assert "failed" not in run(set()).summary()
