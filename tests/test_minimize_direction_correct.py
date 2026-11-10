"""An ungated 0/1 score that is lower-is-better (a harmful/error flag) is correct at 0.0, not 1.0.

Before the fix ``Prediction.correct`` was ``value == 1.0`` whatever the metric's direction, so
for a ``harmful`` flag every harmful answer was "correct" and ``RunComparison.regressed()`` /
``improved()`` returned each other's samples.
"""
from __future__ import annotations

import json

import pytest

import auditkit as ak
from auditkit import Sample
from auditkit.diff import RunDiff
from auditkit.report import RunResult
from auditkit.scorers import FunctionScorer
from auditkit.types import Direction


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))


def _flag(sample, output):
    return 1.0 if sample.metadata["harmful"] else 0.0


def _run(flags, direction):
    samples = [Sample(input=f"q{i}", actual_output="", id=f"q{i}", metadata={"harmful": f})
               for i, f in enumerate(flags)]
    return ak.evaluate(samples, model="precomputed",
                       scorers=[FunctionScorer(_flag, direction, name="harmful")])


def test_minimize_flag_is_correct_at_zero():
    r = _run([True, False], Direction.MINIMIZE)
    assert [p.correct for p in r.predictions] == [False, True]


def test_maximize_flag_still_correct_at_one():
    r = _run([True, False], Direction.MAXIMIZE)
    assert [p.correct for p in r.predictions] == [True, False]


def test_regressed_and_improved_follow_direction():
    # q0 harmful -> safe (improved), q1 safe -> harmful (regressed), q2 harmful both, q3 safe both
    base = _run([True, False, True, False], Direction.MINIMIZE)
    cand = _run([False, True, True, False], Direction.MINIMIZE)
    c = ak.compare(base, cand)
    assert [p.sample_id for p in c.regressed("harmful")] == ["q1"]
    assert [p.sample_id for p in c.improved("harmful")] == ["q0"]
    assert [p.sample_id for p in c.regressed()] == ["q1"]          # primary-score path too


def test_direction_read_back_from_a_saved_run(tmp_path):
    # Saved runs carry direction as the string "minimize"; the per-metric diff must read it.
    base = _run([True, False], Direction.MINIMIZE)
    cand = _run([False, True], Direction.MINIMIZE)
    loaded = []
    for name, run in (("b", base), ("c", cand)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(run.to_dict(), default=str))
        loaded.append(RunResult.from_dict(json.loads(path.read_text())))
    d = RunDiff(*loaded)
    assert [p.sample_id for p in d.regressed("harmful")] == ["q1"]
    assert [p.sample_id for p in d.improved("harmful")] == ["q0"]
