"""Tests for T1 experiment tracking."""

from __future__ import annotations

import pytest

from auditkit.experiment import Experiment, ExperimentDB
from auditkit.report import RunResult, Prediction
from auditkit.errors import ExtraNotInstalled


def _make_run(run_id: str, scores: list[float]) -> RunResult:
    preds = [
        Prediction(
            run_id=run_id,
            task="test",
            sample_id=str(i),
            prompt="",
            raw_output="",
            parsed_answer="",
            expected="",
            correct=None,
            score=s,
        )
        for i, s in enumerate(scores)
    ]
    mean = sum(scores) / len(scores) if scores else 0.0
    return RunResult(
        run_id=run_id,
        fingerprint=run_id,
        stats={},
        predictions=preds,
        headline={"acc": mean},
    )


class TestExperiment:
    def test_add_and_count(self):
        exp = Experiment(name="test")
        exp.add(_make_run("r1", [1.0, 1.0]))
        exp.add(_make_run("r2", [0.5, 0.5]))
        assert len(exp.runs) == 2

    def test_aggregate_all(self):
        exp = Experiment(name="test")
        exp.add(_make_run("r1", [1.0]))
        exp.add(_make_run("r2", [0.0]))
        agg = exp.aggregate()
        assert agg.get("acc", 0) == 0.5

    def test_aggregate_single_metric(self):
        exp = Experiment(name="test")
        exp.add(_make_run("r1", [1.0]))
        agg = exp.aggregate("acc")
        assert agg["acc"] == 1.0

    def test_leaderboard(self):
        exp = Experiment(name="test")
        exp.add(_make_run("r3", [0.3]))
        exp.add(_make_run("r1", [1.0]))
        exp.add(_make_run("r2", [0.6]))
        lb = exp.leaderboard()
        assert lb[0]["acc"] == 1.0
        assert lb[-1]["acc"] == 0.3

    def test_significance(self):
        exp = Experiment(name="test")
        exp.add(_make_run("baseline", [0.8, 0.9]))
        exp.add(_make_run("contrast", [0.7, 0.8]))
        result = exp.significance("acc")
        assert "p_value" in result
        assert "significant" in result
        assert result["n_samples"] == 2

    def test_significance_insufficient_runs(self):
        exp = Experiment(name="test")
        exp.add(_make_run("alone", [0.5]))
        result = exp.significance("acc")
        assert "error" in result

    def test_log_mlflow_extra_not_installed(self):
        exp = Experiment(name="test")
        exp.add(_make_run("r1", [1.0]))
        with pytest.raises(ExtraNotInstalled):
            exp.log_mlflow()


class TestExperimentDB:
    def test_roundtrip(self, tmp_path):
        exp = Experiment(name="roundtrip")
        exp.add(_make_run("r1", [1.0, 0.5]))
        exp.metadata["foo"] = "bar"
        db = ExperimentDB(path=str(tmp_path))
        db.save(exp)
        loaded = db.load("roundtrip")
        assert loaded.name == "roundtrip"
        assert len(loaded.runs) == 1
        assert loaded.metadata["foo"] == "bar"

    def test_list(self, tmp_path):
        db = ExperimentDB(path=str(tmp_path))
        db.save(Experiment(name="a"))
        db.save(Experiment(name="b"))
        names = db.list_experiments()
        assert "a" in names
        assert "b" in names
