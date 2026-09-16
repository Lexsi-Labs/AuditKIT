"""Tests for practical gaps: CLI, RunResult.load, from_dict, Report formatting."""

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
from auditkit.report_format import Report


def _run(**cfg) -> RunResult:
    return Runner().run(RunSpec(
        scenario=ListScenario([Sample(input="a", target="a")]),
        model=EchoModel(),
        adapter=GenerationAdapter(),
        metrics=[ExactMatch()],
        config=RunConfig(**cfg),
        run_name="test",
    ))


# ============================================================================
# RunResult.from_dict / RunResult.load
# ============================================================================

class TestRunResultLoad:
    def test_from_dict_roundtrip(self):
        r = _run()
        d = r.to_dict()
        r2 = RunResult.from_dict(d)
        assert r2.run_id == r.run_id
        assert r2.fingerprint == r.fingerprint
        assert r2.headline == r.headline
        assert len(r2.predictions) == len(r.predictions)
        assert r2.predictions[0].correct == r.predictions[0].correct

    def test_load_from_json(self, tmp_path):
        r = _run()
        p = tmp_path / "run.json"
        r.save(str(p))
        r2 = RunResult.load(str(p))
        assert r2.run_id == r.run_id
        assert r2.headline == r.headline

    def test_load_invalid_path(self):
        with pytest.raises(FileNotFoundError):
            RunResult.load("/nonexistent/path.json")


# ============================================================================
# RunConfig.from_dict
# ============================================================================

class TestRunConfigFromDict:
    def test_from_dict_basic(self):
        cfg = RunConfig.from_dict({"seed": 42, "limit": 10, "temperature": 0.5})
        assert cfg.seed == 42
        assert cfg.limit == 10
        assert cfg.temperature == 0.5

    def test_from_dict_partial(self):
        cfg = RunConfig.from_dict({"seed": 7})
        assert cfg.seed == 7
        assert cfg.limit is None  # default

    def test_from_dict_empty(self):
        cfg = RunConfig.from_dict({})
        assert isinstance(cfg, RunConfig)

    def test_from_dict_extra(self):
        cfg = RunConfig.from_dict({"seed": 1, "unknown_key": "x"})
        assert cfg.seed == 1  # ignores unknown


# ============================================================================
# Report formatting
# ============================================================================

class TestReport:
    def test_report_creation(self):
        r = _run()
        report = Report(r)
        assert report.result is r

    def test_report_string(self):
        r = _run()
        report = Report(r)
        text = str(report)
        assert isinstance(text, str)
        assert "exact_match" in text
        assert "test" in text

    def test_report_metric_rows(self):
        r = _run()
        report = Report(r)
        rows = report.metric_rows()
        assert len(rows) >= 1
        assert any(row["name"] == "exact_match" for row in rows)

    def test_report_markdown(self):
        r = _run()
        report = Report(r)
        md = report.markdown()
        assert isinstance(md, str)
        assert "|" in md  # table format
        assert "exact_match" in md


# ============================================================================
# CLI
# ============================================================================

class TestCLI:
    def test_cli_module_exists(self):
        from auditkit import cli
        assert hasattr(cli, "main")
        assert callable(cli.main)

    def test_cli_help(self):
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-m", "auditkit.cli", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "usage" in result.stdout.lower() or "usage" in result.stderr.lower()
