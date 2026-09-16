"""Integration tests — real evaluate() pipelines with zero-dep backends."""
from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

import auditkit as ak

SRC = os.path.join(os.path.dirname(__file__), "..", "src")


@pytest.fixture
def samples():
    return [ak.Sample(input="hello", target="hello"), ak.Sample(input="world", target="world")]


@pytest.fixture
def tmp():
    d = tempfile.mkdtemp()
    os.environ["XDG_DATA_HOME"] = d
    return d


class TestEvaluatePipeline:
    def test_echo_evaluate(self, samples):
        r = ak.evaluate(samples, model="echo")
        assert r.headline == {"exact_match": 1.0}
        assert len(r.predictions) == 2
        assert all(p.correct for p in r.predictions)

    def test_callable_evaluate(self, samples):
        r = ak.evaluate(samples, model=lambda ps: [p.upper() for p in ps])
        assert r.headline["exact_match"] == 0.0
        assert r.predictions[0].raw_output == "HELLO"

    def test_explicit_metric(self, samples):
        r = ak.evaluate(samples, model="echo", scorers=[ak.Bleu()])
        assert "bleu" in r.headline
        assert r.headline["bleu"] == 1.0

    def test_runconfig_passthrough(self, samples):
        cfg = ak.RunConfig(seed=42, temperature=0.5, max_tokens=100)
        r = ak.evaluate(samples, model="echo", config=cfg)
        assert r.config is not None
        assert r.config.seed == 42

    def test_custom_scorer(self, samples):
        @ak.scorer(direction=ak.Direction.MAXIMIZE)
        def contains_h(sample, output):
            return 1.0 if "hello" in output else 0.0

        r = ak.evaluate([ak.Sample(input="hello world", target="hello")], model="echo", scorers=[contains_h])
        assert r.headline["contains_h"] == 1.0


class TestSerialization:
    def test_save_load_json(self, samples):
        r1 = ak.evaluate(samples, model="echo")
        path = tempfile.mktemp(suffix=".json")
        r1.save(path)
        r2 = ak.RunResult.load(path)
        assert r2.headline == r1.headline
        assert len(r2.predictions) == len(r1.predictions)

    def test_to_dict_roundtrip(self, samples):
        r1 = ak.evaluate(samples, model="echo")
        d = r1.to_dict()
        r2 = ak.RunResult.from_dict(d)
        assert r2.headline == r1.headline

    def test_config_roundtrip(self):
        cfg = ak.RunConfig(seed=42, temperature=0.7, top_p=0.9)
        d = cfg.to_dict()
        cfg2 = ak.RunConfig.from_dict(d)
        assert cfg2.seed == 42
        assert cfg2.temperature == 0.7


class TestDiffAndCompare:
    def test_compare(self, samples):
        r1 = ak.evaluate(samples, model="echo")
        r2 = ak.evaluate(samples[:1], model="echo")
        cmp = ak.compare([r1, r2])
        assert len(cmp) == 2

    def test_diff(self, samples):
        r1 = ak.evaluate(samples, model="echo")
        r2 = ak.evaluate(samples, model="echo")
        diff = ak.RunDiff(r1, r2)
        assert diff.sample_summary()["still_correct"] == 2
        assert diff.grade("exact_match") == ak.DeltaGrade.PASS

    def test_diff_bad_samples(self):
        s1 = ak.Sample(input="a", target="a")
        s2 = ak.Sample(input="b", target="b")
        r1 = ak.evaluate([s1], model="echo")
        r2 = ak.evaluate([s2], model="echo")
        diff = ak.RunDiff(r1, r2)
        assert "unknown" in diff.sample_summary()


class TestExperiment:
    def test_experiment_tracking(self, samples, tmp):
        ak.evaluate(samples, model="echo", experiment_name="test_exp")
        db = ak.ExperimentDB()
        exp = db.load("test_exp")
        agg = exp.aggregate()
        assert agg["exact_match"] == 1.0

    def test_leaderboard(self, samples, tmp):
        ak.evaluate(samples, model="echo", experiment_name="lb_test")
        db = ak.ExperimentDB()
        exp = db.load("lb_test")
        lb = exp.leaderboard()
        assert len(lb) >= 1

    def test_significance(self, samples, tmp):
        ak.evaluate(samples, model="echo", experiment_name="sig_test")
        db = ak.ExperimentDB()
        exp = db.load("sig_test")
        sig = exp.significance("exact_match")
        assert "error" in sig


class TestReport:
    def test_text_report(self, samples):
        r = ak.evaluate(samples, model="echo")
        report = ak.Report(r)
        text = str(report)
        assert "exact_match" in text
        assert "1.0000" in text

    def test_markdown_report(self, samples):
        r = ak.evaluate(samples, model="echo")
        report = ak.Report(r)
        md = report.markdown()
        assert "| Metric | Mean |" in md
        assert "| exact_match |" in md


class TestCLI:
    def test_cli_with_csv(self, tmp):
        import subprocess
        csv_path = os.path.join(tmp, "data.csv")
        with open(csv_path, "w") as f:
            f.write("input,target\nhello,hello\nworld,world\n")
        result = subprocess.run(
            [sys.executable, "-m", "auditkit.cli", "--model", "echo", "--csv", csv_path, "--input-col", "input", "--target-col", "target", "--output", os.path.join(tmp, "out.json")],
            capture_output=True, text=True, env={**os.environ, "PYTHONPATH": SRC},
        )
        assert result.returncode == 0, result.stderr
        with open(os.path.join(tmp, "out.json")) as f:
            data = json.load(f)
        assert "predictions" in data

    def test_cli_echo(self, tmp):
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "auditkit.cli", "--model", "echo", "--output", os.path.join(tmp, "echo_out.json")],
            capture_output=True, text=True, timeout=5, env={**os.environ, "PYTHONPATH": SRC},
        )
        assert result.returncode == 0, result.stderr

    def test_cli_list_annotators(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "auditkit.cli", "list", "annotators"],
            capture_output=True, text=True, timeout=5, env={**os.environ, "PYTHONPATH": SRC},
        )
        assert result.returncode == 0, result.stderr
        assert "Built-in annotators:" in result.stdout
        assert "regex" in result.stdout
        assert "llm" in result.stdout

    def test_cli_list_all_includes_annotators(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "auditkit.cli", "list"],
            capture_output=True, text=True, timeout=5, env={**os.environ, "PYTHONPATH": SRC},
        )
        assert result.returncode == 0, result.stderr
        assert "Built-in annotators:" in result.stdout


class TestScenarioResolution:
    def test_scenarios_registered(self):
        names = ak.SCENARIOS.names()
        assert "mmlu" in names
        assert "gsm8k" in names
        assert "arc" in names
        assert "hellaswag" in names
        assert "humaneval" in names
        assert "truthfulqa" in names

    def test_scenario_create(self):
        sc = ak.SCENARIOS.get("mmlu")
        assert sc.__name__ == "MMLUScenario"


class TestAllMetricFamilies:
    def test_code_metrics(self):
        s = ak.Sample(input="x", target="hello")
        assert ak.Equals().score(s, "hello").value == 1.0
        assert ak.Contains("ell").score(s, "hello").value == 1.0
        assert ak.Regex(r"he.*").score(s, "hello").value == 1.0
        assert isinstance(ak.Levenshtein().score(s, "hello").value, float)

    def test_generation_metrics(self):
        s = ak.Sample(input="x", target="the cat sat")
        assert ak.Bleu().score(s, "the cat sat").value == 1.0
        assert ak.ChrF().score(s, "the cat sat").value == 1.0
        assert ak.WordErrorRate().score(s, "the cat sat").value == 1.0  # 1 - 0 WER

    def test_toxicity_metrics(self):
        s = ak.Sample(input="x", target="nice")
        # Real classifier (default use_model=True) never returns exactly
        # 1.0 -- it's a probability, not a rule match.
        assert ak.ToxicityScore().score(s, "nice message").value > 0.99

    def test_pairwise_metrics(self):
        s = ak.Sample(input="x", target="good")
        ctx = {"candidates": ["bad", "worse"]}
        assert ak.WinRate().score(s, "good", ctx).value == 1.0

    def test_rag_metrics(self):
        s = ak.Sample(input="x", target="target", retrieval_context=["this is target"])
        assert 0.0 <= ak.ContextCoverage().score(s, "target").value <= 1.0

    def test_perf_metrics(self):
        ls = ak.LatencyStats()
        ls.record(100.0)
        assert ls.stats()["count"] == 1

