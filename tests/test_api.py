"""Tests for the public façade (Cycle 8).

Run independently of acceptance tests in test_runner etc. — focuses on the
entry points that users see: evaluate(), scorer, load_csv,
AutoModel.resolve, and the __init__ exports.
"""

from __future__ import annotations


import pytest

import auditkit as ak
from auditkit.api import _to_scenario, _to_metrics
from auditkit.model import CallableModel, EchoModel, AutoModel
from auditkit.scorers import FunctionScorer, ScorerMetric, scorer
from auditkit.sample import Sample
from auditkit.metric import Metric, ExactMatch
from auditkit.report import RunResult
from auditkit.score import Score
from auditkit.errors import AuditKitError
from auditkit.types import Direction


# ============================================================================
# AutoModel.resolve
# ============================================================================

class TestAutoModelResolve:
    def test_model_passed_through(self):
        m = EchoModel()
        assert AutoModel.resolve(m) is m

    def test_callable_wrapped_as_callable_model(self):
        fn = lambda ps: [p.upper() for p in ps]
        model = AutoModel.resolve(fn)
        assert isinstance(model, CallableModel)
        # Default name is derived from the wrapped function (qualname + a code
        # hash), not a fixed literal, so two different callables never collide
        # under one shared identity (this used to cause a real fingerprint/
        # cache collision bug between different unnamed models).
        assert model.name.startswith("callable:")
        assert "<lambda>" in model.name

    def test_callable_wrapped_model_name_distinguishes_different_functions(self):
        fn_a = lambda ps: [p.upper() for p in ps]
        fn_b = lambda ps: [p.lower() for p in ps]
        model_a = AutoModel.resolve(fn_a)
        model_b = AutoModel.resolve(fn_b)
        assert model_a.name != model_b.name

    def test_callable_model_explicit_name_still_honored(self):
        fn = lambda ps: ps
        model = AutoModel.resolve(fn, name="my_custom_name")
        assert model.name == "my_custom_name"

    def test_echo_string(self):
        model = AutoModel.resolve("echo")
        assert isinstance(model, EchoModel)

    def test_unknown_string_raises(self):
        with pytest.raises(AuditKitError, match="echo|model"):
            AutoModel.resolve("no-such-model")

    def test_hf_prefix_resolves_t1_backend(self):
        from auditkit.model.hf_gen import HFGenModel
        model = AutoModel.resolve("hf:gpt2")
        assert isinstance(model, HFGenModel)

    def test_vllm_prefix_resolves_t1_backend(self):
        from auditkit.model.vllm_gen import VLLMModel
        model = AutoModel.resolve("vllm:mistralai/Mistral-7B")
        assert isinstance(model, VLLMModel)

    def test_litellm_prefix_resolves_t1_backend(self):
        from auditkit.model.litellm_gen import LiteLLMModel
        model = AutoModel.resolve("litellm:gpt-4o")
        assert isinstance(model, LiteLLMModel)

    def test_api_prefix_resolves_t1_backend(self):
        from auditkit.model.api_gen import APIModel
        model = AutoModel.resolve("api:openai/gpt-4")
        assert isinstance(model, APIModel)

    def test_lexsi_prefix_resolves_t1_backend(self):
        from auditkit.model.lexsi import LexsiModel
        model = AutoModel.resolve("lexsi:lexsi-3.5")
        assert isinstance(model, LexsiModel)


# ============================================================================
# @scorer decorator
# ============================================================================

class TestScorerDecorator:
    def test_bare_decorator_returns_function_scorer(self):
        @scorer(direction=Direction.MAXIMIZE)
        def my_scorer(sample, output):
            return 1.0 if output == "ok" else 0.0
        assert isinstance(my_scorer, FunctionScorer)
        assert my_scorer.name == "my_scorer"
        result = my_scorer(Sample(input=""), "ok")
        assert result.value == 1.0

    def test_direction_is_required(self):
        with pytest.raises(TypeError):
            @scorer
            def my_scorer(sample, output):
                return 1.0

    def test_parameterized_decorator(self):
        @scorer(name="custom_name", direction=Direction.MAXIMIZE)
        def my_scorer(sample, output):
            return Score(name="custom_name", value=0.5)
        assert isinstance(my_scorer, FunctionScorer)
        assert my_scorer.name == "custom_name"
        result = my_scorer(Sample(input=""), "")
        assert result.value == 0.5

    def test_scorer_returning_float(self):
        @scorer(direction=Direction.MAXIMIZE)
        def length_check(sample, output):
            return float(len(output))
        result = length_check(Sample(input=""), "hello")
        assert isinstance(result, Score)
        assert result.value == 5.0

    def test_scorer_returning_score(self):
        @scorer(direction=Direction.MAXIMIZE)
        def always_half(sample, output):
            return Score(name="always_half", value=0.5)
        result = always_half(Sample(input=""), "anything")
        assert isinstance(result, Score)
        assert result.value == 0.5

    def test_direction_reaches_the_score(self):
        @scorer(direction=Direction.MINIMIZE)
        def latency_scorer(sample, output):
            return 42.0
        result = latency_scorer(Sample(input=""), "x")
        assert result.direction == Direction.MINIMIZE


# ============================================================================
# ScorerMetric adapter
# ============================================================================

class TestScorerMetric:
    def test_wraps_scorer_as_metric(self):
        @scorer(direction=Direction.MAXIMIZE)
        def my_scorer(sample, output):
            return 1.0
        metric = ScorerMetric(my_scorer)
        assert isinstance(metric, Metric)
        assert metric.name == "my_scorer"
        assert metric.direction == Direction.MAXIMIZE
        score = metric.score(Sample(input=""), "out")
        assert score.value == 1.0

    def test_metric_applicable_always_true(self):
        @scorer(direction=Direction.MAXIMIZE)
        def my_scorer(sample, output):
            return 0.0
        metric = ScorerMetric(my_scorer)
        assert metric.applicable(Sample(input="")) is True
        assert metric.applicable(Sample(input="", target="x")) is True


# ============================================================================
# _to_scenario helper
# ============================================================================

class TestToScenario:
    def test_scenario_passed_through(self):
        from auditkit.scenario import ListScenario
        s = ListScenario([Sample(input="hi")])
        assert _to_scenario(s) is s

    def test_list_of_samples(self):
        samples = [Sample(input="a"), Sample(input="b")]
        scn = _to_scenario(samples)
        assert list(scn.samples()) == samples

    def test_callable_yields_samples(self):
        scn = _to_scenario(lambda: [Sample(input="a")])
        assert list(scn.samples()) == [Sample(input="a")]


# ============================================================================
# _to_metrics helper
# ============================================================================

class TestToMetrics:
    def test_none_returns_exact_match_for_golden(self):
        samples = [Sample(input="a", target="a")]
        metrics = _to_metrics(None, samples)
        assert len(metrics) == 1
        assert isinstance(metrics[0], ExactMatch)

    def test_none_returns_empty_for_no_golden(self):
        samples = [Sample(input="a")]
        metrics = _to_metrics(None, samples)
        assert metrics == []

    def test_str_lookup(self):
        metrics = _to_metrics("exact_match", [Sample(input="a", target="a")])
        assert len(metrics) == 1
        assert isinstance(metrics[0], ExactMatch)

    def test_quasi_exact_match_str(self):
        metrics = _to_metrics("quasi_exact_match", [Sample(input="a", target="a")])
        assert len(metrics) == 1
        assert isinstance(metrics[0], Metric)
        assert metrics[0].name == "quasi_exact_match"

    def test_metric_instance_passed_through(self):
        m = ExactMatch()
        metrics = _to_metrics(m, [Sample(input="a", target="a")])
        assert metrics == [m]

    def test_list_of_scorers(self):
        @scorer(direction=Direction.MAXIMIZE)
        def f(s, o):
            return 0.0
        metrics = _to_metrics([f], [Sample(input="a")])
        assert len(metrics) == 1
        assert isinstance(metrics[0], ScorerMetric)
        assert metrics[0].name == "f"

    def test_mixed_list(self):
        metrics = _to_metrics(["exact_match", ExactMatch()], [Sample(input="a", target="a")])
        assert len(metrics) == 2


# ============================================================================
# load_csv
# ============================================================================

class TestLoadCSV:
    def test_load_csv_returns_samples(self, tmp_path):
        p = tmp_path / "data.csv"
        p.write_text("input,target\nhello,world\nfoo,bar\n")
        samples = ak.load_csv(str(p))
        assert len(samples) == 2
        assert samples[0].input == "hello"
        assert samples[0].target == "world"
        assert samples[1].input == "foo"
        assert samples[1].target == "bar"

    def test_load_csv_defaults_missing_target_to_none(self, tmp_path):
        p = tmp_path / "data.csv"
        p.write_text("input\nhello\nfoo\n")
        samples = ak.load_csv(str(p))
        assert len(samples) == 2
        assert samples[0].target is None
        assert samples[1].target is None

    def test_load_csv_custom_columns(self, tmp_path):
        p = tmp_path / "data.csv"
        p.write_text("question,answer\nhello,world\nfoo,bar\n")
        samples = ak.load_csv(str(p), input_col="question", target_col="answer")
        assert samples[0].input == "hello"
        assert samples[0].target == "world"


# ============================================================================
# evaluate() — acceptance tests
# ============================================================================

class TestEvaluate:
    def test_auto_select_exact_match_on_golden_samples(self):
        r = ak.evaluate(
            [ak.Sample(input="4", target="4"), ak.Sample(input="x", target="y")],
            model="echo",
        )
        assert isinstance(r, RunResult)
        assert r.headline["exact_match"] == 0.5

    def test_custom_scorer_integration(self):
        @ak.scorer(direction=Direction.MAXIMIZE)
        def contains_four(sample, output):
            return 1.0 if "4" in output else 0.0

        r = ak.evaluate(
            [ak.Sample(input="4")], model="echo", scorers=[contains_four]
        )
        assert isinstance(r, RunResult)
        assert r.headline["contains_four"] == 1.0

    def test_batch_callable_model_with_string_scorer(self):
        r = ak.evaluate(
            [ak.Sample(input="hi", target="HI")],
            model=lambda ps: [p.upper() for p in ps],
            scorers=["exact_match"],
        )
        assert isinstance(r, RunResult)
        assert r.headline["exact_match"] == 1.0

    def test_evaluate_with_all_scorers_none_selects_exact_match(self):
        r = ak.evaluate(
            [ak.Sample(input="a", target="a"), ak.Sample(input="b", target="c")],
            model="echo",
        )
        assert "exact_match" in r.headline
        assert r.headline["exact_match"] == 0.5

    def test_evaluate_with_no_golden_and_no_scorers_returns_empty_headline(self):
        r = ak.evaluate(
            [ak.Sample(input="a"), ak.Sample(input="b")],
            model="echo",
        )
        assert r.headline == {}


# ============================================================================
# __init__ exports
# ============================================================================

class TestExports:
    """Verify every name listed in HANDOFF §Cycle-8 step 5 is importable."""

    def test_evaluate_exported(self):
        assert hasattr(ak, "evaluate")
        assert callable(ak.evaluate)

    def test_sample_exported(self):
        assert ak.Sample is Sample

    def test_scorer_exported(self):
        assert ak.scorer is scorer

    def test_load_csv_exported(self):
        assert callable(ak.load_csv)

    def test_result_exported(self):
        assert ak.Result is RunResult

    def test_score_exported(self):
        assert hasattr(ak, "Score")

    @pytest.mark.parametrize("name", [
        "TaskKind", "ScoreKind", "DataType", "Direction",
        "Capability", "Source",
    ])
    def test_enum_exported(self, name):
        assert hasattr(ak, name)

    def test_run_result_api_members(self):
        r = ak.evaluate([ak.Sample(input="a")], model="echo")
        assert hasattr(r, "headline")
        assert hasattr(r, "predictions")
        assert hasattr(r, "wrong_only")
        assert hasattr(r, "save")
        assert hasattr(r, "summary")
        assert hasattr(r, "metric_table")
