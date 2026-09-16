"""Token usage capture (real, provider-reported) and cost calculation.

Cost is deliberately never estimated by character/tokenizer guessing and
never priced by a library-bundled table (prices change and there is no live
pricing API) -- only the token *counts* are automatic, read straight from
each backend's real API response; ``$``/token pricing is always supplied by
the caller at the point cost is actually computed.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from auditkit.model import Request
from auditkit.report import RunResult
from auditkit.score import Stat


class TestTokenThroughputDerivedFromUsageAndLatency:
    """output_tokens_per_sec/total_tokens_per_sec are derived from two
    numbers that already existed separately (real elapsed call time, real
    token counts from the backend's own response) but were never combined
    into a tokens/sec figure until now."""

    def test_runner_computes_token_throughput_when_backend_reports_usage(self):
        import time as time_mod
        from auditkit.model import Model, Result_, Generated
        from auditkit.types import Capability
        from auditkit.runner import Runner
        from auditkit.runspec import RunConfig, RunSpec
        from auditkit.scenario import ListScenario
        from auditkit.adapter import GenerationAdapter
        from auditkit.metric import ExactMatch
        from auditkit.sample import Sample

        class UsageReportingModel(Model):
            name = "usage_reporting"

            def capabilities(self):
                return {Capability.GENERATE}

            def generate(self, requests):
                time_mod.sleep(0.02)
                return [
                    Result_(completions=[Generated(text=r.prompt)],
                           usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30})
                    for r in requests
                ]

        spec = RunSpec(
            scenario=ListScenario([Sample(input="4", target="4"), Sample(input="5", target="5")]),
            model=UsageReportingModel(), adapter=GenerationAdapter(),
            metrics=[ExactMatch()], config=RunConfig(track_performance=True),
        )
        result = Runner().run(spec)

        thr = result.perf["throughput"]
        total_time_s = thr["total_time_ms"] / 1000.0
        assert thr["output_tokens_per_sec"] == result.token_usage["completion_tokens"] / total_time_s
        assert thr["total_tokens_per_sec"] == result.token_usage["total_tokens"] / total_time_s
        assert thr["output_tokens_per_sec"] > 0
        assert "out-tok/s" in result.summary()

    def test_token_throughput_is_zero_not_crashing_when_no_usage_reported(self):
        # A local/callable backend reports no usage at all -- 0 tok/s, not
        # a division-by-zero crash or a missing key.
        from auditkit.runner import Runner
        from auditkit.runspec import RunConfig, RunSpec
        from auditkit.scenario import ListScenario
        from auditkit.adapter import GenerationAdapter
        from auditkit.metric import ExactMatch
        from auditkit.model import EchoModel
        from auditkit.sample import Sample

        spec = RunSpec(
            scenario=ListScenario([Sample(input="4", target="4")]),
            model=EchoModel(), adapter=GenerationAdapter(),
            metrics=[ExactMatch()], config=RunConfig(track_performance=True),
        )
        result = Runner().run(spec)
        assert result.perf["throughput"]["output_tokens_per_sec"] == 0.0
        assert result.perf["throughput"]["total_tokens_per_sec"] == 0.0
        assert "out-tok/s" not in result.summary()


class TestOpenAIUsageCapture:
    def test_usage_is_read_from_the_real_response(self):
        from auditkit.model.openai import OpenAIModel

        m = OpenAIModel(api_key="test")
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        m._client = fake_client

        results = m.generate([Request(prompt="hi")])
        assert results[0].usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    def test_missing_usage_on_the_response_does_not_crash(self):
        from auditkit.model.openai import OpenAIModel

        m = OpenAIModel(api_key="test")
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        )
        m._client = fake_client

        results = m.generate([Request(prompt="hi")])
        assert results[0].usage == {}


class TestAnthropicUsageCapture:
    def test_usage_is_normalized_from_input_output_tokens(self):
        from auditkit.model.anthropic import AnthropicModel

        m = AnthropicModel(api_key="test")
        fake_client = MagicMock()
        fake_client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(text="ok")],
            usage=SimpleNamespace(input_tokens=8, output_tokens=4),
        )
        m._client = fake_client

        results = m.generate([Request(prompt="hi")])
        # Anthropic's own field names (input/output) are normalized to the
        # same prompt/completion shape every other backend uses.
        assert results[0].usage == {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12}


class TestGroqUsageCapture:
    def test_usage_is_read_from_the_raw_json_response(self):
        from auditkit.model.groq_gen import GroqModel

        m = GroqModel(api_key="test")
        fake_session = MagicMock()
        fake_resp = MagicMock()
        fake_resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 6, "total_tokens": 26},
        }
        fake_session.post.return_value = fake_resp
        m._session = fake_session

        results = m.generate([Request(prompt="hi")])
        assert results[0].usage == {"prompt_tokens": 20, "completion_tokens": 6, "total_tokens": 26}


class TestRunResultCost:
    def _run(self, prompt_tokens, completion_tokens):
        total = prompt_tokens + completion_tokens
        return RunResult(
            run_id="r", fingerprint="r", stats={"exact_match": Stat("exact_match")},
            predictions=[], headline={},
            token_usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                        "total_tokens": total},
        )

    def test_cost_uses_real_tokens_and_caller_supplied_pricing(self):
        run = self._run(prompt_tokens=1_000_000, completion_tokens=1_000_000)
        pricing = {"input_per_1m": 2.50, "output_per_1m": 10.0}
        assert run.cost(pricing) == 2.50 + 10.0

    def test_cost_is_none_when_no_tokens_were_recorded(self):
        run = self._run(prompt_tokens=0, completion_tokens=0)
        assert run.cost({"input_per_1m": 1.0, "output_per_1m": 1.0}) is None

    def test_no_library_default_pricing_exists(self):
        # cost() requires the pricing dict -- there is no bundled table to
        # silently fall back to (it would go stale).
        run = self._run(prompt_tokens=100, completion_tokens=50)
        import pytest
        with pytest.raises(KeyError):
            run.cost({"input_per_1m": 1.0})  # missing output_per_1m -- not defaulted to 0 or anything


class TestRunComparisonCost:
    def test_cost_compares_both_sides_with_their_own_pricing(self):
        from auditkit.comparison import RunComparison

        base = RunResult(run_id="b", fingerprint="b", stats={"exact_match": Stat("exact_match")},
                         predictions=[], headline={},
                         token_usage={"prompt_tokens": 1_000_000, "completion_tokens": 0, "total_tokens": 1_000_000})
        cand = RunResult(run_id="c", fingerprint="c", stats={"exact_match": Stat("exact_match")},
                         predictions=[], headline={},
                         token_usage={"prompt_tokens": 1_000_000, "completion_tokens": 0, "total_tokens": 1_000_000})
        cmp = RunComparison(base, cand)
        out = cmp.cost(
            baseline_pricing={"input_per_1m": 10.0, "output_per_1m": 30.0},   # e.g. a pricier model
            candidate_pricing={"input_per_1m": 1.0, "output_per_1m": 3.0},    # a cheaper one
        )
        assert out["baseline_cost"] == 10.0
        assert out["candidate_cost"] == 1.0
        assert abs(out["cost_ratio"] - 0.1) < 1e-9

    def test_cost_omits_sides_with_no_recorded_usage(self):
        from auditkit.comparison import RunComparison

        base = RunResult(run_id="b", fingerprint="b", stats={"exact_match": Stat("exact_match")},
                         predictions=[], headline={})  # no token_usage -- local/callable model
        cand = RunResult(run_id="c", fingerprint="c", stats={"exact_match": Stat("exact_match")},
                         predictions=[], headline={},
                         token_usage={"prompt_tokens": 1_000_000, "completion_tokens": 0, "total_tokens": 1_000_000})
        cmp = RunComparison(base, cand)
        out = cmp.cost({"input_per_1m": 1.0, "output_per_1m": 1.0}, {"input_per_1m": 1.0, "output_per_1m": 1.0})
        assert "baseline_cost" not in out
        assert out["candidate_cost"] == 1.0
        assert "cost_ratio" not in out


class TestCompareResultCostTable:
    def test_cost_table_prices_only_models_with_a_pricing_entry(self):
        from auditkit.model_compare import CompareResult

        a = RunResult(run_id="a", fingerprint="a", stats={}, predictions=[], headline={},
                     token_usage={"prompt_tokens": 1_000_000, "completion_tokens": 0, "total_tokens": 1_000_000})
        b = RunResult(run_id="b", fingerprint="b", stats={}, predictions=[], headline={},
                     token_usage={"prompt_tokens": 1_000_000, "completion_tokens": 0, "total_tokens": 1_000_000})
        result = CompareResult(runs={"a": a, "b": b}, dataset_size=1, scorers=[])
        table = {row["model"]: row for row in result.cost_table({"a": {"input_per_1m": 5.0, "output_per_1m": 5.0}})}
        assert table["a"]["cost"] == 5.0
        assert table["b"]["cost"] is None  # no pricing entry for "b" -- not guessed
