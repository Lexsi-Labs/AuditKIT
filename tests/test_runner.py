import json

import pytest

from auditkit.scenario import Scenario, ListScenario
from auditkit.adapter import Adapter, GenerationAdapter, MCQAdapter
from auditkit.runspec import RunConfig, RunSpec
from auditkit.report import RunResult
from auditkit.runner import Runner
from auditkit.model import EchoModel, Model, PrecomputedModel, Request, Result_, Generated, LogLikelihood
from auditkit.metric import ExactMatch
from auditkit.sample import Sample
from auditkit.score import Stat
from auditkit.types import Capability


def _spec(samples, **cfg):
    return RunSpec(
        scenario=ListScenario(samples),
        model=EchoModel(),
        adapter=GenerationAdapter(),
        metrics=[ExactMatch()],
        config=RunConfig(**cfg),
    )


# ---- Scenario / Adapter --------------------------------------------------

def test_scenario_is_abstract():
    with pytest.raises(TypeError):
        Scenario()


def test_list_scenario_yields_its_samples():
    s = [Sample(input="a", target="a"), Sample(input="b", target="b")]
    assert list(ListScenario(s).samples()) == s


def test_adapter_is_abstract():
    with pytest.raises(TypeError):
        Adapter()


def test_generation_adapter_makes_one_request_per_sample():
    reqs = GenerationAdapter().adapt(Sample(input="hi", target="hi"), RunConfig())
    assert len(reqs) == 1
    assert isinstance(reqs[0], Request)
    assert reqs[0].prompt == "hi"


# ---- RunSpec.fingerprint -------------------------------------------------

def test_fingerprint_is_16_hex_chars():
    fp = _spec([Sample(input="a", target="a")]).fingerprint()
    assert len(fp) == 16
    int(fp, 16)  # parses as hex


def test_fingerprint_is_stable_for_same_inputs():
    a = _spec([Sample(input="a", target="a")], seed=0)
    b = _spec([Sample(input="a", target="a")], seed=0)
    assert a.fingerprint() == b.fingerprint()


def test_fingerprint_changes_with_seed():
    a = _spec([Sample(input="a", target="a")], seed=0)
    b = _spec([Sample(input="a", target="a")], seed=7)
    assert a.fingerprint() != b.fingerprint()


def test_fingerprint_distinguishes_directly_constructed_models_with_same_default_name():
    # Every real backend defaults .name to a fixed class-level string ("hf",
    # "openai", ...) unless a caller passes name= explicitly -- only
    # AutoModel.resolve()'s string-spec path does that automatically. Two
    # directly-constructed instances of the same backend class pointing at
    # genuinely different checkpoints used to collide into one fingerprint,
    # since RunSpec.fingerprint() read self.model.name directly instead of
    # going through the same identity()-aware path adapters/metrics/
    # annotators already used. Model.identity() now includes _model_name.
    from auditkit.model.hf_gen import HFGenModel

    m1 = HFGenModel(model="gpt2")
    m2 = HFGenModel(model="gpt2-medium")
    assert m1.name == m2.name == "hf"  # confirms the premise: .name alone can't distinguish them
    fp1 = RunSpec(scenario=ListScenario([Sample(input="a", target="a")]), model=m1,
                  adapter=GenerationAdapter(), metrics=[ExactMatch()]).fingerprint()
    fp2 = RunSpec(scenario=ListScenario([Sample(input="a", target="a")]), model=m2,
                  adapter=GenerationAdapter(), metrics=[ExactMatch()]).fingerprint()
    assert fp1 != fp2


def test_fingerprint_still_matches_for_string_spec_resolved_models():
    # Regression guard: AutoModel.resolve("hf:gpt2") already sets name=spec,
    # so this path must keep working exactly as before.
    from auditkit.model import AutoModel

    a = RunSpec(scenario=ListScenario([Sample(input="a", target="a")]), model=AutoModel.resolve("hf:gpt2"),
                adapter=GenerationAdapter(), metrics=[ExactMatch()]).fingerprint()
    b = RunSpec(scenario=ListScenario([Sample(input="a", target="a")]), model=AutoModel.resolve("hf:gpt2-medium"),
                adapter=GenerationAdapter(), metrics=[ExactMatch()]).fingerprint()
    assert a != b


def test_fingerprint_distinguishes_scoregate_weight_and_threshold():
    from auditkit.scoring import ScoreGate

    a = _spec([Sample(input="a", target="a")])
    a.metrics = [ScoreGate(ExactMatch(), weight=1.0, threshold=None)]
    b = _spec([Sample(input="a", target="a")])
    b.metrics = [ScoreGate(ExactMatch(), weight=5.0, threshold=0.9)]
    assert a.fingerprint() != b.fingerprint()


def test_scoregate_identity_delegates_to_wrapped_metric_identity():
    from auditkit.scoring import ScoreGate
    from auditkit.metrics.judge import LLMJudge

    j1 = ScoreGate(LLMJudge(judge_model="openai:gpt-4o", choices={"a": 1.0}, prompt="v1 {output}"))
    j2 = ScoreGate(LLMJudge(judge_model="openai:gpt-4o", choices={"a": 1.0}, prompt="v2 {output}"))
    # A ScoreGate wrapping two judges that differ only in prompt must not
    # collide either -- it used to (ScoreGate.name only read the wrapped
    # metric's .name, never its .identity()).
    assert j1.identity() != j2.identity()


# ---- Runner stages -------------------------------------------------------

def test_build_requests_respects_limit():
    r = Runner()
    scn = ListScenario([Sample(input=str(i), target=str(i)) for i in range(10)])
    batch = r.build_requests(scn, GenerationAdapter(), RunConfig(limit=3))
    assert len(batch) == 3


def test_execute_aligns_results_to_samples():
    r = Runner()
    samples = [Sample(input="x", target="x"), Sample(input="y", target="y")]
    batch = r.build_requests(ListScenario(samples), GenerationAdapter(), RunConfig())
    executed = r.execute(EchoModel(), batch)
    assert [s.input for s, _ in executed] == ["x", "y"]
    assert executed[0][1][0].text == "x"  # echo


def test_aggregate_groups_scores_into_stats():
    from auditkit.score import Score
    r = Runner()
    stats = r.aggregate([
        Score(name="exact_match", value=1.0),
        Score(name="exact_match", value=0.0),
    ])
    assert isinstance(stats["exact_match"], Stat)
    assert stats["exact_match"].mean == 0.5


# ---- End-to-end run ------------------------------------------------------

def test_run_produces_headline_and_predictions():
    # echo returns the input; matching targets score 1.0, mismatched 0.0
    samples = [Sample(input="4", target="4"), Sample(input="x", target="y")]
    result = Runner().run(_spec(samples))
    assert isinstance(result, RunResult)
    assert result.headline["exact_match"] == pytest.approx(0.5)
    assert len(result.predictions) == 2
    assert result.run_id == result.fingerprint  # deterministic default


def test_run_marks_correctness_per_prediction():
    samples = [Sample(input="4", target="4"), Sample(input="x", target="y")]
    result = Runner().run(_spec(samples))
    by_expected = {p.expected: p for p in result.predictions}
    assert by_expected["4"].correct is True
    assert by_expected["y"].correct is False


def test_wrong_only_filters_incorrect():
    samples = [Sample(input="4", target="4"), Sample(input="x", target="y")]
    result = Runner().run(_spec(samples))
    wrong = result.wrong_only()
    assert len(wrong) == 1
    assert wrong[0].expected == "y"


# ---- RunResult reporting -------------------------------------------------

def test_summary_mentions_metric_and_is_a_string():
    result = Runner().run(_spec([Sample(input="4", target="4")]))
    text = result.summary()
    assert isinstance(text, str)
    assert "exact_match" in text


def test_metric_table_rows():
    result = Runner().run(_spec([Sample(input="4", target="4")]))
    rows = result.metric_table()
    assert any(row["name"] == "exact_match" for row in rows)


def test_save_json_writes_file(tmp_path):
    result = Runner().run(_spec([Sample(input="4", target="4")], track_performance=True))
    out = tmp_path / "run.json"
    result.save(str(out), fmt="json")
    loaded = json.loads(out.read_text())
    assert "headline" in loaded
    assert "predictions" in loaded
    assert "perf" in loaded
    assert loaded["perf"]["latency_ms"]["count"] == 1


def test_perf_survives_to_dict_from_dict_roundtrip():
    result = Runner().run(_spec([Sample(input="4", target="4")], track_performance=True))
    restored = RunResult.from_dict(result.to_dict())
    assert restored.perf == result.perf


def test_prediction_to_doc_roundtrip():
    result = Runner().run(_spec([Sample(input="4", target="4")]))
    doc = result.predictions[0].to_doc()
    assert doc["expected"] == "4"
    assert doc["correct"] is True


def test_hf_publish_card_is_auditkit():
    from auditkit.hf_publish import _LOGO_ASSET, _branding_header, render_dataset_card

    assert _LOGO_ASSET.is_file()
    assert "lexsi.ai" in _branding_header("https://example.com/auditkit_logo.png")
    card = render_dataset_card(
        "ram-lexsi/auditkit-testrun-evaluate",
        method="evaluate",
        model="echo",
        logo_url="https://example.com/auditkit_logo.png",
        built_on="2026-09-01 00:00 UTC",
    )
    assert "Lexsi-Labs/AuditKIT" in card
    assert "aligntune" not in card.lower()


def test_parquet_safe_rows_drop_empty_structs():
    from auditkit.hf_publish import _parquet_safe_rows

    rows = _parquet_safe_rows(
        [{"context": {}, "metadata": {}, "output": "ok", "nested": {"keep": 1, "empty": {}}}]
    )
    assert rows[0]["context"] is None
    assert rows[0]["metadata"] is None
    assert rows[0]["nested"] == {"keep": 1, "empty": None}
    assert rows[0]["output"] == "ok"


# ---- score_one: a crashing metric is skipped, not faked as 0.0 -----------

class _CrashingMetric(ExactMatch):
    """Raises instead of scoring -- simulates a real computation bug in a
    metric (not the model getting the wrong answer)."""

    name = "crashing_metric"

    def score(self, sample, output, context=None):
        raise RuntimeError("simulated metric computation bug")


def test_crashing_metric_is_not_recorded_as_a_zero_score():
    runner = Runner()
    scores, prediction = runner.score_one(
        [_CrashingMetric(), ExactMatch()],
        Sample(input="4", target="4"), "4", context=None, run_id="r1",
        errors=(errs := []),
    )
    # The crashing metric contributes no Score at all -- not a fake 0.0.
    names = [s["name"] for s in prediction.metadata["scores"]]
    assert "crashing_metric" not in names
    assert "exact_match" in names  # the other metric still scored normally
    # The failure is recorded, not silently discarded.
    assert errs == [{"sample_id": None, "metric": "crashing_metric",
                      "error": "simulated metric computation bug"}]
    # correct/score reflect the metric that actually ran, not the crashed one.
    assert prediction.correct is True


# ---- track_performance: on by default, opt-out available ------------------

def test_track_performance_defaults_to_true():
    assert RunConfig().track_performance is True


def test_perf_model_size_token_usage_populate_by_default():
    # Same generative run as test_generative_run_records_perf below, with no
    # explicit track_performance -- perf/model_size/token_usage must all be
    # populated under the current default, and nothing should crash reading
    # the result.
    result = Runner().run(_spec([Sample(input="4", target="4"), Sample(input="5", target="5")]))
    assert result.perf is not None
    assert result.model_size is not None
    assert result.token_usage is not None
    assert result.headline  # the run itself still happened and still scored


def test_track_performance_true_populates_all_three_fields():
    result = Runner().run(_spec(
        [Sample(input="4", target="4")], track_performance=True,
    ))
    assert result.perf is not None
    assert result.model_size is not None
    assert result.token_usage is not None


def test_track_performance_false_leaves_fields_none():
    # Explicit opt-out must still work: perf/model_size/token_usage must all
    # be None, not {}/zeroed, and nothing should crash reading the result.
    result = Runner().run(_spec(
        [Sample(input="4", target="4"), Sample(input="5", target="5")], track_performance=False,
    ))
    assert result.perf is None
    assert result.model_size is None
    assert result.token_usage is None
    assert result.cost({"input_per_1m": 1.0, "output_per_1m": 1.0}) is None
    assert "perf:" not in result.summary()
    assert result.headline  # the run itself still happened and still scored


def test_toggling_track_performance_changes_the_fingerprint():
    # Both fields live in RunConfig.to_dict(), which fingerprint() hashes
    # whole -- toggling this must not silently replay a cached perf-less
    # result once a caller asks for tracking (or vice versa).
    off = _spec([Sample(input="4", target="4")], track_performance=False)
    on = _spec([Sample(input="4", target="4")], track_performance=True)
    assert off.fingerprint() != on.fingerprint()


# ---- perf: latency/throughput measured during a run -----------------------

def test_generative_run_records_perf():
    result = Runner().run(_spec(
        [Sample(input="4", target="4"), Sample(input="5", target="5")], track_performance=True,
    ))
    lat = result.perf["latency_ms"]
    thr = result.perf["throughput"]
    # Non-parallel (concurrency=1, the default): the whole batch is one
    # generate() call, so exactly one latency sample is recorded.
    assert lat["count"] == 1
    assert lat["mean"] >= 0.0
    assert thr["total_requests"] == 2
    assert thr["rps"] >= 0.0
    assert "perf:" in result.summary()


def test_reads_actual_output_run_has_empty_perf():
    # No model call happens on this path at all -- nothing to time.
    result = Runner().run(RunSpec(
        scenario=ListScenario([Sample(input="q", target="a", actual_output="a")]),
        model=PrecomputedModel(),
        adapter=GenerationAdapter(),
        metrics=[ExactMatch()],
        config=RunConfig(track_performance=True),
    ))
    assert result.perf["latency_ms"]["count"] == 0
    assert result.perf["throughput"]["rps"] == 0.0
    assert "perf:" not in result.summary()


class _FlakyModel(Model):
    """Fails its first call, succeeds on retry -- simulates a transient
    backend error to confirm perf records each real attempt, not the
    artificial sleep()-backoff between them."""

    name = "flaky"
    calls = 0

    def capabilities(self):
        return {Capability.GENERATE}

    def generate(self, requests):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("transient failure")
        return [Result_(completions=[Generated(text=r.prompt)]) for r in requests]


def test_perf_records_each_retry_attempt_not_the_backoff_sleep():
    model = _FlakyModel()
    spec = RunSpec(
        scenario=ListScenario([Sample(input="4", target="4")]),
        model=model,
        adapter=GenerationAdapter(),
        metrics=[ExactMatch()],
        config=RunConfig(max_retries=2, retry_delay=0.05, track_performance=True),
    )
    result = Runner().run(spec)
    # One failed attempt + one successful attempt = 2 recorded calls.
    assert result.perf["latency_ms"]["count"] == 2
    # Real generate() calls are near-instant; if the 50ms artificial backoff
    # sleep had leaked into a recorded sample, the max would be >= 50ms.
    assert result.perf["latency_ms"]["max"] < 50.0
    # Previously: the failed attempt also credited its full request count to
    # Throughput, double-counting against the successful retry (1 request,
    # 2 attempts -> total_requests was 2, not the correct 1).
    assert result.perf["throughput"]["total_requests"] == 1


class _LLModel(Model):
    name = "ll_test"

    def capabilities(self):
        return {Capability.GENERATE, Capability.LOGLIKELIHOOD}

    def generate(self, requests):
        return [Result_(completions=[Generated(text="a")]) for _ in requests]

    def loglikelihood(self, requests):
        return [LogLikelihood(logprob=-0.1 if r.params.get("target") == "a" else -5.0)
                for r in requests]


def test_loglikelihood_run_records_perf():
    spec = RunSpec(
        scenario=ListScenario([Sample(input="q", target="0", choices=["a", "b"], id="0")]),
        model=_LLModel(),
        adapter=MCQAdapter(method="mcq_loglikelihood"),
        metrics=[ExactMatch()],
        config=RunConfig(track_performance=True),
    )
    result = Runner().run(spec)
    assert result.perf["latency_ms"]["count"] == 1
    assert result.perf["throughput"]["total_requests"] == 2  # one loglikelihood request per choice
