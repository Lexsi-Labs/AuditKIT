"""Tests for model comparison."""
from __future__ import annotations

import pytest

from auditkit.model_compare import compare_models, CompareResult
from auditkit import Sample
from auditkit.runspec import RunConfig
from auditkit.scorers import scorer
from auditkit.types import Direction


class TestCompareModels:
    def test_compare_two_echo_models(self):
        samples = [
            Sample(input="hello", target="hello"),
            Sample(input="world", target="world"),
        ]
        result = compare_models(
            models=["echo", "echo"],
            dataset=samples,
            model_names=["echo_a", "echo_b"],
        )
        assert isinstance(result, CompareResult)
        assert len(result.runs) == 2
        assert "echo_a" in result.runs
        assert "echo_b" in result.runs
        assert result.dataset_size == 2

    def test_per_metric(self):
        samples = [
            Sample(input="hello", target="hello"),
            Sample(input="world", target="world"),
        ]
        result = compare_models(
            models=["echo", "echo"],
            dataset=samples,
            model_names=["echo_a", "echo_b"],
            scorers=["exact_match"],
        )
        metrics = result.per_metric()
        assert len(metrics) > 0
        for m in metrics:
            assert m.metric is not None
            assert len(m.scores) == 2
            assert m.winner is not None

    def test_winner(self):
        samples = [
            Sample(input="hello", target="hello"),
            Sample(input="world", target="world"),
        ]
        result = compare_models(
            models=["echo", "echo"],
            dataset=samples,
            model_names=["echo_a", "echo_b"],
        )
        w = result.winner()
        assert w is not None

    def test_significance(self):
        samples = [
            Sample(input="hello", target="hello"),
            Sample(input="world", target="world"),
        ]
        result = compare_models(
            models=["echo", "echo"],
            dataset=samples,
            model_names=["echo_a", "echo_b"],
            scorers=["exact_match"],
        )
        sig = result.significance("echo_a", "echo_b")
        assert "p_value" in sig
        assert "delta" in sig

    def test_summary(self):
        samples = [Sample(input="test", target="test")]
        result = compare_models(
            models=["echo", "echo"],
            dataset=samples,
            model_names=["a", "b"],
            scorers=["exact_match"],
        )
        summary = result.summary()
        assert "Model Comparison" in summary
        assert "a" in summary
        assert "b" in summary

    def test_no_coverage_warnings_by_default(self):
        samples = [Sample(input="hello", target="hello"), Sample(input="world", target="world")]
        result = compare_models(
            models=["echo", "echo"], dataset=samples,
            model_names=["a", "b"], scorers=["exact_match"],
        )
        assert result.coverage_warnings() == []
        assert result.has_failures is False
        assert "WARNING" not in result.summary()

    def test_coverage_warning_on_sample_count_mismatch_across_models(self):
        samples = [Sample(input="hello", target="hello"), Sample(input="world", target="world")]
        result = compare_models(
            models=["echo", "echo"], dataset=samples,
            model_names=["a", "b"], scorers=["exact_match"],
            configs={"b": RunConfig(limit=1)},   # b only sees 1 of 2 samples
        )
        warnings = result.coverage_warnings()
        assert any("sample counts differ" in w for w in warnings)
        assert "WARNING" in result.summary()

    def test_comparison_table_has_score_n_std_per_model(self):
        samples = [Sample(input="hello", target="hello"), Sample(input="world", target="world")]
        result = compare_models(
            models=["echo", "echo"], dataset=samples,
            model_names=["a", "b"], scorers=["exact_match"],
        )
        table = result.comparison_table()
        row = next(r for r in table if r["metric"] == "exact_match")
        assert row["a_n"] == 2 and row["b_n"] == 2
        assert row["a_score"] == 1.0 and row["b_score"] == 1.0
        assert row["winner"] in ("a", "b")

    def test_winner_is_direction_aware_for_minimize_metrics(self):
        # Previously: winner()/comparison_table() picked by raw max(), so a
        # MINIMIZE metric (lower = better) crowned the WORST model. "fast"
        # returns a short (low-value) output, "slow" a long (high-value)
        # one -- for a MINIMIZE metric, "fast" should win.
        @scorer(direction=Direction.MINIMIZE)
        def latency_proxy(sample, output):
            return float(len(output))

        samples = [Sample(input=str(i), target=str(i)) for i in range(3)]

        def fast_model(ps):
            return ["a"] * len(ps)

        def slow_model(ps):
            return ["a" * 100] * len(ps)

        result = compare_models(
            [fast_model, slow_model], samples, scorers=[latency_proxy],
            model_names=["fast", "slow"],
        )
        assert result.winner() == "fast"
        table = {r["metric"]: r for r in result.comparison_table()}
        assert table["latency_proxy"]["winner"] == "fast"

    def test_per_task_and_macro_average_tables_are_direction_aware(self):
        @scorer(direction=Direction.MINIMIZE)
        def error_count(sample, output):
            return 0.0 if output == sample.target else 1.0

        data = [
            Sample(input="1", target="1", task="t1"),
            Sample(input="2", target="2", task="t1"),
            Sample(input="3", target="3", task="t2"),
            Sample(input="4", target="4", task="t2"),
        ]

        def good_model(ps):
            return ps

        def bad_model(ps):
            return ["WRONG"] * len(ps)

        result = compare_models(
            [good_model, bad_model], data, scorers=[error_count],
            model_names=["good", "bad"],
        )
        for row in result.per_task_table():
            assert row["winner"] == "good"
        for row in result.task_macro_average_table():
            assert row["winner"] == "good"

    def test_mixed_direction_metrics_each_graded_independently(self):
        @scorer(direction=Direction.MAXIMIZE)
        def accuracy_like(sample, output):
            return 1.0 if output == sample.target else 0.0

        @scorer(direction=Direction.MINIMIZE)
        def error_like(sample, output):
            return 0.0 if output == sample.target else 1.0

        samples = [Sample(input=str(i), target=str(i)) for i in range(3)]

        def good_model(ps):
            return ps

        def bad_model(ps):
            return ["WRONG"] * len(ps)

        result = compare_models(
            [good_model, bad_model], samples, scorers=[accuracy_like, error_like],
            model_names=["good", "bad"],
        )
        table = {r["metric"]: r for r in result.comparison_table()}
        assert table["accuracy_like"]["winner"] == "good"
        assert table["error_like"]["winner"] == "good"

    def test_performance_table_reports_measured_latency_and_throughput(self):
        import time

        def fast_model(ps):
            time.sleep(0.005)
            return ps

        def slow_model(ps):
            time.sleep(0.03)
            return ps

        samples = [Sample(input=str(i), target=str(i)) for i in range(3)]
        result = compare_models(
            [fast_model, slow_model], samples, scorers=["exact_match"],
            model_names=["fast", "slow"],
            config=RunConfig(track_performance=True),
        )
        table = {r["model"]: r for r in result.performance_table()}
        assert table["fast"]["calls"] == 1
        assert table["slow"]["calls"] == 1
        assert table["fast"]["latency_mean_ms"] < table["slow"]["latency_mean_ms"]
        assert "Performance (measured):" in result.summary()

    def test_performance_table_populates_by_default(self):
        samples = [Sample(input=str(i), target=str(i)) for i in range(2)]
        result = compare_models(
            [lambda ps: ps, lambda ps: ps], samples, scorers=["exact_match"],
            model_names=["a", "b"],
        )
        table = {r["model"]: r for r in result.performance_table()}
        assert table["a"]["calls"] is not None
        assert table["a"]["latency_mean_ms"] is not None
        assert table["b"]["calls"] is not None
        result.summary()  # must not crash with perf populated on every model

    def test_performance_table_is_none_when_opted_out_and_does_not_crash(self):
        samples = [Sample(input=str(i), target=str(i)) for i in range(2)]
        result = compare_models(
            [lambda ps: ps, lambda ps: ps], samples, scorers=["exact_match"],
            model_names=["a", "b"], config=RunConfig(track_performance=False),
        )
        table = {r["model"]: r for r in result.performance_table()}
        assert table["a"]["calls"] is None
        assert table["a"]["latency_mean_ms"] is None
        assert table["b"]["calls"] is None
        result.summary()  # must not crash with perf=None on every model

    def test_per_task_table_distinguishes_opposite_failure_modes(self):
        # a: perfect. b: fails all "trivia". c: fails all "arith" -- opposite
        # failure modes that would blend to the identical 0.5 in the plain
        # comparison_table(), indistinguishable from each other there.
        answers = {"2+2": "4", "3+3": "6", "capital of France": "Paris", "capital of Italy": "Rome"}
        arith = [Sample(input=q, target=a, task="arith") for q, a in list(answers.items())[:2]]
        trivia = [Sample(input=q, target=a, task="trivia") for q, a in list(answers.items())[2:]]
        data = arith + trivia

        def model_a(ps): return [answers[p] for p in ps]
        def model_b(ps): return [answers[p] if p in ("2+2", "3+3") else "WRONG" for p in ps]
        def model_c(ps): return [answers[p] if p in ("capital of France", "capital of Italy") else "WRONG" for p in ps]

        result = compare_models(
            [model_a, model_b, model_c], data, scorers=["exact_match"],
            model_names=["a_perfect", "b_bad_trivia", "c_bad_arith"],
        )
        # Blended table can't tell b and c apart.
        blended = {r["metric"]: r for r in result.comparison_table()}["exact_match"]
        assert blended["b_bad_trivia_score"] == blended["c_bad_arith_score"] == 0.5

        # Per-task table does.
        by_task = {(r["task"], r["metric"]): r for r in result.per_task_table()}
        assert by_task[("arith", "exact_match")]["b_bad_trivia_score"] == 1.0
        assert by_task[("arith", "exact_match")]["c_bad_arith_score"] == 0.0
        assert by_task[("trivia", "exact_match")]["b_bad_trivia_score"] == 0.0
        assert by_task[("trivia", "exact_match")]["c_bad_arith_score"] == 1.0
        assert "Per-task:" in result.summary()

    def test_no_per_task_section_with_a_single_task(self):
        samples = [Sample(input="hello", target="hello"), Sample(input="world", target="world")]
        result = compare_models(
            models=["echo", "echo"], dataset=samples,
            model_names=["a", "b"], scorers=["exact_match"],
        )
        assert "Per-task:" not in result.summary()

    def test_macro_average_differs_from_blended_when_task_sizes_differ(self):
        # "small" task has 1 sample, "big" task has 3 -- unequal N on purpose.
        data = [
            Sample(input="s1", target="X", task="small"),
            Sample(input="b1", target="Y", task="big"),
            Sample(input="b2", target="Y", task="big"),
            Sample(input="b3", target="Y", task="big"),
        ]

        def model_fn(ps):
            # correct on "small", wrong on all of "big"
            return ["X" if p == "s1" else "WRONG" for p in ps]

        result = compare_models(
            [model_fn], data, scorers=["exact_match"], model_names=["m"],
        )
        blended = {r["metric"]: r for r in result.comparison_table()}["exact_match"]
        macro = {r["metric"]: r for r in result.task_macro_average_table()}["exact_match"]

        assert blended["m_score"] == 0.25    # micro: 1 correct out of 4 total samples
        assert macro["m_macro_mean"] == 0.5  # macro: mean(small=1.0, big=0.0), each task counted once
        assert "Macro-average across tasks" in result.summary()

    def test_one_model_failing_entirely_does_not_lose_the_others(self):
        # A model that always crashes used to take down the WHOLE
        # compare_models() call, discarding every other model's
        # already-computed results too.
        samples = [Sample(input="hello", target="hello")]

        def crashing_model(ps):
            raise RuntimeError("simulated total model failure")

        result = compare_models(
            ["echo", crashing_model, "echo"], samples, scorers=["exact_match"],
            model_names=["good_a", "bad", "good_b"],
            config=RunConfig(max_retries=0, retry_delay=0.0),
        )
        assert isinstance(result, CompareResult)
        assert "good_a" in result.runs and "good_b" in result.runs
        assert "bad" not in result.runs
        assert "bad" in result.errors
        assert "simulated total model failure" in result.errors["bad"]
        assert result.has_failures is True
        assert any("FAILED TO RUN ENTIRELY" in w for w in result.coverage_warnings())
        assert "DID NOT RUN" in result.summary()
        # The two models that DID succeed are still fully comparable.
        assert result.winner() in ("good_a", "good_b")

    def test_all_models_failing_still_returns_a_result_not_an_exception(self):
        def crashing_model(ps):
            raise RuntimeError("boom")

        result = compare_models(
            [crashing_model, crashing_model], [Sample(input="x", target="x")],
            model_names=["m1", "m2"], config=RunConfig(max_retries=0, retry_delay=0.0),
        )
        assert result.runs == {}
        assert set(result.errors) == {"m1", "m2"}
        assert result.winner() is None

    def test_coverage_warning_per_task_across_models(self):
        # Two tasks, model "b" only sees a subset via configs= -- limit
        # applies to the WHOLE dataset (both tasks combined), so this
        # creates a real per-task imbalance to detect.
        data = [
            Sample(input="q1", target="a1", task="t1"),
            Sample(input="q2", target="a2", task="t1"),
            Sample(input="q3", target="a3", task="t2"),
            Sample(input="q4", target="a4", task="t2"),
        ]
        result = compare_models(
            ["echo", "echo"], data, scorers=["exact_match"],
            model_names=["a", "b"],
            configs={"b": RunConfig(limit=1)},  # b only ever sees task "t1"'s first sample
        )
        warnings = result.coverage_warnings()
        assert any("t1/exact_match" in w or "t2/exact_match" in w for w in warnings)

    def test_per_metric_and_winner_log_a_warning_on_failures(self, caplog):
        import logging
        samples = [Sample(input="hello", target="hello")]

        def crashing_model(ps):
            raise RuntimeError("boom")

        result = compare_models(
            ["echo", crashing_model], samples, model_names=["good", "bad"],
            scorers=["exact_match"], config=RunConfig(max_retries=0, retry_delay=0.0),
        )
        with caplog.at_level(logging.WARNING):
            result.per_metric()
        assert any("failed to run entirely" in r.message for r in caplog.records)


class TestCompareModelsPerModelAdapterAndAnnotators:
    """compare_models(adapter=, annotators=, extract_with=) shared across all
    models, with model_opts[name] able to override any of them for one
    model only -- the gap flagged in the examples/applications/ and examples/
    notebooks (a shared adapter/annotators used to be silently dropped
    entirely; now it's real and overridable per model). scorers is
    deliberately NOT overridable per-model -- comparison assumes a shared
    metric set across models."""

    @pytest.fixture(autouse=True)
    def _isolated_disk_cache(self, tmp_path, monkeypatch):
        # Every run in this class must actually execute (not silently reuse
        # a cache hit from this file's own real ~/.cache/auditkit, possibly
        # left over from a previous test run) -- these tests assert on
        # side effects (a probe's `seen` dict) that a cache hit would skip.
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    def test_shared_adapter_reaches_every_model(self):
        # Two distinct CallableModel instances with distinct explicit names --
        # closures sharing one bytecode body would otherwise collide under
        # the disk cache's fingerprint (same code object -> same identity),
        # which is a real, separate quirk of auto-named callables, not
        # something this test is about.
        from auditkit.adapter import TemplateAdapter
        from auditkit.model import CallableModel

        seen = {}

        def probe1(prompts):
            seen["m1"] = prompts[0]
            return ["a"]

        def probe2(prompts):
            seen["m2"] = prompts[0]
            return ["a"]

        # Distinct sample text (not reused by the sibling test below) so this
        # run's fingerprint never collides with another test's disk-cache entry.
        data = [Sample(input="q-shared-adapter-test", target="a")]
        compare_models(
            [CallableModel(probe1, name="m1"), CallableModel(probe2, name="m2")],
            data, scorers=["exact_match"],
            model_names=["m1", "m2"],
            adapter=TemplateAdapter("SHARED: {input}"),
        )
        assert seen["m1"] == "SHARED: q-shared-adapter-test"
        assert seen["m2"] == "SHARED: q-shared-adapter-test"

    def test_model_opts_adapter_overrides_shared_adapter_for_one_model_only(self):
        from auditkit.adapter import TemplateAdapter
        from auditkit.model import CallableModel

        seen = {}

        def probe1(prompts):
            seen["m1"] = prompts[0]
            return ["a"]

        def probe2(prompts):
            seen["m2"] = prompts[0]
            return ["a"]

        data = [Sample(input="q-adapter-override-test", target="a")]
        compare_models(
            [CallableModel(probe1, name="m1"), CallableModel(probe2, name="m2")],
            data, scorers=["exact_match"],
            model_names=["m1", "m2"],
            adapter=TemplateAdapter("SHARED: {input}"),
            model_opts={"m2": {"adapter": TemplateAdapter("OVERRIDE: {input}")}},
        )
        assert seen["m1"] == "SHARED: q-adapter-override-test"
        assert seen["m2"] == "OVERRIDE: q-adapter-override-test"

    def test_model_opts_annotators_and_extract_with_override_for_one_model(self):
        from auditkit.annotator import RegexAnnotator

        data = [Sample(input="q-annotator-override-test", target="42")]

        def direct(prompts):
            return ["42"]

        def verbose(prompts):
            return ["FINAL ANSWER: 42"]

        extractor = RegexAnnotator(r"FINAL ANSWER:\s*(.+)", group=1, name="final_answer")

        result = compare_models(
            [direct, verbose], data, scorers=["exact_match"],
            model_names=["direct", "verbose"],
            model_opts={"verbose": {"annotators": [extractor], "extract_with": "final_answer"}},
        )
        assert result.runs["direct"].headline["exact_match"] == 1.0
        assert result.runs["verbose"].headline["exact_match"] == 1.0
        assert result.runs["verbose"].predictions[0].parsed_answer == "42"
        assert result.runs["verbose"].predictions[0].raw_output == "FINAL ANSWER: 42"

    def test_model_opts_scorers_is_not_a_pipeline_override(self):
        # scorers is intentionally NOT popped from model_opts -- comparison
        # assumes one shared metric set. Putting "scorers" in a model's
        # opts dict falls through as a bogus backend kwarg and that model
        # fails (recorded in .errors), it does not silently change what
        # gets scored for that model.
        data = [Sample(input="q-scorers-not-overridable-test", target="hello")]

        def m1(prompts):
            return ["hello"]

        def m2(prompts):
            return ["hello"]

        result = compare_models(
            [m1, m2], data,
            model_names=["m1", "m2"],
            scorers=["exact_match"],
            model_opts={"m2": {"scorers": ["quasi_exact_match"]}},
        )
        assert "exact_match" in result.runs["m1"].headline
        assert "m2" in result.errors
        assert "scorers" in result.errors["m2"]

    def test_model_opts_backend_kwargs_still_forwarded_alongside_pipeline_overrides(self):
        from auditkit.model import AutoModel

        captured = []
        original_resolve = AutoModel.resolve.__func__

        def spy(cls, spec, **opts):
            captured.append(dict(opts))
            clean = {k: v for k, v in opts.items() if k != "made_up_backend_kwarg"}
            return original_resolve(cls, spec, **clean)

        AutoModel.resolve = classmethod(spy)
        try:
            data = [Sample(input="q-backend-kwargs-test", target="a")]
            compare_models(
                [lambda ps: ["a"]], data, scorers=["exact_match"], model_names=["m1"],
                model_opts={"m1": {"annotators": [], "made_up_backend_kwarg": "xyz"}},
            )
        finally:
            AutoModel.resolve = classmethod(original_resolve)

        assert any(
            "made_up_backend_kwarg" in opts and "annotators" not in opts
            for opts in captured
        )


class TestModelTeardown:
    """compare_models frees each self-resolved local model before the next
    loads, so peak GPU memory stays at one model (see Model.unload())."""

    def test_self_resolved_models_are_unloaded(self):
        from auditkit.model import EchoModel
        calls = {"n": 0}
        orig = EchoModel.unload
        EchoModel.unload = lambda self: calls.__setitem__("n", calls["n"] + 1)
        try:
            data = [Sample(input="a", target="a"), Sample(input="b", target="b")]
            compare_models(["echo", "echo"], data, scorers=["exact_match"],
                           model_names=["x", "y"])
            assert calls["n"] == 2                     # both string-spec models freed
        finally:
            EchoModel.unload = orig

    def test_caller_passed_instances_are_not_unloaded(self):
        from auditkit.model import EchoModel
        calls = {"n": 0}
        orig = EchoModel.unload
        EchoModel.unload = lambda self: calls.__setitem__("n", calls["n"] + 1)
        try:
            data = [Sample(input="a", target="a")]
            compare_models([EchoModel(), EchoModel()], data, scorers=["exact_match"],
                           model_names=["x", "y"])
            assert calls["n"] == 0                     # caller owns them -> untouched
        finally:
            EchoModel.unload = orig

    def test_hf_unload_drops_pipeline(self):
        # unload() drops the loaded pipeline (and clears the torch cache, guarded
        # so it's a no-op without torch). Doesn't need transformers installed.
        from auditkit.model.hf_gen import HFGenModel
        m = HFGenModel(model="sshleifer/tiny-gpt2", device="cpu")
        m._pipeline = object()                         # simulate a loaded pipeline
        m.unload()
        assert m._pipeline is None
        m.unload()                                     # idempotent

    def test_base_unload_is_noop(self):
        from auditkit.model import EchoModel
        EchoModel().unload()                           # no error, nothing to free


class TestCompareModelsExperimentNameAndTags:
    """Regression: experiment_name=/tags= aren't real compare_models() params
    without this, so they were silently swept into **opts and forwarded to
    AutoModel.resolve() as model-constructor kwargs -- for a real hosted-API
    backend, landing in the live HTTP request body and getting rejected by
    the provider (a real, confirmed Groq 400: "property 'experiment_name' is
    unsupported")."""

    def test_experiment_name_does_not_leak_into_model_construction(self, monkeypatch):
        """Before experiment_name/tags were real params, compare_models()
        swept them into **opts and forwarded them to AutoModel.resolve() --
        i.e. into each model's own constructor kwargs. For "echo"/callables
        that's harmless (the spec is special-cased before **opts is ever
        used), but for a real backend like GroqModel it meant landing in the
        live HTTP request body. Spy directly on AutoModel.resolve() to prove
        experiment_name/tags never reach it, regardless of which backend."""
        import auditkit.model_compare as mc_mod
        from auditkit.model import EchoModel

        captured_kwargs = []
        real_resolve = mc_mod.AutoModel.resolve

        def spy_resolve(spec, **kwargs):
            captured_kwargs.append(kwargs)
            return EchoModel()

        monkeypatch.setattr(mc_mod.AutoModel, "resolve", staticmethod(spy_resolve))

        data = [Sample(input="a", target="a")]
        compare_models(
            ["groq:fake-model-a", "groq:fake-model-b"], data, scorers=["exact_match"],
            model_names=["a", "b"],
            experiment_name="my_experiment", tags=["t1"],
        )
        assert captured_kwargs  # the spy was actually called
        for kwargs in captured_kwargs:
            assert "experiment_name" not in kwargs
            assert "tags" not in kwargs

    def test_experiment_name_reaches_each_underlying_run(self):
        data = [Sample(input="a", target="a")]
        cmp = compare_models(
            ["echo", "echo"], data, scorers=["exact_match"],
            model_names=["model_a", "model_b"],
            experiment_name="my_experiment", tags=["application"],
        )
        assert cmp.runs["model_a"].experiment_name == "my_experiment:model_a"
        assert cmp.runs["model_b"].experiment_name == "my_experiment:model_b"
        assert cmp.runs["model_a"].tags == ["application"]

    def test_no_experiment_name_still_works(self):
        data = [Sample(input="a", target="a")]
        cmp = compare_models(["echo", "echo"], data, scorers=["exact_match"],
                             model_names=["a", "b"])
        assert cmp.runs["a"].experiment_name is None
        assert cmp.runs["a"].tags == []
