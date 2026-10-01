"""Regression tests: a run's fingerprint must not depend on Sample.id mutation.

Runner mutates a sample's `.id` in place the first time it's scored
(`if sample.id is None: sample.id = str(index)`). ListScenario's
auto-generated name used to be derived from `(s.id, s.input, s.target)`, so
re-running the exact same Sample objects a second time -- a completely
natural pattern -- produced a *different* fingerprint for identical content,
because the first run had already mutated `.id`. This defeated DiskCache's
whole purpose: a second, logically-identical run would make a real (wasted)
model call instead of hitting the cache.
"""
from __future__ import annotations

from auditkit.sample import Sample
from auditkit.scenario import ListScenario


class TestListScenarioNameStability:
    def test_name_is_unaffected_by_id_mutation(self):
        s = Sample(input="hi", target="hi")
        name_before = ListScenario([s]).name

        # simulate exactly what Runner does after scoring a sample once
        s.id = "0"

        name_after = ListScenario([s]).name
        assert name_before == name_after

    def test_reusing_the_same_samples_across_two_evaluate_calls_hits_the_cache(self):
        import auditkit as ak
        from auditkit.cache import DiskCache
        from auditkit.runspec import RunConfig

        DiskCache().clear()
        calls = []
        model = lambda prompts: (calls.append(prompts), ["hi" for _ in prompts])[1]

        # concurrency=1: default concurrency=8 makes Runner._chunk() split 1
        # request into 8 chunks (7 empty), and execute() calls generate() on
        # every chunk including empty ones -- a separate, unrelated waste bug
        # that would otherwise inflate the first call's count and obscure
        # what this test is actually checking (does the *second* call add
        # any further real calls beyond whatever the first one made).
        cfg = RunConfig(concurrency=1)
        samples = [Sample(input="hi", target="hi")]  # id left unset, reused below
        r1 = ak.evaluate(samples, model=model, config=cfg)
        calls_after_first_run = len(calls)
        r2 = ak.evaluate(samples, model=model, config=cfg)

        assert r1.fingerprint == r2.fingerprint
        assert len(calls) == calls_after_first_run  # second call added no further real calls


class TestListScenarioActualOutputInFingerprint:
    """Regression test: the "generate once, score many times" flow
    (ak.generate() -> model="precomputed") depends on Sample.actual_output,
    but ListScenario's auto-generated name used to hash only
    (input, target, choices, retrieval_context) -- never actual_output.
    Two datasets sharing the same input/target but carrying DIFFERENT
    generated text (e.g. two different models' real outputs) collided onto
    the identical fingerprint, silently returning the first call's cached
    result for the second, regardless of what the second's actual_output
    actually was. Confirmed live: scoring a correct answer (1.0) then a
    wrong one (should be 0.0) against the same input/target silently
    returned the cached 1.0 for the wrong one too.
    """

    def test_different_actual_output_gets_a_different_fingerprint(self):
        s1 = Sample(input="q", target="positive", actual_output="positive", id="0")
        s2 = Sample(input="q", target="positive", actual_output="negative", id="0")
        assert ListScenario([s1]).name != ListScenario([s2]).name

    def test_precomputed_scoring_is_not_silently_stale(self):
        import auditkit as ak
        from auditkit.cache import DiskCache

        DiskCache().clear()
        correct = [Sample(input="q", target="positive", actual_output="positive", id="0")]
        wrong = [Sample(input="q", target="positive", actual_output="negative", id="0")]

        r_correct = ak.evaluate(correct, model="precomputed", scorers=["exact_match"])
        r_wrong = ak.evaluate(wrong, model="precomputed", scorers=["exact_match"])

        assert r_correct.fingerprint != r_wrong.fingerprint
        assert r_correct.headline["exact_match"] == 1.0
        assert r_wrong.headline["exact_match"] == 0.0


class TestUserSuppliedIdInName:
    """A caller-supplied Sample.id is part of dataset identity; an
    auto-assigned str(index) id is not. Two datasets differing only in a user
    id must hash apart (else the second evaluate() is served the first run's
    cached predictions under the wrong sample_ids), while an id-less re-run of
    the same reused objects -- whose ids the Runner mutates to str(index) in
    place -- must still hit the cache.
    """

    def test_user_supplied_id_changes_the_name(self):
        a = ListScenario([Sample(input="x", target="y", id="q1")]).name
        b = ListScenario([Sample(input="x", target="y", id="q2")]).name
        assert a != b

    def test_index_valued_id_is_treated_as_auto_assigned(self):
        # An id equal to its positional index is indistinguishable from what
        # the Runner writes into an unset id, so it must not change the name.
        plain = ListScenario([Sample(input="x", target="y")]).name
        indexed = ListScenario([Sample(input="x", target="y", id="0")]).name
        assert plain == indexed

    def test_idless_rerun_after_runner_mutation_still_matches(self):
        s0 = Sample(input="a", target="a")
        s1 = Sample(input="b", target="b")
        before = ListScenario([s0, s1]).name
        s0.id, s1.id = "0", "1"  # what Runner writes into unset ids
        after = ListScenario([s0, s1]).name
        assert before == after


class TestTaskAndKindInName:
    """Sample.task and Sample.kind label every cached prediction
    (Prediction.task = task or kind), so two datasets differing only in them must
    hash apart -- else the second evaluate() is served the first run's
    predictions under the first run's labels. Defaults add nothing to the hash,
    so plain datasets keep their exact name.
    """

    def test_task_label_changes_the_name(self):
        a = ListScenario([Sample(input="x", target="y", task="suite-a")]).name
        b = ListScenario([Sample(input="x", target="y", task="suite-b")]).name
        assert a != b

    def test_non_default_kind_changes_the_name(self):
        from auditkit.types import TaskKind
        plain = ListScenario([Sample(input="x", target="y")]).name
        agent = ListScenario([Sample(input="x", target="y", kind=TaskKind.AGENT)]).name
        assert plain != agent

    def test_defaults_keep_the_existing_name(self):
        from auditkit.types import TaskKind
        plain = ListScenario([Sample(input="x", target="y")]).name
        explicit = ListScenario([Sample(input="x", target="y", task="", kind=TaskKind.GENERATIVE)]).name
        assert plain == explicit

    def test_cached_run_carries_its_own_task_label(self, tmp_path, monkeypatch):
        import auditkit as ak
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        mk = lambda t: [Sample(input="q", target="a", actual_output="a", task=t)]  # noqa: E731
        r1 = ak.evaluate(mk("suite-a"), model="precomputed", scorers=["exact_match"])
        r2 = ak.evaluate(mk("suite-b"), model="precomputed", scorers=["exact_match"])
        assert r1.fingerprint != r2.fingerprint
        assert r2.predictions[0].task == "suite-b"


class TestAuditkitVersionInFingerprint:
    """Upgrading auditkit (which may change how a metric scores) must shift
    the fingerprint, so a cached score from the old code is never replayed."""

    def test_version_changes_the_fingerprint(self, monkeypatch):
        import auditkit
        from auditkit.runspec import RunSpec

        class _M:
            name = "echo"

        class _A:
            method = "gen"

        rs = RunSpec(scenario=ListScenario([Sample(input="x", target="y")]),
                     model=_M(), adapter=_A(), metrics=[])
        monkeypatch.setattr(auditkit, "__version__", "0.0.0-test-a")
        fp_a = rs.fingerprint()
        monkeypatch.setattr(auditkit, "__version__", "0.0.0-test-b")
        fp_b = rs.fingerprint()
        assert fp_a != fp_b
