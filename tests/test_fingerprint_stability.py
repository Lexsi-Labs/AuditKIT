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
