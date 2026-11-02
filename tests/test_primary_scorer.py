"""``correct`` no longer depends on the order of scorers when ``primary=`` is given."""
from __future__ import annotations

import itertools

import pytest

import auditkit as ak
from auditkit.runspec import RunSpec

SAMPLES = [ak.Sample(input="q1", target="Paris", id="1"), ak.Sample(input="q2", target="Rome", id="2")]


def reply(prompts):
    return ["Paris" if "q1" in p else "Rome is nice" for p in prompts]


def run(scorers, **kw):
    return ak.evaluate(SAMPLES, model=reply, scorers=scorers, **kw)


def correct(result):
    return {p.sample_id: p.correct for p in result.predictions}


def score(result):
    return {p.sample_id: p.score for p in result.predictions}


class TestOrderIndependence:
    @pytest.mark.parametrize("order", list(itertools.permutations(["exact_match", "f1_score", "token_overlap"])))
    def test_primary_makes_correct_independent_of_order(self, order):
        r = run(list(order), primary="exact_match")
        assert correct(r) == {"1": True, "2": False}
        assert score(r) == {"1": 1.0, "2": 0.0}

    def test_default_follows_the_first_scorer(self):
        from auditkit.registry import METRICS
        from auditkit.runner import Runner

        sample = SAMPLES[1]  # target "Rome", reply "Rome is nice": exact 0.0, f1 0.5
        exact, f1 = METRICS.get("exact_match")(), METRICS.get("f1_score")()
        _, first_exact = Runner().score_one([exact, f1], sample, "Rome is nice", {}, "r")
        _, first_f1 = Runner().score_one([f1, exact], sample, "Rome is nice", {}, "r")
        assert first_exact.score == 0.0 and first_f1.score == pytest.approx(0.5)  # order decides, as documented
        _, named = Runner().score_one([f1, exact], sample, "Rome is nice", {}, "r", primary="exact_match")
        assert named.score == 0.0 and named.correct is False

    def test_primary_picks_a_later_scorer(self):
        r = run(["exact_match", "f1_score"], primary="f1_score")
        assert score(r)["2"] == pytest.approx(0.5) and correct(r)["2"] is False
        assert score(r)["1"] == 1.0 and correct(r)["1"] is True

    def test_all_scores_are_still_recorded(self):
        r = run(["exact_match", "f1_score"], primary="f1_score")
        names = [s["name"] for s in r.predictions[1].metadata["scores"]]
        assert sorted(names) == ["exact_match", "f1_score"]

    def test_unknown_primary_raises_before_running(self):
        with pytest.raises(ValueError, match="not one of the scorers"):
            run(["exact_match"], primary="nope")


class TestFingerprint:
    def _spec(self, names, primary=None):
        from auditkit.registry import METRICS
        ms = [METRICS.get(n)() for n in names]
        return RunSpec(scenario=_Scn(), model=ak.AutoModel.resolve(reply), adapter=ak.GenerationAdapter(),
                       metrics=ms, primary=primary)

    def test_default_fingerprint_is_unchanged_and_order_free(self):
        # by design (tests/agentic_suite): scorer order alone does not change the fingerprint
        assert self._spec(["exact_match", "f1_score"]).fingerprint() == self._spec(["f1_score", "exact_match"]).fingerprint()

    def test_explicit_primary_is_part_of_the_identity(self):
        a = self._spec(["exact_match", "f1_score"], primary="f1_score").fingerprint()
        b = self._spec(["exact_match", "f1_score"], primary="exact_match").fingerprint()
        c = self._spec(["exact_match", "f1_score"]).fingerprint()
        assert len({a, b, c}) == 3

    def test_same_primary_any_order_same_fingerprint(self):
        a = self._spec(["exact_match", "f1_score"], primary="exact_match").fingerprint()
        b = self._spec(["f1_score", "exact_match"], primary="exact_match").fingerprint()
        assert a == b


class _Scn(ak.Scenario):
    def __init__(self):
        super().__init__(name="s")

    def samples(self):
        return iter(SAMPLES)


class TestCrashAndInapplicable:
    def test_failed_primary_makes_the_sample_incorrect(self):
        from auditkit.metric import Metric
        from auditkit.types import Direction
        from auditkit.score import Score

        class Boom(Metric):
            name = "boom"
            direction = Direction.MAXIMIZE

            def score(self, sample, output, context=None):
                raise RuntimeError("x")

        r = ak.evaluate(SAMPLES, model=reply, scorers=["exact_match", Boom()], primary="boom")
        assert correct(r) == {"1": False, "2": False}

    def test_default_primary_unchanged_when_first_scorer_is_the_one_that_decides(self):
        a = run(["exact_match"])
        b = run(["exact_match"], primary="exact_match")
        assert correct(a) == correct(b) and score(a) == score(b)
