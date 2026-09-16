"""Tests for toxicity / representation / bias metrics."""

from __future__ import annotations

from auditkit.sample import Sample
from auditkit.metrics.toxicity import ToxicityScore, RepresentationSkew, HateSpeechScore
from auditkit.metrics.judge import BiasJudge
from auditkit.model import Generated, Result_
from auditkit.types import Direction


S = Sample(input="x", target="y")


class FakeJudge:
    """A judge model that returns canned text (offline)."""

    name = "fake-judge"

    def __init__(self, reply: str):
        self.reply = reply

    def generate(self, requests):
        return [Result_(completions=[Generated(text=self.reply)]) for _ in requests]


class TestToxicityScore:
    def test_clean_output(self):
        # Real classifier (default use_model=True) never returns exactly
        # 1.0 -- it's a probability, not a rule match.
        assert ToxicityScore().score(S, "this is a nice message").value > 0.99
    def test_toxic_output(self):
        v = ToxicityScore().score(S, "you are a stupid idiot").value
        assert v < 1.0
    def test_custom_blacklist(self):
        # blacklist= only takes effect with use_model=False -- the default
        # model path ignores it (documented on ToxicityScore).
        v = ToxicityScore(blacklist=["bad"], use_model=False).score(S, "this is bad").value
        assert v < 1.0
    def test_name(self):
        assert ToxicityScore().name == "toxicity_score"


class TestRepresentationSkew:
    def test_balanced_is_zero(self):
        # men and women mentioned equally -> no skew.
        assert RepresentationSkew().score(S, "he and she are both here").value == 0.0

    def test_one_sided_is_maximal(self):
        # THE regression the rewrite fixes: "he him his" is all-male (three
        # distinct tokens). The old bias_score scored this 0.9 ("clean"),
        # indistinguishable from a balanced mention; it must now be 1.0.
        assert RepresentationSkew().score(S, "he him his").value == 1.0

    def test_single_repeated_token_still_one_sided(self):
        assert RepresentationSkew().score(S, "he he he he he").value == 1.0

    def test_no_demographic_terms_is_zero(self):
        assert RepresentationSkew().score(S, "the cat sat on the mat").value == 0.0

    def test_balanced_distinct_from_one_sided(self):
        skew = RepresentationSkew()
        balanced = skew.score(S, "he and she are both here").value
        one_sided = skew.score(S, "he him his").value
        assert balanced < one_sided        # 0.0 < 1.0 -- the whole point

    def test_content_bias_is_invisible_to_representation(self):
        # Honest limitation, locked as a test: a blatantly sexist sentence that
        # mentions men and women equally scores 0.0 (balanced) -- representation
        # balance is not fairness. This is exactly why BiasJudge exists.
        v = RepresentationSkew().score(
            S, "women are too emotional to be good engineers, while men are natural leaders"
        ).value
        assert v == 0.0

    def test_per_axis_metadata(self):
        s = RepresentationSkew().score(S, "he him his")
        assert s.metadata["axes"] == {"gender": 1.0}

    def test_name_and_direction(self):
        m = RepresentationSkew()
        assert m.name == "representation_skew"
        assert m.direction == Direction.MINIMIZE


class TestBiasJudge:
    def test_biased_content_scores_high(self):
        reply = ("- women are too emotional to lead [BIASED] - gender generalization\n"
                 "- men are natural leaders [BIASED] - gender generalization\n"
                 "SCORE: 1.0")
        v = BiasJudge(judge_model=FakeJudge(reply)).score(
            S, "women are too emotional to lead, while men are natural leaders").value
        assert v == 1.0

    def test_fair_content_scores_zero(self):
        reply = "- the team is diverse and skilled [FAIR] - no prejudice\nSCORE: 0.0"
        v = BiasJudge(judge_model=FakeJudge(reply)).score(S, "the team is diverse and skilled").value
        assert v == 0.0

    def test_no_opinions_scores_zero(self):
        reply = "NO OPINIONS\nSCORE: 0.0"
        v = BiasJudge(judge_model=FakeJudge(reply)).score(S, "the CEO was appointed in 2021").value
        assert v == 0.0

    def test_partial_fraction(self):
        reply = ("- women can't do math [BIASED]\n- the sky is blue [FAIR]\nSCORE: 0.5")
        v = BiasJudge(judge_model=FakeJudge(reply)).score(S, "...").value
        assert v == 0.5

    def test_reason_keeps_breakdown(self):
        reply = "- women can't lead [BIASED] - generalization\nSCORE: 1.0"
        s = BiasJudge(judge_model=FakeJudge(reply)).score(S, "...")
        assert "BIASED" in s.reason

    def test_unknown_escape(self):
        s = BiasJudge(judge_model=FakeJudge("I could not tell")).score(S, "...")
        assert s.metadata.get("unknown") is True

    def test_name_and_direction(self):
        m = BiasJudge(judge_model=FakeJudge("SCORE: 0.0"))
        assert m.name == "bias_judge"
        assert m.direction == Direction.MINIMIZE


class TestHateSpeechScore:
    def test_clean(self):
        v = HateSpeechScore().score(S, "nice message").value
        assert v > 0.5
    def test_name(self):
        assert HateSpeechScore().name == "hate_speech_score"
