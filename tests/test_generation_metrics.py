"""Tests for T1 generation quality metrics (BLEU, ROUGE-L, chrF, WER, Perplexity, BERTScore)."""

from __future__ import annotations

import pytest

from auditkit.sample import Sample
from auditkit.metrics.generation import Bleu, RogueL, ChrF, WordErrorRate, Perplexity, BertScore
from auditkit.errors import ExtraNotInstalled


S = Sample(input="x", target="the cat sat on the mat")


def _importable(module_name: str) -> bool:
    try:
        __import__(module_name)
    except ImportError:
        return False
    return True


class TestBleu:
    def test_identical(self):
        assert Bleu().score(S, "the cat sat on the mat").value == 1.0

    def test_no_match(self):
        assert Bleu().score(S, "xyz").value < 0.01

    def test_partial(self):
        v = Bleu().score(S, "the cat sat").value
        assert 0.0 < v < 1.0

    def test_short_ref(self):
        short = Sample(input="x", target="short")
        v = Bleu().score(short, "short phrase here").value
        assert 0.0 < v < 1.0

    def test_name(self):
        assert Bleu().name == "bleu"

    def test_requires_target(self):
        assert "target" in Bleu().required_fields


class TestRogueL:
    def test_identical(self):
        assert RogueL().score(S, "the cat sat on the mat").value == 1.0

    def test_partial_lcs(self):
        v = RogueL().score(S, "the cat sat").value
        assert 0.5 < v < 1.0

    def test_no_match(self):
        assert RogueL().score(S, "xyz").value < 0.01

    def test_name(self):
        assert RogueL().name == "rouge_l"


class TestChrF:
    def test_identical(self):
        assert ChrF().score(S, "the cat sat on the mat").value >= 0.99

    def test_no_match(self):
        assert ChrF().score(S, "xyz").value < 0.1

    def test_default_beta(self):
        c = ChrF()
        assert c._beta == 1.0

    def test_name(self):
        assert ChrF().name == "chrf"

    def test_identical_short_strings_score_1(self):
        # Previously: ChrF didn't cap n-gram order by string length like its
        # sibling Bleu does, so a short identical pair still averaged in
        # n=3..6 character n-grams that can't exist at all in a 2-char
        # string, understating a perfect match (scored 0.333, not 1.0).
        from auditkit.sample import Sample
        s = Sample(input="x", target="hi")
        assert ChrF().score(s, "hi").value == 1.0


class TestWordErrorRate:
    def test_identical(self):
        assert WordErrorRate().score(S, "the cat sat on the mat").value == 1.0

    def test_different(self):
        v = WordErrorRate().score(S, "the dog ran").value
        assert 0.0 <= v < 1.0

    def test_name(self):
        assert WordErrorRate().name == "wer"


class TestPerplexity:
    def test_extra_not_installed(self):
        if _importable("transformers"):
            pytest.skip("transformers is installed in this environment -- nothing to assert")
        with pytest.raises(ExtraNotInstalled):
            Perplexity().score(S, "hello world")

    def test_name(self):
        assert Perplexity().name == "perplexity"


class TestBertScore:
    def test_extra_not_installed(self):
        if _importable("bert_score"):
            pytest.skip("bert-score is installed in this environment -- nothing to assert")
        with pytest.raises(ExtraNotInstalled):
            BertScore().score(S, "hello world")

    def test_name(self):
        assert BertScore().name == "bert_score"
