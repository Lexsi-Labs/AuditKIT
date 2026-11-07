"""A curly apostrophe must not hide a refusal from the keyword checks."""
from __future__ import annotations

import pytest

from auditkit._text import fold_apostrophes
from auditkit.metrics.rag_stress import _looks_like_refusal
from auditkit.redteam.detectors.builtin import RefusalDetector


@pytest.mark.parametrize("curly", ["’", "‘", "ʼ", "′"])
def test_fold_straightens_each_look_alike_and_nothing_else(curly):
    assert fold_apostrophes(f"can{curly}t") == "can't"
    assert fold_apostrophes('Say "no" — it\'s ok.') == 'Say "no" — it\'s ok.'
    assert fold_apostrophes(None) == ""


@pytest.mark.parametrize("text", ["I can’t help with that.", "I’m sorry, but I can’t do that.",
                                  "I canʼt help with that."])
def test_refusal_detector_reads_curly_apostrophes(text):
    assert RefusalDetector().detect("p", text).detected is True


def test_refusal_detector_is_unchanged_for_ascii_and_for_answers():
    d = RefusalDetector()
    assert d.detect("p", "I can't help with that.").detected is True
    assert d.detect("p", "Here is the answer you asked for.").detected is False


@pytest.mark.parametrize("marker_text", ["I don’t have that information.", "I’m unable to answer that.",
                                         "I’m not able to say."])
def test_rag_stress_reads_curly_apostrophes(marker_text):
    assert _looks_like_refusal(marker_text) is True


def test_rag_stress_is_unchanged_for_a_real_answer_that_mentions_a_refusal_later():
    assert _looks_like_refusal("Paris is the capital. Some say I don't have to explain.") is False
