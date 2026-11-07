"""LLMJudge option labels in scripts with combining marks keep their vowel signs, so two labels do not collapse."""

from __future__ import annotations

from auditkit.metrics.judge import _norm


def test_labels_that_differ_by_a_vowel_sign_stay_distinct():
    assert _norm("दान") != _norm("दिन")           # Hindi "charity" and "day"
    assert _norm("काम") != _norm("कम")            # "work" and "less"


def test_punctuation_is_still_stripped_and_case_folded():
    assert _norm(" Yes! ") == "yes"
    assert _norm("दिन.") == _norm("दिन")
