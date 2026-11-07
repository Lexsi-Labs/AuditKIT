from __future__ import annotations

from auditkit.metrics.judge import LLMJudge


def _j(scale=(0, 1)):
    return LLMJudge(judge_model=lambda xs: ["x"] * len(xs), scale=scale)


def test_fraction_is_normalised_not_clamped():
    assert _j()._parse("SCORE: 8/10").value == 0.8
    assert _j((1, 5))._parse("SCORE: 4/5").value == 0.75


def test_fallback_reads_the_last_number_of_the_last_line():
    assert _j()._parse("has 3 factual errors, so overall I rate it 0.1").value == 0.1


def test_plain_scores_unchanged():
    assert _j((1, 5))._parse("SCORE: 4").value == 0.75
    assert _j()._parse("SCORE: 0.5").value == 0.5


def test_out_of_scale_value_is_unknown():
    assert _j()._parse("SCORE: 8").metadata.get("unknown")
    assert _j()._parse("SCORE: 12/10").metadata.get("unknown")
