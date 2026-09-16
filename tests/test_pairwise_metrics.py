"""Tests for pairwise/preference metrics."""

from __future__ import annotations

from typing import Any

from auditkit.sample import Sample
from auditkit.metrics.pairwise import WinRate, EloScore, PreferenceAccuracy


S = Sample(input="x", target="good answer")


class TestWinRate:
    def test_wins_all(self):
        v = WinRate().score(S, "good answer", {"candidates": ["bad", "worse"]}).value
        assert v == 1.0
    def test_wins_none(self):
        v = WinRate().score(S, "bad answer", {"candidates": ["good", "better"]}).value
        assert v == 0.0
    def test_no_candidates(self):
        v = WinRate().score(S, "good answer").value
        assert v == 0.0
    def test_name(self):
        assert WinRate().name == "win_rate"


class TestEloScore:
    def test_basic(self):
        v = EloScore().score(S, "good answer").value
        assert 0 <= v <= 2000
    def test_name(self):
        assert EloScore().name == "elo_score"


class TestPreferenceAccuracy:
    def test_no_data(self):
        v = PreferenceAccuracy().score(S, "answer").value
        assert v == 0.5
    def test_name(self):
        assert PreferenceAccuracy().name == "preference_accuracy"


# -- source= : the annotator-namespace lookup ---------------------------------
#
# Runner.annotate() nests every annotator's output one level down, under the
# annotator's own name (context[annotator.name] = ...), never at the top
# level -- so a bare context.get("candidates", []) can never be reached by
# the real annotators=/extract_with= pipeline. source= names the annotator
# whose nested dict actually holds the field these metrics need.

class TestWinRateSource:
    def test_source_finds_nested_candidates(self):
        ctx = {"my_annotator": {"candidates": ["bad", "worse"]}}
        v = WinRate(source="my_annotator").score(S, "good answer", ctx).value
        assert v == 1.0

    def test_bare_key_ignored_once_source_is_set(self):
        # candidates sits at the top level, but source= points elsewhere --
        # must not fall back to the bare key (that would defeat the point of
        # namespacing: two annotators' outputs could otherwise collide).
        ctx = {"candidates": ["bad", "worse"], "other_annotator": {}}
        v = WinRate(source="other_annotator").score(S, "good answer", ctx).value
        assert v == 0.0

    def test_default_source_none_keeps_legacy_bare_key_behavior(self):
        v = WinRate().score(S, "good answer", {"candidates": ["bad", "worse"]}).value
        assert v == 1.0

    def test_identity_includes_source(self):
        assert WinRate(source="my_annotator").identity()["source"] == "my_annotator"
        assert WinRate().identity()["source"] is None

    def test_end_to_end_through_a_real_runner_shaped_context(self):
        # Reproduces exactly what Runner.annotate() produces for a real
        # Annotator, rather than a hand-built context dict.
        from auditkit.annotator import Annotator

        class CandidatesAnnotator(Annotator):
            name = "candidates_annotator"

            def annotate(self, sample: Sample, results: list) -> dict[str, Any]:
                return {"candidates": ["bad", "worse"]}

        annotator = CandidatesAnnotator()
        # This is exactly Runner.annotate()'s own construction (runner.py):
        # context[annotator.name] = annotator.annotate(sample, results)
        context = {annotator.name: annotator.annotate(S, [])}

        v = WinRate(source="candidates_annotator").score(S, "good answer", context).value
        assert v == 1.0


class TestEloScoreSource:
    def test_source_finds_nested_pairwise_results(self):
        # A single match between two equally-rated (both start at 1000.0)
        # players averages back to exactly 1000.0 -- distinguishable from
        # the token-overlap fallback branch, which for this exact
        # (output, target) pair gives 1016.0 (confirmed directly). Landing
        # on 1000.0 here proves source= actually located the nested
        # pairwise_results and took the real Elo-update branch, not the
        # fallback that fires when pairwise_results can't be found at all.
        ctx = {"judge": {"pairwise_results": [{"winner": "a", "loser": "b"}]}}
        v = EloScore(source="judge").score(S, "good answer", ctx).value
        assert v == 1000.0
        fallback_v = EloScore(source="wrong_name").score(S, "good answer", ctx).value
        assert fallback_v == 1016.0

    def test_identity_includes_source(self):
        assert EloScore(source="judge").identity()["source"] == "judge"


class TestPreferenceAccuracySource:
    def test_source_finds_nested_preference_data(self):
        ctx = {"judge": {"preference_data": [{"chosen": "good answer", "rejected": "bad"}]}}
        v = PreferenceAccuracy(source="judge").score(S, "good answer", ctx).value
        assert v == 1.0

    def test_identity_includes_source(self):
        assert PreferenceAccuracy(source="judge").identity()["source"] == "judge"
