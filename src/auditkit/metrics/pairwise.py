"""Pairwise comparison and preference metrics."""
from __future__ import annotations

from typing import Any, Optional

from auditkit.metric import Metric
from auditkit.registry import METRICS
from auditkit.sample import Sample
from auditkit.score import Score
from auditkit.types import Direction, ScoreKind


def _token_overlap(a: str, b: str) -> float:
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    if not a_tokens and not b_tokens:
        return 1.0
    return len(a_tokens & b_tokens) / max(len(a_tokens), len(b_tokens))


def _lookup(context: Any, source: Optional[str], field: str) -> list:
    """Read *field* (a list) out of *context*, optionally namespaced under an
    annotator's name.

    Runner.annotate() always nests an annotator's output one level down --
    ``context[annotator.name] = annotator.annotate(...)`` -- never at the top
    level (see runner.py). So a bare ``context.get(field, [])`` can only ever
    be populated by a caller building ``context`` by hand, never by the
    normal ``annotators=``/``extract_with=`` pipeline. ``source=`` names the
    annotator whose nested dict actually holds *field*, matching the same
    "look up a named thing in context" pattern ``extracted_by`` already uses
    in ``Runner.score_one()``. ``source=None`` preserves the original
    bare-top-level-key lookup for backward compatibility.
    """
    ctx = context or {}
    if source:
        ctx = ctx.get(source, {})
        if not isinstance(ctx, dict):
            return []
    value = ctx.get(field, [])
    return value if isinstance(value, list) else []


@METRICS.register("win_rate")
class WinRate(Metric):
    name = "win_rate"
    kind = ScoreKind.BENCHMARK
    direction = Direction.MAXIMIZE
    is_deterministic = True
    required_fields = frozenset({"target"})

    def __init__(self, comparator: str = "exact", source: Optional[str] = None) -> None:
        self._comparator = comparator
        self._source = source

    def identity(self) -> dict:
        return {"name": self.name, "comparator": self._comparator, "source": self._source}

    def score(self, sample: Sample, output: str, context: Any = None) -> Score:
        candidates = _lookup(context, self._source, "candidates")
        if not candidates:
            return Score(name=self.name, value=0.0, kind=self.kind)

        target = sample.target or ""
        wins = 0
        for candidate in candidates:
            output_score = self._compare(output, target)
            candidate_score = self._compare(candidate, target)
            if output_score > candidate_score:
                wins += 1

        return Score(name=self.name, value=wins / len(candidates), kind=self.kind)

    def _compare(self, a: str, b: str) -> float:
        if self._comparator == "exact":
            return 1.0 if a.strip() == b.strip() else 0.0
        return _token_overlap(a, b)


@METRICS.register("elo_score")
class EloScore(Metric):
    name = "elo_score"
    kind = ScoreKind.BENCHMARK
    direction = Direction.MAXIMIZE
    is_deterministic = True
    required_fields = frozenset({"target"})

    def __init__(self, k: int = 32, initial_rating: float = 1000.0, source: Optional[str] = None) -> None:
        self._k = k
        self._initial_rating = initial_rating
        self._source = source

    def identity(self) -> dict:
        return {
            "name": self.name, "k": self._k, "initial_rating": self._initial_rating,
            "source": self._source,
        }

    def score(self, sample: Sample, output: str, context: Any = None) -> Score:
        pairwise_results = _lookup(context, self._source, "pairwise_results")

        if pairwise_results:
            ratings: dict[str, float] = {}
            for result in pairwise_results:
                w = result.get("winner", "")
                l = result.get("loser", "")
                if w not in ratings:
                    ratings[w] = self._initial_rating
                if l not in ratings:
                    ratings[l] = self._initial_rating
                expected_w = 1.0 / (1.0 + 10.0 ** ((ratings[l] - ratings[w]) / 400.0))
                ratings[w] += self._k * (1.0 - expected_w)
                ratings[l] += self._k * (0.0 - (1.0 - expected_w))
            rating = sum(ratings.values()) / len(ratings) if ratings else self._initial_rating
        else:
            target = sample.target or ""
            overlap = _token_overlap(output, target)
            rating = self._initial_rating + self._k * (overlap - 0.5)

        rating = max(0.0, min(2000.0, rating))
        return Score(name=self.name, value=rating, kind=self.kind)


@METRICS.register("preference_accuracy")
class PreferenceAccuracy(Metric):
    name = "preference_accuracy"
    kind = ScoreKind.BENCHMARK
    direction = Direction.MAXIMIZE
    is_deterministic = True
    required_fields = frozenset({"target"})

    def __init__(self, source: Optional[str] = None) -> None:
        self._source = source

    def identity(self) -> dict:
        return {"name": self.name, "source": self._source}

    def score(self, sample: Sample, output: str, context: Any = None) -> Score:
        preference_data = _lookup(context, self._source, "preference_data")
        if not preference_data:
            return Score(name=self.name, value=0.5, kind=self.kind)

        correct = 0
        for pair in preference_data:
            chosen = pair.get("chosen", "")
            rejected = pair.get("rejected", "")
            chosen_overlap = _token_overlap(output, chosen)
            rejected_overlap = _token_overlap(output, rejected)
            if chosen_overlap > rejected_overlap:
                correct += 1

        accuracy = correct / len(preference_data)
        return Score(name=self.name, value=accuracy, kind=self.kind)
