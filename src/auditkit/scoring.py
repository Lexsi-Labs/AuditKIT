from __future__ import annotations

from typing import Any

from auditkit.metric import Metric
from auditkit.sample import Sample
from auditkit.score import Score
from auditkit.types import Direction


class ScoreGate(Metric):
    """Wraps a Metric with weight and threshold."""

    def __init__(self, metric: Metric, weight: float = 1.0, threshold: float | None = None) -> None:
        self._metric = metric
        self.weight = weight
        self._threshold = threshold

    @property
    def name(self) -> str:
        return f"gate({self._metric.name})"

    def identity(self) -> dict:
        """Includes weight/threshold (previously unhashed -- changing either
        and rerunning would silently reuse a stale result scored under the
        old settings) and delegates to the wrapped metric's own identity()
        if it has one, rather than just its .name -- otherwise a ScoreGate
        wrapping e.g. an LLMJudge would lose that judge's prompt/choices
        from the fingerprint too, the same gap this class had for itself."""
        wrapped_identity = getattr(self._metric, "identity", None)
        wrapped = wrapped_identity() if callable(wrapped_identity) else {"name": self._metric.name}
        return {"name": self.name, "weight": self.weight, "threshold": self._threshold, "wrapped": wrapped}

    @property
    def kind(self):
        return self._metric.kind

    @property
    def direction(self) -> Direction:
        """Delegates to the wrapped metric -- a gate doesn't change what
        "better" means, only adds a weight/threshold on top. Previously this
        unconditionally overwrote direction to MAXIMIZE whenever a threshold
        was set, silently flipping the pass/fail sense of any MINIMIZE
        metric (latency, error rate, ...) wrapped in a ScoreGate."""
        return self._metric.direction

    @property
    def required_fields(self):
        return self._metric.required_fields

    def applicable(self, sample: Sample) -> bool:
        return self._metric.applicable(sample)

    def score(self, sample: Sample, output: str, context: Any = None) -> list[Score]:
        result = self._metric.score(sample, output, context)
        if isinstance(result, Score):
            scores = [result]
        else:
            scores = list(result)
        for s in scores:
            s.weight = self.weight
            s.direction = self.direction
            if self._threshold is not None:
                s.threshold = self._threshold
        return scores


class WeightedSum:
    def __init__(self, name: str = "weighted_sum") -> None:
        self.name = name

    def compute(self, scores: list[Score]) -> Score:
        if not scores:
            return Score(name=self.name, value=0.0)
        total_weight = sum(s.weight for s in scores)
        if total_weight == 0.0:
            return Score(name=self.name, value=0.0)
        weighted_value = sum(s.value * s.weight for s in scores) / total_weight
        return Score(name=self.name, value=weighted_value)
