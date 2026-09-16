"""Hallucination and factual consistency metrics."""
from __future__ import annotations

from typing import Any

from auditkit.metric import Metric
from auditkit.registry import METRICS
from auditkit.sample import Sample
from auditkit.score import Score
from auditkit.types import Direction, ScoreKind
from auditkit.errors import ExtraNotInstalled


@METRICS.register("factual_consistency")
class FactualConsistency(Metric):
    name = "factual_consistency"
    kind = ScoreKind.BENCHMARK
    direction = Direction.MAXIMIZE
    is_deterministic = False
    required_fields = frozenset({"target"})

    def __init__(self, model_name: str = "microsoft/deberta-base-mnli") -> None:
        self._model_name = model_name
        self._pipe = None

    def identity(self) -> dict:
        return {"name": self.name, "model_name": self._model_name}

    def score(self, sample: Sample, output: str, context: Any = None) -> Score:
        try:
            from transformers import pipeline
        except ImportError:
            raise ExtraNotInstalled("transformers", "pip install auditkit[transformers]")
        if self._pipe is None:
            self._pipe = pipeline("text-classification", model=self._model_name)
        premise = sample.target or ""
        result = self._pipe(f"{premise} [SEP] {output}")
        label = result[0]["label"].upper()
        if "ENTAIL" in label:
            value = 1.0
        elif "CONTRADICT" in label:
            value = 0.0
        else:
            value = 0.5
        return Score(name=self.name, value=value, kind=self.kind)
