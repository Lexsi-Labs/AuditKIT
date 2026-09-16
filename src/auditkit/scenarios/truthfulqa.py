"""TruthfulQA benchmark scenario (lazy datasets import)."""
from __future__ import annotations

from auditkit.scenario import Scenario
from auditkit.sample import Sample
from auditkit.types import TaskKind
from auditkit.errors import ExtraNotInstalled
from auditkit.registry import SCENARIOS


@SCENARIOS.register("truthfulqa")
class TruthfulQAScenario(Scenario):
    name = "truthfulqa"

    def __init__(self, split: str = "validation"):
        self.split = split
        super().__init__(name=f"truthfulqa_{split}")

    def samples(self):
        try:
            import datasets
        except ImportError:
            raise ExtraNotInstalled("interop", "pip install auditkit[interop]")
        data = datasets.load_dataset("truthfulqa", "multiple_choice", split=self.split)
        return [
            Sample(
                input=row["question"],
                choices=row["mc1_targets"]["choices"],
                target=row["mc1_targets"]["labels"].index(1) if 1 in row["mc1_targets"]["labels"] else "0",
                kind=TaskKind.MCQ,
            )
            for row in data
        ]
