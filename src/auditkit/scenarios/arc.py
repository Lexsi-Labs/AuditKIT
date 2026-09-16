"""ARC-Challenge benchmark scenario (lazy datasets import)."""
from __future__ import annotations

from auditkit.scenario import Scenario
from auditkit.sample import Sample
from auditkit.types import TaskKind
from auditkit.errors import ExtraNotInstalled
from auditkit.registry import SCENARIOS


@SCENARIOS.register("arc")
class ARCScenario(Scenario):
    name = "arc"

    def __init__(self, split: str = "test"):
        self.split = split
        super().__init__(name=f"arc_{split}")

    def samples(self):
        try:
            import datasets
        except ImportError:
            raise ExtraNotInstalled("interop", "pip install auditkit[interop]")
        data = datasets.load_dataset("arc", "ARC-Challenge", split=self.split)
        return [
            Sample(
                input=row["question"],
                choices=list(row["choices"]["text"]),
                target=row["answerKey"],
                kind=TaskKind.MCQ,
            )
            for row in data
        ]
