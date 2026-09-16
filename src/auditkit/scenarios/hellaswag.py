"""HellaSwag benchmark scenario (lazy datasets import)."""
from __future__ import annotations

from auditkit.scenario import Scenario
from auditkit.sample import Sample
from auditkit.types import TaskKind
from auditkit.errors import ExtraNotInstalled
from auditkit.registry import SCENARIOS


@SCENARIOS.register("hellaswag")
class HellaSwagScenario(Scenario):
    name = "hellaswag"

    def __init__(self, split: str = "validation"):
        self.split = split
        super().__init__(name=f"hellaswag_{split}")

    def samples(self):
        try:
            import datasets
        except ImportError:
            raise ExtraNotInstalled("interop", "pip install auditkit[interop]")
        data = datasets.load_dataset("hellaswag", split=self.split)
        return [
            Sample(
                input=row["ctx"],
                choices=list(row["endings"]),
                target=str(row["label"]),
                kind=TaskKind.MCQ,
            )
            for row in data
        ]
