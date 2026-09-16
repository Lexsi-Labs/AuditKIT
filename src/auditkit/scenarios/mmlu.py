"""MMLU benchmark scenario (lazy datasets import)."""
from __future__ import annotations

from auditkit.scenario import Scenario
from auditkit.sample import Sample
from auditkit.types import TaskKind
from auditkit.errors import ExtraNotInstalled
from auditkit.registry import SCENARIOS


@SCENARIOS.register("mmlu")
class MMLUScenario(Scenario):
    name = "mmlu"

    def __init__(self, subject: str = "all", split: str = "test"):
        self.subject = subject
        self.split = split
        super().__init__(name=f"mmlu_{subject}_{split}")

    def samples(self):
        try:
            import datasets
        except ImportError:
            raise ExtraNotInstalled("interop", "pip install auditkit[interop]")
        data = datasets.load_dataset("mmlu", self.subject, split=self.split)
        return [
            Sample(
                input=row["question"],
                choices=list(row["choices"]),
                target=row["answer"],
                kind=TaskKind.MCQ,
            )
            for row in data
        ]
