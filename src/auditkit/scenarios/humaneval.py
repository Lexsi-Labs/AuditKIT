"""HumanEval benchmark scenario (lazy datasets import)."""
from __future__ import annotations

from auditkit.scenario import Scenario
from auditkit.sample import Sample
from auditkit.types import TaskKind
from auditkit.errors import ExtraNotInstalled
from auditkit.registry import SCENARIOS


@SCENARIOS.register("humaneval")
class HumanEvalScenario(Scenario):
    name = "humaneval"

    def __init__(self, split: str = "test"):
        self.split = split
        super().__init__(name=f"humaneval_{split}")

    def samples(self):
        try:
            import datasets
        except ImportError:
            raise ExtraNotInstalled("interop", "pip install auditkit[interop]")
        data = datasets.load_dataset("openai_humaneval", split=self.split)
        return [
            Sample(
                input=row["prompt"],
                target=row["canonical_solution"],
                kind=TaskKind.CODE,
            )
            for row in data
        ]
