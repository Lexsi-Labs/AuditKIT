"""GSM8K benchmark scenario (lazy datasets import)."""
from __future__ import annotations

from auditkit.scenario import Scenario
from auditkit.sample import Sample
from auditkit.types import TaskKind
from auditkit.errors import ExtraNotInstalled
from auditkit.registry import SCENARIOS


@SCENARIOS.register("gsm8k")
class GSM8KScenario(Scenario):
    name = "gsm8k"
    # The target is the final number; the reply may show its working.
    default_scorers = ["numeric_match"]

    def __init__(self, split: str = "test"):
        self.split = split
        super().__init__(name=f"gsm8k_{split}")

    def samples(self):
        try:
            import datasets
        except ImportError:
            raise ExtraNotInstalled("interop", "pip install auditkit[interop]")
        data = datasets.load_dataset("openai/gsm8k", "main", split=self.split)
        return [
            Sample(
                input=row["question"],
                # "<worked solution>\n#### 72": the number after #### is the answer
                target=row["answer"].rsplit("####", 1)[-1].strip().replace(",", ""),
                kind=TaskKind.GENERATIVE,
                metadata={"solution": row["answer"]},
            )
            for row in data
        ]
