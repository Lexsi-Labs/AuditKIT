from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from auditkit.scenario import Scenario
    from auditkit.runner import RunConfig
    from auditkit.score import Score
    from auditkit.model import Model
    from auditkit.sample import Prediction


class Evaluator(ABC):
    technique: str = "generic"

    @abstractmethod
    def evaluate(
        self, scenario: Scenario, model: Model, config: RunConfig
    ) -> tuple[list[Score], list[Prediction]]: ...
