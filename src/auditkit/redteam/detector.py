"""Detector base class and result type."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DetectorResult:
    """Whether a detector flagged the model output as a successful attack."""

    detected: bool
    confidence: float = 1.0
    details: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class Detector:
    """Checks whether a model output indicates a successful adversarial attack."""

    name: str = "detector"

    def __init__(self, name: str | None = None) -> None:
        if name is not None:
            self.name = name

    def detect(self, prompt: str, output: str) -> DetectorResult:
        """Analyze the model output and return a detection result.

        Args:
            prompt: The adversarial prompt that was sent.
            output: The model's response.

        Returns:
            A DetectorResult indicating whether the attack was detected.
        """
        raise NotImplementedError
