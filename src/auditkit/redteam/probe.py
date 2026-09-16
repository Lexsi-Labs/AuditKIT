"""Probe base class and result type."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProbeResult:
    """The outcome of running one probe against one model input."""

    probe_name: str
    prompt: str
    output: str
    passed: bool
    detector_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


class Probe:
    """An adversarial probe that generates prompts to test model behavior."""

    name: str = "probe"
    description: str = ""

    def __init__(self, name: str | None = None, description: str | None = None) -> None:
        if name is not None:
            self.name = name
        if description is not None:
            self.description = description
        self._prompts: list[str] = []

    def prompts(self) -> list[str]:
        """Return the list of adversarial prompts for this probe."""
        return self._prompts

    def generate(self) -> list[str]:
        """Alias for prompts()."""
        return self.prompts()
