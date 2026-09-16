"""Named plugin registries.

Every extension point on the spine — scenarios, adapters, models, metrics,
annotators, evaluators — is a :class:`Registry`. A technique registers its
classes by name at import time; the runner and the public façade look them up by
name or build them from an :class:`ObjectSpec`. This is what lets a new modality
or technique be a plugin: the spine never imports it directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .errors import RegistryError


@dataclass
class ObjectSpec:
    """A declarative "build me this": a registered name plus constructor args."""

    class_name: str
    args: dict[str, Any] = field(default_factory=dict)


class Registry:
    """A name → class table for one kind of plugin."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: dict[str, type] = {}

    def register(self, name: str) -> Callable[[type], type]:
        """Decorator: register a class under ``name`` (must be unique)."""

        def decorate(cls: type) -> type:
            if name in self._entries:
                raise RegistryError(f"{self.kind} '{name}' already registered")
            self._entries[name] = cls
            return cls

        return decorate

    def get(self, name: str) -> type:
        try:
            return self._entries[name]
        except KeyError:
            raise RegistryError(
                f"unknown {self.kind} '{name}'; known: {self.names()}"
            ) from None

    def create(self, spec: ObjectSpec) -> Any:
        """Instantiate the class named by ``spec`` with its args."""
        return self.get(spec.class_name)(**spec.args)

    def names(self) -> list[str]:
        return sorted(self._entries)


SCENARIOS = Registry("scenario")
ADAPTERS = Registry("adapter")
MODELS = Registry("model")
METRICS = Registry("metric")
ANNOTATORS = Registry("annotator")
EVALUATORS = Registry("evaluator")
