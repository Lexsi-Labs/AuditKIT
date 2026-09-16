"""Scenarios: a task and its data, yielding :class:`Sample`.

Users pass a task name, a list of samples, or a callable to ``evaluate()``; the
façade wraps whatever they pass into a :class:`Scenario` so the runner sees one
shape. :class:`ListScenario` is the trivial wrapper around an in-memory list;
:class:`CallableScenario` defers to a function that yields samples (a custom
loader, a generator, a streaming source).
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable

from .sample import Sample


class Scenario(ABC):
    """A task plus its data, as an iterable of samples."""

    name: str = "scenario"

    def __init__(self, name: str = "scenario", metadata: dict[str, Any] | None = None) -> None:
        self.name = name
        self.metadata: dict[str, Any] = metadata or {}

    @abstractmethod
    def samples(self) -> Iterable[Sample]:
        ...


class ListScenario(Scenario):
    """A scenario backed by an in-memory list of samples."""

    def __init__(self, samples: Iterable[Sample], name: str | None = None,
                 metadata: dict[str, Any] | None = None) -> None:
        samples = list(samples)
        if name is None:
            # Deliberately excludes s.id: Runner mutates a sample's id in
            # place the first time it's scored (`if sample.id is None:
            # sample.id = str(index)`), so including id here meant re-running
            # the exact same Sample objects a second time produced a
            # different auto-generated name -- and therefore a different
            # RunSpec.fingerprint() -- for identical content, defeating the
            # cache purely from that side effect. Identity for caching
            # purposes should track the data, not a bookkeeping field that
            # changes as a side effect of a prior run.
            # Covers every field a built-in Adapter actually reads (input,
            # target for all; choices for MCQAdapter; retrieval_context for
            # RAGAdapter) -- input/target alone let two samples that differ
            # only in retrieval_context or choices collide onto the same
            # name/fingerprint and silently share a DiskCache entry.
            #
            # actual_output is included for the same reason: it's what
            # PrecomputedModel (reads_actual_output=True) actually scores in
            # the documented "generate once, score many times" flow
            # (ak.generate() -> model="precomputed"). Without it, two
            # datasets sharing the same input/target but carrying DIFFERENT
            # generated text (e.g. two different models' real outputs, or a
            # re-generation after a prompt change) hash to the identical
            # name/fingerprint -- confirmed live: scoring "positive" (correct,
            # 1.0) then "negative" (wrong, should be 0.0) against the same
            # input/target silently returned the FIRST call's cached 1.0 for
            # the second, wrong-answer sample.
            h = hashlib.sha256(json.dumps(
                [(s.input, s.target, s.choices, s.retrieval_context, s.actual_output) for s in samples],
                default=str, sort_keys=True,
            ).encode()).hexdigest()[:8]
            name = f"inline_{len(samples)}_{h}"
        super().__init__(name=name, metadata=metadata)
        self._samples = samples

    def samples(self) -> list[Sample]:
        return list(self._samples)


class CallableScenario(Scenario):
    """A scenario that calls a function to produce samples each time it's read."""

    def __init__(self, fn: Callable[[], Iterable[Sample]], name: str = "callable",
                 metadata: dict[str, Any] | None = None) -> None:
        super().__init__(name=name, metadata=metadata)
        self._fn = fn

    def samples(self) -> list[Sample]:
        return list(self._fn())
