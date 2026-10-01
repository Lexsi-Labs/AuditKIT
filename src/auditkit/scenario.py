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
from .trace import decode_reference


def _stable(o: Any) -> Any:
    """``json.dumps(default=)`` hook that can't leak a per-process value into a
    cache key: a set's ``str`` order follows ``PYTHONHASHSEED`` and a default
    ``__repr__`` embeds the object's address. Sets fold to a sorted list (of
    each element's canonical JSON); an opaque object (no custom ``__repr__``)
    folds to its type name plus its ``__dict__`` state, so two instances that
    differ only in their attributes get different fingerprints."""
    if isinstance(o, (set, frozenset)):
        return sorted(json.dumps(x, default=_stable, sort_keys=True) for x in o)
    if type(o).__repr__ is object.__repr__:
        d = getattr(o, "__dict__", None)
        if d:  # recurse via default=_stable; a self-referential __dict__ trips
            return {type(o).__qualname__: d}  # json's cycle guard -> repr fallback
        # ponytail: __slots__-only / stateless objects still fold to qualname
        # (no __dict__ to capture); a narrowed, documented ceiling.
        return type(o).__qualname__
    return str(o)


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
            # A USER-supplied Sample.id enters the hash; an auto-assigned one
            # does not (see _row). Two datasets that differ only in a caller's
            # id ("q1" vs "q2") must hash apart, or the second evaluate() is
            # served the first run's cached predictions under the wrong
            # sample_ids. But the Runner writes str(index) into an UNSET id in
            # place the first time it scores a sample, so an id-less re-run of
            # the SAME objects must still hit the cache -- hence the positional
            # exclusion below, not a blanket "drop id".
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
            # Fields added after the original five are appended only when set,
            # so existing datasets keep their fingerprints (and caches), while
            # two datasets differing only in, e.g., expected tool calls or
            # metadata (which feeds judge templates and agent messages) never
            # share a cache entry.
            def _row(s: Sample, i: int) -> tuple:
                # An id equal to its own positional index is treated as
                # auto-assigned (identical to the str(index) the Runner writes
                # into an unset id) and excluded, so an id-less re-run of the
                # same reused objects keeps its fingerprint. Any other id is a
                # real user id and enters the hash via `extra`, so old
                # id-less datasets keep their exact name.
                # ponytail: positional check -- a shuffled split writes
                # non-positional ids, so re-running the SAME reused objects
                # under a random split misses the cache once (a safe MISS,
                # never a wrong HIT); acceptable unless that reuse matters.
                user_id = s.id if (s.id is not None and s.id != str(i)) else None
                base = (s.input, s.target, s.choices, s.retrieval_context, s.actual_output)
                # task / kind label every cached prediction (Prediction.task =
                # task or kind), so they are identity too: hashed only when set to
                # a non-default value, which keeps plain datasets' names unchanged.
                kind = getattr(s.kind, "value", s.kind)
                extra = {k: v for k, v in (
                    ("id", user_id),
                    ("task", s.task or None),
                    ("kind", kind if kind != "generative" else None),
                    # a JSON-string reference hashes like its decoded form (#44)
                    ("tools", s.tools), ("expected_tool_calls", decode_reference(s.expected_tool_calls)),
                    ("reference_contexts", s.reference_contexts),
                    ("actual_trace", s.actual_trace), ("metadata", s.metadata or None),
                    ("images", s.images or None),
                ) if v is not None}
                return base + (extra,) if extra else base
            rows = [_row(s, i) for i, s in enumerate(samples)]
            try:
                # JSON-native rows (the common case) never call _stable and
                # have no dicts for sort_keys to touch, so old datasets keep
                # their exact fingerprint. _stable makes sets/opaque objects
                # process-stable.
                blob = json.dumps(rows, default=_stable, sort_keys=True)
            except (TypeError, ValueError):
                # Free-form metadata with mixed/non-str dict keys or a cycle:
                # sort_keys/json can't encode it. repr can't raise and is
                # address-free for primitives, so a cache MISS at worst, never
                # a wrong HIT (GUIDE.md documents metadata as "dict -- anything").
                blob = repr(rows)
            h = hashlib.sha256(blob.encode()).hexdigest()[:8]
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
