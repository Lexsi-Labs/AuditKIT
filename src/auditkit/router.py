"""Automatic adapter selection from the dataset's shape.

Routing is a pure function of the **task** — what each sample carries — never of
the model:

- ``choices`` present            -> :class:`MCQAdapter`
- ``retrieval_context`` present  -> :class:`RAGAdapter`
- otherwise                      -> :class:`GenerationAdapter`

Why not look at the model? Chat-vs-base formatting (an instruct model's chat
template vs a base model's raw text) is applied by the **model backend** at
generate-time, so any adapter's output is formatted correctly for whatever model
runs it. That means routing never has to know the model, never has to download a
tokenizer to probe it, and an MCQ/RAG/plain task is handled correctly on both
base and instruct models without picking a different adapter.

:class:`ChatAdapter` is intentionally **never** auto-selected: its only job now
is to inject a specific system prompt, which is an explicit user choice — not
something to guess (and auto-injecting a generic "You are a helpful assistant"
would silently confound an eval).
"""

from __future__ import annotations

from typing import Any, Sequence, Union

from .adapter import Adapter, GenerationAdapter, MCQAdapter, RAGAdapter
from .sample import Sample


def _first_sample(samples: Union[Sample, Sequence[Sample]]) -> Sample:
    if isinstance(samples, Sample):
        return samples
    for s in samples:
        return s
    raise ValueError("route_adapter needs at least one sample to inspect")


def route_adapter(samples: Union[Sample, Sequence[Sample]], model: Any = None,
                  **_ignored: Any) -> Adapter:
    """Pick an adapter for *samples* from their shape.

    Routes on the field each adapter actually needs (``choices`` for MCQ,
    ``retrieval_context`` for RAG), so a sample can never be routed to an
    adapter that would then fail for lack of that field. ``model`` is accepted
    for backward compatibility but ignored — routing is task-based; the model
    backend handles model-specific formatting.
    """
    sample = _first_sample(samples)
    if sample.choices:
        return MCQAdapter()
    if sample.retrieval_context:
        return RAGAdapter()
    return GenerationAdapter()
