"""The unit of evaluation: a :class:`Sample`.

A sample is one thing to ask a model and (optionally) what a correct answer
looks like. Most fields are optional and task-specific — an MCQ sample carries
``choices``, a RAG sample carries ``retrieval_context`` — so one dataclass
covers every task the spine handles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union

from .types import TaskKind


@dataclass
class Sample:
    """One evaluation item: what to ask, and (optionally) the reference answer."""

    input: str
    target: Optional[Union[str, int]] = None
    id: Optional[str] = None
    task: str = ""
    kind: TaskKind = TaskKind.GENERATIVE
    choices: Optional[list[str]] = None
    retrieval_context: Optional[list[str]] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    actual_output: Optional[str] = None
    # Agent / tool-use fields. ``tools`` are the OpenAI-format tool schemas
    # offered to the model. ``expected_tool_calls`` is the reference as a list
    # of TURNS, each a list of calls ``{"name", "arguments"}`` -- a turn with
    # two or more calls is a parallel group, which is what makes parallel
    # tool calling evaluable at all. A flat list of calls is read as ONE turn
    # (the calls expected in a single response, BFCL-style). ``[]`` means
    # "no tool should be called" (irrelevance). A dict names several equally
    # correct routes, ``{"any_of": [route, ...]}``, and a str is a JSON-encoded
    # reference of either shape (a CSV cell); see
    # :func:`auditkit.trace.to_reference_paths`.
    tools: Optional[list[dict[str, Any]]] = None
    expected_tool_calls: Optional[Union[list[Any], dict[str, Any], str]] = None
    # Gold retrieval targets (chunk ids or texts) for ranked retrieval
    # metrics; a ``{id: grade}`` dict gives graded relevance for nDCG.
    reference_contexts: Optional[Union[list[str], dict[str, float]]] = None
    # A pre-recorded agent/RAG run to score offline, paired with
    # ``actual_output``: ``{"messages": [...OpenAI chat messages...],
    # "tool_calls": [[...turns...]], "retrieved_contexts": [...]}`` (any subset).
    actual_trace: Optional[dict[str, Any]] = None
    # Images for vision models (PIL images, paths or URLs), sent alongside
    # ``input`` by backends that support them (``hf:`` on a vision model, ``api:`` in chat mode).
    images: Optional[list[Any]] = None

    @property
    def is_golden(self) -> bool:
        """True when the sample carries a reference answer to score against."""
        return self.target is not None

    @property
    def input_text(self) -> str:
        """The input as plain text — what a text-only backend is prompted with."""
        return self.input
