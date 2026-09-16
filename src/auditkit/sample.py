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

    @property
    def is_golden(self) -> bool:
        """True when the sample carries a reference answer to score against."""
        return self.target is not None

    @property
    def input_text(self) -> str:
        """The input as plain text — what a text-only backend is prompted with."""
        return self.input
