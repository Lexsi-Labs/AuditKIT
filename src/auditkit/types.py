"""Core enumerations for AuditKIT.

These are the fixed vocabulary of the evaluation spine: what an evaluation asks
the model to do (:class:`TaskKind`), what a score means (:class:`ScoreKind`,
:class:`DataType`, :class:`Direction`), what a model can do (:class:`Capability`),
and where a run came from (:class:`Source`).

Every enum is ``str``-backed so members compare equal to their string value and
serialize to plain JSON without a custom encoder.
"""

from __future__ import annotations

from enum import Enum


class TaskKind(str, Enum):
    """What an evaluation asks the model to do."""

    MCQ = "mcq"
    GENERATIVE = "generative"
    RAG = "rag"
    SECURITY = "security"
    PERFORMANCE = "performance"
    LANGUAGE_MODELING = "language_modeling"


class ScoreKind(str, Enum):
    """How a score was produced."""

    BENCHMARK = "benchmark"
    JUDGE = "judge"
    CODE = "code"
    SECURITY = "security"
    PERF = "perf"
    HUMAN = "human"
    RAG = "rag"


class DataType(str, Enum):
    """The value domain of a score."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


class Direction(str, Enum):
    """Which way is better for a metric."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class Capability(str, Enum):
    """What a model backend can be asked for."""

    GENERATE = "generate"
    LOGLIKELIHOOD = "loglikelihood"
    CHAT = "chat"
    EMBED = "embed"


class Source(str, Enum):
    """Where a run or score originated."""

    SDK = "sdk"
    UI = "ui"
    ONLINE = "online"
