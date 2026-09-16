"""Typed errors AuditKIT raises.

Everything derives from :class:`AuditKitError` so a caller can catch the whole
family with one ``except``. The platform maps each type to a ``#08-xxx`` code at
its service boundary; the library itself stays framework-free.
"""

from __future__ import annotations


class AuditKitError(Exception):
    """Base class for every error AuditKIT raises."""


class CapabilityError(AuditKitError):
    """A model was asked for something it doesn't declare (e.g. loglikelihood)."""


class RegistryError(AuditKitError):
    """A name was not found in, or already taken in, a registry."""


class ExtraNotInstalled(AuditKitError):
    """An optional backend/metric was used without its extra installed.

    *hint*, when given, is the complete, ready-to-run fix (e.g. ``"pip
    install auditkit[vllm]"``) and becomes the message verbatim -- every
    caller already passes a full actionable instruction here, so prefixing
    a second, separately-generated ``"install auditkit[{extra}]"`` in front
    of it (the previous behavior) just produced a redundant, confusing
    message like ``"install auditkit[vllm] for pip install auditkit[vllm]"``.
    Falls back to ``"install auditkit[{extra}]"`` only when no *hint* is
    given at all.
    """

    def __init__(self, extra: str, hint: str = "") -> None:
        super().__init__(hint if hint else f"install auditkit[{extra}]")
        self.extra = extra


class ModelError(AuditKitError):
    """Model backend failure (possibly transient)."""


class MetricError(AuditKitError):
    """Metric computation failure."""


class SampleSkipped(AuditKitError):
    """Sample-level error (non-fatal, result still returned)."""


class ModelTimeout(AuditKitError):
    """Model call exceeded the configured timeout."""
