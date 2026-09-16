"""Metrics: the internal, richer form of a scorer.

A :class:`Metric` turns one sample plus one model output into one or more
:class:`Score`. It knows which sample fields it needs (``required_fields``), so
the runner can skip a metric that doesn't apply instead of crashing — an
exact-match metric is silently skipped for an open-ended sample with no gold.
The two here are deterministic string matchers; judge/code/security metrics
arrive as later techniques but wear this same interface.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

from .registry import METRICS
from .sample import Sample
from .score import Score
from .types import Direction, ScoreKind
from ._identity_guard import warn_if_identity_incomplete


class Metric(ABC):
    """Score a model output against a sample."""

    name: str = "metric"
    kind: ScoreKind = ScoreKind.BENCHMARK
    is_deterministic: bool = True
    required_fields: frozenset[str] = frozenset()

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        warn_if_identity_incomplete(cls, Metric, "score")
        if not hasattr(cls, "direction"):
            raise TypeError(
                f"{cls.__name__} must declare 'direction' (Direction.MAXIMIZE or "
                f"Direction.MINIMIZE) -- required so RunComparison/compare_models() "
                f"grade it in the right sense (a metric where lower is better, e.g. "
                f"latency or error rate, graded as MAXIMIZE would report a regression "
                f"as an improvement). Set it as a class attribute:\n\n"
                f"    class {cls.__name__}(Metric):\n"
                f"        direction = Direction.MAXIMIZE  # or Direction.MINIMIZE\n\n"
                f"See docs/best_practices/scorers.md#direction for a full example."
            )

    def applicable(self, sample: Sample) -> bool:
        """False when any required field is absent on the sample (→ skipped)."""
        for field_name in self.required_fields:
            if getattr(sample, field_name, None) is None:
                return False
        return True

    @abstractmethod
    def score(self, sample: Sample, output: str, context: Any = None) -> Union[Score, list[Score]]:
        """Produce a Score (or list) for ``output`` against ``sample``."""
        ...

    async def ascore(self, sample: Sample, output: str, context: Any = None):
        """Async entry point; deterministic metrics just defer to :meth:`score`."""
        return self.score(sample, output, context)

    def identity(self) -> dict:
        """The config that defines this metric's scoring, for the run fingerprint.

        Deterministic metrics are fully described by their name. Judges override
        this to include the judge model + prompt + choices, so changing the judge
        (a different "ruler") changes the run fingerprint and never silently
        reuses a cached result scored by a different judge. ``__init_subclass__``
        above warns at class-definition time when a subclass takes constructor
        arguments but skips this override (unless it instead makes ``self.name``
        itself parameter-derived, which already protects the fingerprint — see
        ``_identity_guard.py``)."""
        return {"name": self.name}


@METRICS.register("exact_match")
class ExactMatch(Metric):
    """1.0 iff the output equals the target after trimming outer whitespace."""

    name = "exact_match"
    direction = Direction.MAXIMIZE
    required_fields = frozenset({"target"})

    def score(self, sample: Sample, output: str, context: Any = None) -> Score:
        hit = output.strip() == (sample.target or "").strip()
        return Score(name=self.name, value=1.0 if hit else 0.0, kind=self.kind)


_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^\w\s]")


def _normalize(text: str) -> str:
    """Lowercase, drop punctuation and articles, collapse whitespace."""
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _ARTICLE_RE.sub(" ", text)
    return " ".join(text.split())


@METRICS.register("quasi_exact_match")
class QuasiExactMatch(Metric):
    """Exact match after normalization (case, punctuation, articles, spacing)."""

    name = "quasi_exact_match"
    direction = Direction.MAXIMIZE
    required_fields = frozenset({"target"})

    def score(self, sample: Sample, output: str, context: Any = None) -> Score:
        hit = _normalize(output) == _normalize(sample.target or "")
        return Score(name=self.name, value=1.0 if hit else 0.0, kind=self.kind)


def _resolve_choice_index(value: Any, choices: list) -> Optional[int]:
    """Resolve a target or model-output value to a 0-based index into ``choices``.

    ``value`` legitimately shows up in three encodings across scenarios and
    custom loaders: a bare letter ("B", only meaningful for <=26 choices), a
    0-based index into ``choices`` (``1`` or ``"1"``), or the literal text of
    the correct choice. All three are accepted here so ``Acc``/``AccNorm``
    score correctly regardless of which one a given dataset uses, instead of
    silently mis-scoring (or crashing on a non-str target, e.g. an int).

    The canonical return value is a 0-based **index**, not a letter — letters
    run out at 26 choices (``chr(65+26)`` is ``'['``, not a letter) and
    collide once you reach lowercase range (index 32 -> ``'a'``, which
    case-normalizes to the same key as index 0's ``'A'``). An index has no
    such ceiling, so it's what both the adapter's prompt labels (see
    :class:`~auditkit.adapter.MCQAdapter`) and this resolver key off of.
    """
    if value is None:
        return None
    choices = choices or []
    if isinstance(value, str):
        s = value.strip()
        # Compare against *stripped* choices, not raw ones -- MCQ choices
        # commonly carry a leading space (the correct convention for
        # GPT-2/BPE-style loglikelihood scoring, so " Paris" tokenizes as a
        # proper word-initial continuation of the prompt), and comparing an
        # already-stripped candidate against unstripped choices can never
        # match even on a perfect pick.
        for i, c in enumerate(choices):
            if isinstance(c, str) and c.strip() == s:
                return i
        if s.lstrip("-").isdigit():
            idx = int(s)
            if 0 <= idx < len(choices):
                return idx
        if len(s) == 1 and s.isalpha():
            idx = ord(s.upper()) - ord("A")
            if 0 <= idx < len(choices):
                return idx
    elif isinstance(value, int) and not isinstance(value, bool):
        if 0 <= value < len(choices):
            return value
    return None


def _target_index(sample: Sample) -> int:
    idx = _resolve_choice_index(sample.target, sample.choices or [])
    if idx is None:
        raise ValueError(
            f"cannot resolve sample.target={sample.target!r} against "
            f"sample.choices={sample.choices!r}; expected a 0-based index, "
            f"a choice letter (e.g. 'B', only for <=26 choices), or the "
            f"exact text of the correct choice"
        )
    return idx


@METRICS.register("acc")
class Acc(Metric):
    """1.0 iff the output resolves to the same choice index as the target."""

    name = "acc"
    direction = Direction.MAXIMIZE
    required_fields = frozenset({"target", "choices"})

    def score(self, sample: Sample, output: str, context: Any = None) -> Score:
        target_idx = _target_index(sample)
        pred_idx = _resolve_choice_index(output, sample.choices or [])
        hit = pred_idx is not None and pred_idx == target_idx
        return Score(name=self.name, value=1.0 if hit else 0.0, kind=self.kind)


@METRICS.register("acc_norm")
class AccNorm(Metric):
    """1.0 iff the predicted choice text maps to the correct target choice."""

    name = "acc_norm"
    direction = Direction.MAXIMIZE
    required_fields = frozenset({"target", "choices"})

    def score(self, sample: Sample, output: str, context: Any = None) -> Score:
        target_idx = _target_index(sample)
        pred_idx = _resolve_choice_index(output, sample.choices or [])
        hit = pred_idx is not None and pred_idx == target_idx
        return Score(name=self.name, value=1.0 if hit else 0.0, kind=self.kind)
