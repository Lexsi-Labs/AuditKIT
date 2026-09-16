"""Scores, running statistics, and the pass@k / pass^k estimators.

A :class:`Score` is one measurement of one sample (a value plus what it means).
A :class:`Stat` accumulates many values into a mean/stderr/percentile summary
without holding every value in memory longer than it needs to. :func:`pass_at_k`
and :func:`pass_hat_k` turn "``c`` of ``n`` trials were correct" into the two
numbers people actually want: capability (can it ever?) and reliability (does it
every time?).
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

from .types import DataType, Direction, ScoreKind, Source


@dataclass
class Score:
    """One measurement of one sample.

    ``value`` is the number; the rest says how to read it. When ``threshold`` is
    set, :attr:`passed` turns the number into a gate honoring ``direction`` —
    higher-is-better for ``MAXIMIZE``, lower-is-better for ``MINIMIZE``.
    """

    name: str
    value: float
    kind: ScoreKind = ScoreKind.BENCHMARK
    data_type: DataType = DataType.NUMERIC
    direction: Direction = Direction.MAXIMIZE
    label: Optional[str] = None
    reason: Optional[str] = None
    threshold: Optional[float] = None
    weight: float = 1.0
    source: Source = Source.SDK
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> Optional[bool]:
        """``True``/``False`` against ``threshold``, or ``None`` if ungated."""
        if self.threshold is None:
            return None
        if self.direction == Direction.MINIMIZE:
            return self.value <= self.threshold
        return self.value >= self.threshold


class Stat:
    """A running summary of many values (mean, stderr, spread, percentiles).

    Values are kept so percentiles can be computed exactly; everything else is
    derived on read. Empty is safe: every reduction returns ``0.0``.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._values: list[float] = []
        self._lock = threading.Lock()

    def add(self, value: float) -> "Stat":
        """Record one value; returns self so calls can be chained."""
        with self._lock:
            self._values.append(float(value))
        return self

    @property
    def count(self) -> int:
        return len(self._values)

    @property
    def mean(self) -> float:
        return sum(self._values) / len(self._values) if self._values else 0.0

    @property
    def std(self) -> float:
        """Population standard deviation (0.0 for fewer than two values)."""
        n = len(self._values)
        if n < 2:
            return 0.0
        mu = self.mean
        return math.sqrt(sum((v - mu) ** 2 for v in self._values) / n)

    @property
    def stderr(self) -> float:
        n = len(self._values)
        if n < 2:
            return 0.0
        return self.std / math.sqrt(n)

    @property
    def min(self) -> float:
        return min(self._values) if self._values else 0.0

    @property
    def max(self) -> float:
        return max(self._values) if self._values else 0.0

    def percentile(self, p: float) -> float:
        """The ``p``-th percentile (0..100) by linear interpolation."""
        if not self._values:
            return 0.0
        ordered = sorted(self._values)
        if len(ordered) == 1:
            return ordered[0]
        rank = (p / 100.0) * (len(ordered) - 1)
        lo = math.floor(rank)
        hi = math.ceil(rank)
        if lo == hi:
            return ordered[lo]
        frac = rank - lo
        return ordered[lo] + (ordered[hi] - ordered[lo]) * frac

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "mean": self.mean,
            "stderr": self.stderr,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "values": list(self._values),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Stat:
        s = cls(data.get("name", ""))
        for v in data.get("values", []):
            s.add(v)
        return s


def _check_draw(n: int, c: int, k: int) -> None:
    if k > n:
        raise ValueError(f"cannot draw k={k} from n={n} trials")
    if c > n or c < 0:
        raise ValueError(f"correct count c={c} out of range for n={n}")


def pass_at_k(n: int, c: int, k: int) -> float:
    """Capability: probability that a draw of ``k`` of ``n`` trials has >=1 correct.

    ``1 - C(n-c, k) / C(n, k)`` — the unbiased HumanEval estimator.
    """
    _check_draw(n, c, k)
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """Reliability: probability that all ``k`` of a draw of ``k`` are correct.

    ``C(c, k) / C(n, k)``.
    """
    _check_draw(n, c, k)
    if c < k:
        return 0.0
    return math.comb(c, k) / math.comb(n, k)
