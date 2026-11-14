"""The public scorer decorator and adapters.

A *scorer* is the user-facing way to define a custom metric: decorate a
``(sample, output) -> float | Score`` function with ``@scorer`` and pass it to
:func:`evaluate`. The decorator returns a :class:`FunctionScorer`; the
:class:`ScorerMetric` adapter wraps it as an internal :class:`Metric`.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Optional, Union

from .metric import Metric
from .sample import Sample
from .score import Score
from .types import Direction, ScoreKind


def _accepts_context(fn: Callable[..., Any]) -> bool:
    """True when ``fn(sample, output, context=...)`` binds to ``fn``'s own signature.

    A ``TypeError`` raised by such a call came from inside the scorer, so it must not be retried
    without ``context``. A pure ``*args`` forwarder (``functools.wraps`` sets ``__wrapped__``) is judged
    by the function it wraps; a forwarder with nothing behind it is treated as not accepting it, so a
    two-argument scorer behind it still works. A callable with no readable signature is treated as not
    accepting it.
    """
    try:
        signature = inspect.signature(fn, follow_wrapped=False)
        signature.bind(None, None, context=None)
    except (TypeError, ValueError):  # does not bind, or some builtins expose no signature
        return False
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in signature.parameters.values()):
        inner = getattr(fn, "__wrapped__", None)
        return _accepts_context(inner) if inner is not None else False
    return True


class FunctionScorer:
    """A user-defined scorer returned by the ``@scorer`` decorator.

    Attributes
    ----------
    name : str
        The scorer name (defaults to the function's ``__name__``).
    fn : Callable[[Sample, str], Union[float, Score]]
        The underlying callable.
    direction : Direction
        Whether a higher (``MAXIMIZE``) or lower (``MINIMIZE``) value is
        better -- required, not guessed, so ``RunComparison``/
        ``compare_models()`` grade this scorer in the right sense.
    """

    def __init__(
        self,
        fn: Callable[[Sample, str], Union[float, Score]],
        direction: Direction,
        name: Optional[str] = None,
    ) -> None:
        self.fn = fn
        self.name = name or fn.__name__
        self.direction = direction

    def __call__(self, sample: Sample, output: str, context: Any = None) -> Score:
        try:
            result = self.fn(sample, output, context=context)
        except TypeError:
            if _accepts_context(self.fn):     # the call bound, so the scorer raised this itself
                raise
            result = self.fn(sample, output)  # a two-argument scorer
        if isinstance(result, Score):
            result.direction = self.direction
            return result
        return Score(name=self.name, value=float(result), direction=self.direction)

    def __repr__(self) -> str:
        return f"<FunctionScorer {self.name!r}>"


class ScorerMetric(Metric):
    """Wraps a :class:`FunctionScorer` as a :class:`Metric`."""

    def __init__(self, scorer: FunctionScorer) -> None:
        self._scorer = scorer
        self.name = scorer.name

    @property
    def kind(self) -> ScoreKind:
        return ScoreKind.CODE

    @property
    def direction(self) -> Direction:
        return self._scorer.direction

    def score(self, sample: Sample, output: str, context: Any = None) -> Score:
        return self._scorer(sample, output, context=context)

    def applicable(self, sample: Sample) -> bool:
        return True

    def identity(self) -> dict:
        """Name, direction and a fingerprint of the scorer's code and captured values.

        The name alone is not enough: editing the function body (or a value it closes
        over) under the same name must not be served the old run's cached scores.
        """
        from .model import _default_callable_name
        return {"name": self.name, "direction": self.direction.value,
                "scorer": _default_callable_name(self._scorer.fn)}


ScorerType = Union[FunctionScorer, Callable[[Sample, str], Union[float, Score]]]


def scorer(
    fn: Optional[Callable[[Sample, str], Union[float, Score]]] = None,
    *,
    direction: Direction,
    name: Optional[str] = None,
) -> Union[FunctionScorer, Callable[..., FunctionScorer]]:
    """Decorate a function as a scorer usable with :func:`evaluate`.

    ``direction`` is required -- there is no library default, because only
    you know whether your scorer's return value means "higher is better"
    (``Direction.MAXIMIZE``, e.g. a similarity/accuracy-style score) or
    "lower is better" (``Direction.MINIMIZE``, e.g. a latency/error-count-
    style score). Getting this wrong silently flips every ``RunComparison``/
    ``compare_models()`` verdict for the metric. See
    ``docs/best_practices/scorers.md#direction`` for a full example.

    Always called with parens, ``direction=`` required (the old bare
    ``@scorer`` form with no arguments no longer works -- there is nowhere
    to put ``direction`` in that form)::

        @scorer(direction=ak.Direction.MAXIMIZE)
        def my_scorer(sample, output): ...

        @scorer(direction=ak.Direction.MAXIMIZE, name="custom")
        def another(sample, output): ...

    An existing function (a bare callable is not accepted by :func:`evaluate`) is wrapped with
    ``scorer(fn, direction=...)``::

        wrapped = ak.scorer(existing_fn, direction=ak.Direction.MINIMIZE)
    """
    if fn is not None:
        return FunctionScorer(fn, direction=direction, name=name or fn.__name__)

    def decorator(
        f: Callable[[Sample, str], Union[float, Score]],
    ) -> FunctionScorer:
        return FunctionScorer(f, direction=direction, name=name or f.__name__)

    return decorator
