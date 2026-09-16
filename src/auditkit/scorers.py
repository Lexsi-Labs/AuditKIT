"""The public scorer decorator and adapters.

A *scorer* is the user-facing way to define a custom metric: decorate a
``(sample, output) -> float | Score`` function with ``@scorer`` and pass it to
:func:`evaluate`. The decorator returns a :class:`FunctionScorer`; the
:class:`ScorerMetric` adapter wraps it as an internal :class:`Metric`.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Union

from .metric import Metric
from .sample import Sample
from .score import Score
from .types import Direction, ScoreKind


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
            result = self.fn(sample, output)
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
    """
    if fn is not None:
        return FunctionScorer(fn, direction=direction, name=name or fn.__name__)

    def decorator(
        f: Callable[[Sample, str], Union[float, Score]],
    ) -> FunctionScorer:
        return FunctionScorer(f, direction=direction, name=name or f.__name__)

    return decorator
