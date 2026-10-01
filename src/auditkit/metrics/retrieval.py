"""Deterministic retrieval-ranking metrics (no model calls).

The ranked retrieved list is the trace's ``retrieved_contexts`` when a RAG
pipeline returned them at run time (an ``agent:`` endpoint, or
``Sample.actual_trace``), else ``Sample.retrieval_context``. Relevance comes
from ``Sample.reference_contexts``: a list of relevant chunk ids/texts
(binary relevance) or a ``{id: grade}`` dict (graded, for nDCG). Items are
compared as exact strings after stripping whitespace; a retrieved duplicate
counts once (at its first rank) but still holds its slot -- later ranks keep
their original positions, so a repeat never lifts a following item up a rank.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from auditkit.metric import Metric
from auditkit.registry import METRICS
from auditkit.sample import Sample
from auditkit.score import Score
from auditkit.trace import get_trace
from auditkit.types import Direction, ScoreKind


def retrieved_contexts(sample: Sample, context: Any) -> list[str]:
    """The ranked contexts a run actually retrieved for *sample*."""
    trace = get_trace(context)
    ctx = trace.get("retrieved_contexts")
    if ctx is None:
        ctx = sample.retrieval_context
    if isinstance(ctx, str):  # one chunk, not a list of characters
        ctx = [ctx]
    return [str(c) for c in (ctx or [])]


def _grades(reference: Any) -> dict[str, float]:
    if isinstance(reference, dict):
        grades = {str(k).strip(): float(v) for k, v in reference.items()}
        bad = [k for k, g in grades.items() if not math.isfinite(g)]
        if bad:  # inf/inf would make ndcg NaN (and the run file non-strict JSON)
            raise ValueError(f"reference_contexts grades must be finite; got non-finite for {bad[:5]}")
        return {k: g for k, g in grades.items() if g > 0}
    if isinstance(reference, str):
        reference = [reference]
    return {str(k).strip(): 1.0 for k in (reference or [])}


def ranking_scores(ranked: list[str], grades: dict[str, float], k: Optional[int] = None) -> dict[str, float]:
    """hit_rate / precision / recall / mrr / ap / ndcg over one ranked list."""
    if not all(math.isfinite(g) for g in grades.values()):
        raise ValueError("relevance grades must be finite")
    # Cut to the top k, then keep every item at its ORIGINAL rank: a repeat of an
    # already-seen chunk holds its slot at zero relevance (counted once, at its
    # first rank) so items after it do not move up a rank and inflate
    # mrr/ap/ndcg (finding 11).
    top = ranked[:k] if k else ranked
    seen: set[str] = set()
    rel: list[float] = []
    for r in top:
        key = r.strip()
        rel.append(0.0 if key in seen else grades.get(key, 0.0))
        seen.add(key)
    hits = [g > 0 for g in rel]
    n_rel = len(grades)
    first = next((i for i, h in enumerate(hits) if h), None)
    # Average precision over the list, divided by ALL relevant items, so
    # relevant chunks that were never retrieved pull it down (RAGAS divides
    # by retrieved-relevant only, which hides missed documents).
    ap, found = 0.0, 0
    for i, h in enumerate(hits):
        if h:
            found += 1
            ap += found / (i + 1)
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(rel))
    ideal = sorted(grades.values(), reverse=True)[:k] if k else sorted(grades.values(), reverse=True)
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    denom = k if k else len(top)
    return {
        "hit_rate": 1.0 if first is not None else 0.0,
        "precision": sum(hits) / denom if denom else 0.0,
        "recall": sum(hits) / n_rel,
        "mrr": 1.0 / (first + 1) if first is not None else 0.0,
        "average_precision": ap / (min(k, n_rel) if k else n_rel),
        "ndcg": dcg / idcg if idcg else 0.0,
    }


@METRICS.register("retrieval")
class RetrievalMetrics(Metric):
    """Ranked retrieval quality against gold ``reference_contexts``.

    Emits ``hit_rate``, ``precision``, ``recall``, ``mrr``,
    ``average_precision`` and ``ndcg``; with ``k`` set, the cut-off metrics
    are computed on the top ``k`` and every name gets an ``@k`` suffix
    (e.g. ``recall@5``). ``precision@k`` divides by ``k``
    even when fewer than ``k`` items came back. Samples without a gold set
    are skipped (no fake 0); an empty retrieval scores 0 everywhere.
    """

    name = "retrieval"
    kind = ScoreKind.RAG
    direction = Direction.MAXIMIZE
    required_fields = frozenset({"reference_contexts"})

    def __init__(self, k: Optional[int] = None) -> None:
        if k is not None and k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.name = f"retrieval@{k}" if k else "retrieval"

    def identity(self) -> dict:
        return {"name": self.name, "k": self.k}

    def score(self, sample: Sample, output: str, context: Any = None) -> list[Score]:
        grades = _grades(sample.reference_contexts)
        if not grades:
            return []
        ranked = retrieved_contexts(sample, context)
        vals = ranking_scores(ranked, grades, self.k)
        at = f"@{self.k}" if self.k else ""
        return [Score(name=key + at, value=v, kind=self.kind,
                      metadata={"n_retrieved": len(ranked), "n_relevant": len(grades)} if key == "hit_rate" else {})
                for key, v in vals.items()]
