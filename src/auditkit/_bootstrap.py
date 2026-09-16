"""Shared paired-bootstrap significance test.

Both the model-comparison path (`model_compare.py`) and the experiment path
(`experiment.py`) need the same test — a two-sided paired bootstrap over
per-sample (baseline, candidate) score pairs — so it lives here once instead of
being reimplemented (and drifting) in each. The old copies also aligned the two
runs' scores by list *position*, which silently mispaired samples whenever the
runs' prediction order diverged (partial failures, retries, caching); callers
here align **by sample_id** via :func:`score_pairs`.
"""

from __future__ import annotations

import random
from typing import Any, Optional

from .report import RunResult


def score_pairs(
    baseline: RunResult, candidate: RunResult, metric: Optional[str] = None
) -> list[tuple[float, float]]:
    """Per-sample ``(baseline_value, candidate_value)`` pairs, aligned by ``sample_id``.

    ``metric=None`` uses each prediction's primary ``.score`` (works for every
    engine, including lm-eval benchmark runs whose predictions don't carry a
    per-metric ``metadata["scores"]`` breakdown). A metric name reads that named
    score out of ``metadata["scores"]`` (native runs).
    """
    def by_id(run: RunResult) -> dict[str, float]:
        out: dict[str, float] = {}
        for p in run.predictions:
            if metric is None:
                if p.score is not None:
                    out[p.sample_id] = float(p.score)
            else:
                for s in p.metadata.get("scores", []):
                    if s.get("name") == metric and s.get("value") is not None:
                        out[p.sample_id] = float(s["value"])
                        break
        return out

    b, c = by_id(baseline), by_id(candidate)
    return [(b[i], c[i]) for i in sorted(b.keys() & c.keys())]


def paired_bootstrap(
    pairs: list[tuple[float, float]], *, n_resamples: int = 1000, seed: int = 42
) -> dict[str, Any]:
    """Two-sided paired bootstrap over ``(baseline, candidate)`` value pairs.

    Returns ``{n, mean_baseline, mean_candidate, delta, p_value, significant}``
    where ``delta = mean_candidate - mean_baseline`` (positive ⇒ candidate scored
    higher). Fewer than 2 pairs can't be resampled meaningfully → ``{"error": ...}``.
    """
    n = len(pairs)
    if n < 2:
        return {"error": "need at least 2 aligned sample pairs", "n": n}
    diffs = [c - b for b, c in pairs]
    mean_baseline = sum(b for b, _ in pairs) / n
    mean_candidate = sum(c for _, c in pairs) / n
    observed = sum(diffs) / n
    rng = random.Random(seed)
    count = 0
    for _ in range(n_resamples):
        resample = [rng.choice(diffs) for _ in range(n)]
        if abs(sum(resample) / n) >= abs(observed):
            count += 1
    p = (count + 1) / (n_resamples + 1)
    return {
        "n": n,
        "mean_baseline": round(mean_baseline, 6),
        "mean_candidate": round(mean_candidate, 6),
        "delta": round(observed, 6),
        "p_value": round(p, 6),
        "significant": p < 0.05,
    }
