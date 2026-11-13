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

import math
import random
from typing import Any, Optional

from .report import RunResult
from .score import _measured_value


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
                    # a flagged placeholder is not a pair to compare
                    if s.get("name") == metric and _measured_value(s) is not None:
                        out[p.sample_id] = _measured_value(s)
                        break
        return out

    b, c = by_id(baseline), by_id(candidate)
    return [(b[i], c[i]) for i in sorted(b.keys() & c.keys())]


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolation percentile (numpy's default method) of an ascending list."""
    pos = q * (len(sorted_vals) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def paired_bootstrap(
    pairs: list[tuple[float, float]], *, n_resamples: int = 1000, seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Two-sided paired bootstrap over ``(baseline, candidate)`` value pairs.

    Returns ``{n, mean_baseline, mean_candidate, delta, p_value, significant, ci, ci_level}``
    where ``delta = mean_candidate - mean_baseline`` (positive ⇒ candidate scored
    higher) and ``ci = [lo, hi]`` is the percentile bootstrap interval on ``delta`` at
    ``ci_level = confidence``, taken from the un-centred resampled means (the p-value uses
    centred ones). Fewer than 2 pairs can't be resampled meaningfully → ``{"error": ...}``.
    ``n_resamples`` below 1 raises ``ValueError``: with none the p-value is not defined, and a negative
    count gave a negative p-value. ``confidence`` must be strictly between 0 and 1.
    """
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be at least 1, got {n_resamples}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be strictly between 0 and 1, got {confidence}")
    n = len(pairs)
    if n < 2:
        return {"error": "need at least 2 aligned sample pairs", "n": n}
    diffs = [c - b for b, c in pairs]
    # math.fsum is correctly rounded on every Python; the built-in float sum() changed in 3.12.
    mean_baseline = math.fsum(b for b, _ in pairs) / n
    mean_candidate = math.fsum(c for _, c in pairs) / n
    observed = math.fsum(diffs) / n
    rng = random.Random(seed)
    # Null hypothesis: no difference. Centre the differences on zero before resampling; resampling the raw
    # differences centres the resampled means on `observed`, which puts p near 0.5 whatever the effect.
    centred = [d - observed for d in diffs]
    # Discrete scores (0/1 flags, exact match) make resampled means that tie the observed one
    # exactly; centring leaves rounding error in each value, so an exact >= would decide those ties
    # by float noise. A tie counts as "at least as extreme", as in exact arithmetic.
    threshold = abs(observed) - 1e-9 * max(1.0, abs(observed))
    count = 0
    boot_deltas: list[float] = []
    idx = range(n)
    for _ in range(n_resamples):
        # rng.choice(range(n)) draws the same stream as rng.choice(centred) did, so p-values
        # are unchanged; the same indices give the un-centred resample for the interval.
        picks = [rng.choice(idx) for _ in range(n)]
        if abs(math.fsum(centred[i] for i in picks) / n) >= threshold:
            count += 1
        boot_deltas.append(math.fsum(diffs[i] for i in picks) / n)
    p = (count + 1) / (n_resamples + 1)
    boot_deltas.sort()
    tail = (1.0 - confidence) / 2
    return {
        "n": n,
        "mean_baseline": round(mean_baseline, 6),
        "mean_candidate": round(mean_candidate, 6),
        "delta": round(observed, 6),
        "p_value": round(p, 6),
        "significant": p < 0.05,
        "ci": [round(_percentile(boot_deltas, tail), 6), round(_percentile(boot_deltas, 1.0 - tail), 6)],
        "ci_level": confidence,
    }
