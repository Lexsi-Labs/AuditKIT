"""paired_bootstrap gives the exact-arithmetic p-value on discrete scores, on every Python version.

0/1 scores make resampled means that tie the observed mean exactly. The old code compared float
means with ``>=`` after built-in ``sum()``, so float rounding decided those ties: the p-value
changed with the Python version (float ``sum()`` became compensated in 3.12) and differed from
exact arithmetic. Seen live: one 100-pair comparison gave 0.008991 on 3.11 and 0.010989 on 3.12.
"""
from __future__ import annotations

import random
from fractions import Fraction

import pytest

import auditkit._bootstrap as boot


def exact_p(pairs, n_resamples=1000, seed=42):
    """Reference: the same resampling draws, in exact rational arithmetic."""
    n = len(pairs)
    diffs = [Fraction(c) - Fraction(b) for b, c in pairs]
    observed = sum(diffs) / n
    centred = [d - observed for d in diffs]
    rng = random.Random(seed)
    count = sum(abs(sum(rng.choice(centred) for _ in range(n)) / n) >= abs(observed)
                for _ in range(n_resamples))
    return round((count + 1) / (n_resamples + 1), 6)


def flags(improved, worsened, n, seed):
    """Paired 0/1 scores: `improved` pairs go 1 -> 0, `worsened` go 0 -> 1, the rest stay 0."""
    pairs = [(1.0, 0.0)] * improved + [(0.0, 1.0)] * worsened + [(0.0, 0.0)] * (n - improved - worsened)
    random.Random(seed).shuffle(pairs)
    return pairs


# (improved, worsened, n, shuffle seed): cases where the old code's p depended on float summation
CASES = [(22, 7, 100, 1), (14, 7, 100, 2), (9, 2, 40, 0), (6, 1, 30, 0), (12, 4, 60, 0)]


def _naive_sum(xs, start=0.0):          # float sum() before Python 3.12
    for x in xs:
        start += x
    return start


@pytest.mark.parametrize("case", CASES)
def test_matches_exact_arithmetic(case):
    pairs = flags(*case)
    assert boot.paired_bootstrap(pairs)["p_value"] == exact_p(pairs)


@pytest.mark.parametrize("case", CASES)
def test_independent_of_builtin_sum(case, monkeypatch):
    pairs = flags(*case)
    p_default = boot.paired_bootstrap(pairs)["p_value"]
    monkeypatch.setattr(boot, "sum", _naive_sum, raising=False)
    assert boot.paired_bootstrap(pairs)["p_value"] == p_default


def test_continuous_scores_unchanged_in_spirit():
    rng = random.Random(0)
    pairs = [(rng.random(), rng.random() + 0.3) for _ in range(50)]
    r = boot.paired_bootstrap(pairs)
    assert r["significant"] and r["p_value"] == pytest.approx(1 / 1001)
