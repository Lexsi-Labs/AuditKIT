"""paired_bootstrap: the p-value must separate a real effect from noise."""

from __future__ import annotations

import random

from auditkit._bootstrap import paired_bootstrap


def test_large_effect_is_significant():
    rng = random.Random(1)
    pairs = [(rng.random() * 0.2, 0.7 + rng.random() * 0.2) for _ in range(50)]
    out = paired_bootstrap(pairs)
    assert out["delta"] > 0.5 and out["p_value"] < 0.01 and out["significant"] is True


def test_noise_is_not_significant():
    rng = random.Random(2)
    pairs = [(0.5, 0.5 + (rng.random() - 0.5) * 0.1) for _ in range(50)]
    out = paired_bootstrap(pairs)
    assert out["p_value"] > 0.05 and out["significant"] is False


def test_identical_runs_have_p_one():
    assert paired_bootstrap([(0.4, 0.4)] * 10)["p_value"] == 1.0


def test_too_few_pairs_report_an_error():
    assert "error" in paired_bootstrap([(1.0, 0.0)])
