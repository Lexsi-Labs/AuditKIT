"""Tests for hallucination/factual consistency metrics."""

from __future__ import annotations

import pytest

from auditkit.sample import Sample
from auditkit.metrics.hallucination import FactualConsistency
from auditkit.errors import ExtraNotInstalled


S = Sample(input="x", target="the cat sat on the mat", retrieval_context=["cats like to sit on mats"])


def _importable(module_name: str) -> bool:
    try:
        __import__(module_name)
    except ImportError:
        return False
    return True


class TestFactualConsistency:
    def test_extra_not_installed(self):
        if _importable("transformers"):
            pytest.skip("transformers is installed in this environment -- nothing to assert")
        with pytest.raises(ExtraNotInstalled):
            FactualConsistency().score(S, "the cat sat on the mat")
    def test_name(self):
        assert FactualConsistency().name == "factual_consistency"
