"""Shared fixtures for the non-metric suite (PRs #1-#4)."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Every test gets its own run cache, so no result is ever served from ~/.cache/auditkit."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


def per_sample(result):
    """{sample_id: {score_name: value}} from a RunResult."""
    return {p.sample_id: {s["name"]: s["value"] for s in p.metadata.get("scores", [])} for p in result.predictions}
