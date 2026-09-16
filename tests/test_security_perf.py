"""Tests for T6 Security + Performance metrics."""

from __future__ import annotations


from auditkit.sample import Sample
from auditkit.metrics.security import (
    KeywordDetector, DefconGrade, ThreatCategory,
)
from auditkit.metrics.perf import LatencyStats, Throughput


# ============================================================================
# ThreatCategory (OWASP/MITRE crosswalk)
# ============================================================================

class TestThreatCategory:
    def test_str_enum(self):
        assert ThreatCategory.PROMPT_INJECTION.value == "prompt_injection"
        assert ThreatCategory.JAILBREAK.value == "jailbreak"
        assert ThreatCategory.DATA_LEAKAGE.value == "data_leakage"
        assert ThreatCategory.HARMFUL_CONTENT.value == "harmful_content"

    def test_categories_exist(self):
        names = {c.value for c in ThreatCategory}
        assert "prompt_injection" in names


# ============================================================================
# KeywordDetector — checks output for blacklisted keywords
# ============================================================================

class TestKeywordDetector:
    def test_detects_blacklisted(self):
        m = KeywordDetector(blacklist=["bad", "dangerous"])
        s = Sample(input="q")
        score = m.score(s, "this is bad")
        assert score.value == 0.0  # 0 = failed (unsafe content detected)

    def test_passes_clean_output(self):
        m = KeywordDetector(blacklist=["bad"])
        s = Sample(input="q")
        score = m.score(s, "this is fine")
        assert score.value == 1.0

    def test_case_insensitive(self):
        m = KeywordDetector(blacklist=["BAD"])
        s = Sample(input="q")
        score = m.score(s, "this is bad")
        assert score.value == 0.0

    def test_whitelist_overrides_blacklist(self):
        m = KeywordDetector(blacklist=["bad", "harmful"], whitelist=["bad"])
        s = Sample(input="q")
        score = m.score(s, "bad but not harmful")
        # "bad" is whitelisted, "harmful" is not → fails
        assert score.value == 0.0

    def test_name(self):
        assert KeywordDetector(blacklist=["x"]).name == "keyword_detector"


# ============================================================================
# DefconGrade — maps numeric scores to DEFCON levels
# ============================================================================

class TestDefconGrade:
    def test_grade_5_at_zero(self):
        assert DefconGrade.from_score(0.0) == DefconGrade.DEFCON_5

    def test_grade_map(self):
        assert DefconGrade.from_score(1.0) == DefconGrade.DEFCON_1
        assert DefconGrade.from_score(0.8) == DefconGrade.DEFCON_2
        assert DefconGrade.from_score(0.6) == DefconGrade.DEFCON_3
        assert DefconGrade.from_score(0.4) == DefconGrade.DEFCON_4
        assert DefconGrade.from_score(0.0) == DefconGrade.DEFCON_5

    def test_as_metadata(self):
        grade = DefconGrade.from_score(0.85)
        meta = grade.to_metadata()
        assert "defcon" in meta
        assert "label" in meta


# ============================================================================
# LatencyStats — collect and bucket latency measurements
# ============================================================================

class TestLatencyStats:
    def test_aggregates_latencies(self):
        ls = LatencyStats()
        ls.record(100.0)
        ls.record(200.0)
        ls.record(300.0)
        stats = ls.stats()
        assert stats["count"] == 3
        assert stats["mean"] == 200.0
        assert stats["min"] == 100.0
        assert stats["max"] == 300.0
        assert stats["p50"] == 200.0
        assert stats["p95"] is not None

    def test_empty(self):
        ls = LatencyStats()
        stats = ls.stats()
        assert stats["count"] == 0

    def test_single_value(self):
        ls = LatencyStats()
        ls.record(42.0)
        stats = ls.stats()
        assert stats["mean"] == 42.0
        assert stats["p50"] == 42.0


# ============================================================================
# Throughput — requests per second
# ============================================================================

class TestThroughput:
    def test_throughput_calculation(self):
        t = Throughput()
        t.record(total_ms=1000.0, num_requests=10)
        stats = t.stats()
        assert stats["rps"] == 10.0

    def test_zero_time(self):
        t = Throughput()
        t.record(total_ms=0.0, num_requests=10)
        stats = t.stats()
        assert stats["rps"] == 0.0

    def test_multiple_records(self):
        t = Throughput()
        t.record(total_ms=2000.0, num_requests=10)
        t.record(total_ms=1000.0, num_requests=20)
        stats = t.stats()
        assert stats["total_requests"] == 30
        assert stats["total_time_ms"] == 3000.0
