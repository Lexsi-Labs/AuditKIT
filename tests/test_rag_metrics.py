"""Tests for T2 RAG metrics (LexicalGroundedness, ContextCoverage, ContextOverlap, AnswerOverlap)."""

from __future__ import annotations


from auditkit.sample import Sample
from auditkit.metrics.rag import (
    LexicalGroundedness,
    ContextCoverage,
    ContextOverlap,
    AnswerOverlap,
)


# ============================================================================
# LexicalGroundedness
# ============================================================================

class TestLexicalGroundedness:
    def test_fully_supported(self):
        m = LexicalGroundedness()
        s = Sample(input="What is X?", target="X is a car",
                   retrieval_context=["X is a type of car", "Cars have wheels"])
        score = m.score(s, "X is a car")
        assert score.value == 1.0

    def test_partially_supported(self):
        m = LexicalGroundedness()
        s = Sample(input="Q", target="X and Y",
                   retrieval_context=["X is here", "Z is nowhere"])
        score = m.score(s, "X and Y")
        assert 0.0 < score.value < 1.0

    def test_no_support(self):
        m = LexicalGroundedness()
        s = Sample(input="Q", target="Z",
                   retrieval_context=["X", "Y"])
        score = m.score(s, "Z")
        assert score.value == 0.0

    def test_requires_retrieval_context(self):
        m = LexicalGroundedness()
        assert "retrieval_context" in m.required_fields
        assert m.applicable(Sample(input="q")) is False
        assert m.applicable(Sample(input="q", retrieval_context=["a"])) is True

    def test_name(self):
        assert LexicalGroundedness().name == "lexical_groundedness"


# ============================================================================
# ContextCoverage
# ============================================================================

class TestContextCoverage:
    def test_full_recall(self):
        m = ContextCoverage()
        s = Sample(input="Q", target="quick brown fox",
                   retrieval_context=["the quick brown fox jumps"])
        score = m.score(s, "quick brown fox")
        assert score.value == 1.0

    def test_partial_recall(self):
        m = ContextCoverage()
        s = Sample(input="Q", target="quick brown fox",
                   retrieval_context=["slow brown bear"])
        score = m.score(s, "quick brown fox")
        assert 0.0 < score.value < 1.0

    def test_no_recall(self):
        m = ContextCoverage()
        s = Sample(input="Q", target="xyz",
                   retrieval_context=["abc", "def"])
        score = m.score(s, "")
        assert score.value == 0.0

    def test_requires_retrieval_context_and_target(self):
        m = ContextCoverage()
        assert "retrieval_context" in m.required_fields
        assert "target" in m.required_fields

    def test_name(self):
        assert ContextCoverage().name == "context_coverage"


# ============================================================================
# ContextOverlap
# ============================================================================

class TestContextOverlap:
    def test_all_relevant(self):
        m = ContextOverlap()
        s = Sample(input="Q", target="cat",
                   retrieval_context=["cat is an animal", "cats are furry"])
        score = m.score(s, "cat")
        assert score.value == 1.0

    def test_some_relevant(self):
        m = ContextOverlap()
        s = Sample(input="Q", target="cat",
                   retrieval_context=["cat is an animal", "weather is nice"])
        score = m.score(s, "cat")
        assert 0.0 < score.value < 1.0

    def test_none_relevant(self):
        m = ContextOverlap()
        s = Sample(input="Q", target="cat",
                   retrieval_context=["dogs are fun", "weather"])
        score = m.score(s, "cat")
        assert score.value == 0.0

    def test_requires_retrieval_context(self):
        m = ContextOverlap()
        assert "retrieval_context" in m.required_fields

    def test_name(self):
        assert ContextOverlap().name == "context_overlap"


# ============================================================================
# AnswerOverlap
# ============================================================================

class TestAnswerOverlap:
    def test_fully_relevant(self):
        m = AnswerOverlap()
        s = Sample(input="What color is the sky?", target="blue")
        score = m.score(s, "The sky is blue.")
        assert score.value == 1.0

    def test_partially_relevant(self):
        m = AnswerOverlap()
        s = Sample(input="What color is the sky?", target="blue")
        score = m.score(s, "The sky is blue and grass is green.")
        assert 0.0 < score.value <= 1.0

    def test_not_relevant(self):
        m = AnswerOverlap()
        s = Sample(input="What color is the sky?", target="blue")
        score = m.score(s, "I like pizza.")
        assert score.value == 0.0

    def test_requires_target(self):
        m = AnswerOverlap()
        assert "target" in m.required_fields
        assert m.applicable(Sample(input="q")) is False

    def test_name(self):
        assert AnswerOverlap().name == "answer_overlap"
