"""Tests for T2 LLM-as-judge infrastructure (JudgeMetric, G-Eval scaffold)."""

from __future__ import annotations

import pytest

from auditkit.sample import Sample
from auditkit.score import Score
from auditkit.metrics.judge import JudgeMetric, GEval, RubricItem


# ============================================================================
# RubricItem
# ============================================================================

class TestRubricItem:
    def test_rubric_item_creation(self):
        item = RubricItem(criterion="coherence", weight=1.0, description="Is the text coherent?")
        assert item.criterion == "coherence"
        assert item.weight == 1.0
        assert item.description == "Is the text coherent?"

    def test_default_weight(self):
        item = RubricItem(criterion="relevance")
        assert item.weight == 1.0


# ============================================================================
# JudgeMetric ABC
# ============================================================================

class TestJudgeMetric:
    def test_abstract(self):
        with pytest.raises(TypeError):
            JudgeMetric()

    def test_subclass_must_implement_judge(self):
        class Incomplete(JudgeMetric):
            pass
        with pytest.raises(TypeError):
            Incomplete()

    def test_default_properties(self):
        class Concrete(JudgeMetric):
            def judge(self, sample, output, context):
                return Score(name="test", value=1.0)
        m = Concrete()
        assert m.name == "judge"
        assert m.is_deterministic is False
        assert m.kind.value == "judge"

    def test_judge_subclass_works_as_metric(self):
        class Concrete(JudgeMetric):
            name = "my_judge"
            def judge(self, sample, output, context):
                return Score(name="my_judge", value=0.5)
        m = Concrete()
        result = m.score(Sample(input="q"), "output")
        assert result.value == 0.5
        assert result.kind.value == "judge"


# ============================================================================
# GEval scaffold
# ============================================================================

class TestGEval:
    def test_geval_creates_with_rubric(self):
        rubric = [
            RubricItem(criterion="correctness", weight=2.0),
            RubricItem(criterion="fluency", weight=1.0),
        ]
        g = GEval(rubric=rubric, name="g-eval-test")
        assert g.name == "g-eval-test"
        assert len(g._rubric) == 2

    def test_geval_without_judge_model_raises_on_score(self):
        g = GEval(rubric=[RubricItem(criterion="x")], name="empty")
        with pytest.raises(NotImplementedError, match="judge_model"):
            g.score(Sample(input="q"), "output")

    def test_geval_accepts_judge_model(self):
        class FakeJudge:
            def generate(self, reqs):
                from auditkit.model import Result_, Generated
                return [Result_(completions=[Generated(text="5")]) for _ in reqs]
        g = GEval(rubric=[
            RubricItem(criterion="quality", weight=1.0, description="Rate quality 1-5"),
        ], judge_model=FakeJudge(), name="g-eval")
        result = g.score(Sample(input="Q", target="answer"), "model output")
        assert isinstance(result, Score)
        assert result.name == "g-eval"

    def test_geval_requires_rubric(self):
        with pytest.raises(ValueError, match="rubric"):
            GEval(rubric=[], name="bad")

    def test_geval_kind_is_judge(self):
        g = GEval(rubric=[RubricItem(criterion="x")], name="g")
        assert g.kind.value == "judge"
