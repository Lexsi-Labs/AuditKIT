"""Tests for T1 pre-built benchmark scenarios."""

from __future__ import annotations


from auditkit.registry import SCENARIOS


class TestScenarioRegistry:
    def test_mmlu_registered(self):
        from auditkit.scenarios.mmlu import MMLUScenario
        assert SCENARIOS.get("mmlu") is MMLUScenario

    def test_gsm8k_registered(self):
        from auditkit.scenarios.gsm8k import GSM8KScenario
        assert SCENARIOS.get("gsm8k") is GSM8KScenario

    def test_arc_registered(self):
        from auditkit.scenarios.arc import ARCScenario
        assert SCENARIOS.get("arc") is ARCScenario

    def test_hellaswag_registered(self):
        from auditkit.scenarios.hellaswag import HellaSwagScenario
        assert SCENARIOS.get("hellaswag") is HellaSwagScenario

    def test_truthfulqa_registered(self):
        from auditkit.scenarios.truthfulqa import TruthfulQAScenario
        assert SCENARIOS.get("truthfulqa") is TruthfulQAScenario

    def test_humaneval_registered(self):
        from auditkit.scenarios.humaneval import HumanEvalScenario
        assert SCENARIOS.get("humaneval") is HumanEvalScenario

    def test_registered_count(self):
        assert len(SCENARIOS.names()) >= 6
