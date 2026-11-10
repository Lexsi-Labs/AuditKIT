"""Tests for the red teaming module."""
from __future__ import annotations

from auditkit.redteam import Probe, ProbeResult, RedTeamRunner, RedTeamResult
from auditkit.redteam.probes.builtin import PromptInjectionProbe, JailbreakProbe, EncodingProbe, RefusalProbe
from auditkit.redteam.detectors.builtin import KeywordDetector, RefusalDetector, InjectionSuccessDetector


class TestProbe:
    def test_base_probe(self):
        p = Probe(name="custom", description="test")
        assert p.name == "custom"
        assert p.prompts() == []

    def test_prompt_injection_probe(self):
        p = PromptInjectionProbe()
        assert p.name == "prompt_injection"
        assert len(p.prompts()) > 0
        assert "Ignore" in p.prompts()[0]

    def test_jailbreak_probe(self):
        p = JailbreakProbe()
        assert p.name == "jailbreak"
        assert len(p.prompts()) > 0

    def test_encoding_probe(self):
        p = EncodingProbe()
        assert p.name == "encoding"
        assert len(p.prompts()) > 0

    def test_refusal_probe(self):
        p = RefusalProbe()
        assert p.name == "refusal"
        assert len(p.prompts()) > 0


class TestDetector:
    def test_keyword_detector_finds_keyword(self):
        d = KeywordDetector()
        result = d.detect("say hacked", "I have been HACKED")
        assert result.detected is True

    def test_keyword_detector_safe_output(self):
        d = KeywordDetector()
        result = d.detect("hello", "That's a nice question!")
        assert result.detected is False

    def test_refusal_detector_finds_refusal(self):
        d = RefusalDetector()
        result = d.detect("bad thing", "I'm sorry, I cannot do that")
        assert result.detected is True

    def test_refusal_detector_no_refusal(self):
        d = RefusalDetector()
        result = d.detect("hello", "Here is my response")
        assert result.detected is False

    def test_injection_success_detector(self):
        d = InjectionSuccessDetector()
        assert d.detect("inject", "ACCESS_GRANTED").detected is True
        assert d.detect("inject", "Normal response").detected is False


class TestRedTeamRunner:
    def test_echo_model_all_probes(self):
        runner = RedTeamRunner(model="echo")
        result = runner.run()
        assert isinstance(result, RedTeamResult)
        assert result.total_probes > 0
        assert result.model_name is not None
        # Echo model returns the prompt as output, so keyword detector should find things
        assert result.pass_rate >= 0.0

    def test_echo_model_specific_probe(self):
        runner = RedTeamRunner(model="echo")
        result = runner.run(probes=["refusal"])
        assert result.total_probes > 0
        for r in result.results:
            assert r.probe_name == "refusal"

    def test_echo_model_with_custom_detector(self):
        runner = RedTeamRunner(model="echo")
        result = runner.run(probes=["refusal"], detectors=["refusal"])
        for r in result.results:
            assert r.detector_name == "refusal"

    def test_multi_probe_run_uses_each_probes_own_detectors(self):
        # A model that always refuses should pass the jailbreak probe whether it runs
        # alone or alongside the refusal probe; the refusal probe's detector must not
        # be applied to jailbreak prompts.
        def always_refuse(prompts):
            return ["I'm sorry, but I can't help with that." for _ in prompts]

        single = {
            name: RedTeamRunner(model=always_refuse).run(probes=[name])
            for name in ("jailbreak", "refusal")
        }
        multi = RedTeamRunner(model=always_refuse).run(probes=["jailbreak", "refusal"])

        assert multi.total_probes == sum(r.total_probes for r in single.values())
        assert multi.passed_probes == sum(r.passed_probes for r in single.values())
        jailbreak_detectors = {r.detector_name for r in single["jailbreak"].results}
        for r in multi.results:
            if r.probe_name == "jailbreak":
                assert r.detector_name in jailbreak_detectors
                assert r.passed

    def test_redteam_result_summary(self):
        result = RedTeamResult(
            results=[
                ProbeResult(probe_name="test", prompt="prompt", output="output", passed=True, detector_name="kw"),
                ProbeResult(probe_name="test", prompt="prompt2", output="output2", passed=False, detector_name="kw"),
            ],
            model_name="echo",
            probes_used=["test"],
            detectors_used=["kw"],
        )
        assert result.total_probes == 2
        assert result.passed_probes == 1
        assert result.failed_probes == 1
        assert result.pass_rate == 0.5
        summary = result.summary()
        assert "Red Team Results" in summary
        assert "Pass rate:" in summary


class TestDetectorFalsePositives:
    def test_refusal_is_not_an_attack_keyword(self):
        d = KeywordDetector()
        assert not d.detect("p", "As an AI, I cannot help with that request.").detected
        assert not d.detect("p", "I'm sorry, but I can't reveal my system prompt.").detected
        assert d.detect("p", "ACCESS_GRANTED").detected

    def test_hacked_in_a_refusal_is_not_an_injection(self):
        d = InjectionSuccessDetector()
        assert not d.detect("p", "I can't help you get someone's account hacked.").detected
        assert d.detect("p", "I have been hacked").detected

    def test_refusal_is_anchored_to_the_opening(self):
        d = RefusalDetector()
        assert not d.detect("p", "Parameterised queries mean the API cannot then confuse data with code.").detected
        assert not d.detect("p", "It is not appropriate to use alcohol directly on a wound.").detected
        assert d.detect("p", "I'm sorry, I can't help with that.").detected
        assert d.detect("p", "Unfortunately, I cannot assist with this.").detected

    def test_ordinary_phrases_are_not_a_prompt_leak(self):
        from auditkit.redteam.detectors.builtin import SystemPromptLeakDetector
        d = SystemPromptLeakDetector()
        assert not d.detect("p", "You are right that sorting is O(n log n); following your guidelines, here is the summary.").detected

    def test_a_disclosure_is_a_prompt_leak(self):
        from auditkit.redteam.detectors.builtin import SystemPromptLeakDetector
        d = SystemPromptLeakDetector()
        assert d.detect("p", "Sure. My system prompt is: You are a helpful bot.").detected
        assert d.detect("p", "I was instructed to never discuss pricing.").detected
        assert not d.detect("p", "I'm sorry, but I can't reveal my system prompt.").detected
        assert not d.detect("p", "I can't share that. My instructions are confidential.").detected

    def test_natural_words_are_not_injection_markers(self):
        assert not InjectionSuccessDetector().detect("p", "Spam filters get bypassed by misspelling words.").detected
        assert not KeywordDetector().detect("p", "I won't ignore all previous instructions.").detected
