"""Tests for T1 benchmark technique: MCQAdapter, loglikelihood, acc/acc_norm."""

from __future__ import annotations

import pytest

from auditkit.sample import Sample
from auditkit.adapter import MCQAdapter
from auditkit.model import Request, Model, EchoModel, LogLikelihood, CapabilityError
from auditkit.metric import Acc, AccNorm
from auditkit.runner import Runner
from auditkit.runspec import RunConfig, RunSpec
from auditkit.scenario import ListScenario
from auditkit.report import RunResult
from auditkit.score import Score
from auditkit.types import Capability


# ============================================================================
# MCQAdapter
# ============================================================================

class TestMCQAdapter:
    def test_mcq_joint_creates_single_request(self):
        samples = [Sample(input="What is 2+2?", choices=["3", "4", "5"], target=1)]
        adapter = MCQAdapter(method="mcq_joint")
        reqs = adapter.adapt(samples[0], RunConfig())
        assert len(reqs) == 1
        assert reqs[0].request_type == "generate"
        prompt = reqs[0].prompt
        assert "0." in prompt
        assert "1." in prompt
        assert "2." in prompt

    def test_mcq_joint_prompt_format(self):
        s = Sample(input="Q", choices=["alpha", "beta", "gamma"], target=0)
        adapter = MCQAdapter(method="mcq_joint")
        req = adapter.adapt(s, RunConfig())[0]
        p = str(req.prompt)
        assert "Q" in p
        assert "0." in p and "alpha" in p
        assert "1." in p and "beta" in p
        assert "2." in p and "gamma" in p

    def test_mcq_joint_labels_past_26_choices_stay_numeric(self):
        choices = [f"choice_{i}" for i in range(30)]
        s = Sample(input="pick", choices=choices, target=27)
        adapter = MCQAdapter(method="mcq_joint")
        p = str(adapter.adapt(s, RunConfig())[0].prompt)
        assert "26. choice_26" in p
        assert "27. choice_27" in p
        assert "29. choice_29" in p

    def test_mcq_loglikelihood_creates_one_request_per_choice(self):
        s = Sample(input="Q", choices=["a", "b", "c"], target="B")
        adapter = MCQAdapter(method="mcq_loglikelihood")
        reqs = adapter.adapt(s, RunConfig())
        assert len(reqs) == 3
        for r in reqs:
            assert r.request_type == "loglikelihood"
            assert "target" in r.params

    def test_mcq_loglikelihood_sets_correct_targets(self):
        s = Sample(input="Q", choices=["alpha", "beta"], target="A")
        adapter = MCQAdapter(method="mcq_loglikelihood")
        reqs = adapter.adapt(s, RunConfig())
        targets = [r.params["target"] for r in reqs]
        assert targets == ["alpha", "beta"]

    def test_mcq_adapter_method_property(self):
        assert MCQAdapter(method="mcq_joint").method == "mcq_joint"
        assert MCQAdapter(method="mcq_loglikelihood").method == "mcq_loglikelihood"

    def test_mcq_adapter_requires_choices(self):
        adapter = MCQAdapter(method="mcq_joint")
        with pytest.raises(ValueError, match="choices"):
            adapter.adapt(Sample(input="Q"), RunConfig())


# ============================================================================
# EchoModel.loglikelihood (test double)
# ============================================================================

class TestEchoModelLogLikelihood:
    def test_echo_model_does_not_support_loglikelihood_by_default(self):
        m = EchoModel()
        assert m.supports(Capability.LOGLIKELIHOOD) is False

    def test_echo_model_loglikelihood_raises_capability_error(self):
        m = EchoModel()
        with pytest.raises(CapabilityError):
            m.loglikelihood([Request(prompt="x", request_type="loglikelihood",
                                     params={"target": "y"})])


# ============================================================================
# Acc metric
# ============================================================================

class TestAcc:
    def test_acc_exact_match(self):
        m = Acc()
        s = Sample(input="Q", choices=["a", "b"], target="A")
        score = m.score(s, "A")
        assert isinstance(score, Score)
        assert score.name == "acc"
        assert score.value == 1.0

    def test_acc_mismatch(self):
        m = Acc()
        s = Sample(input="Q", choices=["a", "b"], target="A")
        score = m.score(s, "B")
        assert score.value == 0.0

    def test_acc_no_target_raises_skip(self):
        m = Acc()
        s = Sample(input="Q", choices=["a", "b"])
        assert m.applicable(s) is False

    def test_acc_no_choices_skips(self):
        m = Acc()
        s = Sample(input="Q", target="A")
        assert m.applicable(s) is False

    def test_acc_required_fields(self):
        m = Acc()
        assert "target" in m.required_fields
        assert "choices" in m.required_fields

    def test_acc_strips_output(self):
        m = Acc()
        s = Sample(input="Q", choices=["a", "b"], target="A")
        assert m.score(s, "  A  ").value == 1.0

    def test_acc_accepts_lowercase_output(self):
        m = Acc()
        s = Sample(input="Q", choices=["a", "b"], target="A")
        assert m.score(s, "a").value == 1.0

    def test_acc_matches_when_choices_carry_leading_whitespace(self):
        # Real bug, found while live-testing HFGenModel.loglikelihood():
        # GPT-2/BPE-style loglikelihood scoring requires choices with a
        # leading space (e.g. " Paris") to tokenize as a correct word-initial
        # continuation of the prompt -- _resolve_choice_index() stripped the
        # candidate output before comparing but never stripped `choices`,
        # so an exact, correct pick could never match. Confirmed live: a
        # real gpt2 model picking the objectively correct MCQ choice every
        # time still scored acc=0.0 before this fix.
        m = Acc()
        s = Sample(input="Q", choices=[" a banana", " Paris"], target=1)
        assert m.score(s, " Paris").value == 1.0
        assert m.score(s, " a banana").value == 0.0


# ============================================================================
# AccNorm metric (loglikelihood-based accuracy)
# ============================================================================

class TestAccNorm:
    def test_accnorm_returns_score_with_correct_name(self):
        m = AccNorm()
        s = Sample(input="Q", choices=["a", "b"], target="A")
        score = m.score(s, "a", context={"choice_likelihoods": [-1.0, -3.0]})
        assert isinstance(score, Score)
        assert score.name == "acc_norm"

    def test_accnorm_picks_argmax(self):
        m = AccNorm()
        s = Sample(input="Q", choices=["a", "b", "c"], target="B")
        # choice "b" (index 1) has highest logprob → expected answer letter "B"
        score = m.score(s, "b", context={"choice_likelihoods": [-5.0, -0.5, -3.0]})
        assert score.value == 1.0

    def test_accnorm_wrong_pick(self):
        m = AccNorm()
        s = Sample(input="Q", choices=["a", "b"], target="A")
        score = m.score(s, "b", context={"choice_likelihoods": [-4.0, -0.5]})
        assert score.value == 0.0

    def test_accnorm_no_choices_skips(self):
        m = AccNorm()
        s = Sample(input="Q", target="A")
        assert m.applicable(s) is False

    def test_accnorm_no_target_skips(self):
        m = AccNorm()
        s = Sample(input="Q", choices=["a", "b"])
        assert m.applicable(s) is False

    def test_accnorm_required_fields(self):
        m = AccNorm()
        assert "target" in m.required_fields
        assert "choices" in m.required_fields


# ============================================================================
# Runner — loglikelihood execution path
# ============================================================================

class TestRunnerLogLikelihood:
    def test_runner_handles_mcq_joint_end_to_end(self):
        samples = [
            Sample(input="1+1=?", choices=["1", "2", "3"], target="B"),
            Sample(input="2+2=?", choices=["3", "4", "5"], target="B"),
        ]
        spec = RunSpec(
            scenario=ListScenario(samples),
            model=EchoModel(),
            adapter=MCQAdapter(method="mcq_joint"),
            metrics=[Acc()],
        )
        result = Runner().run(spec)
        assert isinstance(result, RunResult)
        assert "acc" in result.headline
        assert len(result.predictions) == 2

    def test_runner_loglikelihood_with_echo_fails_capability(self):
        samples = [
            Sample(input="Q", choices=["a", "b"], target="A"),
        ]
        spec = RunSpec(
            scenario=ListScenario(samples),
            model=EchoModel(),
            adapter=MCQAdapter(method="mcq_loglikelihood"),
            metrics=[AccNorm()],
        )
        with pytest.raises(CapabilityError):
            Runner().run(spec)


# ============================================================================
# LogLikelihoodModel — a test model that supports loglikelihood
# ============================================================================

class LogLikelihoodModel(Model):
    """A test model that supports loglikelihood — returns logprobs from params."""

    name = "ll_test"

    def capabilities(self):
        return {Capability.GENERATE, Capability.LOGLIKELIHOOD}

    def generate(self, requests):
        from auditkit.model import Result_, Generated
        return [Result_(completions=[Generated(text="A")]) for _ in requests]

    def loglikelihood(self, requests):
        # Every request scores -1.0, regardless of target -- an error path.
        return [LogLikelihood(logprob=-1.0) for _ in requests]


class TestLogLikelihoodModel:
    def test_capabilities(self):
        m = LogLikelihoodModel()
        assert m.supports(Capability.GENERATE)
        assert m.supports(Capability.LOGLIKELIHOOD)

    def test_loglikelihood_returns_probs(self):
        m = LogLikelihoodModel()
        reqs = [Request(prompt="Q", request_type="loglikelihood", params={"target": "a"})]
        lls = m.loglikelihood(reqs)
        assert len(lls) == 1
        assert lls[0].logprob == -1.0


# ============================================================================
# PreferentialLogLikelihoodModel — picks the right answer
# ============================================================================

class PreferentialLogLikelihoodModel(Model):
    """Returns higher logprob for the correct answer."""

    name = "ll_good"

    def capabilities(self):
        return {Capability.GENERATE, Capability.LOGLIKELIHOOD}

    def generate(self, requests):
        from auditkit.model import Result_, Generated
        return [Result_(completions=[Generated(text="A")]) for _ in requests]

    def loglikelihood(self, requests):
        results = []
        for r in requests:
            target = r.params.get("target", "")
            prompt = r.prompt
            # Simulate higher logprob for a specific correct answer
            # This model knows that choices with "right" in the prompt are correct
            if "right" in str(prompt).lower() or target == "correct":
                results.append(LogLikelihood(logprob=-0.1))
            else:
                results.append(LogLikelihood(logprob=-5.0))
        return results


# ============================================================================
# MCQ end-to-end with loglikelihood model
# ============================================================================

class TestMCQEndToEnd:
    def test_mcq_joint_end_to_end_with_acc(self):
        samples = [
            Sample(input="Q1", choices=["a", "b"], target="A"),
            Sample(input="Q2", choices=["a", "b"], target="B"),
        ]
        spec = RunSpec(
            scenario=ListScenario(samples),
            model=EchoModel(),
            adapter=MCQAdapter(method="mcq_joint"),
            metrics=[Acc()],
        )
        result = Runner().run(spec)
        assert "acc" in result.headline

    def test_mcq_loglikelihood_with_ll_model(self):
        samples = [
            Sample(input="right", choices=["wrong", "correct"], target="B"),
            Sample(input="right", choices=["wrong", "correct"], target="A"),
        ]
        spec = RunSpec(
            scenario=ListScenario(samples),
            model=PreferentialLogLikelihoodModel(),
            adapter=MCQAdapter(method="mcq_loglikelihood"),
            metrics=[AccNorm()],
        )
        result = Runner().run(spec)
        assert "acc_norm" in result.headline


# ============================================================================
# AutoModel.resolve — T1 prefixes now dispatch properly
# ============================================================================

class TestAutoModelResolveT1:
    def test_hf_prefix_resolves_t1_backend(self):
        from auditkit.model import AutoModel
        from auditkit.model.hf_gen import HFGenModel
        model = AutoModel.resolve("hf:gpt2")
        assert isinstance(model, HFGenModel)
