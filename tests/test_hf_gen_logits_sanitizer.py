"""Regression tests for HFGenModel._logits_processor()'s real bug: a
previous version blanket-clamped every logit (finite or not) to a fixed
[-50, 50] range, which silently broke generation for checkpoints whose
real, non-NaN logit scale exceeds that range -- confirmed live:
bigscience/bloom-560m and EleutherAI/pythia-160m both produced EMPTY
generated text (not a crash, not an error -- just silently wrong output)
because clamping collapsed their real top-~30 logits down to the same
ceiling value, creating an artificial tie that argmax/greedy then broke
by lowest token ID, favoring a low-ID special token (EOS) over the
model's real best pick.

Fixed: only positions that are ACTUALLY non-finite (nan/inf) are ever
touched now; every genuinely finite logit, regardless of magnitude,
passes through completely unchanged. This file locks in both halves --
the original NaN-crash fix must still work, and it must never again
touch a legitimately large-but-finite value.
"""
from __future__ import annotations

import pytest


def _importable(module_name: str) -> bool:
    try:
        __import__(module_name)
    except ImportError:
        return False
    return True


NEEDS_TRANSFORMERS = pytest.mark.skipif(
    not _importable("transformers"), reason="needs the transformers package"
)


@NEEDS_TRANSFORMERS
class TestSanitizeLogitsProcessorUnit:
    """Direct unit tests of the processor's logic -- no real model needed."""

    def _processor(self):
        from auditkit.model.hf_gen import HFGenModel
        m = HFGenModel.__new__(HFGenModel)  # bypass __init__, don't need a real pipeline for this
        return m._logits_processor()

    def test_large_but_finite_logits_are_left_completely_unchanged(self):
        """The exact bug: a value like 430.0 (BLOOM's real top-token logit
        for a real prompt) must never be touched, even though it's far
        outside the old [-50, 50] clamp range."""
        import torch
        proc = self._processor()
        scores = torch.tensor([[430.0, 429.5, 420.25, -60.28, 0.0]])
        out = proc(input_ids=None, scores=scores)
        assert torch.equal(out, scores)

    def test_nan_is_replaced_not_left_as_is(self):
        import torch
        proc = self._processor()
        scores = torch.tensor([[1.0, float("nan"), 3.0]])
        out = proc(input_ids=None, scores=scores)
        assert not torch.isnan(out).any()
        assert out[0, 0].item() == 1.0  # untouched finite values stay untouched
        assert out[0, 2].item() == 3.0

    def test_positive_and_negative_inf_are_replaced_with_finite_sentinels(self):
        import torch
        proc = self._processor()
        scores = torch.tensor([[float("inf"), float("-inf"), 5.0]])
        out = proc(input_ids=None, scores=scores)
        assert torch.isfinite(out).all()
        assert out[0, 2].item() == 5.0  # untouched

    def test_all_finite_tensor_returns_the_same_object_unmodified(self):
        """No unnecessary cloning/mutation when there's nothing to fix --
        confirms the early-return path."""
        import torch
        proc = self._processor()
        scores = torch.tensor([[1.0, 2.0, 3.0]])
        out = proc(input_ids=None, scores=scores)
        assert out is scores


@NEEDS_TRANSFORMERS
class TestSanitizeLogitsProcessorRealModels:
    """Real, live tests against the actual checkpoints that exposed this
    bug -- no mocks. Confirms both real model families that were broken
    are fixed, and that the original gpt2-medium crash-prevention purpose
    still works."""

    def test_bloom_produces_real_text_not_empty_string(self):
        """bigscience/bloom-560m: real logits reach ~430 for a normal
        prompt's top token -- confirmed live to previously produce an
        empty generated_text under the old blanket clamp."""
        from auditkit.model.hf_gen import HFGenModel
        from auditkit.model import Request

        model = HFGenModel(model="bigscience/bloom-560m", name="bloom", device="cpu")
        result = model.generate([Request(
            prompt="The capital of France is", request_type="generate",
            params={"max_tokens": 10, "temperature": 0.0},
        )])
        text = result[0].completions[0].text
        assert text.strip() != ""
        assert "Paris" in text

    def test_pythia_produces_real_text_not_empty_string(self):
        """EleutherAI/pythia-160m: same root cause, confirmed live."""
        from auditkit.model.hf_gen import HFGenModel
        from auditkit.model import Request

        model = HFGenModel(model="EleutherAI/pythia-160m", name="pythia", device="cpu")
        result = model.generate([Request(
            prompt="The capital of France is", request_type="generate",
            params={"max_tokens": 10, "temperature": 0.0},
        )])
        text = result[0].completions[0].text
        assert text.strip() != ""

    def test_gpt2_medium_real_sampling_still_does_not_crash(self):
        """The original purpose of this processor: gpt2-medium emits real
        nan/inf logits for certain prompts, and torch.multinomial (real
        sampling) crashes outright on a distribution containing any.
        Confirmed live (this session): 'The', 'I think that', 'In the
        beginning' all reliably produce nan/inf logits for this
        checkpoint. Must still not crash after the fix."""
        from auditkit.model.hf_gen import HFGenModel
        from auditkit.model import Request

        model = HFGenModel(model="gpt2-medium", name="gpt2-medium", device="cpu")
        for prompt in ("The", "I think that", "In the beginning"):
            result = model.generate([Request(
                prompt=prompt, request_type="generate",
                params={"max_tokens": 10, "temperature": 0.9},
            )])
            # No crash is the actual assertion here; a real (if
            # low-quality, given this checkpoint's known numerical
            # instability on these specific prompts) string comes back.
            assert isinstance(result[0].completions[0].text, str)
