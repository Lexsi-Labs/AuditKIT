"""``HFGenModel.loglikelihood`` takes its softmax in float32, whatever dtype the model emits."""
from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from auditkit.model import Request  # noqa: E402
from auditkit.model.hf_gen import HFGenModel  # noqa: E402


class _Tok:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 50 for c in text]}


def _model_with_logits(logits):
    class _Net:
        def __call__(self, input_ids):
            return types.SimpleNamespace(logits=logits[None, : input_ids.shape[1]])

    m = HFGenModel.__new__(HFGenModel)
    m._ensure_pipeline = lambda: None
    m._pipeline = types.SimpleNamespace(model=_Net(), tokenizer=_Tok(), device="cpu")
    return m


def test_bf16_logits_score_like_float32():
    g = torch.Generator().manual_seed(0)
    logits32 = torch.randn(8, 50, generator=g) * 6
    logits16 = logits32.to(torch.bfloat16)
    req = [Request(prompt="abc", params={"target": "de"})]
    got = _model_with_logits(logits16).loglikelihood(req)[0].logprob
    # float32 reference computed from the same (already rounded) bf16 logits
    want = _model_with_logits(logits16.float()).loglikelihood(req)[0].logprob
    assert got == pytest.approx(want, abs=1e-6)
