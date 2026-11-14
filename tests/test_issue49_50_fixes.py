"""Issues #49 (a bare callable in ``scorers=``) and #50 (the HF run fingerprint)."""
from __future__ import annotations

import os

import pytest

import auditkit as ak
from auditkit.metric import ExactMatch
from auditkit.model import CallableModel, Result_, Generated
from auditkit.model.hf_gen import HFGenModel
from auditkit.model.vllm_gen import VLLMModel

SAMPLES = [ak.Sample(input="2+2?", target="4", id="a")]


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


# ---- #49 -----------------------------------------------------------------

def test_bare_callable_scorer_is_refused_with_the_fix_in_the_message():
    with pytest.raises(TypeError, match=r"bare callable.*direction.*@ak\.scorer"):
        ak.evaluate(SAMPLES, model=lambda ps: ["4"] * len(ps), scorers=[lambda s, o: 1.0], cache=False)


def test_a_metric_class_is_asked_for_as_an_instance():
    with pytest.raises(TypeError, match=r"pass an instance: ExactMatch\(\)"):
        ak.evaluate(SAMPLES, model=lambda ps: ["4"] * len(ps), scorers=ExactMatch, cache=False)


def test_declared_scorers_still_work():
    @ak.scorer(direction=ak.Direction.MAXIMIZE)
    def one(sample, output):
        return 1.0

    def bare(sample, output):
        return 0.0

    for scorers in (one, [one], ak.scorer(bare, direction=ak.Direction.MINIMIZE)):
        res = ak.evaluate(SAMPLES, model=lambda ps: ["4"] * len(ps), scorers=scorers, cache=False)
        assert "one: " in res.summary() or "bare: " in res.summary()


# ---- #50 -----------------------------------------------------------------

@pytest.fixture
def ckpt(tmp_path):
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"a" * 10)
    return d


def test_load_and_call_settings_change_the_identity(ckpt):
    base = HFGenModel(model=str(ckpt)).identity()
    assert HFGenModel(model=str(ckpt), batch_size=8).identity() != base
    assert HFGenModel(model=str(ckpt), dtype="float16").identity() != base
    assert HFGenModel(model=str(ckpt), device="cpu").identity() != base
    assert HFGenModel(model=str(ckpt), revision="abc").identity() != base
    assert HFGenModel(model=str(ckpt), batch_size=1).identity() != HFGenModel(model=str(ckpt), batch_size=8).identity()


def test_equivalent_configs_share_an_identity(ckpt):
    a = HFGenModel(model=str(ckpt), dtype="float16", batch_size=4)
    b = HFGenModel(model=str(ckpt), batch_size=4, dtype="torch.float16", device="auto", hf_token="secret")
    assert a.identity() == b.identity()                      # key order, torch. prefix, auto device
    assert "secret" not in str(b.identity())                 # the token never enters the fingerprint


def test_a_hub_model_with_default_settings_keeps_its_identity():
    assert HFGenModel(model="no/such-hub-model").identity() == {"name": "hf", "model": "no/such-hub-model"}


def test_rewriting_the_local_weights_changes_the_identity(ckpt):
    m = HFGenModel(model=str(ckpt))
    before = m.identity()
    assert m.identity() == before                            # stable while nothing changes
    (ckpt / "model.safetensors").write_bytes(b"b" * 20)
    assert m.identity() != before


def test_vllm_identity_sees_engine_kwargs_and_weights(ckpt):
    base = VLLMModel(model=str(ckpt)).identity()
    assert VLLMModel(model=str(ckpt), dtype="float16").identity() != base
    (ckpt / "model.safetensors").write_bytes(b"b" * 20)
    assert VLLMModel(model=str(ckpt)).identity() != base


def test_evaluate_does_not_replay_another_batch_size(ckpt):
    calls = []

    class _Fake(HFGenModel):
        def generate(self, requests):
            calls.append(self._extra_kwargs.get("batch_size"))
            return [Result_(completions=[Generated(text="4")]) for _ in requests]

    for bs in (1, 8, 8):
        ak.evaluate(SAMPLES, model=_Fake(model=str(ckpt), batch_size=bs))
    assert calls == [1, 8]                                   # the third is a genuine cache hit
