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


# ---- #50, hosted backends ------------------------------------------------

import json

from auditkit.model import _identity_value
from auditkit.model.anthropic import AnthropicModel
from auditkit.model.groq_gen import GroqModel
from auditkit.model.litellm_gen import LiteLLMModel
from auditkit.model.lexsi import LexsiModel
from auditkit.model.openai import OpenAIModel
from auditkit.model.openrouter_gen import OpenRouterModel

DEFAULTS = [
    (OpenAIModel, "gpt-4o-mini", "openai"),
    (AnthropicModel, "claude-sonnet-4-20250514", "anthropic"),
    (GroqModel, "llama-3.3-70b-versatile", "groq"),
    (LiteLLMModel, "gpt-4o-mini", "litellm"),
    (OpenRouterModel, "openai/gpt-4o-mini", "openrouter"),
    (LexsiModel, "lexsi-3.5", "lexsi"),
]
SECRET = "sk-live-0123456789abcdef"


@pytest.mark.parametrize("cls,model,name", DEFAULTS)
def test_hosted_default_config_keeps_its_identity(cls, model, name, monkeypatch):
    for var in ("GROQ_API_KEY", "OPENROUTER_API_KEY", "LEXSI_API_KEY"):
        monkeypatch.setenv(var, "k")
    assert cls().identity() == {"name": name, "model": model}
    assert cls(api_key="a").identity() == cls(api_key="b").identity()         # a key never keys the run


@pytest.mark.parametrize("kw", [{"organization": "org-b"}, {"project": "p"}, {"default_query": {"api-version": "2"}},
                                {"reasoning_effort": "high"}, {"api_base": "https://proxy.example/v1"}])
def test_openai_settings_change_the_identity(kw):
    assert OpenAIModel(**kw).identity() != OpenAIModel().identity()


@pytest.mark.parametrize("cls,default_base", [
    (GroqModel, "https://api.groq.com/openai/v1"),
    (OpenRouterModel, "https://openrouter.ai/api/v1"),
    (LexsiModel, "https://api.lexsi.ai/v1"),
])
def test_requests_backends_see_endpoint_and_body_fields(cls, default_base):
    base = cls().identity()
    assert cls(api_base="https://other.example/v1").identity() != base
    assert cls(api_base=default_base + "/").identity() == base                # the default, trailing slash or not
    assert cls(reasoning_effort="high").identity() != base
    assert cls(timeout=5, verify=False).identity() == base                    # transport settings stay out


def test_litellm_and_anthropic_see_their_kwargs():
    assert LiteLLMModel(api_base="https://x.example").identity() != LiteLLMModel().identity()
    assert AnthropicModel(base_url="https://x.example").identity() != AnthropicModel().identity()
    assert AnthropicModel(thinking={"type": "enabled", "budget_tokens": 1000}).identity() != AnthropicModel().identity()
    assert AnthropicModel(thinking={"budget_tokens": 1}).identity() != AnthropicModel(thinking={"budget_tokens": 2}).identity()
    assert OpenAIModel(max_retries=1, timeout=3).identity() == OpenAIModel().identity()


@pytest.mark.parametrize("make", [
    lambda s: OpenAIModel(default_headers={"Authorization": f"Bearer {s}", "X-Route": "a"}),
    lambda s: OpenAIModel(default_headers={"x-api-key": s}),
    lambda s: OpenAIModel(api_base=f"https://user:{s}@proxy.example/v1?api_key={s}"),
    lambda s: AnthropicModel(default_headers={"X-Custom": s}, base_url=f"https://h.example/?key={s}"),
    lambda s: LiteLLMModel(api_key=s, extra_headers={"Authorization": s}),
    lambda s: GroqModel(extra_headers={"api-key": s}),
    lambda s: LexsiModel(api_base=f"https://{s}@h.example/v1"),
    lambda s: OpenRouterModel(default_query={"token": s}),
])
def test_a_secret_never_reaches_the_identity_or_the_cache_key(make):
    ident = make(SECRET).identity()
    assert SECRET not in json.dumps(ident, default=str)
    assert make(SECRET).identity() == make("sk-rotated-9999999999999999").identity()   # rotation keeps the cache
    assert ident != type(make(SECRET))().identity()                           # the setting itself still keys the run


def test_a_secret_never_reaches_the_cache_files(tmp_path):
    m = OpenAIModel(default_headers={"Authorization": f"Bearer {SECRET}"})
    ak.evaluate(SAMPLES, model=_Echo(m))
    blob = b"".join(p.read_bytes() for p in (tmp_path / "cache").rglob("*") if p.is_file())
    assert blob and SECRET.encode() not in blob


class _Echo:
    """An OpenAIModel that answers locally, so the run caches without a network."""
    def __new__(cls, model):
        model.generate = lambda requests: [Result_(completions=[Generated(text="4")]) for _ in requests]
        return model


def test_output_changing_names_that_contain_token_are_kept():
    assert _identity_value({"max_completion_tokens": 5}) == {"max_completion_tokens": 5}
    assert _identity_value({"budget_tokens": 5, "access_token": "x"}) == {"budget_tokens": 5, "access_token": "<redacted>"}


def test_the_api_backend_hides_secrets_too():
    from auditkit.model.api_gen import APIModel
    a = APIModel(model="m", api_base="http://h/v1", extra_headers={"Authorization": SECRET}).identity()
    assert SECRET not in json.dumps(a, default=str)
