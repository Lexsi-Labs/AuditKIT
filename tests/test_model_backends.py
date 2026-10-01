"""Tests for T1 model backend resolution."""

from __future__ import annotations

import pytest

from auditkit.model import AutoModel, EchoModel, CallableModel
from auditkit.model.openai import OpenAIModel
from auditkit.model.anthropic import AnthropicModel
from auditkit.model.hf_gen import HFGenModel
from auditkit.model.lexsi import LexsiModel
from auditkit.model.vllm_gen import VLLMModel
from auditkit.model.litellm_gen import LiteLLMModel
from auditkit.errors import AuditKitError


class TestHFGenModelDeviceResolution:
    """device="auto" used to be forwarded verbatim to torch.device(...),
    which crashes immediately (RuntimeError: unrecognized device string
    "auto") -- "auto" belongs to a different parameter (device_map) in the
    HF ecosystem, not this one. _resolve_device() now detects a real device
    instead of forwarding the literal string.
    """

    def test_default_resolves_to_a_real_device_not_the_literal_string_auto(self):
        m = HFGenModel(model="x")
        assert m._resolve_device() in ("cuda", "mps", "cpu")

    def test_none_also_triggers_detection(self):
        m = HFGenModel(model="x", device=None)
        assert m._resolve_device() in ("cuda", "mps", "cpu")

    def test_explicit_device_is_never_overridden(self):
        m = HFGenModel(model="x", device="cpu")
        assert m._resolve_device() == "cpu"

    def test_explicit_cuda_index_is_never_overridden(self):
        m = HFGenModel(model="x", device="cuda:0")
        assert m._resolve_device() == "cuda:0"


class TestAutoModelResolve:
    def test_echo(self):
        m = AutoModel.resolve("echo")
        assert isinstance(m, EchoModel)

    def test_openai_prefix(self):
        m = AutoModel.resolve("openai:gpt-4o")
        assert isinstance(m, OpenAIModel)

    def test_anthropic_prefix(self):
        m = AutoModel.resolve("anthropic:claude-3")
        assert isinstance(m, AnthropicModel)

    def test_hf_prefix(self):
        m = AutoModel.resolve("hf:gpt2")
        assert isinstance(m, HFGenModel)

    def test_lexsi_prefix(self):
        m = AutoModel.resolve("lexsi:lexsi-3.5")
        assert isinstance(m, LexsiModel)

    def test_vllm_prefix(self):
        m = AutoModel.resolve("vllm:meta-llama/Llama-2-7b")
        assert isinstance(m, VLLMModel)

    def test_litellm_prefix(self):
        m = AutoModel.resolve("litellm:gpt-4o")
        assert isinstance(m, LiteLLMModel)

    def test_api_prefix_resolves(self):
        from auditkit.model.api_gen import APIModel
        m = AutoModel.resolve("api:openai/gpt-4o")
        assert isinstance(m, APIModel)

    def test_model_instance_passthrough(self):
        m = EchoModel()
        assert AutoModel.resolve(m) is m

    def test_callable(self):
        m = AutoModel.resolve(lambda prompts: prompts)
        assert isinstance(m, CallableModel)

    def test_unknown_string_raises(self):
        with pytest.raises(AuditKitError, match="unknown"):
            AutoModel.resolve("nonexistent:")

    # #43 part 3: a bare prefix is decided per backend, not by one blanket guard.
    @pytest.mark.parametrize("spec", ["api:", "hf:", "vllm:", "api:  "])
    def test_bare_prefix_that_needs_a_name_raises_with_the_spec(self, spec):
        with pytest.raises(AuditKitError, match=f"empty model spec {spec!r}"):
            AutoModel.resolve(spec)

    def test_named_api_spec_resolves_as_before(self):
        pytest.importorskip("requests")
        assert AutoModel.resolve("api:x")._model_name == "x"

    def test_bare_hosted_prefix_uses_the_backend_default(self):
        pytest.importorskip("requests")
        assert AutoModel.resolve("groq:")._model_name == "llama-3.3-70b-versatile"

    def test_bare_agent_prefix_takes_the_url_from_opts(self):
        m = AutoModel.resolve("agent:", url="http://127.0.0.1:9/agent")
        assert m.name == "agent:"
        with pytest.raises(AuditKitError, match="URL"):
            AutoModel.resolve("agent:")

    def test_hf_local_directory_still_resolves(self, tmp_path):
        assert AutoModel.resolve(f"hf:{tmp_path}")._model_name == str(tmp_path)

    def test_precomputed_and_echo_have_no_name_and_still_resolve(self):
        assert AutoModel.resolve("precomputed").name == "precomputed"
        assert AutoModel.resolve("echo").name == "echo"


class TestThreadsafe:
    def test_echo_threadsafe(self):
        assert EchoModel().threadsafe is True

    def test_callable_threadsafe(self):
        m = AutoModel.resolve(lambda p: p)
        assert m.threadsafe is True

    def test_openai_threadsafe(self):
        m = OpenAIModel(api_key="test")
        assert m.threadsafe is False

    def test_vllm_threadsafe(self):
        m = VLLMModel(model="gpt2")
        assert m.threadsafe is False

    def test_litellm_threadsafe(self):
        m = LiteLLMModel()
        assert m.threadsafe is True
