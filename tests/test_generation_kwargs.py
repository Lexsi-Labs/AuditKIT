"""Stress test for the Request.params -> real backend generation-kwargs fix.

Before this fix, every backend read temperature/top_p/stop_sequences/etc.
only from its own constructor-time attributes (self._temperature, ...),
completely ignoring whatever RunConfig/adapters built into Request.params on
each request. This file drives every one of the 7 real backends' generate()
methods with mocked clients/sessions/pipelines and asserts the *actual*
outgoing call received the per-request override -- not just that the code
runs without error.

No network access, no API keys, no optional extras required beyond what's
already a project dependency for tests -- every external client is replaced
with a lightweight fake object that records what it was called with.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from auditkit.model import Request, resolve_params, DEFAULT_TEMPERATURE
from auditkit.errors import AuditKitError


# ---------------------------------------------------------------------------
# Unit tests for the shared resolve_params() helper itself
# ---------------------------------------------------------------------------
class TestResolveParams:
    def test_falls_back_to_defaults_when_request_has_no_override(self):
        req = Request(prompt="hi", params={})
        out = resolve_params(req, {"temperature": 0.0}, {"temperature": "temperature"})
        assert out == {"temperature": 0.0}

    def test_request_param_overrides_default(self):
        req = Request(prompt="hi", params={"temperature": 0.9})
        out = resolve_params(req, {"temperature": 0.0}, {"temperature": "temperature"})
        assert out == {"temperature": 0.9}

    def test_key_is_renamed_per_target_api(self):
        req = Request(prompt="hi", params={"stop_sequences": ["\n"]})
        out = resolve_params(req, {}, {"stop_sequences": "stop"})
        assert out == {"stop": ["\n"]}

    def test_keys_outside_the_map_are_never_forwarded(self):
        # timeout/max_retries are Runner-only settings that live in
        # Request.params (GenerationAdapter puts them there) but must never
        # reach a real provider API call -- they're just absent from key_map.
        req = Request(prompt="hi", params={"timeout": 30, "max_retries": 5, "temperature": 0.5})
        out = resolve_params(req, {"temperature": 0.0}, {"temperature": "temperature"})
        assert out == {"temperature": 0.5}
        assert "timeout" not in out
        assert "max_retries" not in out

    def test_none_valued_params_do_not_override_default(self):
        req = Request(prompt="hi", params={"top_p": None, "temperature": 0.7})
        out = resolve_params(
            req, {"temperature": 0.0, "top_p": 1.0},
            {"temperature": "temperature", "top_p": "top_p"},
        )
        assert out == {"temperature": 0.7, "top_p": 1.0}


# ---------------------------------------------------------------------------
# Generation settings are no longer accepted as model constructor kwargs --
# every backend must reject them clearly (not silently swallow them into
# **kwargs and crash later with a confusing duplicate-keyword TypeError once
# resolve_params() also supplies the same key from Request.params).
# ---------------------------------------------------------------------------
class TestConstructorRejectsGenerationKwargs:
    def _assert_rejects(self, model_cls, **extra_ctor_kwargs):
        with pytest.raises(AuditKitError, match="temperature"):
            model_cls(temperature=0.7, **extra_ctor_kwargs)

    def test_openai_model_rejects_temperature(self):
        from auditkit.model.openai import OpenAIModel
        self._assert_rejects(OpenAIModel, api_key="test")

    def test_anthropic_model_rejects_temperature(self):
        from auditkit.model.anthropic import AnthropicModel
        self._assert_rejects(AnthropicModel, api_key="test")

    def test_groq_model_rejects_temperature(self):
        from auditkit.model.groq_gen import GroqModel
        self._assert_rejects(GroqModel, api_key="test")

    def test_openrouter_model_rejects_temperature(self):
        from auditkit.model.openrouter_gen import OpenRouterModel
        self._assert_rejects(OpenRouterModel, api_key="test")

    def test_litellm_model_rejects_temperature(self):
        from auditkit.model.litellm_gen import LiteLLMModel
        self._assert_rejects(LiteLLMModel)

    def test_api_model_rejects_temperature(self):
        from auditkit.model.api_gen import APIModel
        self._assert_rejects(APIModel)

    def test_hf_gen_model_rejects_temperature(self):
        from auditkit.model.hf_gen import HFGenModel
        self._assert_rejects(HFGenModel)

    def test_vllm_model_rejects_temperature(self):
        from auditkit.model.vllm_gen import VLLMModel
        self._assert_rejects(VLLMModel)

    def test_lexsi_model_rejects_temperature(self):
        from auditkit.model.lexsi import LexsiModel
        self._assert_rejects(LexsiModel, api_key="test")

    def test_unrelated_extra_kwargs_still_pass_through(self):
        # Non-generation extras (e.g. HF's repetition_penalty) aren't in the
        # reserved set and must still reach _extra_kwargs untouched.
        from auditkit.model.hf_gen import HFGenModel
        m = HFGenModel(repetition_penalty=1.2)
        assert m._extra_kwargs == {"repetition_penalty": 1.2}


# ---------------------------------------------------------------------------
# Stress test: drive every real backend's generate() with a fake client and
# confirm the actual outgoing call reflects Request.params, not just
# whatever the Model was constructed with.
# ---------------------------------------------------------------------------
class TestOpenAIModelHonorsRequestParams:
    def test_temperature_override_reaches_the_real_api_call(self):
        from auditkit.model.openai import OpenAIModel

        m = OpenAIModel(api_key="test")
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )
        m._client = fake_client  # bypass _ensure_client's real openai.OpenAI() call

        req = Request(prompt="hi", params={"temperature": 0.9, "stop_sequences": ["\n"], "num_completions": 3})
        m.generate([req])

        _, kwargs = fake_client.chat.completions.create.call_args
        assert kwargs["temperature"] == 0.9      # overridden, not the shared default
        assert kwargs["stop"] == ["\n"]           # renamed stop_sequences -> stop
        assert kwargs["n"] == 3                   # renamed num_completions -> n

    def test_client_kwargs_reach_openai_constructor_not_the_call(self):
        """timeout/max_retries/default_headers/organization/project are real
        openai.OpenAI() CLIENT-construction kwargs -- chat.completions.create()
        doesn't accept max_retries/default_headers/organization/project at
        all. Previously these were blindly spread into **self._extra_kwargs
        on the per-request call instead. Mocks the openai module itself to
        exercise the real _ensure_client() code path, not a pre-populated
        m._client stand-in like the test above."""
        from types import ModuleType
        from unittest.mock import patch

        recorded_client_kwargs = {}

        class FakeOpenAI:
            def __init__(self, **kwargs):
                recorded_client_kwargs.update(kwargs)
                self.chat = MagicMock()
                self.chat.completions.create.return_value = SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
                )

        fake_openai_module = ModuleType("openai")
        fake_openai_module.OpenAI = FakeOpenAI

        with patch.dict(sys.modules, {"openai": fake_openai_module}):
            from auditkit.model.openai import OpenAIModel

            m = OpenAIModel(api_key="test", max_retries=5, organization="org-123")
            m.generate([Request(prompt="hi", params={})])

        assert recorded_client_kwargs["api_key"] == "test"
        assert recorded_client_kwargs["max_retries"] == 5
        assert recorded_client_kwargs["organization"] == "org-123"
        _, call_kwargs = m._client.chat.completions.create.call_args
        assert "max_retries" not in call_kwargs
        assert "organization" not in call_kwargs

    def test_shared_default_temperature_used_when_no_override(self):
        # Generation settings are no longer model-constructor kwargs -- there's
        # one shared fallback (DEFAULT_TEMPERATURE, sourced from RunConfig's own
        # dataclass default) used only when a request carries no override at all.
        from auditkit.model.openai import OpenAIModel

        m = OpenAIModel(api_key="test")
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )
        m._client = fake_client

        m.generate([Request(prompt="hi", params={})])
        _, kwargs = fake_client.chat.completions.create.call_args
        assert kwargs["temperature"] == DEFAULT_TEMPERATURE

    def test_runner_only_keys_never_reach_the_real_api_call(self):
        from auditkit.model.openai import OpenAIModel

        m = OpenAIModel(api_key="test")
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )
        m._client = fake_client

        # exactly what GenerationAdapter actually builds from a RunConfig
        req = Request(prompt="hi", params={
            "temperature": 0.0, "seed": 0, "max_retries": 3, "timeout": 30,
        })
        m.generate([req])
        _, kwargs = fake_client.chat.completions.create.call_args
        assert "max_retries" not in kwargs
        assert "timeout" not in kwargs


class TestAnthropicModelHonorsRequestParams:
    def test_stop_sequences_and_top_k_reach_the_real_api_call(self):
        from auditkit.model.anthropic import AnthropicModel

        m = AnthropicModel(api_key="test")
        fake_client = MagicMock()
        fake_client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(text="ok")]
        )
        m._client = fake_client

        req = Request(prompt="hi", params={"top_k": 40, "stop_sequences": ["<END>"], "presence_penalty": 0.5})
        m.generate([req])

        _, kwargs = fake_client.messages.create.call_args
        assert kwargs["top_k"] == 40
        assert kwargs["stop_sequences"] == ["<END>"]
        # Anthropic's real API has no presence_penalty -- must be silently dropped,
        # not forwarded (which would raise a real TypeError against the SDK).
        assert "presence_penalty" not in kwargs

    def test_client_kwargs_reach_anthropic_constructor_not_the_call(self):
        """max_retries/default_headers are real anthropic.Anthropic()
        CLIENT-construction kwargs -- messages.create() doesn't accept them
        at all. Previously blindly spread into **self._extra_kwargs on the
        per-request call instead. Mocks the anthropic module itself to
        exercise the real _ensure_client() code path."""
        from types import ModuleType
        from unittest.mock import patch

        recorded_client_kwargs = {}

        class FakeAnthropic:
            def __init__(self, **kwargs):
                recorded_client_kwargs.update(kwargs)
                self.messages = MagicMock()
                self.messages.create.return_value = SimpleNamespace(
                    content=[SimpleNamespace(text="ok")]
                )

        fake_anthropic_module = ModuleType("anthropic")
        fake_anthropic_module.Anthropic = FakeAnthropic

        with patch.dict(sys.modules, {"anthropic": fake_anthropic_module}):
            from auditkit.model.anthropic import AnthropicModel

            m = AnthropicModel(api_key="test", max_retries=5)
            m.generate([Request(prompt="hi", params={})])

        assert recorded_client_kwargs["api_key"] == "test"
        assert recorded_client_kwargs["max_retries"] == 5
        _, call_kwargs = m._client.messages.create.call_args
        assert "max_retries" not in call_kwargs

    def test_unsupported_openai_only_keys_are_dropped_not_forwarded(self):
        from auditkit.model.anthropic import AnthropicModel

        m = AnthropicModel(api_key="test")
        fake_client = MagicMock()
        fake_client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(text="ok")]
        )
        m._client = fake_client

        req = Request(prompt="hi", params={"num_completions": 5, "seed": 1, "frequency_penalty": 0.2})
        m.generate([req])
        _, kwargs = fake_client.messages.create.call_args
        assert "n" not in kwargs
        assert "seed" not in kwargs
        assert "frequency_penalty" not in kwargs


class TestLiteLLMModelHonorsRequestParams:
    def test_temperature_and_stop_reach_the_real_call(self, monkeypatch):
        fake_completion = MagicMock(return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        ))
        fake_module = SimpleNamespace(completion=fake_completion)
        monkeypatch.setitem(sys.modules, "litellm", fake_module)

        from auditkit.model.litellm_gen import LiteLLMModel
        m = LiteLLMModel()
        req = Request(prompt="hi", params={"temperature": 0.6, "stop_sequences": ["END"]})
        m.generate([req])

        _, kwargs = fake_completion.call_args
        assert kwargs["temperature"] == 0.6
        assert kwargs["stop"] == ["END"]


class TestGroqModelHonorsRequestParams:
    def test_verify_and_timeout_are_session_concerns_not_body_fields(self):
        """verify=/proxies=/cert=/timeout= are real requests.Session/post()
        concerns -- previously blindly spread into the JSON request BODY
        instead, so e.g. GroqModel(verify=False) silently did nothing rather
        than disabling TLS verification."""
        from auditkit.model.groq_gen import GroqModel

        m = GroqModel(api_key="test", verify=False, timeout=5)
        fake_session = MagicMock()
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        fake_session.post.return_value = fake_resp
        m._session = fake_session  # bypass _ensure_session's real requests.Session()

        m.generate([Request(prompt="hi", params={})])

        _, call_kwargs = fake_session.post.call_args
        assert call_kwargs["timeout"] == 5
        assert "verify" not in call_kwargs["json"]

    def test_verify_reaches_the_real_session_attribute(self):
        from auditkit.model.groq_gen import GroqModel

        m = GroqModel(api_key="test", verify=False)
        m._ensure_session()

        assert m._session.verify is False

    def test_http_error_message_includes_response_body(self):
        """requests' default HTTPError message drops the response body,
        which is where Groq actually explains what went wrong (e.g. an
        unsupported param for that model) -- surface it instead of a bare
        '400 Client Error: Bad Request'."""
        import requests
        from auditkit.model.groq_gen import GroqModel

        m = GroqModel(api_key="test")
        fake_session = MagicMock()
        fake_resp = MagicMock()
        fake_resp.text = '{"error":{"message":"seed is not supported by this model"}}'
        fake_resp.raise_for_status.side_effect = requests.HTTPError(
            "400 Client Error: Bad Request for url: https://api.groq.com/openai/v1/chat/completions",
            response=fake_resp,
        )
        fake_session.post.return_value = fake_resp
        m._session = fake_session

        with pytest.raises(requests.HTTPError) as exc_info:
            m.generate([Request(prompt="hi", params={})])

        assert "seed is not supported by this model" in str(exc_info.value)


class TestAPIModelHonorsRequestParams:
    def test_body_reflects_request_params_override(self, monkeypatch):
        from auditkit.model.api_gen import APIModel

        m = APIModel()
        fake_session = MagicMock()
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        fake_session.post.return_value = fake_resp
        m._session = fake_session

        req = Request(prompt="hi", params={"temperature": 0.8, "stop_sequences": ["\n\n"]})
        m.generate([req])

        _, call_kwargs = fake_session.post.call_args
        body = call_kwargs["json"]
        assert body["temperature"] == 0.8
        assert body["stop"] == ["\n\n"]

    def test_verify_and_timeout_are_session_concerns_not_body_fields(self):
        """verify=/proxies=/cert=/timeout= are real requests.Session/post()
        concerns -- previously these were blindly spread into the JSON
        request BODY instead, so e.g. APIModel(verify=False) silently did
        nothing (sent {"verify": false, ...} as an ignored extra field to
        the actual LLM API) rather than disabling TLS verification."""
        from auditkit.model.api_gen import APIModel

        m = APIModel(verify=False, timeout=5)
        fake_session = MagicMock()
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        fake_session.post.return_value = fake_resp
        m._session = fake_session  # bypass _ensure_session's real requests.Session()

        m.generate([Request(prompt="hi", params={})])

        _, call_kwargs = fake_session.post.call_args
        assert call_kwargs["timeout"] == 5
        assert "verify" not in call_kwargs["json"]

    def test_verify_reaches_the_real_session_attribute(self):
        from auditkit.model.api_gen import APIModel

        m = APIModel(verify=False, proxies={"https": "http://proxy:8080"})
        m._ensure_session()

        assert m._session.verify is False
        assert m._session.proxies == {"https": "http://proxy:8080"}

    def test_http_error_message_includes_response_body(self):
        import requests
        from auditkit.model.api_gen import APIModel

        m = APIModel()
        fake_session = MagicMock()
        fake_resp = MagicMock()
        fake_resp.text = '{"error":"invalid request: unknown field"}'
        fake_resp.raise_for_status.side_effect = requests.HTTPError(
            "400 Client Error: Bad Request", response=fake_resp,
        )
        fake_session.post.return_value = fake_resp
        m._session = fake_session

        with pytest.raises(requests.HTTPError) as exc_info:
            m.generate([Request(prompt="hi", params={})])

        assert "invalid request: unknown field" in str(exc_info.value)


class TestOpenRouterModelHonorsRequestParams:
    def test_body_reflects_request_params_override(self, monkeypatch):
        from auditkit.model.openrouter_gen import OpenRouterModel

        m = OpenRouterModel(api_key="test")
        fake_session = MagicMock()
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        fake_session.post.return_value = fake_resp
        m._session = fake_session

        req = Request(prompt="hi", params={"temperature": 0.8, "stop_sequences": ["\n\n"]})
        m.generate([req])

        call_args, call_kwargs = fake_session.post.call_args
        assert call_args[0] == "https://openrouter.ai/api/v1/chat/completions"
        body = call_kwargs["json"]
        assert body["model"] == "openai/gpt-4o-mini"
        assert body["temperature"] == 0.8
        assert body["stop"] == ["\n\n"]

    def test_verify_and_timeout_are_session_concerns_not_body_fields(self):
        from auditkit.model.openrouter_gen import OpenRouterModel

        m = OpenRouterModel(api_key="test", verify=False, timeout=5)
        fake_session = MagicMock()
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        fake_session.post.return_value = fake_resp
        m._session = fake_session

        m.generate([Request(prompt="hi", params={})])

        _, call_kwargs = fake_session.post.call_args
        assert call_kwargs["timeout"] == 5
        assert "verify" not in call_kwargs["json"]

    def test_null_content_from_reasoning_model_becomes_empty_string_not_none(self, monkeypatch):
        """Reasoning models (e.g. some OpenRouter-routed models) can return an
        explicit `"content": null` -- not a missing key -- when max_tokens is
        hit while the model is still inside its internal `reasoning` field.
        dict.get(key, "") only falls back to "" when the key is *absent*, so
        a present-but-null value must be coalesced explicitly or it passes
        through as literal None."""
        from auditkit.model.openrouter_gen import OpenRouterModel

        m = OpenRouterModel(api_key="test")
        fake_session = MagicMock()
        fake_resp = MagicMock()
        fake_resp.json.return_value = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning": "still thinking...",
                },
                "finish_reason": "length",
            }],
            "usage": {},
        }
        fake_session.post.return_value = fake_resp
        m._session = fake_session

        results = m.generate([Request(prompt="hi", params={})])

        assert results[0].completions[0].text == ""
        assert results[0].completions[0].text is not None

    def test_http_error_message_includes_response_body(self, monkeypatch):
        """requests' default HTTPError message drops the response body, which
        is where OpenRouter (and the underlying provider it routed to)
        actually explains what went wrong -- surface it instead of a bare
        '400 Client Error: Bad Request'."""
        import requests
        from auditkit.model.openrouter_gen import OpenRouterModel

        m = OpenRouterModel(api_key="test")
        fake_session = MagicMock()
        fake_resp = MagicMock()
        fake_resp.text = '{"error":{"message":"seed is not supported by this model"}}'
        fake_resp.raise_for_status.side_effect = requests.HTTPError(
            "400 Client Error: Bad Request for url: https://openrouter.ai/api/v1/chat/completions",
            response=fake_resp,
        )
        fake_session.post.return_value = fake_resp
        m._session = fake_session

        with pytest.raises(requests.HTTPError) as exc_info:
            m.generate([Request(prompt="hi", params={})])

        assert "seed is not supported by this model" in str(exc_info.value)


class TestHFGenModelHonorsRequestParams:
    def test_pipeline_receives_overridden_temperature_and_max_new_tokens(self):
        from auditkit.model.hf_gen import HFGenModel

        m = HFGenModel()
        fake_pipeline = MagicMock(return_value=[{"generated_text": "ok"}])
        m._pipeline = fake_pipeline  # bypass _ensure_pipeline's real model download

        req = Request(prompt="hi", params={"temperature": 0.7, "max_tokens": 256})
        m.generate([req])

        _, kwargs = fake_pipeline.call_args
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_new_tokens"] == 256  # renamed max_tokens -> max_new_tokens

    def test_stop_sequences_reach_generate_as_stop_strings_with_tokenizer(self):
        from auditkit.model.hf_gen import HFGenModel

        m = HFGenModel()
        fake_pipeline = MagicMock(return_value=[{"generated_text": "ok"}])
        fake_pipeline.tokenizer = MagicMock(name="fake_tokenizer")
        m._pipeline = fake_pipeline

        req = Request(prompt="hi", params={"stop_sequences": ["\n\n"]})
        m.generate([req])

        _, kwargs = fake_pipeline.call_args
        assert kwargs["stop_strings"] == ["\n\n"]
        # generate()'s stop_strings criterion needs the tokenizer passed
        # alongside it -- the pipeline doesn't supply this on its own.
        assert kwargs["tokenizer"] is fake_pipeline.tokenizer

    def test_load_time_kwargs_reach_pipeline_construction_not_the_call(self):
        """torch_dtype/trust_remote_code/device_map/... are transformers.
        pipeline() CONSTRUCTION kwargs -- the pipeline's __call__ doesn't
        accept them at all. Previously these were blindly spread into
        **self._extra_kwargs on the generate() call instead, which would
        raise TypeError for any of them. Mocks the transformers module
        itself to exercise the real _ensure_pipeline() code path."""
        from types import ModuleType
        from unittest.mock import patch

        recorded_pipeline_kwargs = {}

        def fake_pipeline_ctor(task, **kwargs):
            recorded_pipeline_kwargs.update(kwargs)
            fake_pipe = MagicMock(return_value=[{"generated_text": "ok"}])
            fake_pipe.model = MagicMock(generation_config=None)
            fake_pipe.generation_config = None
            return fake_pipe

        class _FakeLogitsProcessor:
            pass

        fake_transformers_module = ModuleType("transformers")
        fake_transformers_module.pipeline = fake_pipeline_ctor
        fake_transformers_module.set_seed = MagicMock()
        fake_transformers_module.LogitsProcessor = _FakeLogitsProcessor

        with patch.dict(sys.modules, {"transformers": fake_transformers_module}):
            from auditkit.model.hf_gen import HFGenModel

            m = HFGenModel(model="gpt2", device="cpu", torch_dtype="bfloat16", trust_remote_code=True)
            req = Request(prompt="hi", params={})
            m.generate([req])

        # Load-time kwargs reached pipeline() construction...
        assert recorded_pipeline_kwargs["torch_dtype"] == "bfloat16"
        assert recorded_pipeline_kwargs["trust_remote_code"] is True
        # ...and did NOT reach the per-call generate() invocation.
        _, call_kwargs = m._pipeline.call_args
        assert "torch_dtype" not in call_kwargs
        assert "trust_remote_code" not in call_kwargs

    def test_unrelated_extra_kwargs_still_reach_the_generate_call(self):
        """Real generation-time kwargs (repetition_penalty, num_beams, ...)
        aren't in _LOAD_TIME_KWARGS, so they must still reach the pipeline
        CALL, unaffected by the load-time-kwargs routing fix above."""
        from auditkit.model.hf_gen import HFGenModel

        m = HFGenModel(repetition_penalty=1.2)
        fake_pipeline = MagicMock(return_value=[{"generated_text": "ok"}])
        m._pipeline = fake_pipeline

        req = Request(prompt="hi", params={})
        m.generate([req])

        _, kwargs = fake_pipeline.call_args
        assert kwargs["repetition_penalty"] == 1.2

    def test_seed_resets_the_global_rng_before_generating(self, monkeypatch):
        from auditkit.model.hf_gen import HFGenModel
        import transformers

        m = HFGenModel()
        fake_pipeline = MagicMock(return_value=[{"generated_text": "ok"}])
        m._pipeline = fake_pipeline

        recorded_seed = {}
        monkeypatch.setattr(transformers, "set_seed", lambda s: recorded_seed.setdefault("seed", s))

        req = Request(prompt="hi", params={"seed": 42})
        m.generate([req])

        assert recorded_seed["seed"] == 42

    def test_no_seed_means_set_seed_never_called(self, monkeypatch):
        from auditkit.model.hf_gen import HFGenModel
        import transformers

        m = HFGenModel()
        fake_pipeline = MagicMock(return_value=[{"generated_text": "ok"}])
        m._pipeline = fake_pipeline

        called = MagicMock()
        monkeypatch.setattr(transformers, "set_seed", called)

        req = Request(prompt="hi", params={})
        m.generate([req])

        called.assert_not_called()


class TestVLLMModelHonorsRequestParams:
    def test_sampling_params_receives_overridden_values(self):
        from auditkit.model.vllm_gen import VLLMModel

        m = VLLMModel()
        recorded = {}

        def fake_sampling_params(**kwargs):
            recorded.update(kwargs)
            return kwargs

        fake_llm = MagicMock()
        fake_out = SimpleNamespace(outputs=[SimpleNamespace(text="ok")])
        fake_llm.generate.return_value = [fake_out]
        m._llm = fake_llm
        m._sampling_params = fake_sampling_params

        req = Request(prompt="hi", params={"temperature": 0.9, "stop_sequences": ["\n"]})
        m.generate([req])

        assert recorded["temperature"] == 0.9
        assert recorded["stop"] == ["\n"]

    def test_constructor_kwargs_reach_llm_not_sampling_params(self):
        """Engine-level vLLM args (gpu_memory_utilization, dtype,
        tensor_parallel_size, ...) passed to VLLMModel(**kwargs) must reach
        vllm.LLM()'s constructor, not SamplingParams() -- SamplingParams
        doesn't accept them at all and previously raised TypeError for any
        of them. Mocks the vllm module itself (not installed in this dev
        environment) to exercise the real _ensure_llm() code path, not just
        a pre-populated m._llm stand-in like the test above."""
        from types import ModuleType
        from unittest.mock import patch

        recorded_llm_kwargs = {}

        def fake_llm_ctor(model=None, **kwargs):
            recorded_llm_kwargs["model"] = model
            recorded_llm_kwargs.update(kwargs)
            return MagicMock()

        fake_vllm_module = ModuleType("vllm")
        fake_vllm_module.LLM = fake_llm_ctor
        fake_vllm_module.SamplingParams = MagicMock()

        with patch.dict(sys.modules, {"vllm": fake_vllm_module}):
            from auditkit.model.vllm_gen import VLLMModel

            m = VLLMModel(model="gpt2", gpu_memory_utilization=0.3, dtype="float16")
            m._ensure_llm()

        assert recorded_llm_kwargs["model"] == "gpt2"
        assert recorded_llm_kwargs["gpu_memory_utilization"] == 0.3
        assert recorded_llm_kwargs["dtype"] == "float16"

    def test_model_info_falls_back_to_identity_only_when_no_known_path_resolves(self):
        from auditkit.model.vllm_gen import VLLMModel

        m = VLLMModel()
        m._llm = MagicMock(spec=[])  # no attributes at all -> every candidate path raises AttributeError

        info = m.model_info()
        assert info == {"is_local": True, "model_name": "gpt2"}

    def test_model_info_reports_real_params_when_the_engine_path_resolves(self):
        import torch
        from auditkit.model.vllm_gen import VLLMModel

        m = VLLMModel()
        fake_module = torch.nn.Linear(4, 4, bias=False)  # 16 real params, known size
        m._llm = SimpleNamespace(
            llm_engine=SimpleNamespace(
                model_executor=SimpleNamespace(
                    driver_worker=SimpleNamespace(
                        model_runner=SimpleNamespace(model=fake_module)
                    )
                )
            )
        )

        info = m.model_info()
        assert info["is_local"] is True
        assert info["total_params"] == 16
        assert info["nonzero_params"] <= 16
        assert info["size_mb"] > 0


class TestVLLMModelEnvironmentDefaults:
    """VLLMModel bakes in fixes for real environment issues confirmed live
    on Colab (CUDA-fork crash, Jupyter stdout incompatibility, HF Hub Xet
    404s) so ordinary users never need to know about them, on any platform."""

    def test_environment_defaults_set_multiproc_and_xet(self, monkeypatch):
        # Calling the function directly (rather than reloading the module)
        # avoids creating a new VLLMModel class object -- module reload
        # would break isinstance checks in other already-imported test
        # files holding a reference to the original class.
        from auditkit.model.vllm_gen import _apply_environment_defaults

        monkeypatch.delenv("VLLM_ENABLE_V1_MULTIPROCESSING", raising=False)
        monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)

        _apply_environment_defaults()

        import os
        assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"
        assert os.environ["HF_HUB_DISABLE_XET"] == "1"

    def test_environment_defaults_do_not_override_an_explicit_caller_value(self, monkeypatch):
        from auditkit.model.vllm_gen import _apply_environment_defaults

        monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")

        _apply_environment_defaults()

        import os
        assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "1"

    def test_stdout_fix_swaps_to_real_streams_and_restores_after(self):
        import sys
        from auditkit.model.vllm_gen import _stdout_fix

        fake_stdout = object()
        original_stdout = sys.stdout
        sys.stdout = fake_stdout
        try:
            with _stdout_fix():
                assert sys.stdout is sys.__stdout__
            assert sys.stdout is fake_stdout  # restored after the block
        finally:
            sys.stdout = original_stdout

    def test_stdout_fix_restores_even_if_the_wrapped_code_raises(self):
        import sys
        from auditkit.model.vllm_gen import _stdout_fix

        fake_stdout = object()
        original_stdout = sys.stdout
        sys.stdout = fake_stdout
        try:
            with pytest.raises(ValueError):
                with _stdout_fix():
                    raise ValueError("boom")
            assert sys.stdout is fake_stdout
        finally:
            sys.stdout = original_stdout

    def test_ensure_llm_constructs_llm_inside_stdout_fix(self, monkeypatch):
        """A fake sys.stdout lacking fileno() must not crash _ensure_llm()
        -- confirms the real construction call site is actually wrapped,
        not just that the helper works in isolation."""
        import sys
        import types
        from auditkit.model.vllm_gen import VLLMModel

        class NoFilenoStream:
            def fileno(self):
                raise OSError("fileno")

        fake_vllm = types.ModuleType("vllm")

        class FakeLLM:
            def __init__(self, model):
                # Exercising this inside the real _stdout_fix() context is
                # the point -- sys.stdout must be the real stream here.
                sys.stdout.fileno()
                self.model = model

        fake_vllm.LLM = FakeLLM
        fake_vllm.SamplingParams = MagicMock()
        monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

        original_stdout = sys.stdout
        sys.stdout = NoFilenoStream()
        try:
            m = VLLMModel(model="gpt2")
            m._ensure_llm()  # would raise OSError here if not wrapped in _stdout_fix()
            assert m._llm.model == "gpt2"
        finally:
            sys.stdout = original_stdout


class TestLexsiModelHonorsRequestParams:
    def test_conservative_key_map_only_forwards_temperature_and_max_tokens(self):
        from auditkit.model.lexsi import LexsiModel

        m = LexsiModel(api_key="test")
        fake_session = MagicMock()
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"choices": [{"text": "ok"}]}
        fake_session.post.return_value = fake_resp
        m._session = fake_session

        # stop_sequences is deliberately NOT in Lexsi's key_map (unverified
        # proprietary API) -- must not appear in the outgoing body at all.
        req = Request(prompt="hi", params={"temperature": 0.5, "stop_sequences": ["\n"]})
        m.generate([req])

        _, call_kwargs = fake_session.post.call_args
        body = call_kwargs["json"]
        assert body["temperature"] == 0.5
        assert "stop" not in body
        assert "stop_sequences" not in body

    def test_verify_and_timeout_are_session_concerns_not_body_fields(self):
        from auditkit.model.lexsi import LexsiModel

        m = LexsiModel(api_key="test", verify=False, timeout=5)
        fake_session = MagicMock()
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"choices": [{"text": "ok"}]}
        fake_session.post.return_value = fake_resp
        m._session = fake_session

        m.generate([Request(prompt="hi", params={})])

        _, call_kwargs = fake_session.post.call_args
        assert call_kwargs["timeout"] == 5
        assert "verify" not in call_kwargs["json"]

    def test_http_error_message_includes_response_body(self):
        import requests
        from auditkit.model.lexsi import LexsiModel

        m = LexsiModel(api_key="test")
        fake_session = MagicMock()
        fake_resp = MagicMock()
        fake_resp.text = '{"error":"invalid request"}'
        fake_resp.raise_for_status.side_effect = requests.HTTPError(
            "400 Client Error: Bad Request", response=fake_resp,
        )
        fake_session.post.return_value = fake_resp
        m._session = fake_session

        with pytest.raises(requests.HTTPError) as exc_info:
            m.generate([Request(prompt="hi", params={})])

        assert "invalid request" in str(exc_info.value)
