"""Tests for generic API model backend."""

from __future__ import annotations

import pytest

from auditkit.model import AutoModel
from auditkit.model.api_gen import APIModel
from auditkit.model.groq_gen import GroqModel
from auditkit.model.openrouter_gen import OpenRouterModel
from auditkit.errors import AuditKitError


class TestAPIBackend:
    def test_api_prefix_resolves(self):
        m = AutoModel.resolve("api:gpt-4o")
        assert isinstance(m, APIModel)
        assert m._model_name == "gpt-4o"

    def test_defaults(self):
        m = APIModel()
        assert m._api_base == "https://api.openai.com/v1"
        assert m._chat_template is True

    def test_completion_mode(self):
        m = APIModel(chat_template=False)
        assert m._chat_template is False

    def test_name(self):
        assert APIModel().name == "api"

    def test_no_more_t1_stubs(self):
        """All prefixes should now resolve, none should raise T1 errors."""
        for prefix in ["openai:", "anthropic:", "hf:", "lexsi:", "groq:", "openrouter:", "vllm:", "litellm:", "api:"]:
            try:
                AutoModel.resolve(f"{prefix}test")
            except AuditKitError as e:
                if "not in T0" in str(e):
                    pytest.fail(f"Prefix {prefix} still raises T1 error")


class TestGroqBackend:
    def test_groq_prefix_resolves(self):
        m = AutoModel.resolve("groq:llama-3.3-70b-versatile")
        assert isinstance(m, GroqModel)
        assert m._model_name == "llama-3.3-70b-versatile"

    def test_defaults(self):
        m = GroqModel()
        assert m._api_base == "https://api.groq.com/openai/v1"
        assert m._model_name == "llama-3.3-70b-versatile"

    def test_name(self):
        assert GroqModel().name == "groq"

    def test_api_key_falls_back_to_env_var(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test123")
        m = GroqModel()
        assert m._api_key == "gsk_test123"

    def test_explicit_api_key_wins_over_env_var(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_env")
        m = GroqModel(api_key="gsk_explicit")
        assert m._api_key == "gsk_explicit"


class TestOpenRouterBackend:
    def test_openrouter_prefix_resolves(self):
        m = AutoModel.resolve("openrouter:openai/gpt-4o-mini")
        assert isinstance(m, OpenRouterModel)
        assert m._model_name == "openai/gpt-4o-mini"

    def test_provider_slash_model_name_parses_correctly(self):
        """OpenRouter's own model names contain a '/' (provider/model) --
        confirms that doesn't get mangled by AutoModel's prefix-stripping,
        which only splits on the first ':'."""
        m = AutoModel.resolve("openrouter:anthropic/claude-3.5-sonnet")
        assert m._model_name == "anthropic/claude-3.5-sonnet"

    def test_defaults(self):
        m = OpenRouterModel()
        assert m._api_base == "https://openrouter.ai/api/v1"
        assert m._model_name == "openai/gpt-4o-mini"

    def test_name(self):
        assert OpenRouterModel().name == "openrouter"

    def test_api_key_falls_back_to_env_var(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "or_test123")
        m = OpenRouterModel()
        assert m._api_key == "or_test123"

    def test_explicit_api_key_wins_over_env_var(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "or_env")
        m = OpenRouterModel(api_key="or_explicit")
        assert m._api_key == "or_explicit"

    def test_optional_attribution_headers(self):
        m = OpenRouterModel(site_url="https://example.com", app_name="MyApp")
        m._ensure_session()
        assert m._session.headers["HTTP-Referer"] == "https://example.com"
        assert m._session.headers["X-Title"] == "MyApp"

    def test_attribution_headers_omitted_by_default(self):
        m = OpenRouterModel(api_key="test")
        m._ensure_session()
        assert "HTTP-Referer" not in m._session.headers
        assert "X-Title" not in m._session.headers
