"""OpenAI model backend (optional extra: auditkit[openai])."""
from __future__ import annotations

from typing import Any

from . import (
    Model, Request, Result_, Generated, resolve_params, resolve_messages,
    DEFAULT_TEMPERATURE, reject_generation_kwargs,
)
from ..errors import ExtraNotInstalled

# AuditKit-internal RunConfig key -> OpenAI chat.completions.create() kwarg name.
# Only these keys are ever pulled from Request.params; timeout/max_retries/
# top_k/best_of are either Runner-only or unsupported by this API, so they're
# deliberately absent and can never reach the real API call.
_KEY_MAP = {
    "temperature": "temperature",
    "top_p": "top_p",
    "max_tokens": "max_tokens",
    "stop_sequences": "stop",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
    "num_completions": "n",
    "seed": "seed",
}

# Real openai.OpenAI() CLIENT-construction kwargs -- must reach the client
# constructor in _ensure_client(), not chat.completions.create(). Previously
# blindly spread into **self._extra_kwargs on the per-request call instead,
# where max_retries/default_headers/default_query/organization/project
# don't exist at all and raise TypeError (timeout happens to be valid on
# both, so it alone never surfaced this). Anything else in **kwargs is
# assumed to be a real chat.completions.create()-level kwarg and keeps
# going there, unaffected. api_key/base_url are already handled via their
# own explicit constructor params, not this set.
_CLIENT_KWARGS = frozenset({
    "timeout", "max_retries", "default_headers", "default_query",
    "organization", "project",
})


class OpenAIModel(Model):
    """Call the OpenAI chat completions API."""

    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None,
                 api_base: str | None = None, name: str = "openai", **kwargs: Any) -> None:
        reject_generation_kwargs(kwargs, "OpenAIModel")
        self._model_name = model
        self._extra_kwargs = kwargs
        self._client = None
        self._api_key = api_key
        self._api_base = api_base
        self.name = name

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            import openai
        except ImportError:
            raise ExtraNotInstalled("openai", "pip install auditkit[openai]")
        client_kwargs = {k: v for k, v in self._extra_kwargs.items() if k in _CLIENT_KWARGS}
        self._client = openai.OpenAI(api_key=self._api_key, base_url=self._api_base, **client_kwargs)

    def generate(self, requests: list[Request]) -> list[Result_]:
        self._ensure_client()
        results = []
        for r in requests:
            defaults = {"temperature": DEFAULT_TEMPERATURE, "max_tokens": 1024}
            gen_kwargs = resolve_params(r, defaults, _KEY_MAP)
            call_kwargs = {k: v for k, v in self._extra_kwargs.items() if k not in _CLIENT_KWARGS}
            resp = self._client.chat.completions.create(
                model=self._model_name,
                messages=resolve_messages(r),
                **gen_kwargs,
                **call_kwargs,
            )
            text = resp.choices[0].message.content or ""
            usage = {}
            resp_usage = getattr(resp, "usage", None)
            if resp_usage is not None:
                usage = {
                    "prompt_tokens": resp_usage.prompt_tokens,
                    "completion_tokens": resp_usage.completion_tokens,
                    "total_tokens": resp_usage.total_tokens,
                }
            results.append(Result_(completions=[Generated(text=text)], usage=usage))
        return results
