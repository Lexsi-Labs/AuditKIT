"""Anthropic model backend (optional extra: auditkit[anthropic])."""
from __future__ import annotations

from typing import Any

from . import (
    Model, Request, Result_, Generated, resolve_params, resolve_messages,
    DEFAULT_TEMPERATURE, reject_generation_kwargs,
)
from ..errors import ExtraNotInstalled

# Anthropic's Messages API supports far fewer knobs than OpenAI's -- no
# presence/frequency penalty, no seed, no multi-completion (n). Only list
# what it actually accepts; everything else in Request.params is ignored.
_KEY_MAP = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "max_tokens": "max_tokens",
    "stop_sequences": "stop_sequences",  # Anthropic uses this name natively
}

# Real anthropic.Anthropic() CLIENT-construction kwargs -- must reach the
# client constructor in _ensure_client(), not messages.create(). Previously
# blindly spread into **self._extra_kwargs on the per-request call instead,
# where max_retries/default_headers/default_query don't exist at all and
# raise TypeError (timeout happens to be valid on both, so it alone never
# surfaced this). Anything else in **kwargs is assumed to be a real
# messages.create()-level kwarg and keeps going there, unaffected.
_CLIENT_KWARGS = frozenset({
    "base_url", "timeout", "max_retries", "default_headers", "default_query",
})


class AnthropicModel(Model):
    """Call the Anthropic messages API."""

    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: str | None = None,
                 name: str = "anthropic", **kwargs: Any) -> None:
        reject_generation_kwargs(kwargs, "AnthropicModel")
        self._model_name = model
        self._extra_kwargs = kwargs
        self._client = None
        self._api_key = api_key
        self.name = name

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            import anthropic
        except ImportError:
            raise ExtraNotInstalled("anthropic", "pip install auditkit[anthropic]")
        client_kwargs = {k: v for k, v in self._extra_kwargs.items() if k in _CLIENT_KWARGS}
        self._client = anthropic.Anthropic(api_key=self._api_key, **client_kwargs)

    def generate(self, requests: list[Request]) -> list[Result_]:
        self._ensure_client()
        results = []
        for r in requests:
            defaults = {"temperature": DEFAULT_TEMPERATURE, "max_tokens": 1024}
            gen_kwargs = resolve_params(r, defaults, _KEY_MAP)
            # Anthropic's Messages API has no "system" role inside `messages` --
            # unlike OpenAI, the system prompt is its own top-level `system=`
            # kwarg. Pull any system turn(s) out before sending the rest.
            turns = resolve_messages(r)
            system = "\n\n".join(m["content"] for m in turns if m.get("role") == "system")
            messages = [m for m in turns if m.get("role") != "system"]
            if system:
                gen_kwargs["system"] = system
            call_kwargs = {k: v for k, v in self._extra_kwargs.items() if k not in _CLIENT_KWARGS}
            resp = self._client.messages.create(
                model=self._model_name,
                messages=messages,
                **gen_kwargs,
                **call_kwargs,
            )
            text = "".join(b.text for b in resp.content if hasattr(b, "text"))
            usage = {}
            resp_usage = getattr(resp, "usage", None)
            if resp_usage is not None:
                # Anthropic names these input_tokens/output_tokens, not
                # prompt/completion -- normalize to the same shape every
                # other backend uses so cost calculation doesn't need to
                # know per-provider field names.
                usage = {
                    "prompt_tokens": resp_usage.input_tokens,
                    "completion_tokens": resp_usage.output_tokens,
                    "total_tokens": resp_usage.input_tokens + resp_usage.output_tokens,
                }
            results.append(Result_(completions=[Generated(text=text)], usage=usage))
        return results
