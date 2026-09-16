"""OpenRouter model backend (optional extra: auditkit[requests]).

OpenRouter's API is a documented OpenAI-compatible endpoint
(https://openrouter.ai/api/v1, standard Bearer auth, /chat/completions
request/response shape) -- this backend is a thin, pre-configured wrapper so
you don't have to pass api_base=/api_key= through the generic api: backend
by hand every time. Same shape as GroqModel; OpenRouter itself is a router
in front of many providers (OpenAI, Anthropic, Meta, Mistral, Google, ...),
selected via a "provider/model" name (e.g. "openai/gpt-4o",
"anthropic/claude-3.5-sonnet", "meta-llama/llama-3.1-70b-instruct") rather
than one fixed model family.
"""
from __future__ import annotations

import os
import time
from typing import Any

from . import (
    Model, Request, Result_, Generated, resolve_params, resolve_messages,
    DEFAULT_TEMPERATURE, reject_generation_kwargs, split_session_kwargs,
)
from ..errors import ExtraNotInstalled

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


class OpenRouterModel(Model):
    """Call the OpenRouter chat completions API.

    Parameters
    ----------
    model
        Model name on OpenRouter, in ``"provider/model"`` form (e.g.
        ``"openai/gpt-4o"``, ``"anthropic/claude-3.5-sonnet"``,
        ``"meta-llama/llama-3.1-70b-instruct"``). The provider prefix is
        OpenRouter's own routing syntax, unrelated to AuditKit's
        ``"openrouter:"`` spec prefix -- ``"openrouter:openai/gpt-4o"``
        parses correctly since only the text before the *first* colon is
        AuditKit's own prefix.
    api_key
        OpenRouter API key. Falls back to the ``OPENROUTER_API_KEY`` env var.
    api_base
        Base URL (default: ``https://openrouter.ai/api/v1``).
    site_url, app_name
        Optional -- OpenRouter's own leaderboard/analytics attribution
        headers (``HTTP-Referer``/``X-Title``). Not required for the API
        to work; harmless to omit.
    """

    name = "openrouter"
    threadsafe = False

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        api_key: str | None = None,
        api_base: str = "https://openrouter.ai/api/v1",
        site_url: str | None = None,
        app_name: str | None = None,
        name: str = "openrouter",
        **kwargs: Any,
    ) -> None:
        reject_generation_kwargs(kwargs, "OpenRouterModel")
        self._model_name = model
        self._api_base = api_base.rstrip("/")
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self._site_url = site_url
        self._app_name = app_name
        # verify=/proxies=/cert=/timeout= are real requests.Session concerns,
        # not API request-body fields -- split them out so e.g.
        # OpenRouterModel(verify=False) actually disables TLS verification
        # instead of silently becoming an extra, ignored JSON field.
        self._session_kwargs, self._extra_kwargs = split_session_kwargs(kwargs)
        self._session = None
        self.name = name

    def _ensure_session(self) -> None:
        if self._session is not None:
            return
        try:
            import requests
        except ImportError:
            raise ExtraNotInstalled("requests", "pip install auditkit[requests]")
        self._session = requests.Session()
        headers = {
            "Authorization": f"Bearer {self._api_key}" if self._api_key else "",
            "Content-Type": "application/json",
        }
        if self._site_url:
            headers["HTTP-Referer"] = self._site_url
        if self._app_name:
            headers["X-Title"] = self._app_name
        self._session.headers.update(headers)
        for attr in ("verify", "proxies", "cert"):
            if attr in self._session_kwargs:
                setattr(self._session, attr, self._session_kwargs[attr])

    def generate(self, requests: list[Request]) -> list[Result_]:
        self._ensure_session()
        results = []
        for r in requests:
            defaults = {"temperature": DEFAULT_TEMPERATURE, "max_tokens": 1024}
            gen_kwargs = resolve_params(r, defaults, _KEY_MAP)
            body = {
                "model": self._model_name,
                "messages": resolve_messages(r),
                **gen_kwargs,
                **self._extra_kwargs,
            }
            timeout = self._session_kwargs.get("timeout", 120)
            t0 = time.monotonic()
            resp = self._session.post(f"{self._api_base}/chat/completions", json=body, timeout=timeout)
            try:
                resp.raise_for_status()
            except Exception as e:
                # requests' default HTTPError message drops the response body,
                # which is where OpenRouter (and the underlying provider it
                # routed to) actually explains what's wrong -- an invalid
                # model slug, an unsupported param for that specific model,
                # rate limiting, etc. Surface it instead of a bare "400
                # Client Error: Bad Request".
                raise type(e)(f"{e} -- response body: {resp.text}") from None
            latency_ms = (time.monotonic() - t0) * 1000
            data = resp.json()
            message = data.get("choices", [{}])[0].get("message", {}) or {}
            # Reasoning models (e.g. some OpenRouter-routed models) can return
            # an explicit `"content": null` -- not a missing key -- when
            # max_tokens is hit while the model is still inside its internal
            # `reasoning` field. dict.get(key, "") only falls back to "" when
            # the key is *absent*, so a present-but-null value passes through
            # as None otherwise; `or ""` catches both cases.
            text = message.get("content") or ""
            raw_usage = data.get("usage") or {}
            usage = {}
            if raw_usage:
                usage = {
                    "prompt_tokens": raw_usage.get("prompt_tokens", 0),
                    "completion_tokens": raw_usage.get("completion_tokens", 0),
                    "total_tokens": raw_usage.get("total_tokens", 0),
                }
            results.append(Result_(completions=[Generated(text=text)], usage=usage, latency_ms=latency_ms))
        return results
