"""Groq model backend (optional extra: auditkit[requests]).

Groq's API is a documented OpenAI-compatible endpoint
(https://api.groq.com/openai/v1, standard Bearer auth, /chat/completions
request/response shape) -- this backend is a thin, pre-configured wrapper so
you don't have to pass api_base=/api_key= through the generic api: backend
by hand every time.
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


class GroqModel(Model):
    """Call the Groq chat completions API.

    Parameters
    ----------
    model
        Model name on Groq (e.g. ``"llama-3.3-70b-versatile"``).
    api_key
        Groq API key. Falls back to the ``GROQ_API_KEY`` env var.
    api_base
        Base URL (default: ``https://api.groq.com/openai/v1``).
    """

    name = "groq"
    threadsafe = False

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: str | None = None,
        api_base: str = "https://api.groq.com/openai/v1",
        name: str = "groq",
        **kwargs: Any,
    ) -> None:
        reject_generation_kwargs(kwargs, "GroqModel")
        self._model_name = model
        self._api_base = api_base.rstrip("/")
        self._api_key = api_key or os.environ.get("GROQ_API_KEY")
        # verify=/proxies=/cert=/timeout= are real requests.Session concerns,
        # not API request-body fields -- split them out so e.g.
        # GroqModel(verify=False) actually disables TLS verification instead
        # of silently becoming an extra, ignored JSON field in the payload.
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
        self._session.headers.update({
            "Authorization": f"Bearer {self._api_key}" if self._api_key else "",
            "Content-Type": "application/json",
        })
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
                # requests' default HTTPError message drops the response
                # body, which is where Groq actually explains what's wrong
                # (an invalid model slug, an unsupported param, rate
                # limiting, ...) -- surface it instead of a bare "400
                # Client Error: Bad Request".
                raise type(e)(f"{e} -- response body: {resp.text}") from None
            latency_ms = (time.monotonic() - t0) * 1000
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
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
