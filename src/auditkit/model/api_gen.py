"""Generic OpenAI-compatible HTTP API backend (optional extra: auditkit[requests])."""
from __future__ import annotations

import os
from typing import Any

from . import (
    Model, Request, Result_, Generated, resolve_params, resolve_messages,
    DEFAULT_TEMPERATURE, reject_generation_kwargs, split_session_kwargs,
)
from ..errors import ExtraNotInstalled

# OpenAI-compatible endpoints (the whole point of this backend) use
# OpenAI's own kwarg names.
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


class APIModel(Model):
    """Call any OpenAI-compatible HTTP API.

    Parameters
    ----------
    model
        Model name sent in the API request body.
    api_base
        Base URL of the API endpoint (default: ``https://api.openai.com/v1``).
    api_key
        API key. Falls back to ``API_KEY`` env var, then ``OPENAI_API_KEY``.
    chat_template
        If True (default), uses ``/chat/completions`` with messages format.
        If False, uses ``/completions`` with raw prompt.
    """

    name = "api"
    threadsafe = False

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_base: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        chat_template: bool = True,
        name: str = "api",
        **kwargs: Any,
    ) -> None:
        reject_generation_kwargs(kwargs, "APIModel")
        self._model_name = model
        self._api_base = api_base.rstrip("/")
        self._api_key = api_key or os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
        self._chat_template = chat_template
        # verify=/proxies=/cert=/timeout= are real requests.Session concerns,
        # not API request-body fields -- split them out so e.g.
        # APIModel(verify=False) actually disables TLS verification (useful
        # for a self-signed internal endpoint) instead of silently becoming
        # an extra, ignored JSON field in the chat-completions payload.
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
            prompt = r.prompt if isinstance(r.prompt, str) else str(r.prompt)
            defaults = {"temperature": DEFAULT_TEMPERATURE, "max_tokens": 1024}
            gen_kwargs = resolve_params(r, defaults, _KEY_MAP)
            if self._chat_template:
                endpoint = f"{self._api_base}/chat/completions"
                body = {
                    "model": self._model_name,
                    "messages": resolve_messages(r),
                    **gen_kwargs,
                }
            else:
                endpoint = f"{self._api_base}/completions"
                body = {
                    "model": self._model_name,
                    "prompt": prompt,
                    **gen_kwargs,
                }
            body.update(self._extra_kwargs)
            timeout = self._session_kwargs.get("timeout", 120)
            import time
            t0 = time.monotonic()
            resp = self._session.post(endpoint, json=body, timeout=timeout)
            try:
                resp.raise_for_status()
            except Exception as e:
                # requests' default HTTPError message drops the response
                # body, which is where the real API actually explains what's
                # wrong. Surface it instead of a bare "400 Client Error".
                raise type(e)(f"{e} -- response body: {resp.text}") from None
            latency_ms = (time.monotonic() - t0) * 1000
            data = resp.json()
            if self._chat_template:
                text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                text = data.get("choices", [{}])[0].get("text", "")
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
