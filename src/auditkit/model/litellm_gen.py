"""LiteLLM unified API backend (optional extra: auditkit[litellm])."""
from __future__ import annotations

from typing import Any

from . import (
    Model, Request, Result_, Generated, resolve_params, resolve_messages,
    DEFAULT_TEMPERATURE, reject_generation_kwargs,
)
from ..errors import ExtraNotInstalled

# LiteLLM normalizes most providers to OpenAI-style kwarg names.
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


class LiteLLMModel(Model):
    """Call any LLM through the LiteLLM unified interface."""

    name = "litellm"
    threadsafe = True

    def __init__(self, model: str = "gpt-4o-mini",
                 name: str = "litellm", **kwargs: Any) -> None:
        reject_generation_kwargs(kwargs, "LiteLLMModel")
        self._model_name = model
        self._extra_kwargs = kwargs
        self.name = name

    def generate(self, requests: list[Request]) -> list[Result_]:
        try:
            from litellm import completion
        except ImportError:
            raise ExtraNotInstalled("litellm", "pip install auditkit[litellm]")
        results = []
        for r in requests:
            defaults = {"temperature": DEFAULT_TEMPERATURE, "max_tokens": 1024}
            gen_kwargs = resolve_params(r, defaults, _KEY_MAP)
            resp = completion(
                model=self._model_name,
                messages=resolve_messages(r),
                **gen_kwargs,
                **self._extra_kwargs,
            )
            text = resp.choices[0].message.content or ""
            usage = {}
            resp_usage = getattr(resp, "usage", None)
            if resp_usage is not None:
                # LiteLLM normalizes every provider's response to this
                # OpenAI-shaped usage object regardless of backend.
                usage = {
                    "prompt_tokens": resp_usage.prompt_tokens,
                    "completion_tokens": resp_usage.completion_tokens,
                    "total_tokens": resp_usage.total_tokens,
                }
            results.append(Result_(completions=[Generated(text=text)], usage=usage))
        return results
