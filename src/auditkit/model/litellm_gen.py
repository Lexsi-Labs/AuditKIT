"""LiteLLM unified API backend (optional extra: auditkit[litellm])."""
from __future__ import annotations

from typing import Any, Sequence

from . import (
    Model, Request, Result_, Generated, resolve_params, resolve_messages,
    DEFAULT_TEMPERATURE, reject_generation_kwargs, remote_identity,
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
                 name: str = "litellm",
                 identity_extra: Sequence[str] = (), **kwargs: Any) -> None:
        reject_generation_kwargs(kwargs, "LiteLLMModel")
        self._model_name = model
        self._extra_kwargs = kwargs
        self._identity_extra = tuple(identity_extra)
        self.name = name

    def _env_base_names(self) -> tuple[str, ...]:
        """The base-URL variables litellm reads for this model's provider (``azure/...`` ->
        ``AZURE_API_BASE``, no prefix -> OpenAI's)."""
        provider = str(self._extra_kwargs.get("custom_llm_provider")
                       or (self._model_name.split("/", 1)[0] if "/" in self._model_name else "openai"))
        p = provider.upper().replace("-", "_")
        return (f"{p}_API_BASE", f"{p}_BASE_URL") + (("AZURE_OPENAI_ENDPOINT",) if p == "AZURE" else ())

    def identity(self) -> dict:
        # api_base and an allowlist of extra kwargs (see remote_identity); the rest stay out.
        return {**super().identity(), **remote_identity(self, None, env_base=self._env_base_names())}

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
