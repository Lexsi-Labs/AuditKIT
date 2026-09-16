"""Adapters: how a :class:`Sample` becomes model :class:`Request` objects.

The adapter is the technique of *prompting* — the same sample can be elicited as
a plain generation, as multiple-choice by joint prompt or by loglikelihood, as a
chat turn, as an adversarial probe, and so on. Each ``method`` is a different way
to turn a sample into requests. :class:`GenerationAdapter` is the plainest: ask
the model to generate an answer to the input.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod

from .model import Request
from .registry import ADAPTERS
from .runspec import RunConfig
from .sample import Sample
from ._identity_guard import warn_if_identity_incomplete


# The RunConfig fields that are *generation* settings. Every adapter forwards
# exactly this set into each Request's params, so which knob reaches the model
# never depends on which adapter you picked (previously GenerationAdapter
# forwarded the full set while the other five silently dropped presence_penalty/
# frequency_penalty/top_k/num_completions/best_of). timeout/max_retries are
# deliberately excluded — they're Runner-only, not generation params. Each
# backend's own key-map still decides which of these its API actually accepts;
# forwarding one a backend doesn't support is a safe no-op.
_GEN_PARAM_ATTRS = (
    "temperature", "top_p", "top_k", "max_tokens", "stop_sequences",
    "presence_penalty", "frequency_penalty", "num_completions", "best_of", "seed",
)


def _gen_params(config: RunConfig) -> dict:
    """Generation settings from *config* to put on a Request (skips unset ones)."""
    return {
        attr: getattr(config, attr)
        for attr in _GEN_PARAM_ATTRS
        if getattr(config, attr, None) is not None
    }


class Adapter(ABC):
    """Turn one sample into the requests that elicit an answer."""

    method: str = "generation"

    @abstractmethod
    def adapt(self, sample: Sample, config: RunConfig) -> list[Request]:
        ...

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        warn_if_identity_incomplete(cls, Adapter, "adapt", name_attr="method")

    def identity(self) -> dict:
        """The config that defines this adapter's prompts, for the run fingerprint.

        Two adapters with the same identity produce the same prompts, so they are
        the same "ruler". Prompt-bearing adapters override this to include their
        template/system prompt — otherwise changing a prompt would leave the
        fingerprint unchanged and a stale cached result would be returned.
        ``__init_subclass__`` above warns at class-definition time when a
        subclass takes constructor arguments but skips this override (unless
        it instead makes ``self.method`` itself parameter-derived, which
        already protects the fingerprint — see ``_identity_guard.py``)."""
        return {"method": self.method}


@ADAPTERS.register("generation")
class GenerationAdapter(Adapter):
    """One free-form generation request per sample."""

    method = "generation"

    def adapt(self, sample: Sample, config: RunConfig) -> list[Request]:
        params = _gen_params(config)
        return [Request(prompt=sample.input, request_type="generate", params=params)]


@ADAPTERS.register("mcq")
class MCQAdapter(Adapter):
    """One sample → one joint-prompt request (mcq_joint) or N loglikelihood requests (mcq_loglikelihood)."""

    def __init__(self, method: str = "mcq_joint") -> None:
        self.method = method

    @property
    def method(self) -> str:
        return self._method

    @method.setter
    def method(self, value: str) -> None:
        self._method = value

    def adapt(self, sample: Sample, config: RunConfig) -> list[Request]:
        if not sample.choices:
            raise ValueError("MCQAdapter requires choices")
        if self._method == "mcq_joint":
            # Numeric, 0-based labels: unlike chr(65+i) letters, these never run
            # out (letters break past 26 choices — chr(91) is '[', not a
            # letter — and collide once index >= 32 wraps into lowercase
            # ASCII). 0-based to match the index convention used everywhere
            # else a choice is identified (sample.target, _resolve_choice_index).
            prompt = sample.input_text + "\n\n"
            for i, choice in enumerate(sample.choices):
                prompt += f"{i}. {choice}\n"
            prompt += "\nAnswer:"
            return [Request(prompt=prompt, request_type="generate")]
        return [
            Request(prompt=sample.input, request_type="loglikelihood", params={"target": c})
            for c in sample.choices
        ]


@ADAPTERS.register("chat")
class ChatAdapter(Adapter):
    """Formats samples as chat messages using a system prompt template."""

    method = "chat"

    def __init__(self, system_prompt: str = "You are a helpful assistant.") -> None:
        self.system_prompt = system_prompt

    def identity(self) -> dict:
        return {"method": self.method, "system_prompt": self.system_prompt}

    def adapt(self, sample: Sample, config: RunConfig) -> list[Request]:
        params = _gen_params(config)
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": sample.input}]
        # request.params["messages"] is now read by 5 real backends (via
        # resolve_messages(): openai.py, anthropic.py, groq.py, litellm.py,
        # chat-mode api.py) and hf_gen.py's own chat-template rendering --
        # but EchoModel/CallableModel and any bare-callable model= still only
        # ever read request.prompt, so the flattened "System: ...\n\nUser: ..."
        # text stays the one guaranteed-reachable form; messages is kept
        # alongside it for backends that prefer real chat turns.
        prompt = f"System: {self.system_prompt}\n\nUser: {sample.input}"
        return [Request(prompt=prompt, request_type="chat", params={**params, "messages": messages})]


@ADAPTERS.register("fewshot")
class FewShotAdapter(Adapter):
    """Prepends few-shot examples from the dataset before each sample.

    ``num_shots`` is this adapter's own default shot count. ``RunConfig.num_fewshot``
    -- the run-level knob every other engine treats as authoritative (it's the
    ``lm-eval`` name too) -- overrides it when set, so ``evaluate(..., adapter=
    FewShotAdapter(), config=RunConfig(num_fewshot=5))`` actually uses 5 shots
    instead of silently keeping the adapter's own default.

    ``pool`` is a plain public attribute, not a private ``_pool`` -- unlike
    every other adapter's config, it needs to be externally assignable:
    ``Runner.build_requests()`` auto-populates it from ``RunConfig.split``'s
    train fold (``if hasattr(adapter, "pool"): adapter.pool = train``), which
    silently never fired while this was named ``_pool``.
    """

    method = "fewshot"

    def __init__(self, num_shots: int = 3, separator: str = "\n\n",
                 pool: list[Sample] | None = None) -> None:
        self.num_shots = num_shots
        self.separator = separator
        self.pool = pool

    def identity(self) -> dict:
        pool_hash = None
        if self.pool:
            pool_hash = hashlib.sha256(json.dumps(
                [(s.input, s.target) for s in self.pool],
                default=str, sort_keys=True,
            ).encode()).hexdigest()[:12]
        return {
            "method": self.method,
            "num_shots": self.num_shots,
            "separator": self.separator,
            "pool_size": len(self.pool) if self.pool else 0,
            "pool_hash": pool_hash,
        }

    def adapt(self, sample: Sample, config: RunConfig) -> list[Request]:
        params = _gen_params(config)
        n = config.num_fewshot if config.num_fewshot is not None else self.num_shots
        pool = self.pool or []
        if n > 0 and len(pool) < n:
            raise ValueError(
                f"FewShotAdapter needs a pool of at least {n} example(s) to build "
                f"{n}-shot prompts, but the pool has only {len(pool)}. Pass a "
                f"bigger pool=, or lower num_shots / RunConfig.num_fewshot."
            )
        prefix = ""
        for fs in pool[:n]:
            prefix += f"{fs.input}\n{fs.target or ''}{self.separator}"
        return [Request(prompt=f"{prefix}{sample.input}", request_type="generate", params=params)]


@ADAPTERS.register("instruction")
class InstructionAdapter(Adapter):
    """Prepends an instruction/system prompt to each sample input."""

    method = "instruction"

    def __init__(self, instruction: str = "Answer the following question:") -> None:
        self.instruction = instruction

    def identity(self) -> dict:
        return {"method": self.method, "instruction": self.instruction}

    def adapt(self, sample: Sample, config: RunConfig) -> list[Request]:
        params = _gen_params(config)
        return [Request(prompt=f"{self.instruction}\n{sample.input}", request_type="generate", params=params)]


@ADAPTERS.register("rag")
class RAGAdapter(Adapter):
    """Prepends retrieval_context to the sample input."""

    method = "rag"

    def __init__(self, context_separator: str = "\nContext:\n", max_context_chars: int | None = None) -> None:
        self.context_separator = context_separator
        self.max_context_chars = max_context_chars

    def identity(self) -> dict:
        return {"method": self.method, "context_separator": self.context_separator,
                "max_context_chars": self.max_context_chars}

    def adapt(self, sample: Sample, config: RunConfig) -> list[Request]:
        params = _gen_params(config)
        ctx = sample.retrieval_context or []
        if not ctx:
            raise ValueError(
                "RAGAdapter requires sample.retrieval_context; this sample has "
                "none. Pass samples with retrieval_context=[...], or use a "
                "different adapter if retrieval isn't part of this task."
            )
        context_str = self.context_separator + "\n".join(ctx)
        if self.max_context_chars and len(context_str) > self.max_context_chars:
            context_str = context_str[:self.max_context_chars] + "..."
        return [Request(prompt=f"{sample.input}{context_str}", request_type="generate", params=params)]


@ADAPTERS.register("template")
class TemplateAdapter(Adapter):
    """Uses a Python format string with {input}, {target}, {context} placeholders."""

    method = "template"

    def __init__(self, template: str = "{input}") -> None:
        self.template = template

    def identity(self) -> dict:
        return {"method": self.method, "template": self.template}

    def adapt(self, sample: Sample, config: RunConfig) -> list[Request]:
        params = _gen_params(config)
        ctx = "\n".join(sample.retrieval_context or [])
        prompt = self.template.format(input=sample.input, target=sample.target or "", context=ctx)
        return [Request(prompt=prompt, request_type="generate", params=params)]
