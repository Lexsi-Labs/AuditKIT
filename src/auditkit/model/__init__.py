"""The model layer: a uniform, batched interface over any way of generating text.

Everything downstream sees the same :class:`Model` — one abstract :meth:`generate`
that takes a batch of :class:`Request` and returns a batch of :class:`Result_`.
Backends (hf, vllm, litellm, api, lexsi, ...) live in ``model/`` as optional
extras; the two defined here have no dependencies: :class:`CallableModel` wraps
any ``list[str] -> list[str]`` function, and :class:`EchoModel` is a
deterministic test double. Loglikelihood is opt-in — a backend that doesn't
declare it raises :class:`CapabilityError`, which is how MC-by-loglikelihood
greys out for chat-only APIs.
"""

from __future__ import annotations

import hashlib
import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..errors import AuditKitError, CapabilityError, ExtraNotInstalled
from ..runspec import RunConfig
from ..types import Capability

# The one shared fallback for generation settings a backend uses when a
# request carries no override at all (calling generate() directly, bypassing
# the adapter/RunConfig pipeline). Sourced from RunConfig's own dataclass
# default instead of each backend inventing its own literal, so "default
# temperature" has one definition project-wide. max_tokens isn't included --
# RunConfig.max_tokens defaults to None (no cap), but a real API call/local
# generation call needs a concrete number, so each backend still picks its
# own (based on cost/context: small for local inference, larger for hosted
# APIs).
DEFAULT_TEMPERATURE = RunConfig().temperature

# Every generation-behavior field an adapter can put in Request.params (union
# of every backend's _KEY_MAP). No longer accepted as model constructor
# kwargs -- RunConfig is the one place to set them. Rejected explicitly at
# construction time so passing e.g. temperature=0.7 the old way raises a
# clear error immediately, instead of being silently swallowed into a
# backend's **kwargs passthrough and crashing later with a confusing
# "got multiple values for keyword argument" TypeError at generate() time.
_RESERVED_GENERATION_KWARGS = frozenset({
    "temperature", "top_p", "top_k", "max_tokens", "max_new_tokens",
    "stop_sequences", "presence_penalty", "frequency_penalty",
    "num_completions", "seed",
})


def reject_generation_kwargs(kwargs: dict[str, Any], backend_name: str) -> None:
    """Raise a clear error if *kwargs* (a backend's ``**kwargs`` catch-all)
    contains a generation-behavior setting that belongs in ``RunConfig``."""
    bad = _RESERVED_GENERATION_KWARGS & kwargs.keys()
    if bad:
        raise AuditKitError(
            f"{backend_name} received {sorted(bad)} as constructor kwargs, but "
            f"generation settings are no longer set on the model -- pass them "
            f"via RunConfig instead, e.g. "
            f"ak.evaluate(..., config=RunConfig({bad.pop()}=...))."
        )


def _prompt_text(prompt: str) -> str:
    """Render a request prompt as plain text for text-only backends."""
    return prompt


# requests.Session()-based backends (api_gen/groq_gen/openrouter_gen/lexsi)
# take no interesting constructor kwargs on Session() itself -- these knobs
# are real requests.Session attributes / per-call requests.post() kwargs
# instead. Previously every one of these backends spread its entire
# **extra_kwargs catch-all into the JSON request BODY sent to the actual
# LLM API with no exceptions, so e.g. APIModel(verify=False) (a very
# reasonable attempt to skip TLS verification for a self-signed internal
# endpoint) silently sent {"verify": false, ...} as an extra field in the
# chat-completions payload instead -- doing nothing, with no error, while
# looking like it worked. timeout was also hardcoded to 120 in every one of
# these backends with no way to override it at all.
_SESSION_KWARGS = frozenset({"timeout", "verify", "proxies", "cert"})


def split_session_kwargs(kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a requests-based backend's ``**kwargs`` catch-all into
    ``(session_kwargs, body_kwargs)``. ``session_kwargs`` holds real
    ``requests`` concerns (``verify``, ``proxies``, ``cert`` as
    ``Session`` attributes; ``timeout`` as a per-call override) --
    everything else is assumed to be a real API request-body field and
    flows through to the JSON payload unchanged, exactly as before."""
    session_kwargs = {k: v for k, v in kwargs.items() if k in _SESSION_KWARGS}
    body_kwargs = {k: v for k, v in kwargs.items() if k not in _SESSION_KWARGS}
    return session_kwargs, body_kwargs


def is_transient(e: BaseException) -> bool:
    """HTTP 429/5xx or a dropped connection: worth re-sending the request.

    A timeout is NOT transient here -- the server may still be running the
    request, and re-sending it duplicates an agent's side effects.
    """
    status = getattr(e, "status", None)
    if status is None:
        status = getattr(getattr(e, "response", None), "status_code", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    reason = getattr(e, "reason", e)  # urllib's URLError wraps the socket error
    return isinstance(reason, ConnectionError)


def generate_each(
    requests: list["Request"], one: Callable[["Request"], "Result_"], *,
    max_retries: int = 0, retry_delay: float = 1.0, where: str = "", log: Any = None,
) -> list["Result_"]:
    """``[one(r) for r in requests]`` where a failure fails only that request.

    Never raises: the runner's batch retry would re-send every earlier,
    already-answered request (duplicating billed or side-effecting calls).
    A failed request becomes ``finish_reason="error"`` with the reason in
    ``Generated.error``; transient failures (:func:`is_transient`) are retried
    up to *max_retries* times with exponential backoff first.
    """
    import logging
    import time
    log = log or logging.getLogger(__name__)
    out = []
    for r in requests:
        t0 = time.monotonic()
        for attempt in range(max_retries + 1):
            try:
                out.append(one(r))
                break
            except Exception as e:  # noqa: BLE001 -- any failure is this request's alone
                if attempt < max_retries and is_transient(e):
                    log.warning("%s request failed (retry %d/%d): %s", where, attempt + 1, max_retries, e)
                    time.sleep(retry_delay * (2 ** attempt))
                    continue
                log.warning("%s request failed: %s", where, e)
                out.append(Result_(
                    completions=[Generated(text="", finish_reason="error",
                                           error=f"{type(e).__name__}: {e}"[:1000])],
                    latency_ms=(time.monotonic() - t0) * 1000,
                ))
                break
    return out


def resolve_params(
    request: "Request",
    defaults: dict[str, Any],
    key_map: dict[str, str],
) -> dict[str, Any]:
    """Merge a request's per-request generation params over a backend's
    fallback defaults, restricted to keys that backend's real API actually
    accepts.

    Adapters build ``Request.params`` from ``RunConfig`` using AuditKit's own
    internal field names (``stop_sequences``, ``num_completions``, ...), but
    real provider APIs don't all use those names, and none of them accept
    every field (Anthropic has no ``presence_penalty``; OpenAI's chat API
    calls it ``stop`` not ``stop_sequences``; ``timeout``/``max_retries`` are
    Runner-only settings no provider API accepts at all).

    ``defaults`` uses the *target API's* own kwarg names, seeded with each
    backend's own fallback constants (e.g. ``{"temperature": DEFAULT_TEMPERATURE}``)
    used only when a request carries no override at all -- calling
    ``generate()`` directly, bypassing the adapter/``RunConfig`` pipeline.
    Generation settings are no longer accepted as model constructor kwargs:
    ``RunConfig`` (via an adapter) is the one place to set them, since every
    adapter always writes a value into ``Request.params`` for the fields it
    knows about, which would silently shadow a constructor default anyway.
    ``key_map`` maps AuditKit's internal key -> that API's real kwarg name;
    only keys listed here are ever read out of ``request.params``, which is
    what keeps unrelated keys (like ``timeout``/``max_retries``) from ever
    reaching a real API call.
    """
    out = dict(defaults)
    for internal_key, api_key in key_map.items():
        if internal_key in request.params and request.params[internal_key] is not None:
            out[api_key] = request.params[internal_key]
    return out


def template_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A copy of *messages* ready for a chat template: each assistant
    ``tool_calls[].function.arguments`` given as a JSON **string** (OpenAI format,
    AgentTune transcripts) is decoded to a dict. HF chat templates expect a dict;
    Command R7B's would render the string as ``"parameters": "{\\"a\\": 2}"``, a
    malformed call in the model's own history. Undecodable strings are left as-is."""
    import json

    out: list[dict[str, Any]] = []
    for m in messages:
        calls = m.get("tool_calls") if isinstance(m, dict) else None
        if not calls:
            out.append(m)
            continue
        fixed = []
        for c in calls:
            fn = c.get("function") if isinstance(c, dict) else None
            if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
                try:
                    args = json.loads(fn["arguments"] or "{}")
                except ValueError:
                    args = None
                if isinstance(args, dict):
                    c = {**c, "function": {**fn, "arguments": args}}
            fixed.append(c)
        out.append({**m, "tool_calls": fixed})
    return out


def resolve_messages(request: "Request") -> list[dict[str, str]]:
    """The role-tagged chat turns for *request*, for backends with a real
    ``messages``-based API (OpenAI, Anthropic, Groq, LiteLLM, chat-mode
    ``api:``).

    :class:`~auditkit.adapter.ChatAdapter` (and any other adapter that builds
    multi-turn prompts) stores its ``role``/``content`` turns in
    ``request.params["messages"]`` -- this is the one place that structure is
    actually read back out, instead of every backend collapsing it down to
    ``request.prompt`` and losing the system/user role split. Adapters that
    don't set it (``GenerationAdapter``, etc.) still work: a single ``user``
    turn is built from ``request.prompt``, matching prior behavior exactly.

    ``request.params["single_message"] = True`` collapses a real multi-turn
    conversation down to one ``user`` message instead of dropping it back to
    ``request.prompt`` (which would lose the other turns' content entirely).
    Not backend-specific: some hosted providers serve certain models (guard/
    classifier-style ones in particular) through a narrower endpoint contract
    that only accepts a single message, even though the model is reached
    through the same general chat-completions URL as every other model on
    that provider -- confirmed live against Groq's hosted ``llama-guard-3-8b``.
    This is a caller-driven opt-in (e.g. :class:`~auditkit.metrics.guard.
    GuardJudge`'s ``single_message=``), not a hardcoded per-model/per-provider
    table -- whoever knows a given model needs it turns it on for that model.
    """
    messages = request.params.get("messages")
    if not messages:
        return [{"role": "user", "content": _prompt_text(request.prompt)}]
    if request.params.get("single_message") and len(messages) > 1:
        combined = "\n\n".join(f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}" for m in messages)
        return [{"role": "user", "content": combined}]
    return messages


@dataclass
class Request:
    """One thing to send a model. Task-specific knobs live in ``params``
    (``num_completions``, ``stop``, ``target`` for a loglikelihood continuation)."""

    prompt: str
    request_type: str = "generate"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Generated:
    """One completion for one request."""

    text: str = ""
    logprob: Optional[float] = None
    finish_reason: Optional[str] = None
    media: Optional[list[Any]] = None
    # Structured output beyond text, for agents and RAG pipelines:
    # ``{"tool_calls": [[...turns...]], "messages": [...], "retrieved_contexts": [...]}``.
    # The Runner copies it into the scoring context as ``context["trace"]``.
    trace: Optional[dict[str, Any]] = None
    # Why a per-request failure happened (finish_reason="error"); the Runner
    # puts it in RunResult.errors instead of a generic message.
    error: Optional[str] = None


@dataclass
class Result_:
    """A model's response to one request: its completions plus bookkeeping."""

    completions: list[Generated]
    usage: dict[str, Any] = field(default_factory=dict)
    latency_ms: Optional[float] = None

    @property
    def text(self) -> str:
        """The first completion's text (``""`` when there are none)."""
        return self.completions[0].text if self.completions else ""


@dataclass
class LogLikelihood:
    """Log-probability of a target continuation, and whether it was the argmax."""

    logprob: float
    is_greedy: bool = False


# The first transformers release that knows each architecture AuditKit is documented to run,
# where it is newer than the oldest transformers the ecosystem still ships (#51).
_ARCH_MIN_TRANSFORMERS = {"cohere_compass": "5.15"}


def explain_unknown_architecture(err: BaseException, model: str) -> Optional[ValueError]:
    """transformers' "model type `x` but Transformers does not recognize this architecture",
    rewritten to say what actually happened: an installed transformers too old for the model.

    That message reads like a broken checkpoint and cost several Colab runs in #51, where a stale
    auditkit (1.0.0, ``transformers<5``) had quietly downgraded transformers. Returns None for any
    other error.
    """
    import re
    msg = str(err)
    m = re.search(r"model type `([\w.-]+)` but Transformers does not recognize this architecture", msg)
    if not m:
        return None
    model_type = m.group(1)
    try:
        from importlib import metadata
        installed = metadata.version("transformers")
    except Exception:   # noqa: BLE001
        installed = "unknown"
    try:
        from importlib import metadata
        auditkit_version = metadata.version("auditkit")
    except Exception:   # noqa: BLE001
        auditkit_version = None
    floor = _ARCH_MIN_TRANSFORMERS.get(model_type)
    need = f"transformers>={floor}" if floor else "a newer transformers"
    fix = f"transformers>={floor},<6" if floor else "transformers>=5.15,<6"
    return ValueError(
        f"{model!r} is a {model_type!r} model, and the installed transformers {installed} does not know "
        f"that architecture: it needs {need}. The checkpoint is fine. AuditKit's [transformers] extra "
        f"requires transformers>=5.15,<6, so an older one means something else pinned it; the usual "
        f"cause is an auditkit that isn't this one (installed auditkit: {auditkit_version or 'not found'}; "
        f"check `pip show auditkit` and where it was installed from). Fix: pip install '{fix}'."
        f"\n\ntransformers said: {msg}")


def _free_torch_memory() -> None:
    """gc + clear the torch CUDA/MPS allocator cache. Best-effort: a no-op if
    torch isn't importable. Used by local backends' ``unload()`` so a freed
    model's GPU memory is actually released (torch caches freed blocks in its
    own allocator, so dropping the Python reference alone doesn't return it)."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available() and hasattr(torch, "mps"):
            torch.mps.empty_cache()
    except Exception:
        pass


class Model(ABC):
    """A batched text generator. Subclasses implement :meth:`generate`."""

    name: str = "model"
    threadsafe: bool = False
    # True only for backends that load real weights locally (HFGenModel,
    # VLLMModel) -- the only case where parameter count/on-disk size are
    # actually introspectable. Every hosted-API backend (openai, anthropic,
    # groq, litellm, api, lexsi) and the callable/echo test doubles stay
    # False: there is no local checkpoint to measure, by construction, not
    # because someone forgot to wire it up.
    is_local: bool = False

    def capabilities(self) -> set[Capability]:
        return {Capability.GENERATE}

    def model_info(self) -> dict[str, Any]:
        """Size/identity info for comparison views -- never caller-supplied.

        Local backends override this to introspect their loaded checkpoint
        (parameter count, on-disk size, ...). The default here is what every
        hosted-API/callable backend gets: no measurable size, just enough
        identity (``model_name``) for a comparison to say "these are two
        different API-based models" instead of pretending to a size number
        that doesn't exist.
        """
        return {
            "is_local": self.is_local,
            "model_name": getattr(self, "_model_name", self.name),
        }

    def unload(self) -> None:
        """Release any locally-loaded weights and GPU memory.

        Default is a no-op: hosted-API and callable/echo backends load nothing
        locally, so there's nothing to free. Local backends (HFGenModel,
        VLLMModel) override this to drop their loaded pipeline/engine and clear
        the CUDA cache -- used by ``compare_models()`` to free one model before
        loading the next, so peak GPU memory stays at a single model instead of
        accumulating across the comparison. Safe to call more than once; the
        backend lazily reloads on the next ``generate()`` if reused.
        """
        return

    def run_notes(self) -> dict[str, Any]:
        """Facts about the finished run that only the backend knows, merged
        into ``RunResult.metadata``; reading them resets them for the next run.

        ``api:`` reports the server it detected and the request fields it may
        have ignored (``api_server``, ``unverified_request_fields``). Default: none.
        """
        return {}

    @abstractmethod
    def generate(self, requests: list[Request]) -> list[Result_]:
        """Generate for a batch of requests, one :class:`Result_` per request."""
        ...

    def loglikelihood(self, requests: list[Request]) -> list[LogLikelihood]:
        """Score target continuations. Backends opt in; the default declines."""
        raise CapabilityError(f"{self.name} does not declare LOGLIKELIHOOD")

    def supports(self, cap: Capability) -> bool:
        return cap in self.capabilities()

    def identity(self) -> dict:
        """The config that defines this model's identity, for the run
        fingerprint. ``RunSpec.fingerprint()`` used to hash only ``self.name``
        directly -- fine when resolved via a string spec (``AutoModel.
        resolve()`` sets ``name=spec``, e.g. ``"hf:gpt2"``), but every real
        backend defaults ``.name`` to a fixed class-level string (``"hf"``,
        ``"openai"``, ...) unless a caller passes ``name=`` explicitly. Two
        directly-constructed instances of the same backend class pointing at
        genuinely different checkpoints (``HFGenModel(model="gpt2")`` vs.
        ``HFGenModel(model="gpt2-medium")``) then silently collide into one
        cached result. Every real backend consistently stores its checkpoint
        string in ``self._model_name``, so including it here fixes this for
        all of them at once, with no per-backend override needed."""
        return {"name": self.name, "model": getattr(self, "_model_name", None)}


def _stable_capture_repr(value: Any) -> Optional[str]:
    """A deterministic repr for a closure-captured *config* value, or None to skip.

    Only immutable scalars (and tuples/frozensets of them) are folded into a
    callable's identity. These are the values that legitimately identify a model
    -- a model name, a provider, an endpoint string, a numeric setting -- and
    their repr is stable from one run to the next.

    Everything else is skipped: mutable containers (list/dict/set) and arbitrary
    objects are either irrelevant runtime state a closure happens to capture (a
    call-log accumulator, an HTTP client, a logger) or carry an address-based
    repr, and folding those in would make the fingerprint change between two
    logically-identical runs and defeat the disk cache. Skipping can only cause a
    cache *miss* (when a model is identified solely by such a value), never a
    wrong cache *hit*.
    """
    if value is None or isinstance(value, (str, bytes, bool, int, float, complex)):
        return repr(value)
    if isinstance(value, (tuple, frozenset)):
        parts = [_stable_capture_repr(v) for v in value]
        if all(p is not None for p in parts):
            return "(" + ",".join(parts) + ")"
    return None


def _default_callable_name(fn: Callable[..., Any]) -> str:
    """A default name that actually distinguishes different wrapped callables.

    Derived from the function's qualified name plus a short hash of its
    bytecode *and its immutable captured closure values*, so two different
    functions passed as ``model=`` (the common "bring your own model" path) — or
    as a scorer's ``judge_model=`` — don't collide under one generic literal
    name, which previously made :meth:`RunSpec.fingerprint` (and therefore the
    disk cache) treat any two such callables as the same model.
    """
    qualname = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None) or repr(fn)
    parts: list[str] = []
    code = getattr(fn, "__code__", None)
    if code is not None:
        # co_code alone is not enough: CPython stores referenced names
        # (attribute/method names, globals) and literal constants separately
        # in co_names/co_consts, addressed by index -- two functions that
        # call different methods (e.g. .upper() vs .lower()) can have
        # byte-identical co_code while only their co_names differ.
        parts += [repr(code.co_code), repr(code.co_names), repr(code.co_consts),
                  repr(code.co_freevars)]
    else:
        parts.append(repr(fn))
    # Bytecode is identical for every closure built by the same factory: a
    # gateway/judge callable produced by `build_callable(model=...)` closes the
    # differing model over its *free variables*, which live in __closure__ cells,
    # not in co_code/co_names/co_consts. Fold the *immutable* captured values in
    # (paired with their free-var name, so skipping one can't shift the others)
    # so two judges/guards (or models-under-test) that differ only by what they
    # close over -- provider, project_name, model_name -- get distinct names and
    # distinct fingerprints, while identical captures still hash identically so
    # the cache replays a genuinely-repeated run. Mutable/opaque captures are
    # skipped (see _stable_capture_repr): folding those in would make the name
    # change between identical runs and break caching.
    freevars = getattr(code, "co_freevars", ()) if code is not None else ()
    closure = getattr(fn, "__closure__", None) or ()
    captured = []
    for name, cell in zip(freevars, closure):
        try:
            r = _stable_capture_repr(cell.cell_contents)
        except Exception:
            r = None
        if r is not None:
            captured.append((name, r))
    if captured:
        parts.append(repr(captured))
    code_hash = hashlib.sha256("".join(parts).encode()).hexdigest()[:8]
    return f"callable:{qualname}:{code_hash}"


class CallableModel(Model):
    """Wrap any ``list[str] -> list[str]`` function as a model.

    This is the escape hatch: a user's own app, an SDK call, a lambda — anything
    that maps a batch of prompts to a batch of completions plugs straight in.
    """

    threadsafe: bool = True

    def __init__(self, fn: Callable[[list[str]], list[str]], name: Optional[str] = None) -> None:
        self._fn = fn
        self.name = name if name is not None else _default_callable_name(fn)

    def generate(self, requests: list[Request]) -> list[Result_]:
        outputs = self._fn([_prompt_text(r.prompt) for r in requests])
        return [Result_(completions=[Generated(text=o)]) for o in outputs]


class EchoModel(Model):
    """Returns each prompt unchanged — a deterministic double for tests/dry-runs."""

    name = "echo"
    threadsafe: bool = True

    def generate(self, requests: list[Request]) -> list[Result_]:
        return [Result_(completions=[Generated(text=_prompt_text(r.prompt))])
                for r in requests]


class PrecomputedModel(Model):
    """Scores answers that already exist on the samples (``Sample.actual_output``).

    No model is called — the Runner reads each sample's ``actual_output`` as the
    output. This is the second stage of the generate→score flow: generate once
    (``ak.generate``), then score the stored answers with any scorer, repeatedly.
    """

    name = "precomputed"
    threadsafe: bool = True
    reads_actual_output: bool = True

    def generate(self, requests: list[Request]) -> list[Result_]:  # pragma: no cover
        # The Runner special-cases reads_actual_output and never calls this.
        return [Result_(completions=[Generated(text="")]) for _ in requests]


_T1_BACKENDS: dict[str, tuple[str, str, str]] = {
    "openai:": ("model.openai", "OpenAIModel", "openai"),
    "anthropic:": ("model.anthropic", "AnthropicModel", "anthropic"),
    "hf:": ("model.hf_gen", "HFGenModel", "transformers"),
    "lexsi:": ("model.lexsi", "LexsiModel", "requests"),   # pip install auditkit[requests]
    "vllm:": ("model.vllm_gen", "VLLMModel", "vllm"),
    "litellm:": ("model.litellm_gen", "LiteLLMModel", "litellm"),
    "api:": ("model.api_gen", "APIModel", "requests"),   # pip install auditkit[requests]
    "groq:": ("model.groq_gen", "GroqModel", "requests"),   # pip install auditkit[requests]
    "openrouter:": ("model.openrouter_gen", "OpenRouterModel", "requests"),   # pip install auditkit[requests]
    "agent:": ("model.agent_endpoint", "AgentEndpointModel", "core"),   # stdlib only: agent:https://host/path
}

_T1_PREFIXES = ()

# A bare prefix ("api:") is decided per backend (#43):
# - these need a name: a served model or one to load. A constructor default
#   (gpt-4o-mini, gpt2) would evaluate a model nobody asked for, and api: used
#   to send {"model": null};
# - agent: passes None through, since the URL may come as url= and the backend
#   raises itself when neither is given;
# - every other (hosted) prefix falls back to its backend's default model.
_NEEDS_MODEL_NAME = {"api:": "api:Qwen/Qwen2.5-7B-Instruct", "hf:": "hf:CohereLabs/tiny-aya-global",
                     "vllm:": "vllm:Qwen/Qwen2.5-7B-Instruct"}
_BARE_PASSES_NONE = {"agent:"}


class AutoModel:
    """Factory that resolves a spec to a :class:`Model` ready to use.

    T0 built-in backends are resolved inline; everything else raises a clear
    "not in T0" error — those backends land in T1.
    """

    @classmethod
    def resolve(cls, spec: Any, **opts: Any) -> Model:
        """Resolve *spec* to a :class:`Model`.

        Parameters
        ----------
        spec : Model, callable, or str
            - :class:`Model` — returned as-is
            - ``list[str] -> list[str]`` callable — wrapped in :class:`CallableModel`
            - ``"openai:..."``, ``"anthropic:..."``, ``"hf:..."``, ``"lexsi:..."``,
              ``"groq:..."``, ``"openrouter:..."`` — resolved to the corresponding T1 backend
            - ``"vllm:..."``, ``"litellm:..."``, ``"api:..."``: also resolved to their
              T1 backends; ``"agent:<url>"`` resolves to
              :class:`~auditkit.model.agent_endpoint.AgentEndpointModel`, an
              externally deployed agent called over HTTP (stdlib only)
        **opts
            Extra keyword args forwarded to the backend constructor.
        """
        if isinstance(spec, Model):
            return spec
        if callable(spec):
            return CallableModel(spec, **opts)
        if isinstance(spec, str):
            for prefix, (mod, cls_name, extra) in _T1_BACKENDS.items():
                if spec.startswith(prefix):
                    model_name = spec[len(prefix):] if spec[len(prefix):].strip() else None
                    if model_name is None and prefix in _NEEDS_MODEL_NAME:
                        raise AuditKitError(f"empty model spec {spec!r}: name the model, "
                                            f"e.g. {_NEEDS_MODEL_NAME[prefix]!r}")
                    try:
                        m = importlib.import_module(f"auditkit.{mod}")
                        backend_cls = getattr(m, cls_name)
                    except ImportError:
                        raise ExtraNotInstalled(extra, f"pip install auditkit[{extra}]")
                    if model_name is None and prefix not in _BARE_PASSES_NONE:
                        return backend_cls(name=spec, **opts)
                    return backend_cls(model=model_name, name=spec, **opts)
            for prefix in _T1_PREFIXES:
                if spec.startswith(prefix):
                    raise AuditKitError(
                        f"'{spec}' uses a {prefix.rstrip(':')} backend which is not "
                        f"in T0; those backends land in T1."
                    )
            if spec == "echo":
                return EchoModel()
            if spec == "precomputed":
                return PrecomputedModel()
            if spec.startswith("lmeval:"):
                raise AuditKitError(
                    "lm-eval is a benchmark *engine*, not a model backend. Run it "
                    "with ak.run_lmeval(tasks, model='hf:...') or "
                    "ak.evaluate(tasks, model='hf:...', engine='lmeval')."
                )
            raise AuditKitError(f"unknown model spec {spec!r}; known: 'precomputed', 'openai:', 'anthropic:', 'hf:', 'lexsi:', 'groq:', 'openrouter:', 'vllm:', 'litellm:', 'api:', 'agent:'")
        raise AuditKitError(f"cannot resolve model spec: {spec!r}")
