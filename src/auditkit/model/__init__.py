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

import functools
import hashlib
import importlib
import math
import os
import re
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Any, Callable, Optional
from urllib.parse import urlsplit, urlunsplit

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


# --- Secret handling for fingerprints --------------------------------------------------------
# Fail closed: a value is kept out of the fingerprint when its key *or* the value itself looks
# like a credential. Only the name is kept (so adding a key still re-keys the run); the value
# becomes "<redacted>", never a hash. A rotated key therefore does not invalidate the cache.
_REDACTED = "<redacted>"

# Matched as substrings of the key lower-cased with camelCase split and separators dropped, so
# ``api_key``, ``apiKey``, ``x-api-key``, ``AccessToken``, ``clientSecret`` and ``Authorization``
# all hit.
_SECRET_WORDS = ("key", "secret", "token", "password", "passwd", "passphrase", "credential",
                 "auth", "cookie", "bearer", "signature")
# Names that contain a secret word but carry a count, a tokenizer setting or a name, not a
# credential: ``max_tokens``, ``max_new_tokens``, ``budget_tokens``, ``tokenizer_kwargs``,
# ``token_type_ids``, ``keyword``. They change outputs, so they must stay in the fingerprint.
_SAFE_KEY = re.compile(r"tokens$|^tokenizer|tokentypeids$|^keyword|^keep|^authors?$")
# Keys whose values are identifiers, not credentials (a 40-hex commit would otherwise look like a key).
_ID_KEY = re.compile(r"revision$|^(model|engine|deployment(id)?)$")
# Query parameters whose values are settings, not credentials.
_SAFE_QUERY = frozenset({"api-version", "api_version", "version", "deployment", "deployment_id",
                         "model", "region", "location"})
# Values that are credentials whatever they are called.
_SECRET_VALUE = re.compile(
    r"(sk-|sk_|pk-|gsk_|hf_|ghp_|gho_|ghs_|github_pat_|xox[abprs]-|AKIA|ASIA|AIza)[A-Za-z0-9_\-]{8,}"
    r"|eyJ[A-Za-z0-9_\-]{6,}\.[A-Za-z0-9_\-]{6,}"
    r"|(?i:bearer|basic)\s+\S{8,}|-----BEGIN[^\n]*KEY-----")
_KEYLIKE = re.compile(r"[A-Za-z0-9_\-+=.]{24,}")   # no "/" or spaces: model ids, paths and prose stay


def _compact(key: Any) -> str:
    """``apiKey`` / ``x-api-key`` / ``API_KEY`` -> ``apikey`` / ``xapikey`` / ``apikey``."""
    return re.sub(r"[-_. ]+", "", re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(key)).lower())


def _secret_key(key: Any) -> bool:
    c = _compact(key)
    return any(w in c for w in _SECRET_WORDS) and not _SAFE_KEY.search(c)


def _secret_value(text: str, key: Any = None) -> bool:
    if _SECRET_VALUE.search(text):
        return True
    if key is not None and _ID_KEY.search(_compact(key)):
        return False
    if not _KEYLIKE.fullmatch(text) or not (any(c.isdigit() for c in text) and any(c.isalpha() for c in text)):
        return False
    counts = {c: text.count(c) for c in set(text)}
    entropy = -sum(n / len(text) * math.log2(n / len(text)) for n in counts.values())
    return entropy >= 3.5


def _scrub_url(url: str) -> str:
    """*url* without credentials: no ``user:pass@``, no fragment, every query value blanked
    (``&`` and ``;`` separated) except version-like settings, no trailing slash, so
    ``https://h/v1`` and ``https://h/v1/`` agree."""
    try:
        parts = urlsplit(url)
        host = parts.netloc.rpartition("@")[2]
        pairs = [q.partition("=") for q in re.split(r"[&;]", parts.query) if q]
        query = "&".join(f"{k}={v if k.lower() in _SAFE_QUERY and not _secret_value(v) else _REDACTED}"
                         if sep else k for k, sep, v in pairs)
        return urlunsplit((parts.scheme, host, parts.path.rstrip("/"), query, ""))
    except ValueError:
        return _REDACTED


def _pairs(value: Any) -> Optional[list]:
    """*value* as ``[(name, value), ...]`` when it is a mapping or a list/tuple of 2-item
    (name, value) pairs with string names, else None."""
    if isinstance(value, Mapping):
        return list(value.items())
    if (isinstance(value, (list, tuple)) and value and all(
            isinstance(p, (list, tuple)) and len(p) == 2 and isinstance(p[0], str) for p in value)):
        return [tuple(p) for p in value]
    return None


def _identity_value(value: Any, key: Any = None) -> Any:
    """*value* as JSON-stable data for a fingerprint, with secrets removed.

    Mappings and lists of (name, value) pairs (``default_headers=[("Authorization", ...)]``) are
    walked as key/value pairs; objects with ``to_dict()`` (``BitsAndBytesConfig``) are expanded; a
    ``torch.dtype`` is its bare name; any other object is reduced to its type (its repr carries a
    memory address). Anything under a secret-like key (``api_key``, ``apiKey``, ``Authorization``),
    any string that looks like a credential, a ``Name: secret`` / ``name=secret`` string, and the
    credentials in a URL become ``"<redacted>"``. Everything else is kept, normalised.
    """
    if value is None or isinstance(value, bool):
        return value
    if key is not None and _secret_key(key):
        return _REDACTED                  # whatever the value is: a PIN is a number
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        if "://" in value:
            value = _scrub_url(value)     # first, so a key in the userinfo does not hide the host
        if _secret_value(value, key):
            return _REDACTED
        name, sep, rest = value.partition(":") if ": " in value else value.partition("=")
        if "://" not in value and sep and rest.strip() and len(name) <= 40 and " " not in name.strip() \
                and _secret_key(name.strip()):
            return f"{name.strip()}{sep} {_REDACTED}"                 # "Authorization: Bearer ...", "api_key=..."
        return value.removeprefix("torch.")
    pairs = _pairs(value)
    if pairs is not None:
        items = sorted(((str(k), _identity_value(v, k)) for k, v in pairs), key=lambda kv: (kv[0], str(kv[1])))
        return dict(items) if isinstance(value, Mapping) else [list(kv) for kv in items]
    if isinstance(value, (list, tuple)):
        return [_identity_value(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_identity_value(v) for v in value), key=str)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _identity_value(to_dict())
        except Exception:  # noqa: BLE001 -- fall through to the type name
            pass
    text = str(value)
    return text.removeprefix("torch.") if " at 0x" not in text else type(value).__qualname__


# Client settings that change how a call travels, never what the model answers.
_TRANSPORT_ONLY = frozenset({"timeout", "request_timeout", "max_retries", "num_retries"})


def remote_identity(api_base: Optional[str], extra_kwargs: dict[str, Any],
                    default_base: Optional[str] = None) -> dict[str, Any]:
    """What a hosted-API backend adds to :meth:`Model.identity`.

    The endpoint and the extra kwargs decide which deployment answers and how (``organization``,
    ``project``, ``default_query={"api-version": ...}``, ``reasoning_effort``, ``thinking``,
    ``response_format``, a proxy's routing headers ...), so they belong in the run fingerprint.
    Everything goes through :func:`_identity_value`, so credentials (``api_key``, ``token``,
    ``default_headers={"Authorization": ...}``, key-looking values, ``user:pass@`` in a URL) never
    reach it. Left out: timeouts and retry counts. Keys appear only when set and ``api_base``
    only when it is not the backend's own default, so a default configuration keeps its
    fingerprint. Not covered: a base URL or key taken from the environment.
    """
    out: dict[str, Any] = {}
    if api_base and _scrub_url(str(api_base)) != (_scrub_url(default_base) if default_base else None):
        out["api_base"] = _scrub_url(str(api_base))
    extra = {k: v for k, v in extra_kwargs.items() if k not in _TRANSPORT_ONLY}
    if extra:
        out["extra"] = _identity_value(extra)
    return out


def weights_identity(model_name: Any, load_kwargs: dict[str, Any]) -> dict[str, Any]:
    """What a local-weights backend adds to :meth:`Model.identity`.

    Anything that can change the outputs belongs in the run fingerprint, or a second run is
    served the first run's cached answers: the ``load_kwargs`` (``dtype``, ``batch_size`` and
    padding, ``quantization_config``, ``revision``, ``attn_implementation``, extra generation
    kwargs ...) and, when *model_name* is a local folder or file, a cheap signature of what is on
    disk, so retraining into the same folder is not mistaken for the old model. The signature is
    each top-level file's name, size and modification time (weights, ``config.json``, tokenizer,
    ``generation_config.json``): copying the folder with ``cp`` re-keys it (a spare miss), the same
    folder run again does not. Keys appear only when set, so a hub model with default settings
    keeps its fingerprint. Not covered: a hub id's moving ``main`` (pin ``revision=``) and the
    base model a PEFT adapter folder points to.
    """
    out: dict[str, Any] = {}
    if load_kwargs:
        out["extra"] = _identity_value(load_kwargs)
    try:
        path = os.fspath(model_name)
        if os.path.isdir(path):
            files = [e for e in os.scandir(path) if e.is_file() and not e.name.startswith(".")]
            out["checkpoint"] = sorted([e.name, e.stat().st_size, e.stat().st_mtime_ns] for e in files)
        elif os.path.isfile(path):
            st = os.stat(path)
            out["checkpoint"] = [[os.path.basename(path), st.st_size, st.st_mtime_ns]]
    except (TypeError, OSError):      # not a path, or unreadable: nothing on disk to sign
        pass
    return out


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
    logically-identical runs and defeat the disk cache. The cost: a change in
    skipped state is invisible to the fingerprint, so two runs that differ only
    there are served the same cached result (a wrong *hit*, e.g. a model object
    whose weights were edited in place). :class:`CallableModel` warns when a
    function depends on such state; ``evaluate(..., cache=False)`` opts out.
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


_HIDDEN_STATE_WARNED: set[str] = set()


def _hidden_state(fn: Callable[..., Any]) -> list[str]:
    """What *fn* reads that its default name (and so the run fingerprint) cannot see.

    Objects a function closes over, the object a bound method belongs to, and module-level
    objects its code names. Modules, functions, classes and immutable scalars are visible or
    stable and are not listed. Best effort: state reached through a called function is not followed.
    """
    import inspect
    import types

    import collections.abc
    import logging
    import threading

    # Plain containers are mostly call logs and canned replies; loggers, locks and iterators are
    # plumbing. The case worth a warning is an object that computes the answer (a model, a
    # pipeline, a client) and can change behind the function's back.
    plumbing = (types.ModuleType, types.FunctionType, types.BuiltinFunctionType, types.MethodType,
                type, functools.partial, list, dict, set, frozenset, tuple, bytearray,
                collections.abc.Iterator, logging.Logger, type(threading.Lock()), type(threading.RLock()))

    def opaque(value: Any) -> bool:
        if _stable_capture_repr(value) is not None:
            return False
        return not isinstance(value, plumbing)

    found: list[str] = []
    owner = getattr(fn, "__self__", None)
    if owner is not None and not isinstance(owner, types.ModuleType):
        found.append(f"bound object {type(owner).__name__}")
    target = getattr(fn, "__func__", fn)
    if not inspect.isfunction(target):
        return found + ([f"callable object {type(fn).__name__}"] if not found else [])
    code = target.__code__
    for name, cell in zip(code.co_freevars, target.__closure__ or ()):
        try:
            if opaque(cell.cell_contents):
                found.append(f"captured {name!r}")
        except ValueError:   # empty cell
            continue
    # Globals named by the function and by code nested in it: comprehensions (a nested code
    # object before Python 3.12), generator expressions and inner lambdas.
    names, stack = [], [code]
    while stack:
        c = stack.pop()
        names += [n for n in c.co_names if n not in names]
        stack += [k for k in c.co_consts if isinstance(k, types.CodeType)]
    g = target.__globals__
    for name in names:
        if name in g and opaque(g[name]):
            found.append(f"global {name!r}")
    return found


def _warn_hidden_state(fn: Callable[..., Any]) -> None:
    hidden = _hidden_state(fn)
    key = getattr(fn, "__qualname__", None) or type(fn).__qualname__
    if not hidden or key in _HIDDEN_STATE_WARNED:
        return
    _HIDDEN_STATE_WARNED.add(key)
    warnings.warn(
        f"model function {key!r} reads state the run fingerprint cannot see ({', '.join(hidden[:3])}). "
        f"If that state changes between runs (for example weights edited in place), evaluate() and "
        f"generate() return the earlier run's cached result. Name the state with "
        f"CallableModel(fn, name=...), or pass cache=False.",
        UserWarning, stacklevel=4)


class CallableModel(Model):
    """Wrap any ``list[str] -> list[str]`` function as a model.

    This is the escape hatch: a user's own app, an SDK call, a lambda — anything
    that maps a batch of prompts to a batch of completions plugs straight in.
    """

    threadsafe: bool = True

    def __init__(self, fn: Callable[[list[str]], list[str]], name: Optional[str] = None) -> None:
        self._fn = fn
        self.name = name if name is not None else _default_callable_name(fn)
        if name is None:
            _warn_hidden_state(fn)

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
