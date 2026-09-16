"""The benchmark engine: run lm-evaluation-harness with its *full* task machinery.

Unlike the native spine (which owns prompting and scoring), lm-eval is a whole
pipeline — task + prompt template + output_type + filters + metrics + its own
model backends. So it is wrapped here as a *technique engine*, not a
:class:`~auditkit.model.Model`: :func:`run_benchmark` maps our model spec onto
lm-eval's backend, calls ``lm_eval.simple_evaluate(..., log_samples=True)``, and
maps the aggregate results and per-doc logged samples back into our uniform
:class:`~auditkit.report.RunResult` (headline + per-metric ``Stat`` + one
:class:`~auditkit.report.Prediction` per sample, i.e. the answer browser).

lm-eval is an optional extra: ``pip install auditkit[lmeval]``. The import is
lazy, so this module imports fine without it.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from typing import Any, Optional, Union

from .cache import DiskCache
from .errors import AuditKitError, CapabilityError, ExtraNotInstalled
from .report import Prediction, RunResult
from .score import Stat


@contextlib.contextmanager
def _hf_token_env(token: Optional[str]):
    """Temporarily set ``HF_TOKEN`` for the enclosed call, then restore it.

    Passing ``token`` in lm-eval's ``model_args`` covers gated *model* downloads,
    but gated *datasets* (some task data) resolve through ``datasets``, which
    reads ``HF_TOKEN`` from the environment. This forwards an explicitly-passed
    token to that path for the duration of the run only — no ambient env is read,
    and the previous value is always restored.
    """
    if not token:
        yield
        return
    prev = os.environ.get("HF_TOKEN")
    os.environ["HF_TOKEN"] = token
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("HF_TOKEN", None)
        else:
            os.environ["HF_TOKEN"] = prev


@contextlib.contextmanager
def _temp_env(name: str, value: Optional[str]):
    """Temporarily set ``os.environ[name] = value`` for the enclosed call, then
    restore the previous value (or unset). No-op when ``value`` is falsy. Used to
    mirror a Groq key into ``OPENAI_API_KEY`` for the openai-chat-completions
    backend without touching the ambient environment beyond the run."""
    if not value:
        yield
        return
    prev = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = prev

# Our model-spec prefix → (lm-eval backend name, the arg key that holds the model name).
_BACKEND_MAP: dict[str, tuple[str, str]] = {
    "hf": ("hf", "pretrained"),
    "vllm": ("vllm", "pretrained"),
    "openai": ("openai-chat-completions", "model"),
    "anthropic": ("anthropic-chat", "model"),
    "groq": ("openai-chat-completions", "model"),  # OpenAI-compatible chat endpoint (see below)
    "openrouter": ("openai-chat-completions", "model"),  # same shape as groq, see below
    "api": ("local-completions", "model"),      # OpenAI-compatible server (logprob-capable)
    "lexsi": ("local-completions", "model"),     # Lexsi gateway is OpenAI-compatible
}

# Groq speaks the OpenAI *chat* API (no /completions, no logprobs), so it rides
# the openai-chat-completions backend pointed at this base URL. It authenticates
# via the OPENAI_API_KEY env (that backend's api_key is a read-only property, not
# a model_arg), so run_benchmark() mirrors GROQ_API_KEY/api_key into it for the
# duration of the run only. Being openai-chat-completions, it's already in
# _CHAT_ONLY, so loglikelihood/MC tasks are rejected up front.
_GROQ_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

# OpenRouter is the same shape as Groq -- an OpenAI-compatible chat endpoint,
# authenticated the same way (mirrored into OPENAI_API_KEY for the run only).
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Backends that cannot return logprobs, so loglikelihood/MC tasks can't run on them.
_CHAT_ONLY = {"openai-chat-completions", "anthropic-chat"}

# lm-eval output types that are scored by loglikelihood (need a logprob-capable backend).
_LOGLIKELIHOOD_TYPES = {"loglikelihood", "multiple_choice", "loglikelihood_rolling"}

# Passed straight through to lm-eval's model_args when present in opts.
_PASSTHROUGH_ARGS = (
    "base_url", "api_key", "dtype", "device", "trust_remote_code",
    "tokenizer", "revision", "tensor_parallel_size", "max_length", "peft",
)

# Per-sample metric keys lm-eval commonly logs, in preference order for "correct".
_KNOWN_SAMPLE_METRICS = ("acc", "acc_norm", "exact_match", "em", "f1", "mc1", "mc2")


def map_model_spec(model: Any, **opts: Any) -> tuple[str, dict[str, Any]]:
    """Map an AuditKIT model spec to an lm-eval ``(backend, model_args)`` pair.

    Accepts a string like ``"hf:gpt2"``, ``"vllm:meta-llama/…"``,
    ``"openai:gpt-4o"``, ``"groq:llama-3.3-70b-versatile"``, or ``"api:my-model"``
    (with ``base_url=``). A bare name is treated as ``hf``. A callable/:class:`Model`
    is rejected — lm-eval builds its own backend, so those must use the native
    engine.

    ``groq:`` routes through the openai-chat-completions backend with Groq's base
    URL filled in automatically; it's chat-only, so generation-scored tasks only.
    """
    if not isinstance(model, str):
        raise AuditKitError(
            f"the benchmark engine needs a string model spec (e.g. 'hf:gpt2', "
            f"'vllm:...', 'api:name' with base_url); got {type(model).__name__}. "
            f"lm-eval constructs its own backend — use engine='native' for a "
            f"callable or Model instance."
        )
    if ":" in model:
        prefix, name = model.split(":", 1)
    else:
        prefix, name = "hf", model
    if prefix not in _BACKEND_MAP:
        raise AuditKitError(
            f"model prefix {prefix!r} is not supported by the benchmark engine; "
            f"known: {sorted(_BACKEND_MAP)}"
        )
    backend, name_key = _BACKEND_MAP[prefix]
    args: dict[str, Any] = {name_key: name}
    for k in _PASSTHROUGH_ARGS:
        if opts.get(k) is not None:
            args[k] = opts[k]
    # HF gated models (Llama, Gemma, ...): accept `hf_token` or `token` and pass
    # it to lm-eval as the `token` model_arg (hf/vllm read it for from_pretrained).
    # run_benchmark() also mirrors it into HF_TOKEN so gated *datasets* resolve.
    hf_token = opts.get("hf_token") or opts.get("token")
    if hf_token is not None:
        args["token"] = hf_token
    if prefix == "groq":
        # Fill in Groq's endpoint (a caller-supplied base_url still wins). The
        # key is authenticated via OPENAI_API_KEY env in run_benchmark, never a
        # model_arg -- so drop any api_key that flowed through the passthrough.
        args.pop("api_key", None)
        args.setdefault("base_url", _GROQ_BASE_URL)
    if prefix == "openrouter":
        # Same reasoning as groq above.
        args.pop("api_key", None)
        args.setdefault("base_url", _OPENROUTER_BASE_URL)
    if backend == "local-completions" and "base_url" not in args:
        raise AuditKitError(
            f"'{prefix}:' needs base_url=<OpenAI-compatible endpoint> for the "
            f"benchmark engine (e.g. base_url='https://…/v1/completions')."
        )
    return backend, args


def _task_output_types(tasks: list[str]) -> dict[str, str]:
    """Best-effort map of task name → lm-eval output_type (empty if unknowable)."""
    from lm_eval.tasks import TaskManager, get_task_dict

    td = get_task_dict(tasks, TaskManager())
    types: dict[str, str] = {}

    def walk(d: dict) -> None:
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v)
                continue
            ot = getattr(v, "OUTPUT_TYPE", None)
            if ot is None:
                cfg = getattr(v, "config", None)
                ot = getattr(cfg, "output_type", None)
            if ot:
                types[k] = ot

    walk(td)
    return types


def _assert_chat_compatible(tasks: list[str], model: str) -> None:
    """Raise if a chat-only model is asked to run a loglikelihood/MC task."""
    try:
        types = _task_output_types(tasks)
    except Exception:
        return  # can't introspect; the post-call error catch is the safety net
    bad = sorted(t for t, ot in types.items() if ot in _LOGLIKELIHOOD_TYPES)
    if bad:
        raise CapabilityError(
            f"task(s) {bad} score by loglikelihood, but {model!r} is chat-only "
            f"(no logprobs). Use a logprob-capable model (hf:/vllm:/api: with "
            f"base_url), or a generation-scored task variant."
        )


def _looks_like_loglikelihood_error(exc: Exception) -> bool:
    s = f"{type(exc).__name__}: {exc}".lower()
    return "loglikelihood" in s or "not implement" in s


# lm-eval run-level knobs we forward explicitly (everything else goes via
# lmeval_kwargs). Value None means "don't pass it — use lm-eval's default".
_RUN_KNOBS = ("apply_chat_template", "system_instruction", "gen_kwargs",
              "fewshot_as_multiturn", "use_cache", "cache_requests")


def run_benchmark(
    tasks: Union[str, list[str]],
    model: Any,
    *,
    config: Any = None,
    run_name: str = "",
    model_args: Optional[dict[str, Any]] = None,
    apply_chat_template: Optional[bool] = None,
    system_instruction: Optional[str] = None,
    gen_kwargs: Any = None,
    fewshot_as_multiturn: Optional[bool] = None,
    lmeval_kwargs: Optional[dict[str, Any]] = None,
    **opts: Any,
) -> RunResult:
    """Run one or more lm-eval tasks and return a unified :class:`RunResult`.

    ``tasks`` is a task name, comma-separated string, or list of names. ``model``
    is a string spec mapped by :func:`map_model_spec`. ``config`` is a
    :class:`~auditkit.runspec.RunConfig` (``num_fewshot``, ``limit``, ``seed``,
    ``batch_size`` are honored).

    Extra ``opts`` (``base_url``, ``dtype``, ``device``, ``hf_token``, …) flow
    into lm-eval's ``model_args`` (the model backend). lm-eval *run* knobs are
    exposed directly — ``apply_chat_template`` (set True for instruct/chat/
    fine-tuned models), ``system_instruction``, ``gen_kwargs`` (generation params
    for generative tasks, e.g. ``"temperature=0,max_gen_toks=256"``),
    ``fewshot_as_multiturn`` — and anything else lm-eval's ``simple_evaluate``
    accepts can be passed via ``lmeval_kwargs={...}`` for full parity.
    """
    from .runspec import RunConfig

    if isinstance(tasks, str):
        tasks = [t.strip() for t in tasks.split(",") if t.strip()]
    if not tasks:
        raise AuditKitError("benchmark engine needs at least one task name")
    cfg = config or RunConfig()

    backend, resolved_model_args = map_model_spec(model, **opts)
    # Full model-side parity: an explicit model_args dict merges over the
    # convenience opts, so any lm-eval model arg (load_in_4bit, gptq, parallelize,
    # max_memory, ...) is reachable — important for quantized/sharded models.
    if model_args:
        resolved_model_args.update(model_args)

    try:
        import lm_eval
    except ImportError:
        raise ExtraNotInstalled("lmeval", "pip install auditkit[lmeval]")

    if backend in _CHAT_ONLY:
        _assert_chat_compatible(tasks, model)

    kwargs: dict[str, Any] = {
        "model": backend,
        "model_args": resolved_model_args,
        "tasks": tasks,
        "num_fewshot": cfg.num_fewshot,
        "limit": cfg.limit,
        "log_samples": True,
    }
    if cfg.batch_size is not None:
        kwargs["batch_size"] = cfg.batch_size
    if cfg.seed is not None:
        kwargs["random_seed"] = cfg.seed
    # Explicit run knobs (only forwarded when set, so defaults are unchanged).
    for knob, value in (
        ("apply_chat_template", apply_chat_template),
        ("system_instruction", system_instruction),
        ("gen_kwargs", gen_kwargs),
        ("fewshot_as_multiturn", fewshot_as_multiturn),
    ):
        if value is not None:
            kwargs[knob] = value
    # Full parity: any other simple_evaluate arg (max_batch_size, write_out,
    # predict_only, task_manager, ...). Explicit args above win over this.
    if lmeval_kwargs:
        for k, v in lmeval_kwargs.items():
            kwargs.setdefault(k, v)

    fingerprint = _fingerprint(kwargs, model)
    cache = DiskCache()
    cached = cache.get(fingerprint)
    if cached is not None:
        return cached

    # lm-eval constructs vLLM itself, not VLLMModel, so it never inherits
    # vllm_gen's Colab-safe defaults. Without V1 multiprocessing off, engine
    # core init fails in notebooks (empty Failed core proc(s)). Without the
    # stdout fileno swap, vLLM's LLM() construction crashes in Colab
    # (UnsupportedOperation: fileno on ipykernel's iostream).
    stdout_cm: Any = contextlib.nullcontext()
    if backend == "vllm":
        from .model.vllm_gen import _apply_environment_defaults, _stdout_fix
        _apply_environment_defaults()
        stdout_cm = _stdout_fix()

    hf_token = opts.get("hf_token") or opts.get("token")
    # Groq/OpenRouter both authenticate the openai-chat-completions backend via
    # OPENAI_API_KEY; source the key from api_key= or the provider's own env
    # var and mirror it for the run only.
    provider_key = None
    if isinstance(model, str):
        provider_prefix = model.split(":", 1)[0]
        if provider_prefix == "groq":
            provider_key = opts.get("api_key") or os.environ.get("GROQ_API_KEY")
            if not provider_key:
                raise AuditKitError(
                    "'groq:' benchmark run needs a Groq API key -- set GROQ_API_KEY or "
                    "pass api_key=. (Groq maps to lm-eval's openai-chat-completions "
                    "backend, which authenticates via OPENAI_API_KEY; AuditKIT mirrors "
                    "your Groq key into it for the duration of the run only.)"
                )
        elif provider_prefix == "openrouter":
            provider_key = opts.get("api_key") or os.environ.get("OPENROUTER_API_KEY")
            if not provider_key:
                raise AuditKitError(
                    "'openrouter:' benchmark run needs an OpenRouter API key -- set "
                    "OPENROUTER_API_KEY or pass api_key=. (OpenRouter maps to lm-eval's "
                    "openai-chat-completions backend, which authenticates via "
                    "OPENAI_API_KEY; AuditKIT mirrors your OpenRouter key into it for "
                    "the duration of the run only.)"
                )
    try:
        with _hf_token_env(hf_token), _temp_env("OPENAI_API_KEY", provider_key), stdout_cm:
            raw = lm_eval.simple_evaluate(**kwargs)
    except CapabilityError:
        raise
    except Exception as exc:  # noqa: BLE001 — rewrap only the chat/MC case
        if backend in _CHAT_ONLY and _looks_like_loglikelihood_error(exc):
            raise CapabilityError(
                f"task(s) {tasks} score by loglikelihood, but {model!r} is "
                f"chat-only (lm-eval backend {backend!r}, no logprobs). Use a "
                f"logprob-capable model (hf:/vllm:/api: with base_url)."
            ) from exc
        raise

    result = _to_runresult(raw, tasks, model, cfg, run_name, fingerprint)
    cache.set(fingerprint, result)
    return result


def _metric_bases(task_metrics: dict[str, Any]) -> list[str]:
    """The real metric base names from an lm-eval results dict, de-duplicated.

    lm-eval keys real metrics as ``"{metric},{filter}"`` (``"acc,none"``,
    ``"exact_match,strict-match"``). Bookkeeping keys like ``"alias"`` and
    ``"sample_len"`` carry no comma, so requiring one drops them — otherwise
    ``sample_len`` (the sample count) leaks in as a bogus metric.
    """
    bases: list[str] = []
    for k in task_metrics:
        if "," not in k:
            continue
        base = k.split(",")[0]
        if base.endswith("_stderr") or base in bases:
            continue
        bases.append(base)
    return bases


def _primary_metric(task_metrics: dict[str, Any]) -> str:
    bases = _metric_bases(task_metrics)
    for cand in _KNOWN_SAMPLE_METRICS:
        if cand in bases:
            return cand
    return bases[0] if bases else "acc"


def _extract_prompt(sample: dict) -> str:
    args = sample.get("arguments")
    if args:
        first = args[0]
        if isinstance(first, (list, tuple)) and first:
            return str(first[0])
        if isinstance(first, dict):
            for v in first.values():
                return str(v)
        return str(first)
    doc = sample.get("doc")
    if isinstance(doc, dict):
        for key in ("question", "query", "input", "text", "ctx", "goal"):
            if key in doc:
                return str(doc[key])
    return str(doc) if doc is not None else ""


def _as_number(x: Any) -> Optional[float]:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, (list, tuple)) and x and isinstance(x[0], (int, float)) and not isinstance(x[0], bool):
        return float(x[0])  # lm-eval logs MC choices as [loglikelihood, is_greedy]
    return None


def _extract_output(sample: dict) -> str:
    """The model's answer, made readable.

    For multiple-choice tasks lm-eval's ``filtered_resps`` are per-choice
    loglikelihoods, so the raw first element is a meaningless number; we report
    the argmax **choice index** instead (which lines up with the gold ``target``
    index). For generative tasks we report the response text as-is.
    """
    for key in ("filtered_resps", "resps"):
        r = sample.get(key)
        if not r:
            continue
        nums = [_as_number(item) for item in r]
        if len(nums) > 1 and all(n is not None for n in nums):
            return str(max(range(len(nums)), key=lambda i: nums[i]))  # MC → picked index
        first = r[0] if isinstance(r, (list, tuple)) else r
        if isinstance(first, (list, tuple)) and first:
            return str(first[0])
        return str(first)
    return ""


def _sample_to_prediction(
    sample: dict, task: str, run_id: str, idx: int, primary_base: str
) -> Prediction:
    target = sample.get("target")
    raw_score = sample.get(primary_base)
    score = float(raw_score) if isinstance(raw_score, (int, float, bool)) else None
    return Prediction(
        run_id=run_id,
        task=task,
        sample_id=str(sample.get("doc_id", idx)),
        prompt=_extract_prompt(sample),
        raw_output=_extract_output(sample),
        parsed_answer=_extract_output(sample),
        expected=None if target is None else str(target),
        correct=(score == 1.0) if score is not None else None,
        score=score,
        metadata={"engine": "lmeval", "primary_metric": primary_base},
    )


def _fingerprint(kwargs: dict, model_spec: str) -> str:
    """Identity of a benchmark run, computed from the exact kwargs about to be
    passed to ``lm_eval.simple_evaluate`` -- so it can be checked *before*
    running (cache lookup) and reused as-is for the cached :class:`RunResult`,
    the same way :meth:`Runner.run` fingerprints a native ``RunSpec`` before
    executing it.
    """
    try:
        import lm_eval

        version = getattr(lm_eval, "__version__", "?")
    except Exception:
        version = "?"
    key = {
        "engine": "lmeval",
        "model_spec": model_spec,
        "lm_eval_version": version,
        **{k: (sorted(v) if k == "tasks" and isinstance(v, list) else v) for k, v in kwargs.items()},
    }
    blob = json.dumps(key, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _to_runresult(
    raw: dict, tasks: list[str], model_spec: str, cfg: Any, run_name: str, fingerprint: str
) -> RunResult:
    """Map lm-eval's ``simple_evaluate`` output to a :class:`RunResult`."""
    results = raw.get("results", {}) or {}
    samples_by_task = raw.get("samples", {}) or {}

    run_id = run_name or f"benchmark-{fingerprint}"

    # Headline: the authoritative lm-eval aggregate numbers. Real metrics are
    # keyed "{metric},{filter}"; drop the noise "none" filter for a clean name
    # but keep any other filter (e.g. gsm8k strict-match vs flexible-extract).
    headline: dict[str, float] = {}
    for task, metrics in results.items():
        for key, val in metrics.items():
            if "," not in key or not isinstance(val, (int, float, bool)):
                continue
            metric, filt = key.split(",", 1)
            if metric.endswith("_stderr"):
                continue
            name = f"{task}:{metric}" if filt == "none" else f"{task}:{metric},{filt}"
            headline[name] = float(val)

    # Per-sample: one Prediction per logged doc; per-metric Stats from real values.
    stats: dict[str, Stat] = {}
    predictions: list[Prediction] = []
    for task, samps in samples_by_task.items():
        task_metrics = results.get(task, {})
        primary_base = _primary_metric(task_metrics)
        metric_bases = _metric_bases(task_metrics)
        for idx, s in enumerate(samps):
            predictions.append(_sample_to_prediction(s, task, run_id, idx, primary_base))
            for base in metric_bases:
                v = s.get(base)
                if isinstance(v, (int, float, bool)):
                    stats.setdefault(f"{task}:{base}", Stat(f"{task}:{base}")).add(float(v))

    # Any headline metric with no per-sample values still gets a single-value Stat.
    for name, val in headline.items():
        if name not in stats:
            stats[name] = Stat(name).add(val)

    return RunResult(
        run_id=run_id,
        fingerprint=fingerprint,
        stats=stats,
        predictions=predictions,
        headline=headline,
        config=cfg,
        model_spec=model_spec,
    )


class BenchmarkEvaluator:
    """The lm-eval technique engine as a reusable, named object.

    Thin wrapper over :func:`run_benchmark` so a benchmark run can be composed
    and passed around. Not a :class:`~auditkit.model.Model` — it owns task
    loading and scoring, and produces a full :class:`RunResult` via
    :meth:`run`.
    """

    technique = "benchmark"

    def __init__(self, tasks: Union[str, list[str]], **opts: Any) -> None:
        self.tasks = [tasks] if isinstance(tasks, str) else list(tasks)
        self.opts = opts

    def run(self, model: Any, *, config: Any = None, run_name: str = "", **opts: Any) -> RunResult:
        merged = {**self.opts, **opts}
        return run_benchmark(self.tasks, model, config=config, run_name=run_name, **merged)
