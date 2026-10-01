"""vLLM inference backend (optional extra: auditkit[vllm])."""
from __future__ import annotations

import contextlib
import logging
import os
import sys
from typing import Any, Optional

from . import (
    Capability, Model, Request, Result_, Generated, resolve_params, DEFAULT_TEMPERATURE,
    reject_generation_kwargs, _free_torch_memory, template_messages, explain_unknown_architecture,
)
from ..errors import CapabilityError, ExtraNotInstalled

_log = logging.getLogger(__name__)

def _apply_environment_defaults() -> None:
    """Best-effort environment fixes for real, live-confirmed (on Colab)
    vLLM issues, applied once at import time so ordinary users of this
    backend never need to know about them, on any platform.

    - ``VLLM_ENABLE_V1_MULTIPROCESSING=0``: vLLM's V1 engine forks a
      separate worker process by default, which can't inherit an
      already-initialized CUDA context from the parent -- a real,
      reproducible crash ("Cannot re-initialize CUDA in forked
      subprocess") in any process that touches CUDA before constructing
      an ``LLM`` (nearly guaranteed in a notebook/REPL). Running the
      engine in-process instead sidesteps this unconditionally.
    - ``HF_HUB_DISABLE_XET=1``: ``huggingface_hub``'s newer "Xet Storage"
      download path has 404'd in practice for some legacy repos
      (confirmed live: bare ``"gpt2"``) that aren't Xet-enabled
      server-side. Best-effort only: this only helps if this module is
      imported before user code imports ``transformers``/
      ``huggingface_hub`` directly, since ``huggingface_hub`` caches its
      Xet-availability check at first import -- setting this env var
      later has no effect. If a caller hits a Xet 404 despite this, the
      real fix is setting ``HF_HUB_DISABLE_XET=1`` at the very top of
      their own script/notebook, before any other imports.

    ``setdefault`` throughout, so an explicit value the caller already
    set (e.g. to force real multi-process throughput in a dedicated
    script) always wins.
    """
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


# Must run at import time, not inside a method -- some other import path
# (e.g. `import auditkit` pulling in a different backend first) could
# otherwise import vllm/huggingface_hub before this class is ever touched.
_apply_environment_defaults()


@contextlib.contextmanager
def _stdout_fix():
    """vLLM internally redirects real OS file descriptors (via
    ``sys.stdout.fileno()``) to silence noisy distributed-backend init
    logs during engine construction. Jupyter/Colab/most notebook kernels
    replace ``sys.stdout`` with a stream that forwards over a socket
    instead of a real fd, so that call raises ``UnsupportedOperation`` --
    a real, reproducible crash confirmed live on Colab. Swapping to the
    process's real underlying streams for just the construction call
    fixes it; this is a no-op in a plain terminal, where ``sys.stdout``
    already *is* ``sys.__stdout__``, so it's always safe to apply
    unconditionally rather than only in a detected-notebook branch.
    """
    real_stdout, real_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
    try:
        yield
    finally:
        sys.stdout, sys.stderr = real_stdout, real_stderr


def _call_shutdown(obj: Any) -> bool:
    """Call ``shutdown()`` if *obj* has one. Returns True if it ran to the end.

    A failure is logged, not raised: teardown is best effort, but a silently
    swallowed failure is how the 0.30 leak went unseen."""
    if obj is None:
        return False
    fn = getattr(obj, "shutdown", None)
    if not callable(fn):
        return False
    try:
        fn()
        return True
    except Exception as e:  # noqa: BLE001
        _log.warning("vLLM teardown: %s.shutdown() failed: %s: %s", type(obj).__name__, type(e).__name__, e)
        return False


def _free_gpu_bytes() -> Optional[int]:
    """Free device memory in bytes, or None without CUDA (never initialises it)."""
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
            return int(torch.cuda.mem_get_info()[0])
    except Exception:  # noqa: BLE001
        pass
    return None


def _drop_attr(obj: Any, *names: str) -> None:
    """Drop attributes that keep CUDA tensors reachable after teardown."""
    if obj is None:
        return
    for name in names:
        if hasattr(obj, name):
            with contextlib.suppress(Exception):
                setattr(obj, name, None)


def _shutdown_vllm_engine(llm: Any) -> None:
    """Best-effort vLLM teardown across releases, in-process engine.

    vLLM's own teardown runs first. From the 0.2x/0.30 line, ``engine_core.shutdown()``
    cascades EngineCore -> executor -> ``Worker.shutdown()``, which calls
    ``model_runner.shutdown()`` (drops weights, KV caches, CUDA graphs) and then
    ``CuMemAllocator.release_pools()``. It must run before anything is dropped: the
    runner's shutdown starts with ``self.kv_caches.clear()``, so a ``kv_caches`` set to
    None first raises, and the pool release after it never runs. That was the leak on
    Colab (vLLM 0.30, L4): "Sleep mode freed 4.5 GiB, 16.89 GiB still in use", and the
    next ``vllm serve`` failed its free-memory check.

    Older releases fall back to the manual path. Confirmed live on Colab T4 (vLLM
    0.19.1): ``WorkerBase.shutdown()`` is a no-op there, so weights and KV cache stayed
    allocated. ``LLM.sleep(level=2)`` discards them (it needs ``enable_sleep_mode=True``
    at construction), then the worker/module refs are dropped so CUDA tensors can be
    collected. The runner still holding ``model`` after step 1 is what marks that case.
    """
    engine = getattr(llm, "llm_engine", None) or getattr(llm, "engine", None)
    client = getattr(engine, "engine_core", None)
    core = getattr(client, "engine_core", None)
    executor = (
        getattr(engine, "model_executor", None)
        or getattr(core, "model_executor", None)
    )
    worker = getattr(executor, "driver_worker", None)
    runner = getattr(worker, "model_runner", None)
    if runner is None:
        runner = getattr(getattr(worker, "worker", None), "model_runner", None)

    # 1. vLLM's own teardown, in cascade order; the first that exists does the rest.
    _call_shutdown(llm) or _call_shutdown(client) or _call_shutdown(core) \
        or _call_shutdown(executor) or _call_shutdown(worker)

    # 2. Legacy: the runner still holds its model, so shutdown was a no-op.
    model = getattr(runner, "model", None)
    if model is not None:
        with contextlib.suppress(Exception):
            sleep = getattr(llm, "sleep", None)
            if callable(sleep):
                sleep(level=2)
        with contextlib.suppress(Exception):
            model.to("cpu")
        _drop_attr(runner, "model", "kv_caches", "kv_cache")
        del model
    _drop_attr(executor, "driver_worker")
    _drop_attr(engine, "model_executor")
    _drop_attr(core, "model_executor", "scheduler")
    _drop_attr(engine, "engine_core")
    _drop_attr(llm, "llm_engine", "engine")
    del llm
    with contextlib.suppress(Exception):
        import gc
        gc.unfreeze()
    with contextlib.suppress(Exception):
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
        cleanup_dist_env_and_memory()
    with contextlib.suppress(Exception):
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment, destroy_model_parallel,
        )
        destroy_model_parallel()
        destroy_distributed_environment()


# vLLM's SamplingParams supports most of the common sampling knobs directly.
_KEY_MAP = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "max_tokens": "max_tokens",
    "stop_sequences": "stop",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
    "seed": "seed",
}


class VLLMModel(Model):
    """Run inference using a vLLM engine.

    ``is_local = True`` and ``model_info()`` attempts real parameter
    count/size introspection (same approach as ``HFGenModel``), but vLLM's
    internal engine structure has changed significantly between its V0/V1
    architectures and varies across releases, with no single stable public
    attribute for "the loaded nn.Module". ``model_info()`` tries the known
    internal access paths in order and falls back to identity-only
    reporting (this class's previous, unconditional behavior) if none of
    them resolve on the installed vLLM version, rather than crashing.
    **Not yet verified against a real vLLM install** -- vLLM is CUDA/Linux-
    oriented and won't install in the environment this was written in (see
    performance-metrics.md's "What the library can measure now" section).
    Verify the numeric fields against a real install before trusting them,
    the same way HFGenModel's introspection was verified against real gpt2.
    """

    name = "vllm"
    threadsafe = False
    is_local = True

    def __init__(self, model: str = "gpt2", name: str = "vllm", **kwargs: Any) -> None:
        """``**kwargs`` are real ``vllm.LLM()`` engine-construction arguments
        (e.g. ``gpu_memory_utilization=0.3``, ``dtype="float16"``,
        ``tensor_parallel_size=2``, ``max_model_len=2048``,
        ``trust_remote_code=True``, ``quantization="awq"``) -- forwarded to
        ``LLM()`` at load time in :meth:`_ensure_llm`, not to
        ``SamplingParams`` (a previous version of this class forwarded them
        to ``SamplingParams`` instead, which doesn't accept engine-level
        arguments at all and would raise ``TypeError`` for any of the ones
        above). Generation-time sampling knobs belong on the per-request
        ``RunConfig``/``Request.params`` instead (see ``_KEY_MAP``), not
        here.
        """
        reject_generation_kwargs(kwargs, "VLLMModel")
        self._model_name = model
        self._extra_kwargs = kwargs
        self._llm = None
        self._launch_env: dict[str, str] = {}   # what _apply_launch_advice set
        self.name = name

    def unload(self) -> None:
        """Tear down the loaded vLLM engine and free the GPU, so
        ``compare_models()`` can load the next model without stacking VRAM.

        Confirmed live on Colab T4 (vLLM 0.19.1, in-process V1 via
        ``VLLM_ENABLE_V1_MULTIPROCESSING=0``): dropping ``self._llm`` and
        calling ``torch.cuda.empty_cache()`` leaves weights + KV cache
        allocated. The next ``LLM(...)`` then fails the startup free-memory
        guard (``Free memory on device cuda:0 (3.42/14.56 GiB) ... less than
        desired GPU memory utilization (0.7, 10.19 GiB)``). ``LLM`` on
        0.19.1 has no ``shutdown()``; teardown is
        ``llm.llm_engine.engine_core.shutdown()`` → ``EngineCore`` →
        ``model_executor.shutdown()``. Later vLLM added ``LLM.shutdown()``;
        we prefer that when present, then walk the known attributes.
        """
        if self._llm is None:
            _free_torch_memory()
            return
        before = _free_gpu_bytes()
        llm, self._llm = self._llm, None
        _shutdown_vllm_engine(llm)
        del llm                 # no frame may hold the engine while memory is collected
        _free_torch_memory()
        after = _free_gpu_bytes()
        if before is not None and after is not None:
            _log.info("vllm: unload returned %.1f GiB (%.1f GiB free on the device now)",
                      (after - before) / 2**30, after / 2**30)

    def _find_underlying_model(self) -> Any:
        """Try each known vLLM-internal attribute path to the loaded
        ``nn.Module``, across engine versions/releases -- ``None`` if none
        of them resolve on the installed vLLM version."""
        candidates = (
            lambda: self._llm.llm_engine.model_executor.driver_worker.model_runner.model,
            lambda: self._llm.llm_engine.model_executor.driver_worker.worker.model_runner.model,
            lambda: self._llm.llm_engine.engine_core.model_executor.driver_worker.model_runner.model,
        )
        for get in candidates:
            try:
                return get()
            except AttributeError:
                continue
        return None

    def model_info(self) -> dict[str, Any]:
        """Real parameter count/sparsity/size if the installed vLLM
        version's internal engine structure matches a known access path
        (see :meth:`_find_underlying_model`); identity-only otherwise --
        never a guessed number."""
        self._ensure_llm()
        model = self._find_underlying_model()
        if model is None:
            return {"is_local": True, "model_name": self._model_name}
        params = list(model.parameters())
        total_params = sum(p.numel() for p in params)
        nonzero_params = sum(int((p != 0).sum().item()) for p in params)
        size_bytes = sum(p.numel() * p.element_size() for p in params)
        return {
            "is_local": True,
            "model_name": self._model_name,
            "total_params": total_params,
            "nonzero_params": nonzero_params,
            "sparsity": 1.0 - (nonzero_params / total_params) if total_params else 0.0,
            "size_mb": size_bytes / (1024 * 1024),
        }

    def _ensure_llm(self) -> None:
        if self._llm is not None:
            return
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ExtraNotInstalled("vllm", "pip install auditkit[vllm]")
        self._sampling_params = SamplingParams
        self._apply_launch_advice()
        try:
            with _stdout_fix():
                self._llm = LLM(model=self._model_name, **self._extra_kwargs)
        except ValueError as e:
            better = explain_unknown_architecture(e, self._model_name)
            if better is not None:
                raise better from e
            # vLLM refuses an architecture it has no in-tree class for, including one it dropped
            # ("AyaVisionForConditionalGeneration was supported in vLLM until v0.24.0"). Its
            # Transformers backend can still serve many of them; say so where the user is looking.
            msg = str(e)
            if ("was supported in vLLM until" in msg or "are not supported for now" in msg) \
                    and "model_impl" not in self._extra_kwargs:
                raise ValueError(
                    f"{msg}\n\nvLLM has no native implementation of {self._model_name!r}. Its Transformers "
                    f"backend may still run it: VLLMModel({self._model_name!r}, model_impl='transformers') "
                    f"(or `vllm serve ... --model-impl transformers` behind api:).") from e
            raise

    def _apply_launch_advice(self) -> None:
        """Set what vLLM needs on this machine before the engine starts (``compat.vllm_launch_advice``):
        on an SM 12.x GPU whose CUDA toolkit FlashInfer can't build with, vLLM's own sampler instead of
        FlashInfer's, without which every load fails. A variable the user set is left alone, and what
        was set is reported in ``run_notes()``."""
        try:
            import torch
            from auditkit.compat import vllm_launch_advice
            advice = vllm_launch_advice(torch)
        except Exception:   # noqa: BLE001 -- advice is best effort; the load itself reports real errors
            return
        for key, value in advice.get("env", {}).items():
            if key not in os.environ:
                os.environ[key] = value
                self._launch_env[key] = value
                _log.warning("vllm: %s=%s (%s)", key, value, "; ".join(advice.get("notes", [])))

    def run_notes(self) -> dict[str, Any]:
        return {"vllm_launch_env": dict(self._launch_env)} if self._launch_env else {}

    def _chat_tokenizer(self, request: Request):
        """The tokenizer whose chat template renders *request*, or None for raw text.

        ``chat_template`` counts only as a non-empty str or dict so MagicMock
        engines in existing tests still send the raw prompt. The dict is how
        transformers loads a list of named templates, which every Cohere model
        ships (Tiny Aya: ``default``; Command R7B: ``default``/``tool_use``/``rag``);
        ``apply_chat_template`` picks the right one itself.
        """
        if request.params.get("apply_chat_template") is False:
            return None
        tokenizer = None
        getter = getattr(self._llm, "get_tokenizer", None) if self._llm is not None else None
        if callable(getter):
            try:
                tokenizer = getter()
            except Exception:
                tokenizer = None
        chat_template = getattr(tokenizer, "chat_template", None) if tokenizer is not None else None
        if not isinstance(chat_template, (str, dict)) or not chat_template:
            return None
        if not callable(getattr(tokenizer, "apply_chat_template", None)):
            return None
        return tokenizer

    def capabilities(self) -> set[Capability]:
        # TOOLS: schemas go to the chat template's ``tools=`` (see _template_kwargs), as for hf:
        return {Capability.GENERATE, Capability.TOOLS}

    def _template_kwargs(self, request: Request, tokenizer: Any) -> dict[str, Any]:
        """Extra ``apply_chat_template`` kwargs for *request*: its native tool schemas
        as ``tools=``, then ``params["chat_template_kwargs"]``, which win on a clash
        (the same rule as HFGenModel)."""
        extra = dict(request.params.get("chat_template_kwargs") or {})
        tools = request.params.get("tools")
        if not tools:
            return extra
        # ponytail: substring check on the template source, as in HFGenModel
        if "tools" not in str(getattr(tokenizer, "chat_template", "") or ""):
            raise CapabilityError(
                f"{self._model_name!r}: its chat template does not render tool schemas "
                f"(e.g. Tiny Aya, Aya Expanse, Aya Vision, North). Use ToolCallAdapter(mode='prompt') to "
                f"describe the tools in the prompt instead.")
        return {"tools": tools, **extra}

    def _render_prompt(self, request: Request) -> str:
        """Same wire-format rule as HFGenModel: instruct models get a chat template."""
        text = request.prompt if isinstance(request.prompt, str) else str(request.prompt)
        tokenizer = self._chat_tokenizer(request)
        if tokenizer is None:
            if request.params.get("tools"):   # never drop tool schemas silently
                raise CapabilityError(
                    f"{self._model_name!r}: no chat template to render tool schemas with. "
                    f"Use ToolCallAdapter(mode='prompt') to describe the tools in the prompt instead.")
            return text
        messages = template_messages(request.params.get("messages") or [{"role": "user", "content": text}])
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             **self._template_kwargs(request, tokenizer))

    def generate(self, requests: list[Request]) -> list[Result_]:
        self._ensure_llm()
        defaults = {"temperature": DEFAULT_TEMPERATURE, "max_tokens": 128}
        results: list[Optional[Result_]] = [None] * len(requests)
        # A rendered chat template already carries the BOS, a raw prompt needs the
        # tokenizer's: the two kinds go in separate engine calls (the rule HFGenModel
        # follows), so a mixed batch never gives a prompt two BOS tokens or none.
        groups: dict[bool, list[int]] = {True: [], False: []}
        for i, r in enumerate(requests):
            groups[self._chat_tokenizer(r) is not None].append(i)
        for templated, idx in groups.items():
            if not idx:
                continue
            prompts = [self._render_prompt(requests[i]) for i in idx]
            # one SamplingParams per request (vLLM takes a list parallel to the prompts),
            # so each request's own max_tokens / temperature / stop are honoured
            params = [self._sampling_params(**resolve_params(requests[i], defaults, _KEY_MAP)) for i in idx]
            extra = {"tokenization_kwargs": {"add_special_tokens": False}} if templated else {}
            outputs = self._llm.generate(prompts, params, **extra)
            for i, out in zip(idx, outputs):
                text = out.outputs[0].text if out.outputs else ""
                results[i] = Result_(completions=[Generated(text=text)])
        return [r for r in results if r is not None]
