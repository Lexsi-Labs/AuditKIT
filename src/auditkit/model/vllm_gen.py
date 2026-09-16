"""vLLM inference backend (optional extra: auditkit[vllm])."""
from __future__ import annotations

import contextlib
import os
import sys
from typing import Any

from . import (
    Model, Request, Result_, Generated, resolve_params, DEFAULT_TEMPERATURE,
    reject_generation_kwargs, _free_torch_memory,
)
from ..errors import ExtraNotInstalled

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
    """Call ``shutdown()`` if *obj* has one. Returns True if it ran."""
    if obj is None:
        return False
    fn = getattr(obj, "shutdown", None)
    if not callable(fn):
        return False
    with contextlib.suppress(Exception):
        fn()
        return True
    return False


def _drop_attr(obj: Any, *names: str) -> None:
    """Drop attributes that keep CUDA tensors reachable after teardown."""
    if obj is None:
        return
    for name in names:
        if hasattr(obj, name):
            with contextlib.suppress(Exception):
                setattr(obj, name, None)


def _shutdown_vllm_engine(llm: Any) -> None:
    """Best-effort vLLM teardown across V0/V1 and in-process vs multiproc.

    Confirmed live on Colab T4 (vLLM 0.19.1): EngineCore/UniProcExecutor
    ``shutdown()`` ends at ``WorkerBase.shutdown()``, which is a no-op, so
    weights + KV cache stay allocated. After model A, 4.14/14.56 GiB was
    free and model B died on the 0.7 utilization guard (needs 10.19 GiB).
    ``LLM.sleep(level=2)`` is the supported discard (needs
    ``enable_sleep_mode=True`` at construction). Then drop the worker/module
    refs so CUDA tensors can actually be collected.
    """
    with contextlib.suppress(Exception):
        sleep = getattr(llm, "sleep", None)
        if callable(sleep):
            sleep(level=2)

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
    model = getattr(runner, "model", None)
    if model is not None:
        with contextlib.suppress(Exception):
            model.to("cpu")
        _drop_attr(runner, "model", "kv_caches", "kv_cache")

    _call_shutdown(llm)
    _call_shutdown(client)
    _call_shutdown(core)
    _call_shutdown(executor)
    _call_shutdown(worker)
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
    oriented and won't install in the environment this was written in.
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
        llm = self._llm
        self._llm = None
        if llm is not None:
            _shutdown_vllm_engine(llm)
        _free_torch_memory()

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
        with _stdout_fix():
            self._llm = LLM(model=self._model_name, **self._extra_kwargs)

    def _render_prompt(self, request: Request) -> str:
        """Same wire-format rule as HFGenModel: instruct models get a chat template.

        ``chat_template`` counts only as a non-empty str so MagicMock engines
        in existing tests still send the raw prompt.
        """
        text = request.prompt if isinstance(request.prompt, str) else str(request.prompt)
        if request.params.get("apply_chat_template") is False:
            return text
        tokenizer = None
        getter = getattr(self._llm, "get_tokenizer", None) if self._llm is not None else None
        if callable(getter):
            try:
                tokenizer = getter()
            except Exception:
                tokenizer = None
        chat_template = getattr(tokenizer, "chat_template", None) if tokenizer is not None else None
        if not isinstance(chat_template, str) or not chat_template:
            return text
        apply = getattr(tokenizer, "apply_chat_template", None)
        if not callable(apply):
            return text
        messages = request.params.get("messages") or [{"role": "user", "content": text}]
        extra = request.params.get("chat_template_kwargs") or {}
        return apply(messages, tokenize=False, add_generation_prompt=True, **extra)

    def generate(self, requests: list[Request]) -> list[Result_]:
        self._ensure_llm()
        prompts = [self._render_prompt(r) for r in requests]
        # One shared SamplingParams for the whole batch -- same caveat as
        # HFGenModel: per-request variation isn't meaningful here.
        defaults = {"temperature": DEFAULT_TEMPERATURE, "max_tokens": 128}
        gen_kwargs = resolve_params(requests[0], defaults, _KEY_MAP) if requests else defaults
        params = self._sampling_params(**gen_kwargs)
        outputs = self._llm.generate(prompts, params)
        results = []
        for out in outputs:
            text = out.outputs[0].text if out.outputs else ""
            results.append(Result_(completions=[Generated(text=text)]))
        return results
