# `vllm:` backend — known issues, bugs, and gaps

Gaps and environment issues for the `vllm:` backend
(`src/auditkit/model/vllm_gen.py`): implementation gaps in `VLLMModel` itself,
and dependency/environment issues with their fixes.

Companion to [Known Issues](known_issues.md): this page has the full
reproduction and reasoning for everything specific to `vllm:`.

---

## Part 1 — Implementation gaps in `VLLMModel` itself

### 1. Chat templates and tool calling — **fixed**

`VLLMModel._render_prompt()` renders every request through the loaded model's own
`tokenizer.chat_template` (from `self._llm.get_tokenizer()`), the same rule as
`HFGenModel`: structured `messages` are formatted by the template, and a base
model with no template gets the flat prompt.

**Native tool calling** works the same way as on `hf:`. `VLLMModel` declares
`Capability.TOOLS`, `ToolCallAdapter()` passes the tool schemas to the template as
`tools=`, and the model's reply is parsed for its calls (Hermes `<tool_call>`, Cohere
action lists, JSON). A template that can't render tools (Tiny Aya, Aya Expanse, Aya
Vision, North), or a model with no template, raises `CapabilityError` with the
advice to use `ToolCallAdapter(mode="prompt")`; the schemas are never dropped
silently. `RunConfig.chat_template_kwargs` (e.g. Qwen3's `{"enable_thinking": False}`)
reaches the template too, and wins over `tools=` on a clash.

Batches mixing chat-templated and raw prompts are split into one engine call each,
so every prompt gets exactly one BOS token (a template already carries it).

**Tool calling through a server instead:** `vllm serve --enable-auto-tool-choice
--tool-call-parser <parser>` reached with `api:` returns structured
`message.tool_calls`; see `examples/colab/03_vllm_integration.ipynb`.

### 2. `model_info()` — real introspection, with an identity-only fallback

**Where:** `VLLMModel._find_underlying_model()`/`model_info()`.

Tries three known internal attribute paths (spanning vLLM's V0/V1 engine
architectures) to reach the loaded `nn.Module`, for real
parameter-count/size introspection matching `HFGenModel.model_info()`'s
approach. Falls back to identity-only reporting (`{"is_local": True,
"model_name": ...}`) if none resolve, rather than crashing or guessing.

Which path resolves depends on the installed vLLM version, so on some versions
`model_info()` reports identity only and size comparisons show "size not
introspected".

### 3. GPU memory not freed by `evaluate()` alone

**Where:** contrast `src/auditkit/api.py`'s `evaluate()` with
`src/auditkit/model_compare.py`'s `compare_models()`.

`compare_models()` resolves each model itself and calls `.unload()` on it
in a `finally` block once done (`model_compare.py:573`) —
`VLLMModel.unload()` frees the loaded engine and clears the torch CUDA
cache. **`evaluate()` has no equivalent.** When `model=` is passed as a
string spec (e.g. `"vllm:gpt2"`), `evaluate()` resolves a fresh model
instance internally, uses it, and returns — the resolved instance (and
whatever GPU memory it holds) is never exposed to the caller and never
explicitly freed.

**Confirmed live:** constructing a `VLLMModel` directly (holding ~20GB of
an L4's 22GB via vLLM's default `gpu_memory_utilization=0.9`), then
calling `ak.evaluate(model="vllm:gpt2", ...)` in a later cell without
freeing the first engine, fails:
```
ValueError: Free memory on device cuda:0 (1.43/22.03 GiB) on startup is
less than desired GPU memory utilization (0.92, 20.27 GiB). Decrease GPU
memory utilization or reduce GPU memory used by other processes.
```
Any script/notebook calling `ak.evaluate(model="vllm:...")` (or any other
large local model) more than once in the same process can hit this — not
limited to two engines existing on purpose.

**Status: real, open, unfixed.** Workaround: construct the `Model`
instance yourself instead of a bare string, and call `.unload()`
explicitly when done:
```python
from auditkit.model.vllm_gen import VLLMModel
model = VLLMModel(model="gpt2")
result = ak.evaluate(samples, model=model, scorers=["exact_match"])
model.unload()  # free GPU memory before the next evaluate()/compare_models() call
```
**What a fix would look like:** `evaluate()` tracking whether it resolved
the model itself (vs. received an already-constructed instance from the
caller, which must never be unloaded out from under them — the same
distinction `compare_models()` already makes) and calling `.unload()` in
a `finally` block accordingly.

### 4. `threadsafe = False` — no concurrent request execution

**Where:** `VLLMModel.threadsafe = False` (class attribute).

`Runner.execute()` only parallelizes at `concurrency>1` for models that
declare `threadsafe=True`; `VLLMModel` doesn't, so every `vllm:` run is
serial regardless of `RunConfig.concurrency`. This is arguably correct as
a default (vLLM's own internal batching/scheduling provides throughput,
not caller-side threading), but it does mean
`concurrency=` has no effect for this backend — worth knowing rather than
assuming a high `concurrency` will speed anything up.

**Status: by design, not necessarily a bug** — flagged here for
completeness since it's a behavioral difference from threadsafe
backends (`LiteLLMModel`, `CallableModel`, `PrecomputedModel`).

### 5. Per-request `SamplingParams` — **fixed**

`VLLMModel.generate()` builds one `SamplingParams` per request and passes the list
to `LLM.generate` (parallel to the prompts), so each request's own `max_tokens`,
`temperature`, `stop`, ... are honoured. It used to apply the first request's
settings to the whole batch. That mattered once several callers shared a batch:
an agent harness step's budget, or a judge's settings.

---

## Part 2 — Dependency/environment issues (install-time and runtime)

Found while first getting `vllm:` running at all, on a real Colab L4 GPU.
None of these are bugs in this repo's logic.

### Fixed permanently in `VLLMModel` (no user action needed, any platform)

#### 2.1 `Cannot re-initialize CUDA in forked subprocess`

**Where:** vLLM's V1 engine, `EngineCoreClient.make_client()` → forks a
separate worker process by default. CUDA contexts can't survive a
`fork()` into a child process — if the parent has already touched CUDA
(which vLLM's own init code does, early, regardless of what the caller
does), the forked worker crashes on `torch.cuda.set_device()`.

**Not Jupyter-specific** — originates in vLLM's own internal init
sequence, not from anything notebook-related.

**Fix:** `_apply_environment_defaults()` sets
`VLLM_ENABLE_V1_MULTIPROCESSING=0` via `os.environ.setdefault(...)` at
import time — runs the engine in-process instead of forking, sidestepping
the problem unconditionally. `setdefault` so an explicit caller override
still wins.

#### 2.2 `sys.stdout.fileno(): UnsupportedOperation`

**Where:** vLLM's `vllm/distributed/parallel_state.py`, `suppress_stdout()`
— redirects real OS file descriptors during engine construction to
silence noisy `gloo`/NCCL init logs.

**Jupyter/Colab-specific.** Jupyter/IPython kernels replace `sys.stdout`
with a stream that forwards output over a socket instead of a real OS fd,
so `.fileno()` raises. Never fires in a plain terminal.

**Fix:** `_ensure_llm()` wraps the `LLM(model=...)` construction call in a
`_stdout_fix()` context manager that swaps `sys.stdout`/`sys.stderr` to
`sys.__stdout__`/`sys.__stderr__` for the duration of that call. No-op in
a plain terminal, so always safe to apply unconditionally.

#### 2.3 HF Hub `404` on Xet-storage read-token

**Where:** `huggingface_hub`'s newer "Xet Storage" download path.

404s in practice for some legacy repos (confirmed live: bare `"gpt2"`)
not yet Xet-enabled server-side. **Not Colab-specific** — a server-side HF
Hub/repo-migration issue.

**Fix (best-effort):** `_apply_environment_defaults()` also sets
`HF_HUB_DISABLE_XET=1` via `setdefault`. **Caveat:** `huggingface_hub`
caches its Xet-availability check at first import — only works if
`auditkit.model.vllm_gen` is imported before the caller's own code
imports `transformers`/`huggingface_hub` directly. If hit despite this,
set `HF_HUB_DISABLE_XET=1` at the very top of your own script/notebook,
before any other imports.

#### 2.7 `FlashInfer requires GPUs with sm75 or higher` (SM 12.x GPUs, CUDA < 12.9)

**Where:** vLLM's default top-k/top-p sampler, which is FlashInfer's.

FlashInfer compiles for SM 12.x GPUs (RTX PRO 6000, RTX 50-series) only with
CUDA >= 12.9. With an older toolkit (Colab ships 12.8) every model fails to
load with this misleading message. `VLLMModel` sets
`VLLM_USE_FLASHINFER_SAMPLER=0` (vLLM's own sampler, no compiler needed)
before the engine starts on such a machine, never over a value you set, and
records it in `RunResult.metadata["vllm_launch_env"]`. For `vllm serve`,
`python -m auditkit.compat --vllm-launch` prints the setting, and
`check_compat(check_cuda=True)` warns. Any other GPU, or CUDA >= 12.9, is unaffected.

### Not fixable from library code (documented only)

#### 2.4 `libcudart.so.13: cannot open shared object file`

**Where:** `vllm._C_stable_libtorch` (vLLM's own compiled CUDA
extension), imported transitively the moment `import vllm` runs.

**Root cause:** a genuine version mismatch, not a missing library. Colab
pre-ships `torch` built for CUDA 12.x. `vllm`'s own precompiled kernels
want CUDA 13.x. `pip install vllm` doesn't force-replace an already-
present `torch` that loosely satisfies its version bound, so the two
disagree, and the real `.so` vLLM needs (present on disk, e.g.
`/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/libcudart.so.13`)
never ends up on the search path the already-installed `torch` uses.

**Not Colab-specific** — could happen on any machine with a pre-existing
mismatched `torch` before `vllm` is installed into it (a shared ML box, a
Docker image with a pre-baked `torch`). A clean environment
(no pre-existing `torch`) avoids it, since `pip install vllm` then
resolves its own matching `torch` fresh.

**Two failed workaround attempts, for the record (do not use these):**
- Setting `LD_LIBRARY_PATH` via `os.environ` from an already-running
  process — doesn't work. glibc's dynamic linker reads it once, at
  process startup, and caches it internally.
- `ctypes.CDLL(path, mode=RTLD_GLOBAL)` preloading the `.so` — makes the
  *import* succeed, but doesn't resolve the deeper ABI mismatch
  underneath.

**Real fix (manual, one-time per fresh environment), requires a full
restart afterward.** The mechanism: uninstall the pre-existing (mismatched)
`torch` so `pip` can install the exact build `vllm` needs *fresh*, then
install through `auditkit[vllm]` so the pinned `vllm` version (and therefore
the exact `torch` it depends on) is what gets resolved:
```python
import subprocess, sys

def _run(cmd):
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=False)

_run([sys.executable, "-m", "pip", "uninstall", "-y",
      "torch", "torchvision", "torchaudio",
      "nvidia-cuda-runtime", "nvidia-cuda-runtime-cu12"])
# Install through the extra so the pyproject pin (vllm>=0.30) drives the
# version, and with it the matching torch.
_run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "auditkit[vllm]"])
```

> **With the current pin.** `pyproject.toml` pins `vllm>=0.30` (the first
> vLLM with North Micro Vision's `cohere_compass`, on transformers 5.x). On fresh Colab runtimes,
> `pip install -e ".[vllm,requests]"` resolved `vllm 0.30.0` with `torch 2.13.0` (CUDA 13), and it
> loaded the Cohere models on both an A100 and an RTX PRO 6000 **without** this uninstall: the only
> step needed was removing Colab's CUDA-12 `torchaudio` (2.6 below), before anything imported torch.
> `tests/integrations_suite/colab_vllm_cohere_matrix.ipynb` does exactly that. Keep the uninstall
> above for a machine whose pre-installed torch still disagrees with vLLM's; the CUDA build tag
> (`+cu12x` vs `+cu13x`) can't be expressed in a version pin, so that case stays an environment fix.

#### 2.6 `PyTorch and TorchAudio were compiled with different CUDA versions`

**Where:** any `transformers` import (its `audio_utils` imports `torchaudio`
whenever it is installed), so `hf:` and `vllm:` model loads fail, after
`pip install "auditkit[vllm]"` on Colab.

**Root cause:** vLLM upgrades `torch` to a CUDA 13 build. Colab's
preinstalled `torchaudio` (CUDA 12.8) is not a vLLM dependency, so pip leaves
it in place, and it refuses to load against the new `torch`.

**Fix:** nothing in AuditKit needs audio. After installing `[vllm]`, and before
importing `transformers` or `vllm`:
```bash
pip uninstall -y torchaudio      # or install the torchaudio build matching torch's CUDA
```
`ak.check_compat(check_cuda=True)` reports this mismatch as an error.

#### 2.5 `auditkit` `ModuleNotFoundError` after an editable install

**Where:** any Colab/Jupyter bootstrap cell running
`pip install -e /path/to/repo` from inside an already-running kernel.

**Jupyter/Colab-notebook-bootstrap-specific — does not apply to normal
local installs.** `pip install -e` registers itself via a `.pth`/finder
file Python's `site` module only reads once, at interpreter *startup*.
Running the install from a live kernel means that process never re-reads
`site-packages`, so `pip show auditkit` reports it installed (metadata on
disk) while `import auditkit` still fails in that same process.

A normal local install (`pip install -e '.[vllm]'` in a terminal, before
starting Python) never hits this.

**Fix (notebook-only):** the bootstrap cell falls back to
`sys.path.insert(0, f"{REPO_DIR}/src")` after `pip install -e`.

---

## Summary table

| # | Issue | Kind | Scope | Status |
|---|---|---|---|---|
| 1 | Chat templates, native tool calling, one BOS per prompt | Implementation gap | Always applied | **Fixed** in `VLLMModel` |
| 2 | `model_info()` may report identity only | Implementation gap | Depends on installed vLLM version | Falls back to identity-only reporting |
| 3 | GPU memory not freed via `evaluate()` | Implementation gap | Any repeated local-model use via `evaluate()` | **Open** — workaround documented |
| 4 | `threadsafe=False`, no concurrency | Implementation gap | By design | Not a bug — documented behavior |
| 5 | Shared `SamplingParams` per batch | Implementation gap | Harness steps and judges sharing a batch | **Fixed** (one per request) |
| 2.1 | CUDA re-init in forked subprocess | Environment | General (vLLM-internal) | **Fixed** in `VLLMModel` |
| 2.2 | `sys.stdout.fileno()` crash | Environment | Jupyter/Colab-specific | **Fixed** in `VLLMModel` |
| 2.3 | HF Hub Xet `404` | Environment | General (server-side/repo-specific) | **Fixed** (best-effort) in `VLLMModel` |
| 2.7 | FlashInfer can't build on SM 12.x with CUDA < 12.9 | Environment | RTX PRO 6000 / RTX 50-series with an older toolkit | **Fixed** in `VLLMModel` (own sampler) |
| 2.6 | torch / torchaudio CUDA mismatch after `[vllm]` | Environment | Colab (preinstalled torchaudio) | **Documented**; `check_compat(check_cuda=True)` flags it |
| 2.4 | `libcudart.so.13` version mismatch | Environment | General (pre-existing mismatched `torch`) | **Not fixable from code / not fixable by version pins** — manual reinstall + restart where it happens; not hit with `auditkit[vllm]` (vLLM 0.30, torch 2.13 cu13) on Colab, 2026-10-01 |
| 2.5 | `ModuleNotFoundError` after editable install | Environment | Notebook-bootstrap-specific | **Fixed** in the notebook (not a library concern) |

See `examples/10_vllm_smoke_test.ipynb` for the live, working sequence
incorporating the environment fixes (2.1–2.3, 2.5) and the
still-open implementation gaps (1–3) demonstrated/flagged in context.

## Reproducing these

Every reproduction above is copy-pasteable and needs a real CUDA/Linux GPU
(Colab's free tier works).
