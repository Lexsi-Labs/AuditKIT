# `vllm:` backend — known issues, bugs, and gaps

`VLLMModel` (`src/auditkit/model/vllm_gen.py`) had real, non-stub code and
mocked unit-test coverage, but had never been run against a real, installed
vLLM engine until it was live-tested for the first time on a real Colab L4
GPU. That first run — and everything found since — is consolidated
here: implementation-level gaps in `vllm_gen.py` itself, and separately, the
dependency/environment issues that got in the way of running it at all.

Companion to [`docs/BUGS.md`](BUGS.md) and
[`docs/known_issues.md`](known_issues.md), which each have one-line
summaries of some of these — this file has the full reproduction and
reasoning for every one of them, specific to `vllm:`.

---

## Part 1 — Implementation gaps in `VLLMModel` itself

### 1. No chat-template support — sends the raw flat prompt always

**Where:** `VLLMModel.generate()`:
```python
prompts = [r.prompt if isinstance(r.prompt, str) else str(r.prompt) for r in requests]
```

Unlike `HFGenModel._render_prompt()` (which checks the loaded model's own
`tokenizer.chat_template` and formats structured `messages` through it when
one exists, falling back to flat text only for genuine base models),
`VLLMModel` never looks at a chat template at all — it always sends
`request.prompt` as-is. For an instruct/chat-tuned model served via
`vllm:`, this means the prompt is under-formatted compared to how the same
checkpoint would be formatted via `hf:` or a hosted chat API
(`openai:`/`anthropic:`/`groq:`/`litellm:`).

**Status: real, open, unfixed.** Documented in `docs/known_issues.md` and
`docs/BUGS.md` prior to this session; confirmed still true by reading
current code. **Not yet demonstrated live** — the obvious test (compare
`hf:<model>` vs `vllm:<model>` prompts for the same instruct checkpoint)
was attempted in `examples/10_vllm_smoke_test.ipynb` section 3, but using
`gpt2` (a base model with no chat template at all) on both sides, which
can't demonstrate this gap — see the note in that notebook section. A
live demonstration needs an instruction-tuned model with a
`tokenizer.chat_template` on the `hf:` side.

**What a fix would look like:** extend `VLLMModel` with the same
chat-template detection/rendering `HFGenModel._render_prompt()` already
does — vLLM's own tokenizer is reachable via `self._llm.get_tokenizer()`
after the engine is constructed, which should expose the same
`chat_template` attribute a `transformers` tokenizer does.

### 2. `model_info()` — real introspection, but fallback path never confirmed to resolve

**Where:** `VLLMModel._find_underlying_model()`/`model_info()`.

Tries three known internal attribute paths (spanning vLLM's V0/V1 engine
architectures) to reach the loaded `nn.Module`, for real
parameter-count/size introspection matching `HFGenModel.model_info()`'s
approach. Falls back to identity-only reporting (`{"is_local": True,
"model_name": ...}`) if none resolve, rather than crashing or guessing.

**Status: implemented, but which branch actually fires on a real install
was never confirmed** — `examples/10_vllm_smoke_test.ipynb` section 4
(the live test for exactly this) has not been run yet as of this writing.
Until it is, treat vLLM-side `model_info()` output as unverified: it may
report real params/size, or may silently fall back to identity-only on
the installed vLLM version (`0.26.0` at time of writing) — both are
"working as designed," but which one actually happens hasn't been checked.

### 3. GPU memory never freed by `evaluate()`/`compare_models()` alone

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

### 5. One shared `SamplingParams` per batch, not per-request

**Where:** `VLLMModel.generate()`:
```python
# One shared SamplingParams for the whole batch -- same caveat as
# HFGenModel: per-request variation isn't meaningful here.
```

If a batch of requests somehow carried different
`temperature`/`max_tokens`/etc. per sample, only the first request's
params would apply to the whole batch. In practice this never happens
(one `evaluate()` run shares one `RunConfig`, so every request in a batch
already has identical params) — documented here as a known constraint,
not an active bug, matching the identical, already-accepted behavior in
`HFGenModel`.

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
# Install through the extra so the pyproject pin (vllm>=0.15,<0.20) drives
# the version — NOT a bare `pip install vllm`, which pulls latest (0.26+,
# CUDA 13) and defeats the pin.
_run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "auditkit[vllm]"])
```

> **Why through the extra, and a re-verification caveat.** `pyproject.toml`
> now pins `vllm>=0.15,<0.20` (see the `[project.optional-dependencies]` comment in `pyproject.toml`) to stay coherent with the
> `transformers<5` cap the rest of the library needs. The **original live
> confirmation of this fix used an *unpinned* `pip install vllm`**, which
> resolved to `torch==2.11.0+cu130` + `vllm==0.26.0` — the modern CUDA-13
> stack — and `VLLMModel(model="gpt2").generate(...)` then produced real
> text with zero further workarounds. With the current pin,
> `auditkit[vllm]` instead resolves to the `vllm 0.15–0.19` line
> (`torch 2.9.1–2.10.0`, CUDA 12.x). That older line is **not yet
> live-verified in this repo** — and, being CUDA-12-era, it may actually
> *match* Colab's pre-shipped CUDA-12 `torch` and sidestep this issue
> entirely rather than needing the uninstall dance at all. Both are
> plausible; **re-verify on a fresh Colab GPU with the pinned combo** and
> update this note with the result (which path actually fires) rather than
> assuming. The CUDA build tag (`+cu12x` vs `+cu13x`) is *not* expressible
> in a pyproject version pin, so this remains an environment-level fix no
> matter what we pin.

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
| 1 | No chat-template support | Implementation gap | Real limitation, always applies | **Open** — documented, not yet fixed |
| 2 | `model_info()` fallback path unconfirmed | Implementation gap | Depends on installed vLLM version | **Unverified** — implemented, live test pending |
| 3 | GPU memory not freed via `evaluate()` | Implementation gap | Any repeated local-model use via `evaluate()` | **Open** — workaround documented |
| 4 | `threadsafe=False`, no concurrency | Implementation gap | By design | Not a bug — documented behavior |
| 5 | Shared `SamplingParams` per batch | Implementation gap | Never actually triggered in practice | Not a bug — documented constraint |
| 2.1 | CUDA re-init in forked subprocess | Environment | General (vLLM-internal) | **Fixed** in `VLLMModel` |
| 2.2 | `sys.stdout.fileno()` crash | Environment | Jupyter/Colab-specific | **Fixed** in `VLLMModel` |
| 2.3 | HF Hub Xet `404` | Environment | General (server-side/repo-specific) | **Fixed** (best-effort) in `VLLMModel` |
| 2.4 | `libcudart.so.13` version mismatch | Environment | General (pre-existing mismatched `torch`) | **Not fixable from code / not fixable by version pins** — manual reinstall + restart; re-verify with the pinned `vllm<0.20` combo |
| 2.5 | `ModuleNotFoundError` after editable install | Environment | Notebook-bootstrap-specific | **Fixed** in the notebook (not a library concern) |

See `examples/10_vllm_smoke_test.ipynb` for the live, working sequence
incorporating the environment fixes (2.1–2.3, 2.5) and the
still-open implementation gaps (1–3) demonstrated/flagged in context.

## How to re-verify any entry yourself

Every reproduction above is copy-pasteable, and requires a real CUDA/Linux
GPU (Colab's free tier works). If one no longer reproduces by the time
you're reading this, it's fixed — update or remove that entry rather than
leaving a stale report in place.
