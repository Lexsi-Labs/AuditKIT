# SGLang

AuditKit evaluates SGLang-served models over HTTP. SGLang runs its
OpenAI-compatible server in its own environment, and AuditKit's `api:`
backend talks to it. AuditKit's environment needs only `auditkit[requests]`.

Walkthrough notebook: [`examples/13_sglang_compatibility.ipynb`](https://github.com/Lexsi-Labs/AuditKIT/blob/main/examples/13_sglang_compatibility.ipynb).

## Quick start

In the SGLang environment (Linux, NVIDIA GPU):

```bash
python -m venv ~/sglang-env
~/sglang-env/bin/pip install "sglang>=0.5.18,<0.6"
~/sglang-env/bin/python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-1.5B-Instruct --port 30000 --tool-call-parser qwen25
```

In the AuditKit environment:

```python
import auditkit as ak

result = ak.evaluate(
    samples,
    model="api:Qwen/Qwen2.5-1.5B-Instruct",
    api_base="http://localhost:30000/v1",
    api_key="EMPTY",
)
```

### On any GPU

SGLang runs with the machine's own CUDA toolkit. Two things go wrong on some GPUs, both measured on Colab:

- **Hopper and Blackwell:** SGLang enables DeepGEMM, which JIT-compiles at startup with `nvcc` ≥ 12.9. On Colab's RTX PRO 6000 every model then failed to load with `NVCC version must be at least 12.9`. DeepGEMM serves FP8 GEMMs only, so for bf16/fp16 models, switch it off: `SGLANG_ENABLE_JIT_DEEPGEMM=0`.
- **GPUs without bf16 (T4, V100):** `--dtype auto` keeps a checkpoint's bf16, which these GPUs can't run. Serve in `--dtype float16`.
- **The same FlashInfer limit hits vLLM**, whose default top-k/top-p sampler is FlashInfer's: on Colab's RTX PRO 6000 (CUDA 12.8) every vLLM model failed to load with the same message. AuditKIT's `vllm:` backend sets `VLLM_USE_FLASHINFER_SAMPLER=0` there by itself (vLLM's own sampler, no compiler needed; reported in `RunResult.metadata["vllm_launch_env"]`); for `vllm serve`, `compat.py --vllm-launch` prints it. `hf:` doesn't use FlashInfer. With CUDA ≥ 12.9, or on any other GPU, nothing is changed.
- **SM 12.x GPUs (RTX PRO 6000, RTX 50-series) with a CUDA toolkit older than 12.9:** FlashInfer 0.6.18, SGLang's default attention and sampling, compiles for SM 12.x only with CUDA ≥ 12.9. With an older toolkit it swallows that error, targets no architecture, and every load fails with the misleading `FlashInfer requires GPUs with sm75 or higher`. Serve with `--attention-backend triton --sampling-backend pytorch`; neither needs a CUDA compiler. Run `compat.py --patch-sglang` first: SGLang's `Cohere2ForCausalLM` (Tiny Aya, Command R7B) doesn't report its sliding window to the runner. The Triton backend then never builds its window metadata, and every sliding-window layer fails with `'NoneType' object has no attribute 'type'` (`kv_indptr=None`). The patch adds the missing method, returning the layers' own window. SGLang's generic Transformers backend, which serves Aya Vision, has the same gap: Triton decode fails with `kv_indptr=None`. The patch adds the method there too, returning a window only for models that have sliding-window layers. Both are still unfixed on sglang `main`. Measured on the RTX PRO 6000 with Triton and the patches: Aya Expanse 8B and 32B 5/5, Aya Vision 8B 5/5 with the image check passing, and Tiny Aya 4/5 each.

Ask AuditKit what this machine needs, from inside the SGLang env, and launch with that:

```bash
# fix the two SGLang source bugs below, then print the launch command for this GPU
~/sglang-env/bin/python path/to/auditkit/compat.py --patch-sglang --sglang-launch CohereLabs/aya-expanse-8b
# -> e.g. SGLANG_ENABLE_JIT_DEEPGEMM=0 python -m sglang.launch_server --model-path CohereLabs/aya-expanse-8b
```

If a model then fails to load with a kernel build error (`Ninja build failed`, `ptxas fatal`, `cannot find -lcudart`), FlashInfer couldn't compile its attention kernels with this machine's CUDA toolkit. Launch with SGLang's Triton attention instead. Neither Triton attention nor PyTorch sampling needs a CUDA compiler:

```bash
... python -m sglang.launch_server --model-path ... --attention-backend triton --sampling-backend pytorch
```

`auditkit.compat.sglang_launch_advice()` returns the same thing as `{"env": ..., "args": ..., "notes": ...}`. `check_compat(check_cuda=True)` warns about the DeepGEMM case. The Colab matrix notebooks launch every server this way.

Don't point `CUDA_HOME` at CUDA packages installed from pip (`nvidia-cuda-nvcc` and friends). They aren't a complete toolkit: `nvidia-cuda-nvcc` doesn't pin its front end (`nvidia-nvvm`), and FlashInfer's link step needs a `libcudart.so` those wheels don't ship. Both failed on Colab.

`evaluate()` passes `api_base=` and `api_key=` through to `APIModel`. To reuse one
model object across runs, build it once with
`ak.AutoModel.resolve("api:...", api_base=..., api_key=...)` and pass that instead.

## Why SGLang gets its own environment

SGLang and vLLM pin their GPU stacks exactly, and the pins disagree. Numbers below come
from PyPI metadata and `uv pip compile` (linux x86_64, Python 3.11):

| | torch | transformers | xgrammar | llguidance |
|---|---|---|---|---|
| sglang 0.5.20 (latest) | ==2.13.0 | ==5.12.1 | ==0.2.1 | >=1.7.6,<2 |
| sglang 0.5.9 (last on transformers 4.x) | ==2.9.1 | ==4.57.1 | ==0.1.27 | <0.8 |
| vllm 0.15-0.16 (the old `auditkit[vllm]`) | ==2.9.1 | >=4.56,<5 | ==0.1.29 | >=1.3,<1.4 |
| vllm 0.17-0.19.1 (the old `auditkit[vllm]`) | ==2.10.0 | <5 | | |
| vllm 0.30 (`auditkit[vllm]`: `vllm>=0.30`) | ==2.11.0 | >=5.15 | | |
| `auditkit[transformers]` | >=2.4 | >=5.15,<6 | | |
| auditkit 1.0.0 on PyPI, and any copy of the repo from before 1.2 | >=2.4 | >=4.44,<5 | | |

The last row is a real trap: an install of an older auditkit (PyPI's only release, or a
stale copy of this repo) pins `transformers<5`, so pip picks 4.57.6 without an error, and
`cohere_compass` (North) then fails to load. `pip show auditkit` tells you which one you have.

- **sglang and vllm never share an environment.** The llguidance ranges alone rule out
  every pair of releases.
- **Unpinned installs downgrade without an error.** Plain `sglang` next to
  `auditkit[vllm]` resolves to sglang 0.5.2, a frontend with no server runtime. Next to
  the old `auditkit[transformers]` (`<5`) it resolved to 0.5.9; not re-measured against the
  current `>=5.15,<6`.
- sglang 0.5.10 moved to transformers 5.x, 0.5.11 to torch 2.11 with CUDA 13 (needs an
  NVIDIA driver >= 580), and 0.5.18 to torch 2.13. 0.5.10 is the last CUDA 12 build.
- Platform: Linux only (manylinux_2_34 wheels, so glibc >= 2.34), CPython 3.10-3.12.
  sglang's own wheels cover 3.13, but its `outlines==0.1.11` pin needs
  `outlines_core 0.1.26`, which has wheels up to cp312 only, so on 3.13 pip tries a
  Rust source build that fails. On a 3.13 host (current Colab), build the SGLang venv
  on 3.12 with uv:
  ```bash
  pip install uv
  uv venv --python 3.12 --seed /content/sglang-env
  /content/sglang-env/bin/python -m pip install "sglang>=0.5.18,<0.6"
  ```
  On macOS, pip falls back to an old release (0.5.10).
- `openai==2.6.1` (pinned by sglang) holds `litellm` back to 1.80.0 in the same env.

### The `auditkit[sglang]` extra

`auditkit[sglang]` installs `sglang>=0.5.18,<0.6`. Server mode doesn't need it: it is
for your own code that drives `sglang.Engine` in-process. The version floor makes the
downgrades above fail at resolve time instead. The extra conflicts with `[vllm]` and
`[transformers]` (declared in `[tool.uv] conflicts`) and is left out of `all`.

sglang 0.5.18+ pins a pre-release (`cuda-tile==1.6.0rc5`). pip accepts it as is. uv
needs `--prerelease=allow`, or it refuses `sglang>=0.5.18`; for unpinned `sglang`, it
picks 0.5.9 without an error.

## Gotchas

- **Always pass `api_key`.** Without it, `APIModel` falls back to `API_KEY`, then
  `OPENAI_API_KEY`, and sends your real OpenAI key to the self-hosted server. SGLang
  ignores the key unless launched with `--api-key`, so any string such as `"EMPTY"`
  works.
- **No extra `:` in the model name.** SGLang treats the first `:` in the request's
  `model` field as a `base:lora_adapter` separator. `api:` strips only its own prefix,
  so `api:my-model:v2` sends `my-model:v2`, which SGLang reads as adapter `v2`.
- **Reasoning models.** With `--reasoning-parser`, thinking goes to
  `message.reasoning_content`. `APIModel` scores `message.content` only. Qwen3-style
  hybrid models think unless told not to: pass
  `RunConfig(chat_template_kwargs={"enable_thinking": False})`, which `api:` sends as
  the request's `chat_template_kwargs`. Otherwise a small `max_tokens` is spent
  thinking, and the empty answer is recorded as an error (`finish_reason="length"`).
- **Aborted requests.** SGLang answers an aborted request with `finish_reason:
  "abort"`. An empty reply with that finish reason is recorded as an error, never
  scored as an empty answer.
- **A tool call cut off at `max_tokens`.** When the tool parser can't complete a call,
  SGLang returns the raw markup (`<tool_call>{"name": ...`) as `content`. AuditKIT
  reports it as a `truncated` call: invalid, never matched, and never "no call".
  Raise `max_tokens` if it shows up.
- **Latency.** `api:` times each request, so latency statistics are per request,
  and throughput is measured over each batch.
- **Python.** Build the SGLang venv on **Python 3.12** (see *Why SGLang gets its own
  environment*); 3.13 has no `outlines_core` wheel at sglang's pin.

## Tool calling

Start the server with the parser for your model family (`--tool-call-parser qwen25`
for Qwen2.5, `llama3`, `mistral`, `deepseekv3`, `pythonic`, ...). Tool calls come back
as structured `message.tool_calls` with `finish_reason: "tool_calls"` and empty
`content`. Several entries in one response are a parallel call.

`ak.ToolCallAdapter()` (native mode) sends `Sample.tools` as the request's `tools`
field, plus `tool_choice`/`parallel_tool_calls` when set. `APIModel` keeps the returned
calls in `Generated.trace`, and the agent metrics read them from the scoring context:

```python
result = ak.evaluate(
    samples,                       # Sample(tools=[...], expected_tool_calls=[[...]])
    model="api:Qwen/Qwen2.5-1.5B-Instruct",
    api_base="http://localhost:30000/v1", api_key="EMPTY",
    adapter=ak.ToolCallAdapter(),
    scorers=[ak.ToolCallF1(arg_mode="subset"), ak.ParallelToolCalls(arg_mode="subset")],
)
for p in result.predictions:
    print(p.sample_id, ak.to_turns((p.context or {}).get("trace")))
```

### Cohere models

| Model | Architecture | On sglang 0.5.20 (measured, Colab A100 40 GB with FlashInfer, and RTX PRO 6000 95 GB with Triton, 2026-09-30) | Tool calling |
|---|---|---|---|
| Tiny Aya (Global / Fire / Water / Earth) | `cohere2` | all four load (native `commandr.py`). A100: 3/5 each, **identical to `hf:` on the same GPU** (see below). RTX PRO 6000 on Triton, patched (rule l): 4/5 each | template has no tool slot: `ToolCallAdapter(mode="prompt")` |
| Aya Expanse 8B | `cohere` | loads (native); **5/5** on both GPUs, the same as `hf:` | prompt mode (no tool slot) |
| Aya Expanse 32B | `cohere` | RTX PRO 6000 (Triton): **5/5** | prompt mode (no tool slot) |
| Aya Vision 8B | `aya_vision` | **works, patched**: loads through SGLang's Transformers backend (no native class); text 5/5, image check passes, on both GPUs (on Triton also with rule m). Unpatched, it crashes on the first request (an SGLang bug; `compat.py --patch-sglang`, below) | prompt mode (no tool slot) |
| Aya Vision 32B | `aya_vision` | **works, patched** (Transformers backend, as 8B): text, image and layout checks pass (2026-10-01; its 62 GB download needs the disk cleared first) | as 8B |
| North Micro Vision | `cohere_compass` | fails on the pinned transformers 5.12.1 (can't read `cohere_compass`). With transformers 5.16.1 in the SGLang env and `--patch-sglang` (rules j and n, below), it **works** through the Transformers backend: text, image and layout checks pass (2026-10-01). Non-default: SGLang's own pin can't run it | `vllm:` runs it natively, `hf:` too |
| Command R7B / Command A | `cohere2` | expected | native: `--tool-call-parser cohere_command4`, which reads the `<\|START_ACTION\|>[...]<\|END_ACTION\|>` format into `message.tool_calls` |

`tests/integrations_suite/colab_sglang_cohere_matrix.ipynb` produced these rows (two runs,
2026-09-30; the second with the patch) and re-measures them.

**Aya Vision: an SGLang bug, not a missing model.** SGLang has no `aya_vision` class, but it
falls back to its generic Transformers backend, and that loaded the model. For a multimodal
model on that backend, SGLang disables the radix cache, and its scheduler then calls
`UnifiedRadixCache.dec_lock_ref(node)` without the `params` argument the method requires:

```
TypeError: UnifiedRadixCache.dec_lock_ref() missing 1 required positional argument: 'params'
```

The bug is in `schedule_policy.py` `_lock_node`, and it is also on sglang `main` as of
2026-09-30. `check_compat()` detects it from the installed source and says so. Run this in
the SGLang env to fix it:

```bash
~/sglang-env/bin/python path/to/auditkit/compat.py --patch-sglang
```

It defaults `params` to `None`. The disabled path never reads it (it returns before), and every
other caller passes it. Patched, Aya Vision 8B answers 5/5 on the text probes and passes the
image check.

**North Micro Vision** fails on SGLang's `transformers` pin first, and there is no native
`cohere_compass` class behind it. With transformers 5.16.1 in the SGLang env, it reaches the
Transformers backend, and that raises `Unsupported TP style 'embedding_rowwise'`. transformers
5.15 (the first release with `cohere_compass`) adds `"embed_tokens": "embedding_rowwise"` to the
tensor-parallel plan of every tied-embedding model. SGLang's backend reads that plan only for
`nn.Linear` layers, so the entry is inert, but its style table rejects the name before it gets
there. `compat.py --patch-sglang` maps it to `replicate` as well; `check_compat()` warns when the
env has transformers >= 5.15 and the unpatched table.

Patched for that, North fails next with `'NoneType' object has no attribute 'get'` (measured on
the RTX PRO 6000). North's `rope_parameters` is keyed by layer type, and its full-attention layers
have `None`: they use no RoPE. SGLang's Transformers backend calls `.get` on every entry while
deciding whether torch.compile is safe. `--patch-sglang` skips the entries without RoPE (rule n;
also on sglang `main`). SGLang then serves North with plain positions, because it looks for
`mrope_section` only at the top level and North nests it per layer type. For text that's exact:
North expands plain positions to all its rotary axes, as `hf:` does for text tokens. Image tokens
get sequential positions rather than their grid positions, so treat North's image answers on
SGLang as approximate. Measured with both patches (2026-10-01): North loads, passes the text
probes and the image check, and passes a layout check (a red left half and a blue right half,
asked for each side's colour), so the approximation didn't show on it.

**Tiny Aya on SGLang is the same as on `hf:`.** A three-way comparison on one A100 gave the
same answer on all five prompts through SGLang's chat API, through SGLang fed `hf:`'s exact token
ids, and through `hf:` itself. The prompt-token counts were equal too, so there's no extra BOS.
All four variants miss French ("Le Havre") and Hindi ("स्वतंत्रता"). The `hf:` column in
[model_backends.md](model_backends.md) has 4/5 for Global and Water. That was a different GPU and
transformers version (G4, 5.17), and Tiny Aya's greedy answer to the Hindi prompt differs between
the two setups. It's a property of the model on borderline prompts, not of SGLang.

For text-only work, prefer **Aya Expanse 8B**. It scores 5/5 on `hf:` and here, with no
workaround. See [model_backends.md](model_backends.md) for the matrix.

Aya Expanse and Tiny Aya are the same code path in SGLang. `commandr.py` declares

```python
class CohereForCausalLM(nn.Module): ...
class Cohere2ForCausalLM(CohereForCausalLM):
    pass
EntryClass = [CohereForCausalLM, Cohere2ForCausalLM]
```

so `cohere2` is a bare subclass of `cohere` and the two behave identically.
SGLang also ships `zaya.py` (`ZayaForCausalLM`), a newer renamed implementation of
the same family; the Hub checkpoints still declare `CohereForCausalLM`, so
`commandr.py` is the one that matches. Two implementations, one reachable name.

The action markers are special tokens. SGLang keeps them in the decoded text whenever a
request carries `tools` (`serving_chat.py` sets `skip_special_tokens=False` unless
`tool_choice="none"`), so they reach the `cohere_command4` parser in native mode. That's from
the sglang 0.5.20 source; a live Command R7B run is part of the S0 triage.

In prompt mode the model writes its calls as text, either `<tool_call>` blocks or a Cohere
action list, and AuditKIT parses them from the reply, as it does for `hf:`. No `tools` are
sent then, so SGLang strips the special tokens and the action list arrives as bare JSON,
which AuditKIT reads with or without the markers:

```python
result = ak.evaluate(samples, model="api:CohereLabs/tiny-aya-global",
                     api_base="http://localhost:30000/v1", api_key="EMPTY",
                     adapter=ak.ToolCallAdapter(mode="prompt"), scorers=[ak.ToolCallF1()])
```

Send chat requests (the default): the server renders the model's own chat template
once, with one BOS token. With `APIModel(..., chat_template=False)` the request goes to
`/v1/completions`, and AuditKIT sends the prompt exactly as given, with no template and
no BOS added. So don't pre-render a template into it; the server's tokenizer adds the
BOS itself.

`tests/integrations_suite/colab_sglang.ipynb` runs the live suite against real servers.

`parallel_tool_calls` defaults to true on SGLang. Setting it to false caps a response
at one call only when output is grammar-constrained (`tool_choice="required"` or a named
function). With `tool_choice="auto"`, false has no effect.

### Fields a server may ignore

An OpenAI-style reply never says whether `chat_template_kwargs` or `parallel_tool_calls`
was applied. So a run that sends either lists it in
`RunResult.metadata["unverified_request_fields"]` and in `summary()`, and logs one
warning. The parallel scores then carry `metadata["cap_unverified"]`. The field is
still sent, and the run never fails over it.

`api:` probes `GET /v1/models` once and records the server in
`RunResult.metadata["api_server"]` (SGLang reports `owned_by: "sglang"`). The probe only
makes the note specific; it never clears it. `owned_by` belongs to the deployment, not
to the project: a proxy or gateway in front changes it, so `"vllm"` or `"sglang"` there
is not a guarantee. To clear the note, declare the server yourself:

```python
model = ak.AutoModel.resolve("api:Qwen/Qwen3-1.7B", api_base="http://localhost:30000/v1",
                             api_key="EMPTY", server="sglang")
```

A declared `sglang` counts `chat_template_kwargs` as applied, and `parallel_tool_calls`
only when it is true or `tool_choice` forces a call; `vllm` counts both.

## Checking an environment: `check_compat()`

```python
from auditkit.compat import check_compat

report = check_compat()            # stdlib only; check_cuda=True also imports torch
print(report)                      # versions table + findings
report.ok                          # False if any error-level finding
report.raise_on_error()            # raises CompatError
```

`report.findings` is a list of `Finding(level, package, message)`. `report.versions`
maps python, platform, glibc, auditkit, torch, transformers, vllm, sglang,
flashinfer-python, xgrammar, llguidance, openai and litellm to their installed version,
or `None`. Rules:

| Level | Condition |
|---|---|
| error | an installed dependency violates a pin declared by the installed sglang/vllm |
| error | sglang and vllm installed together |
| error | sglang on non-Linux; glibc < 2.34 with sglang >= 0.5.11; Python outside 3.10-3.13 |
| warn | sglang on Python 3.13 (no cp313 `outlines_core` wheel at sglang's pin; use a 3.12 venv) |
| warn | sglang with transformers < 5.15: North Micro Vision (`cohere_compass`) cannot load |
| warn | sglang < 0.5.10 (usually a silent downgrade) or sglang without torch (frontend only) |
| warn | transformers outside 5.15-5.x (AuditKit's `hf:` backend is validated on `>=5.15,<6`) |
| warn | vllm older than 0.30 |
| warn | accelerate < 1.0 with transformers 5 (refuses `device_map`; lm-eval's hf backend) |
| info/warn | `check_cuda=True`: no usable GPU, or a pre-CUDA-13 torch under sglang >= 0.5.11 |
| error | `check_cuda=True`: torchaudio built for a different CUDA than torch (breaks every transformers import) |

Requirements with environment markers are skipped when checking pins. `compat.py`
imports nothing from AuditKit, so it also runs as a script in an environment where
AuditKit is not installed:

```bash
~/sglang-env/bin/python path/to/auditkit/compat.py --cuda   # exit code 1 on errors
```

## Out of scope

AuditKit has no in-process `sglang:` backend. The offline `sglang.Engine` has no
chat-template or tool-call parsing, forces the `spawn` multiprocessing start method,
and its `shutdown()` kills every child process of the host. Server mode through `api:`
is the supported path.
