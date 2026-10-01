# HuggingFace TGI endpoint

AuditKIT evaluates a TGI-served model over HTTP, through its generic
OpenAI-compatible `api:` backend. There is no `tgi:` backend and no TGI client
code: TGI runs in its own environment (in practice a container), the two halves
never share a Python process, and AuditKIT's environment needs only
`auditkit[requests]`.

Walkthrough notebook:
`examples/cohere/tgi_endpoint.ipynb`.

## Quick start

Start TGI (Linux, NVIDIA GPU):

```bash
docker run --gpus all --shm-size 1g -p 8080:80 \
    -v "$HOME/.cache/huggingface:/data" \
    ghcr.io/huggingface/text-generation-inference:3.3.7 \
    --model-id CohereLabs/tiny-aya-global --json-output
```

Then, in the AuditKIT environment:

```python
import auditkit as ak
from auditkit.model.api_gen import APIModel

model = APIModel(
    model="tgi",                          # TGI serves exactly one model per process
    api_base="http://localhost:8080/v1",  # the /v1 base, NOT the server root
    api_key="EMPTY",                      # only checked if TGI was started with --api-key
    name="tgi-tiny-aya",
)

result = ak.evaluate(samples, model=model, scorers=["quasi_exact_match"])
```

`evaluate()` forwards `api_base=` and `api_key=` to the same constructor, so the
spec-string form is equivalent:

```python
result = ak.evaluate(
    samples,
    model="api:tgi",                          # -> APIModel(model="tgi")
    api_base="http://localhost:8080/v1",
    api_key="EMPTY",
)
```

To reuse one model object across runs, build it once with
`ak.AutoModel.resolve("api:tgi", api_base=..., api_key=...)` and pass that
instead.

## When to reach for TGI rather than another backend

| Backend | Where the model runs | Reach for it when |
|---|---|---|
| `hf:` | in-process, `transformers` (`auditkit[transformers]`) | you want the model loaded beside the harness, with no server to launch or reach |
| `vllm:` | in-process vLLM engine (`auditkit[vllm]`, `vllm>=0.30`) | same, but you want vLLM's kernels; see [vLLM Known Issues](VLLM_KNOWN_ISSUES.md) |
| `api:` at TGI | a TGI server you start (usually a container on a GPU host) | the inference host is not the eval host, TGI is already your serving stack, or the model is only reachable as a served endpoint |
| `api:` at a hosted provider | someone else's servers | no GPU and no server to run; you pay per token |

The backend is the same `APIModel` in the last two rows — only `api_base` (and
whether a token is required) changes. Nothing in AuditKIT knows which one you
picked.

## What `api:` actually sends

`api_base` is used verbatim except for a trailing `/`; AuditKIT appends the rest
of the path itself, so **it must include the `/v1` segment**:

| | |
|---|---|
| `api_base` you pass | `http://localhost:8080/v1` |
| chat mode (default), POSTed to | `http://localhost:8080/v1/chat/completions` |
| `chat_template=False`, POSTed to | `http://localhost:8080/v1/completions` |
| passing `http://localhost:8080` instead | 404 — the base is never given a `/v1` for you |

Chat mode posts `{"model", "messages", ...generation settings}`; the raw mode
posts `{"model", "prompt", ...}`. Generation settings are renamed to the OpenAI
names TGI expects, and nothing else is sent:

| AuditKIT | sent as | TGI's `/v1/chat/completions` |
|---|---|---|
| `temperature` | `temperature` | accepted |
| `top_p` | `top_p` | accepted |
| `max_tokens` | `max_tokens` | accepted |
| `stop_sequences` | `stop` | accepted, capped by `--max-stop-sequences` (default 4) |
| `presence_penalty` | `presence_penalty` | accepted, but TGI folds it into its own repetition penalty |
| `frequency_penalty` | `frequency_penalty` | accepted |
| `num_completions` | `n` | accepted but ignored (TGI serves one choice) |
| `seed` | `seed` | accepted |

Anything else you pass to `APIModel(...)` that is not `verify`/`proxies`/`cert`/
`timeout` is merged into the JSON body verbatim, which is how you add a field
TGI wants.

Two per-run knobs are worth knowing:

- **`RunConfig.timeout`** overrides the 120 s default per POST (the same kwarg
  works directly as `APIModel(timeout=...)`).
- **`RunConfig.max_retries` / `retry_delay`** (defaults 3 and 1.0) are handed to
  the backend rather than the batch, so one 429 or dropped connection retries
  only that request; the retry is never batch-wide. Timeouts are *not* retried.

## Launching TGI

The Docker image is the supported route. Its entrypoint script (`tgi-entrypoint.sh`)
runs `text-generation-launcher` and forwards every argument to it, so any
launcher flag can simply be appended to the `docker run` command line. The image
listens on port 80, which is why every example here maps host 8080 onto it with
`-p 8080:80`:

```bash
docker run --gpus all --shm-size 1g -p 8080:80 \
    ghcr.io/huggingface/text-generation-inference:3.3.7 \
    --model-id CohereLabs/tiny-aya-global \
    --max-input-tokens 4096 \
    --api-key "$TGI_KEY"
```

Running the launcher directly (`text-generation-launcher --model-id ... --port
8080`) is exactly what the entrypoint script does, and takes the same flags —
useful when you have TGI installed from source or in a conda image rather than
the published container.

TGI serves an OpenAI-compatible surface on top of its own routes. Two matter
here:

| Route | Use |
|---|---|
| `POST /v1/chat/completions` | OpenAI Chat Completions; what `api:` uses by default |
| `POST /v1/completions` | OpenAI text completions; what `chat_template=False` uses |
| `GET /health`, `GET /info` | readiness and model metadata — poll these before a run |

AuditKIT never launches, supervises or reconfigures TGI. Everything above is
yours to run; AuditKIT only POSTs to the base you hand it.

## Auth

`api_key` is **optional** for a TGI server on localhost. `APIModel` sends
`Authorization: Bearer <api_key>` when it has one and an empty header when it
does not; TGI only enforces a bearer token if it was started with `--api-key`,
which is when you must pass the same value through.

```python
# server started without --api-key: anything works, or pass nothing at all
APIModel(model="tgi", api_base="http://localhost:8080/v1", api_key="EMPTY")

# server started with --api-key "$TGI_KEY"
APIModel(model="tgi", api_base="http://localhost:8080/v1", api_key=os.environ["TGI_KEY"])
```

Pass `api_key` explicitly anyway. Without one, `APIModel` falls back to the
`API_KEY` environment variable and then — only when `api_base` is OpenAI's own
host — to `OPENAI_API_KEY`. A localhost `api_base` never receives your real
OpenAI key, but it also never gets a token by accident, and a run against a
`--api-key` server fails with a 401 instead of working.

**What `api_key` is not:** it is not an HF Hub token. Downloading a gated model
needs `HF_TOKEN` in the *server's* environment (or `HF_TOKEN` on the
`docker run` command line), which is a separate problem from authenticating to
the endpoint.

## Gotchas

- **`model="tgi"`, not the checkpoint id.** TGI serves one model per process and
  its own examples send `{"model": "tgi"}`. AuditKIT puts whatever follows
  `api:` straight into the body's `model` field, so `api:tgi` is the right spec.
  This is the opposite of vLLM/SGLang, where you name the served checkpoint.
  TGI uses a non-`"tgi"` `model` string as a LoRA `adapter_id`, which silently
  does nothing when no adapter of that name is loaded — a checkpoint id is not
  an error, just noise in the logs.
- **A `:` in the model name is fine here** (unlike SGLang, which splits on the
  first `:` and would read `api:my-model:v2` as adapter `v2`). TGI does not
  parse it, though there is nothing to gain by putting one there either.
- **Bound your inputs.** `--max-input-tokens` and `--max-total-tokens` bound what
  a request may carry (both default to roughly 4096, `min(max_position_embeddings,
  ...)`). A long RAG prompt plus a large `RunConfig.max_tokens` can exceed the
  budget; TGI validates every request and answers 422 when it falls outside the
  server's limits.
- **`temperature=0.0` is greedy on TGI**, which turns sampling off entirely
  rather than sampling at zero. That is what AuditKIT's default
  `RunConfig.temperature` is, so an eval is deterministic by default.
- **`stop_sequences` maps to `stop`, capped by `--max-stop-sequences`** (default
  4). Keep the list short or raise the flag at launch.
- **Extra body fields are not validated.** TGI's request struct ignores unknown
  JSON keys, so a field AuditKIT sends that TGI does not implement — most
  relevantly `chat_template_kwargs` — is accepted and dropped without a warning.
  See below.

## Chat-template kwargs

`RunConfig.chat_template_kwargs` is sent in the request body as
`chat_template_kwargs` on the chat route only, and only when you set it. The
code names vLLM and SGLang as the servers that act on it; TGI's chat request
struct has no such field, and unknown keys are ignored rather than rejected, so
against TGI this is a silent no-op. If you rely on a template switch (Qwen3's
`enable_thinking`, say) to make a model behave, verify the model actually changed
its output before trusting the score.

## Tool calling

TGI's `/v1/chat/completions` accepts `tools` and `tool_choice`, and constrains
generation to a JSON schema derived from them, so `ak.ToolCallAdapter()`
(native mode) has a server-side counterpart to work with — calls come back as
structured `message.tool_calls` and `APIModel` keeps them in `Generated.trace`
where the agent metrics read them. Two limits are worth stating: TGI has no
`parallel_tool_calls` field, so AuditKIT's `parallel_tool_calls=False` is ignored
rather than obeyed; and the schema constraint only engages when `tool_choice`
pins a function.

**For the Cohere models (Tiny Aya, Aya Expanse, Aya Vision, North),
none of the four chat templates renders a `tools` slot.** That is a property of
the templates, not of the server: there is nothing for TGI to render the schemas
into, and the model has never seen this format. Use prompt mode, which writes
the schemas into the system message and parses the `<tool_call>` blocks the
model writes back:

```python
result = ak.evaluate(
    tool_samples,                       # Sample(tools=[...], expected_tool_calls=[[...]])
    model="api:tgi",
    api_base="http://localhost:8080/v1", api_key="EMPTY",
    adapter=ak.ToolCallAdapter(mode="prompt"),
    scorers=[ak.ToolCallF1(), ak.ParallelToolCalls(), ak.ToolCallValidity()],
)
```

A native parser only helps a model whose template has a tool slot to begin with;
switching servers does not change the four Cohere templates.

## Chat templates and the double-BOS question

Servers that expose an OpenAI chat route apply the model's own chat template
server-side, and a client that pre-renders the template and posts the result to
a *completions* route can end up with a second BOS token, because the server
tokenizes that text afresh and adds its own special tokens.

**AuditKIT's `api:` backend cannot produce that**, on this path:

- In chat mode (the default) it posts `messages`, and TGI renders the template
  itself with special tokens explicitly *not* re-added to the rendered string.
  AuditKIT never renders a chat template here.
- With `chat_template=False` it posts the raw `Request.prompt` to
  `/v1/completions`. That string is not template-rendered either — for
  `ToolCallAdapter` it is just the conversation flattened to `Role: content`
  lines — so there is no pre-rendered BOS to double up.

What `chat_template=False` costs you is the chat template entirely: the model
receives flat text with no turn markers, which is a quality regression, not a
BOS bug. Keep the default unless a specific reason says otherwise.

None of this has been checked live against a Cohere model on TGI.

## What is not covered

- **No in-process TGI backend.** There is no `tgi:` spec prefix;
  `AutoModel.resolve("tgi:...")` raises `AuditKitError: unknown model spec`. TGI
  ships as a Rust binary in a container, with no Python engine to hold open in
  the harness process, so there is nothing for an in-process backend to wrap.
  Server mode through `api:` is the supported path.
- **No TGI-specific auth.** `APIModel` sends one static bearer token if you give
  it one. It does not exchange an HF token, refresh a credential, or sign a
  request for HF Inference Endpoints, and it knows nothing about `--api-key`
  having been set.
- **No quantisation, sharding or server management.** `--quantize`, `--num-shard`,
  `--max-input-tokens` and friends are TGI's flags and are entirely yours.
  AuditKIT sends only what the OpenAI body accepts plus whatever extra kwargs you
  hand `APIModel`, so a setting that is not a body field is invisible to it.
- **No environment check.** `check_compat()` reports `sglang` and `vllm` pins,
  platform and CUDA findings, and has no `text-generation-inference` package in
  its report and no rule about it. Running TGI in a container keeps AuditKIT's
  environment clean — nothing in AuditKIT imports TGI, and `api:` needs only
  `requests` — but that cleanliness comes from the container, and AuditKIT has
  no equivalent of the SGLang/vLLM pin table to tell you the two halves
  disagree. There is also no `auditkit[tgi]` extra, and none is needed.
- **No live triage.** [SGLang](SGLANG.md) has an open item for triaging which
  Cohere models actually serve and score through `api:`. The same
  triage has not been run for TGI: which of the four models load under
  `--model-id`, and what they score, is unmeasured. Nothing in this document was
  produced by running TGI.
