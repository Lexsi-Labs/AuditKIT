# Model Backends

AuditKIT supports multiple model backends through the `AutoModel.resolve()` factory.

## Built-in (no deps)

| Spec | Class | Description |
|---|---|---|
| `lambda p: ...` | `CallableModel` | Any `list[str] -> list[str]` function |
| `"precomputed"` | `PrecomputedModel` | Returns each sample's own `actual_output`, already set (and `actual_trace`, for agent/RAG metrics) |
| `"agent:http://host/path"` | `AgentEndpointModel` | An externally deployed agent over HTTP, one POST per sample (stdlib only) |

## Remote (require extras)

| Spec | Class | Extra |
|---|---|---|
| `"openai:gpt-4o"` | `OpenAIModel` | `[openai]` |
| `"anthropic:claude-sonnet-5-5"` | `AnthropicModel` | `[anthropic]` |
| `"hf:gpt2"` | `HFGenModel` | `[transformers]` |
| `"lexsi:lexsi-3.5"` | `LexsiModel` | `[openai,mlflow]` |
| `"groq:llama-3.3-70b-versatile"` | `GroqModel` | `[requests]` |
| `"openrouter:openai/gpt-4o-mini"` | `OpenRouterModel` | `[requests]` |
| `"vllm:..."` | `VLLMModel` | `[vllm]` |
| `"litellm:..."` | `LiteLLMModel` | `[litellm]` |
| `"api:openai/gpt-4o"` | `APIModel` | `[requests]` |

## The prefix is required

A string `model=` spec **must** carry one of the prefixes above (or be the
literal `"precomputed"`); there's no bare-name fallback. Everything
after the `:` becomes the checkpoint/model name passed to that backend.

```python
ak.evaluate(dataset, model="hf:gpt2")                       # OK
ak.evaluate(dataset, model="groq:llama-3.3-70b-versatile")  # OK
ak.evaluate(dataset, model="openrouter:openai/gpt-4o-mini")  # OK -- provider/model naming, resolved after the first ':'
ak.evaluate(dataset, model="precomputed")                   # OK -- special-cased, no prefix needed
ak.evaluate(dataset, model="gpt2")                          # AuditKitError: unknown model spec 'gpt2'
```

The same rule applies inside `compare_models(models=[...])`. AuditKIT resolves
every entry in that list the same way, one call to `AutoModel.resolve()` per
model, so a bare, unprefixed name anywhere in the list fails the same way.

## Usage

```python
from auditkit.model import AutoModel

model = AutoModel.resolve(lambda ps: [p.upper() for p in ps])
model = AutoModel.resolve("openai:gpt-4o")  # needs OPENAI_API_KEY
```

Models can be passed directly to `ak.evaluate()`:

```python
result = ak.evaluate(dataset, model="openai:gpt-4o")
```

## Self-hosted OpenAI-compatible servers (vLLM, SGLang)

Point `api:` at the server's `/v1` base URL. `evaluate()` forwards `api_base=`
and `api_key=` to `APIModel`:

```python
result = ak.evaluate(
    dataset,
    model="api:Qwen/Qwen2.5-7B-Instruct",
    api_base="http://localhost:8000/v1",
    api_key="EMPTY",
)
```

**Pass `api_key` (or set `API_KEY`) for a self-hosted server.** `OPENAI_API_KEY`
is used only when `api_base` points at OpenAI itself (`api.openai.com`), so a
custom `api_base` — a self-hosted vLLM/SGLang/Ollama server, or Azure
(`*.openai.azure.com`) — never receives your real OpenAI key; it needs an
explicit `api_key=` or the `API_KEY` env var. vLLM and SGLang ignore the key
unless started with `--api-key`, so any string such as `"EMPTY"` works.

For SGLang specifics (why it gets its own environment, `check_compat()`, the
model-name `:` gotcha), see [SGLang](SGLANG.md).

### Which Cohere models each runtime can load

The nine Cohere models are four architectures, and the runtimes do **not** cover
the same set. The `hf:` column is **measured on a GPU**. The SGLang column is
**measured** for all nine models, on a 40 GB A100 and a 95 GB RTX PRO 6000 (sglang 0.5.20, 2026-09-30 and 2026-10-01,
`tests/integrations_suite/colab_sglang_cohere_matrix.ipynb`);
The `vllm:` column is **measured** for all nine by
`tests/integrations_suite/colab_vllm_cohere_matrix.ipynb`
(vLLM 0.30.0, transformers 5.16.1, 2026-10-01): the seven that fit on a 40 GB A100, and all nine on a
95 GB RTX PRO 6000 with the FlashInfer sampler fix below. The seven scored identically on both GPUs. Both replace the registry readings the columns showed before. Every score uses the same
five-language probes as `hf:`; PASS means 4/5 or better. On vLLM the offline engine and
`vllm serve` gave identical answers.

| Model | architecture | `hf:` measured | `vllm:` | SGLang | HF hosted `api:` |
|---|---|---|---|---|---|
| Tiny Aya Global | `cohere2` | 4/5 | 4/5 | loads, 3/5 † | text |
| Tiny Aya Fire | `cohere2` | 3/5 | 4/5 | loads, 3/5 † | text |
| Tiny Aya Water | `cohere2` | 4/5 | 4/5 | loads, 3/5 † | text |
| Tiny Aya Earth | `cohere2` | 3/5 | 4/5 | loads, 3/5 † | text |
| North Micro Vision Instruct | `cohere_compass` | 3/5 ⚠ | **5/5**, image ✓, layout ✓ | PASS, image ✓ § | text only |
| Aya Expanse 8B | `cohere` | **5/5** | **5/5** | loads, **5/5** | text |
| Aya Expanse 32B | `cohere` | **5/5** | **5/5** ♦ | loads, **5/5** ¶ | text |
| Aya Vision 8B | `aya_vision` | 5/5 | **5/5**, image ✓, layout ✓ ◊ | **5/5**, image ✓ ‡ | text only |
| Aya Vision 32B | `aya_vision` | 4/5 | **5/5**, image ✓, layout ✓ ◊ ♦ | PASS, image ✓ ‡ | text only |

† Identical to `hf:` on the same A100 (a three-way check found the same answers and the same
prompt tokens). The `hf:` column's 4/5 comes from a different GPU and transformers version, where
Tiny Aya's greedy answer to the Hindi prompt differs; see [SGLang](SGLANG.md#cohere-models).
♦ RTX PRO 6000 (CUDA 12.8), 2026-10-01, with `VLLM_USE_FLASHINFER_SAMPLER=0` set by the launch advice: without it,
every model failed to load on that GPU. Offline `vllm:` and `vllm serve` gave the same answers.
◊ Through vLLM's Transformers backend, with no flag: vLLM removed its Aya Vision class after 0.24.0,
and its default `model_impl="auto"` falls back to the Transformers backend on its own (measured on
vLLM 0.30.0). `model_impl="transformers"` forces that path.
‡ Through SGLang's Transformers backend (there is no native class), with `compat.py --patch-sglang`.
Unpatched, it crashes on the first request (an SGLang bug).
§ The pinned transformers 5.12.1 can't read `cohere_compass`. With 5.16.1 in a second SGLang env,
the Transformers backend hits two more SGLang bugs (its `embedding_rowwise` TP style, then the `None`
in its `rope_parameters`), both fixed by `--patch-sglang`. Patched, it passes text, image and a
left/right layout check. PASS = 4/5 or better on the text probes.
¶ On the RTX PRO 6000 with Triton attention (FlashInfer can't build for SM 12.x with Colab's CUDA
12.8). Tiny Aya scores 4/5 each there, and Aya Expanse 8B and Aya Vision 8B score the same as on the A100.

Measured on a 95 GB G4 with `transformers` 5.17, by
`tests/integrations_suite/colab_cohere_models.ipynb`:
every model was asked one question in each of five languages and scored on the
**answer**, not on whether a string came back. "5/5" is five correct answers. All
nine load and answer text-only, including both vision architectures with their
vision towers unused and no image in the request.

### Which model, for text only

**Aya Expanse 32B is the best of the nine (5/5), and Aya Expanse 8B ties it.**

**Aya Vision never beats Aya Expanse** -- 8B ties at 5/5, and at 32B it is *behind*
(4/5 against 5/5). For text-only work there is no case for it: **use Aya
Expanse.** That conclusion rests on the measured numbers alone, so it stands
independently of the registry readings. The secondary reason ("Aya Vision loads on neither
vLLM nor SGLang") turned out **false** when measured: it loads on both, through each one's
Transformers backend: automatically on vLLM, with `--patch-sglang` (and a crash without it)
on SGLang. Aya Expanse runs natively on both with no workaround. So the
answer is unchanged and the reason is now the numbers plus that: use Aya Expanse.

**All four Tiny Aya variants miss the same question.** Asked for the largest port of
France, all four answer *Le Havre*; the answer is *Marseille*. They score 3-4/5. So
Tiny Aya's wider language *list* (27 tags against the Aya models' 18) is breadth,
not depth -- the Aya family is stronger where it counts.

⚠ **North Micro Vision leaks a raw special token into every answer.** Each of its
five replies ends with a literal `<|END_OF_TURN_TOKEN|>`, and it gets Thai wrong
outright. A model that scores on exact match will be marked wrong for punctuation
it did not produce as text, and nothing in the response says why. Strip it (or set
`skip_special_tokens`) before scoring this model. vLLM skips special tokens by default, and there
North scores 5/5 on the same probes.

- **`hf:` covers all nine**, confirmed. It needs `transformers` >= 5.15 for
  `cohere_compass`; see the install trap below.
- **Aya Vision has no native server class, but both servers run it.** vLLM dropped
  `AyaVisionForConditionalGeneration` after 0.24.0 (its registry keeps the annotation
  `"AyaVisionForConditionalGeneration": "0.24.0"`), and SGLang never implemented it. Both
  **load** it through their generic Transformers backends: vLLM on its own (8B measured: 5/5, image
  check passes), SGLang with `compat.py --patch-sglang` (8B and 32B measured, image checks pass).
- **vLLM and SGLang both reuse the Command-R implementation** for `cohere` and
  `cohere2` rather than shipping Cohere-specific code: vLLM maps
  `"Cohere2ForCausalLM" -> ("commandr", "CohereForCausalLM")`, and SGLang declares
  `class Cohere2ForCausalLM(CohereForCausalLM): pass`. Tiny Aya and Aya Expanse
  therefore behave identically on both. vLLM does have a real `cohere_compass`
  module; SGLang does not, so it serves North Micro Vision through its generic Transformers backend,
  which needs transformers >= 5.15 in the SGLang env and `--patch-sglang`.
- **"text only" in the last column is what's measured, not a score.** `api:`
  sends `Sample.images`, and the image checks in the vLLM and SGLang columns went through it.
  Image input on the HF hosted router hasn't been measured for these models.

Eight of the nine are gated on the Hub (`gated: "auto"`): a hosted sweep needs an
`HF_TOKEN` whose account has accepted each licence. North Micro Vision Instruct
is the only open one.

### Install trap: on Colab, `auditkit[transformers]` can resolve to transformers 4.x

`pip install auditkit[transformers,vision]` resolves **transformers to 4.57.6** and
exits 0 -- even on a runtime that already had 5.16.1. 4.x has no `cohere_compass`
registered, so North Micro Vision then fails with a `ValueError` that reads like a
model problem and is not one.

The extra declares `transformers>=5.15,<6`, and it was never at fault. The cap came
from **the auditkit that was installed**: a stale 1.0.0 (`transformers<5`) from another
repository, not this one's 1.2.
`pip show auditkit` says which one you have. `hf:` and `vllm:` now name the model type, the
installed transformers and the version it needs when this happens, instead of an error that
reads like a broken checkpoint.

Either way, install and then pin last, and **check the version took** — `rc=0`
from pip means nothing here:

```bash
pip install 'auditkit[transformers,vision]'
pip install 'transformers>=5.15,<6'
python -c "import transformers; print(transformers.__version__)"   # must start with 5.
```

`colab_cohere_models.ipynb` does exactly this and refuses to report a
`cohere_compass` result if the pin did not hold, so a version artefact is never
recorded as a model finding.

### vLLM on SM 12.x GPUs with an older CUDA toolkit

vLLM samples with FlashInfer by default, and FlashInfer compiles for SM 12.x GPUs (RTX PRO 6000,
RTX 50-series) only with CUDA >= 12.9. With an older toolkit, as on Colab (12.8), every model fails
to load with the misleading `FlashInfer requires GPUs with sm75 or higher`. The `vllm:` backend
detects this and starts vLLM with its own sampler (`VLLM_USE_FLASHINFER_SAMPLER=0`), recorded in
`RunResult.metadata["vllm_launch_env"]`. A value you set yourself is never changed. For
`vllm serve` behind `api:`, set it in the server's env; `python compat.py --vllm-launch` prints
what this machine needs, and `check_compat(check_cuda=True)` warns. SGLang's side of the same
limit is in [SGLang](SGLANG.md).

### Images and vision models

`Sample.images` reaches the model on `hf:` (`hf_gen.py` builds the `image` content parts and
calls `AutoProcessor`) and on `api:` in chat mode (OpenAI `image_url` parts, at the
same position `hf:` uses). The offline `vllm:` backend doesn't read it yet:

| Backend | Image reaches the model? |
|---|---|
| `hf:` | yes |
| `api:` chat mode (HF hosted, `vllm serve`, SGLang) | yes; raw-prompt mode raises |
| `vllm:` (offline engine) | **no, silently** |

For a vision eval on vLLM, serve the model (`vllm serve`) and use `api:`. That is how the
vLLM column's image checks ran.


## Native tool calling (`api:`)

In chat mode (the default), `api:` forwards `tools`, `tool_choice` and
`parallel_tool_calls` from the request, which `ak.ToolCallAdapter()` fills from
`Sample.tools`. The returned `message.tool_calls` are kept as one turn in
`Generated.trace`, and the Runner hands them to the metrics as
`context["trace"]`. Several calls in one response count as a parallel call.
`content` is empty when the model only calls tools.

```python
result = ak.evaluate(
    samples,                                # Sample(tools=[...], expected_tool_calls=[[...]])
    model="api:Qwen/Qwen2.5-7B-Instruct",
    api_base="http://localhost:8000/v1", api_key="EMPTY",
    adapter=ak.ToolCallAdapter(),
    scorers=[ak.ToolCallF1(), ak.ParallelToolCalls()],
)
```

- The server must parse tool calls: vLLM with `--enable-auto-tool-choice
  --tool-call-parser <parser>` (e.g. `hermes`), SGLang with
  `--tool-call-parser <parser>` (e.g. `qwen25`, `llama3`, `mistral`).
- `adapter=None` means `GenerationAdapter`, which never sends tools. Use
  `adapter=ak.ToolCallAdapter()` or `adapter="auto"`.
- Only `api:` (with `chat_template=True`, the default), `agent:` and `hf:` declare
  native tool support. `hf:` passes the schemas to the model's chat template
  (`apply_chat_template(tools=...)`) and parses the calls from the text, including
  Cohere's `<|START_ACTION|>` format (Command R7B); a template that does not render
  tools raises `CapabilityError`. That includes every Cohere model (Tiny
  Aya, Aya Expanse, Aya Vision, North): use `ToolCallAdapter(mode="prompt")` with them.
  On any other backend (including `openai:`, `groq:`,
  `openrouter:`, `litellm:`), the Runner raises `CapabilityError` rather than
  drop the tools. Use `api:` with the provider's base URL, or
  `ak.ToolCallAdapter(mode="prompt")`, which describes the tools in the prompt
  and parses `<tool_call>` blocks from the text.

## Agent endpoints (`agent:`)

`model="agent:<url>"` evaluates an agent that runs its own tool loop behind an
HTTP endpoint (LangGraph, AgentTune, your own service). It POSTs
`{"input", "messages", "tools"?}` and reads `{"output", "messages",
"tool_calls", "contexts"}` back (any subset); an OpenAI chat-completion reply is
detected. Dotted `*_path=` settings or `request_fn=`/`response_fn=` adapt it to
other contracts. The bearer token comes from `api_key=` or `AGENT_API_KEY`,
never `OPENAI_API_KEY`. Generation settings are not sent: the agent owns its
sampling.

```python
result = ak.evaluate(samples, model="agent:http://localhost:8000/run",
                     adapter=ak.ToolCallAdapter(),
                     scorers=[ak.ToolCallF1(), ak.RetrievalMetrics(k=5)])
```

Full contract: [Agents & RAG](agents_and_rag.md#agent-endpoints-agent).

## Wrapping your own app

Any `list[str] -> list[str]` callable plugs straight in via `CallableModel`,
whether it's your own app, an SDK call, or a lambda:

```python
model = ak.model.CallableModel(lambda prompts: [my_app.run(p) for p in prompts], name="my_app")
result = ak.evaluate(dataset, model=model)
```
