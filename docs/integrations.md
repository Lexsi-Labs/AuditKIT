# Integrations

AuditKIT works with any model or service. The examples below connect to
popular LLM providers and frameworks.

## Model Backend Resolution

AuditKIT uses `AutoModel.resolve()` to map a provider prefix to a backend.
Connection-level options (`api_key`, `api_base`, `device`, `hf_token`, ...)
go here; generation settings (`temperature`, `max_tokens`, `top_p`, ...) do
**not** — they're rejected at construction time and must go through
`RunConfig` instead (see below), since a model instance can be reused
across runs with different generation settings.

```python
import auditkit as ak

# Automatically resolved by prefix
model = ak.AutoModel.resolve("openai:gpt-4o")
model = ak.AutoModel.resolve("anthropic:claude-3-opus-20240229")
model = ak.AutoModel.resolve("groq:llama-3.3-70b-versatile")
```

Supported prefix schemes:

| Prefix | Backend | Extra Required |
|--------|---------|---------------|
| `openai:` | OpenAI | `pip install auditkit[openai]` |
| `anthropic:` | Anthropic | `pip install auditkit[anthropic]` |
| `hf:` | HuggingFace Transformers | `pip install auditkit[transformers]` |
| `lexsi:` | Lexsi gateway (OpenAI-compatible) | `pip install auditkit[requests]` |
| `vllm:` | vLLM | `pip install auditkit[vllm]` |
| `litellm:` | LiteLLM proxy (reaches Ollama too, e.g. `litellm:ollama/llama3.1`) | `pip install auditkit[litellm]` |
| `api:` | Generic OpenAI-compatible endpoint (needs `base_url=`) | `pip install auditkit[requests]` |
| `groq:` | Groq | `pip install auditkit[requests]` |

There is no dedicated `ollama:`/`sentence-transformers:` model prefix —
Ollama is reached through `litellm:`. There's no separate
`sentence-transformers` package dependency either: `CosineSimilarity`
computes embeddings directly via `[transformers]` (mean-pooling +
L2-normalization on a plain `AutoModel`, the same recipe
`sentence-transformers` itself used internally) rather than the
`sentence-transformers` package, which transitively pulled in an
unrelated, real, live-confirmed `torchcodec`/FFmpeg environment failure
on some platforms. `TokenOverlap`/`BM25Similarity` need no extra at all
-- pure lexical overlap, no model.

## Generation settings — via `RunConfig`, not the model constructor

```python
from auditkit.runspec import RunConfig

model = ak.AutoModel.resolve("openai:gpt-4o")
result = ak.evaluate(
    dataset, model=model,
    config=RunConfig(temperature=0.7, max_tokens=512),
)
```

## OpenAI

```python
model = ak.AutoModel.resolve("openai:gpt-4o")
```

Requires `OPENAI_API_KEY` environment variable.

## Anthropic

```python
model = ak.AutoModel.resolve("anthropic:claude-3-opus-20240229")
```

Requires `ANTHROPIC_API_KEY` environment variable.

## Groq

```python
model = ak.AutoModel.resolve("groq:llama-3.3-70b-versatile")
```

Requires `GROQ_API_KEY` environment variable.

## Ollama (via LiteLLM)

```python
model = ak.AutoModel.resolve("litellm:ollama/llama3.1")
```

Requires Ollama running locally (`http://localhost:11434`).

## LiteLLM Proxy

If you run a LiteLLM proxy server:

```python
model = ak.AutoModel.resolve("litellm:gpt-4o")
```

## HuggingFace Transformers

```python
model = ak.AutoModel.resolve("hf:mistralai/Mistral-7B-Instruct-v0.3", device="cuda")
```

Requires `pip install auditkit[transformers]`.

## Custom Python Model

Any `list[str] -> list[str]` callable is a valid model — no decorator needed:

```python
def my_model(prompts: list[str]) -> list[str]:
    return [f"Processed: {p}" for p in prompts]

result = ak.evaluate(dataset, model=my_model)
```

## Custom Scorers as Integrations

You can wrap any external evaluation library as a custom scorer:

```python
import auditkit as ak

@ak.scorer(direction=ak.Direction.MAXIMIZE)
def ragas_faithfulness(sample, output):
    # Hypothetical RAGAS integration
    from ragas import evaluate
    score = evaluate(...)
    return score
```

## Experiment Tracking Integration

```python
result = ak.evaluate(dataset, model=model, scorers=[ak.Equals()], experiment_name="my-exp")
exp = ak.ExperimentDB().load("my-exp")
exp.leaderboard()
```

Results are stored locally (`~/.local/share/auditkit/experiments/` by default)
and can be exported via `result.save(...)`. Use `ak.compare([...])` for a
leaderboard across runs, or `ak.compare(baseline, candidate)` for a
baseline-vs-candidate significance test.
