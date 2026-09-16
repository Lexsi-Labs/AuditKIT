# Model Backends

AuditKIT supports multiple model backends through the `AutoModel.resolve()` factory.

## Built-in (no deps)

| Spec | Class | Description |
|---|---|---|
| `lambda p: ...` | `CallableModel` | Any `list[str] -> list[str]` function |
| `"precomputed"` | `PrecomputedModel` | Returns each sample's own `actual_output`, already set |

## Remote (require extras)

| Spec | Class | Extra |
|---|---|---|
| `"openai:gpt-4o"` | `OpenAIModel` | `[openai]` |
| `"anthropic:claude-3-5-sonnet"` | `AnthropicModel` | `[anthropic]` |
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

## Wrapping your own app

Any `list[str] -> list[str]` callable plugs straight in via `CallableModel`,
whether it's your own app, an SDK call, or a lambda:

```python
model = ak.model.CallableModel(lambda prompts: [my_app.run(p) for p in prompts], name="my_app")
result = ak.evaluate(dataset, model=model)
```
