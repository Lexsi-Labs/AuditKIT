# FAQ

## What is AuditKIT?

AuditKIT is a unified evaluation library for AI models. It provides a single API
for evaluating models on benchmark tasks, LLM output quality, RAG pipelines,
and performance profiling.

## Do I need any dependencies?

No. The core library runs on the Python standard library alone. Heavy backends
(OpenAI, Anthropic, HuggingFace, vLLM, etc.) are optional extras that you
install only when needed.

## How is this different from lm-evaluation-harness or HELM?

Those tools focus on academic benchmarks (MMLU, GSM8K, etc.). AuditKIT does that
*and* LLM-as-judge evaluations, RAG evaluation, model comparison, and performance
profiling — all with a consistent API and data model.

## How is this different from Langfuse or Opik?

Those are full-stack observability platforms with backends, UIs, and SDKs.
AuditKIT is a standalone Python library with zero required deps. It can be used
on its own or integrated into any platform.

## How is this different from DeepEval or promptfoo?

DeepEval and promptfoo are evaluation libraries with similar goals. AuditKIT
differentiates through: zero required deps, unified spine across all techniques,
provenance-first fingerprints, and integration with the Lexsi ecosystem
(CuratorKIT, TabTune, AlignTune).

## Do I need an API key?

Only if you use a model backend that requires one (OpenAI, Anthropic, etc.).
The `echo` model and local HuggingFace models work without any keys.

## How do I add a custom metric?

Use the `@ak.scorer` decorator — `direction=` is required, saying whether a
higher or lower value is better:

```python
@ak.scorer(direction=ak.Direction.MAXIMIZE)
def my_metric(sample, output):
    return 1.0 if "keyword" in output else 0.0
```

See the [Custom Scorer example](../examples/custom_scorer.md).

## How do I add a custom model?

Any callable that maps `list[str] -> list[str]` works as a model. Pass it
straight to `ak.evaluate`:

```python
def my_model(prompts):
    return [f"Processed: {p}" for p in prompts]

result = ak.evaluate(dataset, model=my_model, scorers="exact_match")
```

For a named wrapper, use `ak.model.CallableModel(my_model, name="my_app")`. See the
[Model backends guide](../model_backends.md).

## What metrics are available?

10 metric families with 48 individual metrics. See the [Metrics guide](../metrics.md)
for the full list.

## Can I use AuditKIT in CI/CD?

Yes. The CLI returns exit codes and supports JSON output for downstream processing.

## Where can I report issues?

On the [GitHub issue tracker](https://github.com/Lexsi-Labs/AuditKIT/issues).
