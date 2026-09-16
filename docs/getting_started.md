# Getting Started

## Install

```bash
pip install auditkit
```

Optional extras are installed separately:

| Extra | Enables |
|---|---|
| `[openai]` | OpenAI / Azure backend |
| `[anthropic]` | Anthropic Claude |
| `[transformers]` | HuggingFace models, hallucination detection, `CosineSimilarity` |
| `[interop]` | HuggingFace datasets, Croissant loaders |
| `[mlflow]` | MLflow experiment tracking |
| `[vllm]` | vLLM inference backend |
| `[litellm]` | LiteLLM multi-provider backend |
| `[requests]` | Generic API backend |
| `[bert-score]` | BERTScore metric |
| `[dev]` | pytest for development |

## Basic Evaluation

```python
import auditkit as ak

dataset = [
    ak.Sample(input="What is 2+2?", target="4"),
    ak.Sample(input="What is 3+3?", target="6"),
]

result = ak.evaluate(dataset, model=lambda prompts: ["4", "6"])
print(result.headline)      # {"exact_match": 1.0}
print(result.summary())     # formatted table
print(result.wrong_only())  # incorrect predictions
```

## Using a Callable Model

```python
result = ak.evaluate(
    [ak.Sample(input="hello", target="HELLO")],
    model=lambda prompts: [p.upper() for p in prompts],
)
```

## Custom Scorers

```python
@ak.scorer(direction=ak.Direction.MAXIMIZE)  # required: says higher = better
def length_score(sample, output):
    return min(1.0, len(output) / 10.0)

result = ak.evaluate(dataset, model=model, scorers=[length_score])
```

## Sample Object

```python
ak.Sample(
    input="What is 2+2?",
    target="4",
    choices=["3", "4", "5"],         # MCQ
    retrieval_context=["wiki text"], # RAG
)
```
