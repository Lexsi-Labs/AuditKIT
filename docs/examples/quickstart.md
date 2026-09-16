# Quickstart

```python
import auditkit as ak

dataset = [
    ak.Sample(input="hello", target="hello"),
    ak.Sample(input="world", target="world"),
]

# A plain callable that returns the input verbatim -- free, offline, useful
# for verifying the pipeline itself, not real model quality. Swap in a real
# backend (e.g. model="openai:gpt-4o-mini") to evaluate an actual model.
result = ak.evaluate(dataset, model=lambda prompts: prompts)
print(result.headline)   # {"exact_match": 1.0}
print(result.summary())  # formatted table
```
