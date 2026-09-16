# Cookbook: LLM-as-Judge Evaluation

Use an LLM as a judge/evaluator for subjective criteria that deterministic
metrics struggle to capture, like helpfulness, tone, or instruction following.

## Scenario

You have a writing assistant that generates product descriptions. You want to score
each description on: _clarity_, _persuasiveness_, and _brand voice alignment_ — all
of which require semantic judgment.

## Step 1: Define Rubric Items

Each rubric item is a single criterion the judge evaluates. `GEval` combines them
into one weighted, chain-of-thought judge call per sample:

```python
import auditkit as ak

scorer = ak.GEval(
    rubric=[
        ak.RubricItem(
            criterion="clarity",
            description="The description is easy to understand and free of jargon",
            weight=1.0,
        ),
        ak.RubricItem(
            criterion="persuasiveness",
            description="The description makes the product compelling and desirable",
            weight=1.0,
        ),
        ak.RubricItem(
            criterion="brand_voice",
            description="The description matches our brand tone (professional, warm, concise)",
            weight=0.5,
        ),
    ],
    judge_model="openai:gpt-4o-mini",
)
```

## Step 2: Configure the Judge Model

`judge_model=` accepts any model spec string or `Model` instance — the same forms
`evaluate()`'s `model=` accepts:

```python
scorer = ak.GEval(
    rubric=[...],
    judge_model="openai:gpt-4o-mini",   # cheaper judge
    temperature=0.0,                     # deterministic
)
```

Or use a local model:

```python
scorer = ak.GEval(
    rubric=[...],
    judge_model="hf:Qwen/Qwen2.5-3B-Instruct",   # local judge
)
```

## Step 3: Run Evaluation

`model=` is any `list[str] -> list[str]` callable — no decorator needed:

```python
dataset = [
    ak.Sample(input="Write a product description for a wireless mechanical keyboard"),
    ak.Sample(input="Describe our new ergonomic mouse"),
]


def writing_assistant(prompts: list[str]) -> list[str]:
    return [
        f"Introducing our premium {p} — designed for professionals who demand the best."
        for p in prompts
    ]


result = ak.evaluate(dataset, model=writing_assistant, scorers=[scorer])
```

## Step 4: Examine the Scores

`GEval` produces one blended score per sample (the rubric's weighted combination),
with the judge's chain-of-thought reasoning in `reason`:

```python
for p in result.predictions:
    for s in p.metadata["scores"]:
        if s["name"] == "g-eval":
            print(f"output: {p.raw_output!r}")
            print(f"  score: {s['value']:.2f} | reason: {s.get('reason', '')}")
```

## Step 5: Validate Your Judge

LLM judges can have biases. Validate by running a known set through
`model="precomputed"` (scores existing text, makes no new generation call):

```python
gold = [
    ak.Sample(input="test", actual_output="Perfect product", target="high"),
    ak.Sample(input="test", actual_output="Bad unclear", target="low"),
]

val_result = ak.evaluate(gold, model="precomputed", scorers=[scorer])
```

Compare GEval scores against human ratings to catch systematic biases.

## Step 6: Combine with Code Metrics

Mix LLM-judge metrics with deterministic ones for a balanced view:

```python
result = ak.evaluate(dataset, model=writing_assistant, scorers=[
    ak.GEval(rubric=[...], judge_model="openai:gpt-4o-mini"),  # subjective quality
    ak.WordCount(min_words=30, max_words=200),                  # length enforcement
    ak.ToxicityScore(),                                          # safety filter
])
```

## Key Takeaways

- **GEval** replaces expensive human evaluation for subjective criteria
- Keep **temperature=0** for reproducibility
- Validate your judge against a gold set before trusting scores
- Combine GEval with deterministic metrics for robust evaluation
- The judge model doesn't need to match the evaluated model — use cheap judges

## Further Reading

- [Metrics Reference](../metrics.md) — full list of built-in metrics
- [Best Practices for Scorers](../best_practices/scorers.md)
