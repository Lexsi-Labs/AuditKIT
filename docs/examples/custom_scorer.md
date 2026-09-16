# Custom scorers & metrics

AuditKIT ships 40+ metrics, but the point of the spine is that **your own
metric is a first-class citizen** — it plugs in exactly like a built-in, flows
through the same `RunResult`, and is picked up by comparisons and caching with
no core changes. There are two levels of "custom", depending on how much you
need.

## 1. The `@ak.scorer` decorator (the 90% case)

Any `(sample, output) -> float` function becomes a metric. `direction=` is
**required** — it tells the comparison layer which way is better, so a change in
your metric is interpreted correctly next to accuracy, latency, etc.

```python
import auditkit as ak

@ak.scorer(direction=ak.Direction.MAXIMIZE)   # higher = better
def mentions_price(sample, output):
    return 1.0 if "$" in output else 0.0

result = ak.evaluate(
    [ak.Sample(input="Quote the product price", target="$49")],
    model=lambda prompts: ["It costs $49." for _ in prompts],
    scorers=[mentions_price],
)
print(result.headline)   # {"mentions_price": 1.0}
```

You can read the `sample` too — its `target`, `metadata`, or `retrieval_context`
— so a scorer can compare against the gold answer, not just inspect the output:

```python
@ak.scorer(direction=ak.Direction.MAXIMIZE)
def exact_number_match(sample, output):
    """1.0 only if the first number in the output equals the target."""
    import re
    nums = re.findall(r"-?\d+(?:\.\d+)?", output)
    return 1.0 if nums and nums[0] == str(sample.target) else 0.0
```

### A `MINIMIZE` scorer

When lower is better (a cost, a distance, an error count), say so — comparisons
will then correctly read a *rise* as a regression:

```python
@ak.scorer(direction=ak.Direction.MINIMIZE)   # lower = better
def excess_length(sample, output):
    """How many characters over a 280-char budget (0.0 = within budget)."""
    return max(0.0, len(output) - 280) / 280.0
```

## 2. Subclass `Metric` (when you need configuration)

If your metric takes **parameters** (a threshold, a keyword list, a model),
subclass `Metric` and register it. Registration means it's resolvable by name
everywhere — `scorers=["under_limit"]`, the CLI's `--scorers`, YAML configs.

```python
from auditkit.metric import Metric
from auditkit.score import Score
from auditkit.types import Direction, ScoreKind
from auditkit.registry import METRICS

@METRICS.register("under_limit")
class UnderLimit(Metric):
    direction = Direction.MAXIMIZE

    def __init__(self, max_len: int = 280):
        self.max_len = max_len
        # Include config in the name so two differently-configured instances
        # are distinguishable in the headline AND in the run fingerprint.
        self.name = f"under_limit({max_len})"

    def identity(self) -> dict:
        # Anything that changes scoring behavior belongs here — it feeds the
        # run fingerprint, so a cached result is never reused across configs.
        return {"name": self.name, "max_len": self.max_len}

    def score(self, sample, output, context=None) -> Score:
        return Score(
            name=self.name,
            value=float(len(output) <= self.max_len),
            kind=ScoreKind.CODE,
        )

# Use it by name (default config) or as a configured instance:
ak.evaluate(data, model=m, scorers=["under_limit"])
ak.evaluate(data, model=m, scorers=[UnderLimit(max_len=140)])
```

!!! warning "Put config in `identity()`"
    The fingerprint is what makes results cacheable and comparable. If a
    constructor argument affects scoring but isn't returned from `identity()`,
    two differently-configured runs can silently share one cached result.
    AuditKIT warns at class-definition time if a custom `Metric` looks like it
    has this gap — but returning your real config from `identity()` is the fix.

## 3. Mixing custom and built-in scorers

Custom scorers compose with built-ins in the same run — each produces its own
headline entry:

```python
result = ak.evaluate(
    data, model=m,
    scorers=["exact_match", mentions_price, UnderLimit(max_len=140)],
)
print(result.headline)
# {"exact_match": 0.80, "mentions_price": 1.00, "under_limit(140)": 0.65}

# Inspect the per-sample detail for any one of them:
for p in result.predictions[:3]:
    print(p.output, "->", p.metadata["scores"])
```

Because every scorer declares a `direction`, a later
`ak.compare(baseline, candidate)` reports each one correctly — a drop in
`mentions_price` reads as a regression, a rise in `excess_length` reads as one
too.
