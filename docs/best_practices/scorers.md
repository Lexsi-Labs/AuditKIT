# Best Practices: Writing Scorers

How to design metrics (scorers) for evaluating AI outputs with AuditKIT,
adapted from industry best practices.

## Mental Model

Every evaluation has three parts:

1. **Data** — test cases with inputs and optional expected outputs
2. **Task** — the function being evaluated (LLM call, agent, pipeline)
3. **Scorers/Metrics** — functions that measure output quality

AuditKIT metrics map to this model. A metric receives a `Sample` (data) and an `output`
(task result), and returns a `Score`.

## Start with Code-Based Checks

Where possible, use deterministic metrics. They are reliable, fast, and have zero
dependencies.

```python
import auditkit as ak

# Built-in code metrics
ak.Equals().score(sample, output)               # exact match
ak.Contains("keyword").score(sample, output)     # substring check
ak.Regex(r"pattern").score(sample, output)       # regex match
ak.IsJson(require_keys=["name"]).score(sample, output)  # valid JSON
ak.WordCount(min_words=2, max_words=50).score(sample, output)       # length check
ak.F1Score().score(sample, output)                      # token overlap
```

## Custom Scorers for Domain Logic

When built-in metrics don't capture your domain, write a custom scorer:

```python
@ak.scorer(direction=ak.Direction.MAXIMIZE)
def contains_citation(sample, output):
    """Check if the output includes a citation."""
    import re
    return 1.0 if re.search(r"\[.*\]", output) else 0.0


@ak.scorer(name="format_score", direction=ak.Direction.MAXIMIZE)
def check_format(sample, output):
    """Check if output follows expected format."""
    lines = output.strip().split("\n")
    if len(lines) < 2:
        return 0.0
    return 1.0 if lines[0].startswith("#") else 0.5
```

## `direction` is required

Every scorer — built-in or custom — has to say whether a **higher** or
**lower** value is better. `RunComparison`/`compare_models()` read this to
decide which run "won" and whether a change counted as an improvement or a
regression; there's no way to grade correctly without it, so it isn't
optional and there's no default to fall back on.

- **Built-in metrics** (`ak.Equals`, `ak.F1Score`, `ak.Perplexity`, ...)
  already declare the right one — you don't set anything.
- **`@ak.scorer`** requires `direction=` explicitly; the bare `@ak.scorer`
  form (no parens) no longer works, since there'd be nowhere to put it:

  ```python
  import auditkit as ak

  # Higher is better (a similarity/accuracy-style score, 0-1 or otherwise):
  @ak.scorer(direction=ak.Direction.MAXIMIZE)
  def keyword_overlap(sample, output):
      target_words = set((sample.target or "").split())
      output_words = set(output.split())
      return len(target_words & output_words) / max(len(target_words), 1)

  # Lower is better (a cost/latency/error-count-style score):
  @ak.scorer(direction=ak.Direction.MINIMIZE)
  def response_length_penalty(sample, output):
      return float(len(output))  # shorter answers score better here

  result = ak.evaluate(dataset, model=model, scorers=[keyword_overlap])
  ```

- **Custom `Metric` subclasses** must declare `direction` as a class
  attribute — this is enforced at class-definition time, not silently
  defaulted:

  ```python
  from auditkit.metric import Metric
  from auditkit.score import Score
  from auditkit.types import Direction

  class ResponseLength(Metric):
      name = "response_length"
      direction = Direction.MINIMIZE  # shorter output = better, for this metric

      def score(self, sample, output, context=None):
          return Score(name=self.name, value=float(len(output)))
  ```

  Omitting `direction` raises `TypeError` the moment the class is defined
  (at import time), not later when it's actually used — so a missing
  direction is caught immediately, not discovered as a silently-backwards
  leaderboard after a long evaluation run.

## Single-Aspect Metrics

Create separate metrics for each dimension you care about:

```python
result = ak.evaluate(dataset, model=model, scorers=[
    ak.Equals(),           # accuracy
    ak.Contains("ref"),    # citation presence
    custom_relevance,      # domain-specific relevance
])
```

This lets you pinpoint which aspect is failing.

## Combining Metrics with Weights

```python
gate = ak.ScoreGate(metric=ak.Equals(), weight=2.0, threshold=0.5)
ws = ak.WeightedSum(name="overall")
combined = ws.compute(gate.score(sample, output))
```

## Using Generation Metrics

For text generation tasks, use the generation metric family:

```python
# Translation / summarization quality
ak.Bleu().score(sample, output)        # n-gram overlap
ak.RogueL().score(sample, output)      # longest common subsequence
ak.ChrF().score(sample, output)        # character n-gram F-score
ak.WordErrorRate().score(sample, output)  # edit distance
```

## Safety and Bias Checks

```python
ak.ToxicityScore().score(sample, output)      # 1.0 = safe
ak.HateSpeechScore().score(sample, output)    # 1.0 = safe
ak.RepresentationSkew().score(sample, output) # 0.0 = balanced demographic representation
ak.BiasJudge(judge_model="groq:llama-3.3-70b-versatile").score(sample, output)  # 0.0 = no biased opinions
```

## Iterate and Refine

1. Start with 2-3 broad metrics
2. Review low-scoring outputs to identify missing criteria
3. Add targeted metrics for recurring failure modes
4. Re-run and compare across experiments

## Further Reading

- [Metrics Reference](../metrics.md) — full list of built-in metrics
- [Custom Scorer Example](../examples/custom_scorer.md)
