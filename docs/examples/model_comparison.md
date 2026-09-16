# Model Comparison

Compare multiple models on the same dataset.

```python
import auditkit as ak

samples = [
    ak.Sample(input="hello", target="hello"),
    ak.Sample(input="world", target="world"),
]

result = ak.compare_models(
    models=[lambda prompts: prompts, lambda prompts: prompts],
    dataset=samples,
    model_names=["model_a", "model_b"],
)
print(result.summary())
```

## Expected Output

```
Model Comparison — 2 samples, 1 scorers

Runs:
  model_a: run_id=bcafe3ef86cc4610
  model_b: run_id=bcafe3ef86cc4610

Metric                   model_a                   model_b
-----------------------------------------------------------------------------
exact_match              1.0000 (n=2) <-           1.0000 (n=2)

Model size:
  model_a (callable:<lambda>:eada6971): not a local model -- no measurable size
  model_b (callable:<lambda>:eada6971): not a local model -- no measurable size

Performance (measured):
  Model               latency (ms)  req/s     out-tok/s
  --------------------------------------------------------
  model_a             0             212385.58 n/a
  model_b             0             212385.58 n/a

Significance (pairwise bootstrap p < 0.05):
  model_a vs model_b: p=1.0000 (not significant)
```

`<-` marks the winner per metric (real `summary()` output uses plain ASCII,
not `←`). A plain in-process callable is near-instant, so `req/s` is
enormous and noisy run to run. A real backend's numbers are meaningful;
this one's aren't. The significance test uses paired bootstrap resampling to
determine whether differences are statistically significant.

## CLI

```bash
auditkit compare --models openai:gpt-4o,anthropic:claude-3 --csv data.csv
```

## Python API

```python
result = ak.compare_models(
    models=["openai:gpt-4o", "anthropic:claude-3", "hf:mistralai/Mistral-7B"],
    dataset=samples,
    model_names=["GPT-4o", "Claude 3", "Mistral 7B"],
)

# Get per-metric breakdown
for pm in result.per_metric():
    print(f"{pm.metric}: winner is {pm.winner}")

# Check significance
sig = result.significance("GPT-4o", "Claude 3", metric="exact_match")
print(f"p-value: {sig['p_value']}, significant: {sig['significant']}")
```

## Run It

Copy the snippets above into a Python file or notebook cell and run them
directly — see `examples/README.md` for the current set of runnable,
Colab-ready example notebooks (`11_performance_metrics_demo.ipynb` covers
`compare_models()`/`RunComparison` end to end with real models).

## Baseline vs candidate: `RunComparison` (pruned/quantized comparison)

`compare_models()`/`CompareResult` above answer "which of N models wins on
average." A different, more targeted question is: "did *this specific*
candidate (a pruned, quantized, or fine-tuned model) regress against its
baseline, and on *which task specifically*?" That's `RunComparison` — reached
either directly via `ak.compare(baseline_run, candidate_run)`, or via
`CompareResult.pairwise(baseline_name, candidate_name)` if you already ran
several models through `compare_models()`.

```python
import auditkit as ak

# A dataset spanning two tasks on purpose, so per-task deltas have
# something real to isolate (see "Expected Output" below).
data = [
    ak.Sample(input="2+2", target="4", task="arith"),
    ak.Sample(input="3+3", target="6", task="arith"),
    ak.Sample(input="4+4", target="8", task="arith"),
    ak.Sample(input="capital of France", target="Paris", task="trivia"),
    ak.Sample(input="capital of Italy", target="Rome", task="trivia"),
    ak.Sample(input="capital of Germany", target="Berlin", task="trivia"),
]

base = ak.evaluate(data, model="hf:base-checkpoint", scorers=["exact_match"])
pruned = ak.evaluate(data, model="hf:pruned-checkpoint", scorers=["exact_match"])

cmp = ak.compare(base, pruned)   # two RunResults -> RunComparison (not a leaderboard)
print(cmp.summary())
```

### Expected Output

```
Comparison: 1457c9d24db118cc (baseline) -> 67513f400be080e0 (candidate)
  overall grade: FAIL
  per-metric:
    exact_match: 1.0000 -> 0.5000 (delta=-0.5000, n=6->6, std=0.0000->0.5000) [FAIL]
  per-task:
    arith/exact_match: delta=+0.0000 (n=3->3) [PASS]
    trivia/exact_match: delta=-1.0000 (n=3->3) [FAIL]
  samples: regressed 3, improved 0
```

Reading it: the blended `exact_match` line alone (`1.00 -> 0.50`) makes it
*look* like a uniform 50% drop, but the per-task breakdown shows the real
picture — `arith` is untouched (`PASS`), the entire regression is
concentrated in `trivia` (`FAIL`). `n=6->6` and `std=0.0000->0.5000` confirm
both runs scored every sample (no failures silently excluded) and show the
candidate became far less consistent, not just worse on average.

Other views on the same `cmp`:

```python
cmp.grade()                       # DeltaGrade.FAIL -- the single ship/no-ship verdict
cmp.retention("exact_match")      # 0.5 -- candidate kept half of baseline's quality
cmp.regressed()                   # the 3 actual Prediction objects that flipped correct -> wrong
cmp.significance("exact_match")   # paired-bootstrap p-value -- is the delta distinguishable from noise
cmp.tradeoff()                    # everything measured automatically, nothing caller-supplied:
# -> {'retention': 0.5, 'size_ratio': 0.25, 'speedup': 4.0, ...}
#    size/latency come from RunResult.model_size/perf, populated by Runner while it ran.
#    Local backends (hf:/vllm:) get a real size_ratio (introspected params/on-disk MB);
#    if either side is API-based, size_ratio is omitted and 'api_based' names both
#    models instead, since there's no checkpoint to measure.
cmp.performance()                 # pure measured latency/throughput view, no quality/size mixed in

cmp.has_failures            # False here -- no sample crashed during scoring
cmp.failure_summary()       # per-run failed_count/errors, if has_failures were True
cmp.coverage_warnings()     # [] here -- both runs scored the same 6 samples; would flag
                            # any metric where baseline/candidate sample counts differ
```

### Via `compare_models()` instead of two separate `evaluate()` calls

```python
result = ak.compare_models(
    ["hf:base-checkpoint", "hf:pruned-checkpoint"], data, scorers=["exact_match"],
    model_names=["base", "pruned"],
    configs={"pruned": ak.RunConfig(limit=50)},        # per-model RunConfig override
    model_opts={"pruned": {"dtype": "int8"}},           # per-model backend kwargs
)
cmp = result.pairwise("base", "pruned")   # same RunComparison as above
```

`adapter=`/`annotators=`/`extract_with=` are also accepted directly on
`compare_models()` — shared across every model by default — and
`model_opts[name]` can override any of the three for one model only
(alongside backend kwargs like `dtype=` above):

```python
result = ak.compare_models(
    ["hf:base-checkpoint", "hf:verbose-finetune"], data, scorers=["exact_match"],
    model_names=["base", "verbose"],
    model_opts={
        "verbose": {"annotators": [my_extractor], "extract_with": "my_extractor"},
    },
)
```

`scorers` is deliberately **not** overridable per-model this way — comparison
assumes one shared metric set across models.
