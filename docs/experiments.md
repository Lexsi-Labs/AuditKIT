# Experiment Tracking

AuditKIT tracks experiments with built-in comparison and diffing, plus
optional MLflow integration.

## Tagging Runs

```python
result = ak.evaluate(
    dataset,
    model="hf:gpt2",
    experiment_name="my_experiment",
    tags=["baseline", "v1"],
)
```

## Loading Experiments

```python
exp = ak.ExperimentDB().load("my_experiment")
```

## Aggregate

```python
exp.aggregate()
# {"exact_match": 0.85, "bleu": 0.72}
```

## Leaderboard

```python
exp.leaderboard()
# [{"run_id": "...", "exact_match": 0.9, "bleu": 0.75}, ...]
```

## Significance Testing

```python
exp.significance("exact_match")
# {"baseline": "...", "contrast": "...", "delta": 0.05, "p_value": 0.032,
#  "significant": True, "method": "bootstrap", "n_resamples": 1000, "n_samples": 50}
```

Compares the best two runs in the experiment (ranked by the mean of
`exact_match`). It aligns per-sample scores by `sample_id` and runs a paired
bootstrap resample.

## Compare

```python
ranking = ak.compare([result1, result2])
```

## Diff

```python
diff = ak.RunDiff(baseline, contrast)
diff.metric_deltas()
diff.grade("exact_match")  # PASS | WARN | FAIL
diff.summary()
```

## MLflow

`log_mlflow` is a method on `Experiment`, not a standalone function:

```python
exp.log_mlflow(tracking_uri="http://localhost:5000")
```
