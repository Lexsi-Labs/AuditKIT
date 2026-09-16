# Reports

## Formatted Reports

```python
from auditkit import Report

report = Report(result)

# Text table
print(report)

# Markdown
print(report.markdown())
```

Example output:

```
Run f70365d0ccc20fb8  (fingerprint f70365d0ccc20fb8)

Metric                             Mean  ±Stderr      Std      N
--------------------------------------------------------------
exact_match                      1.0000   0.0000   0.0000      2

2 predictions
```

## Result Object

```python
result.headline           # {"exact_match": 1.0}
result.summary()          # formatted table string
result.metric_table()     # list of dicts
result.wrong_only()       # incorrect predictions
result.save("run.json")   # JSON or CSV
result.to_dict()          # full dictionary
RunResult.load("run.json")  # load from file -- a classmethod, returns a new RunResult
```
