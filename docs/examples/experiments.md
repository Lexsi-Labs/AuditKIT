# Experiment Tracking

```python
import os, tempfile
import auditkit as ak

tmpdir = tempfile.mkdtemp()
os.environ["XDG_DATA_HOME"] = tmpdir

dataset = [
    ak.Sample(input="2+2?", target="4"),
    ak.Sample(input="3+3?", target="6"),
]

r1 = ak.evaluate(dataset, model=lambda prompts: ["4", "6"], experiment_name="math")

# Compare
ranking = ak.compare([r1, r1])
print(f"Leaderboard: {ranking}")

# Diff
diff = ak.RunDiff(r1, r1)
print(f"Grade: {diff.grade('exact_match')}")

# Experiment
exp = ak.ExperimentDB().load("math")
print(f"Aggregate: {exp.aggregate()}")
```
