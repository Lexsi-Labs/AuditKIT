# Pre-built Scenarios

AuditKIT ships with 6 pre-built benchmark scenarios that can be used as datasets.
Requires `pip install auditkit[interop]` (the `datasets` package).

| Scenario | Class | Registry Name |
|---|---|---|
| MMLU | `MMLUScenario(subject="all", split="test")` | `mmlu` |
| GSM8K | `GSM8KScenario(split="test")` | `gsm8k` |
| ARC | `ARCScenario(split="test")` — always ARC-Challenge | `arc` |
| HellaSwag | `HellaSwagScenario(split="validation")` | `hellaswag` |
| TruthfulQA | `TruthfulQAScenario(split="validation")` | `truthfulqa` |
| HumanEval | `HumanEvalScenario(split="test")` | `humaneval` |

AuditKIT doesn't re-export these classes at the top-level `auditkit` package.
Import them from their own module, or resolve by name via `SCENARIOS`.

## Usage

```python
import auditkit as ak

# Direct instantiation
from auditkit.scenarios.mmlu import MMLUScenario
scenario = MMLUScenario(subject="abstract_algebra")
result = ak.evaluate(scenario, model="hf:gpt2")

# Via registry
scenario_cls = ak.SCENARIOS.get("mmlu")
scenario = scenario_cls(subject="abstract_algebra")

# List all available
print(ak.SCENARIOS.names())
```

## Custom Scenario

```python
from auditkit.scenario import Scenario
from auditkit import Sample

class MyScenario(Scenario):
    def __init__(self):
        super().__init__(name="my_scenario")

    def samples(self):
        yield Sample(input="hello", target="world")
```
