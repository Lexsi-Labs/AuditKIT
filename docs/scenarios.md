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

Each loads its namespaced Hub dataset (`cais/mmlu`, `openai/gsm8k`, `allenai/ai2_arc`, `Rowan/hellaswag`,
`truthfulqa/truthful_qa`, `openai/openai_humaneval`). HumanEval's target is the reference solution, and
AuditKIT has no code-execution metric, so it can't score pass@k; use `ak.run_lmeval("humaneval", ...)` for that.
GSM8K's target is the final number, and it is scored with `numeric_match` (the number after `####`,
else the last number in the reply) unless you pass `scorers=`.

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

## Multilingual benchmarks

Three loaders return a `list[Sample]` with `task` set to the language, so
`ak.compare(...).per_task_deltas()` reports each metric per language. They need
`pip install auditkit[interop]` and default to a pinned dataset revision, recorded in
`metadata["revision"]`.

| Loader | Dataset | Notes |
|---|---|---|
| `ak.load_multijail(languages=None)` | `DAMO-NLP-SG/MultiJail` | 315 harmful prompts in 10 languages; no reference answers |
| `ak.load_xstest()` | `Paul/XSTest` | 250 safe and 200 unsafe English prompts; `metadata["label"]` says which |
| `ak.load_global_mmlu(languages, split="test")` | `CohereLabs/Global-MMLU` | Multiple choice in 42 languages; `repo=GLOBAL_MMLU_LITE` for 400 questions |

Translations of one item share `metadata["group"]`. MultiJail and the unsafe half of XSTest
are harmful prompts by design, so take care with where the prompts and answers end up.

```python
import auditkit as ak

samples = ak.load_global_mmlu(["en", "hi", "sw"], limit=100)
result = ak.evaluate(samples, "hf:gpt2", ["acc"], adapter="auto")
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
