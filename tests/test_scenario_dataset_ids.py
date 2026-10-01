"""The built-in benchmark scenarios load real Hub datasets: namespaced ids (the bare "mmlu", "arc" and
"truthfulqa" no longer resolve), the config each scenario needs, and a TaskKind that exists.
No network: datasets.load_dataset is replaced by a recorder returning one row of the real schema."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import auditkit as ak
from auditkit.types import TaskKind

ROWS = {  # one row per dataset, with the fields the real Hub schema has
    "cais/mmlu": {"question": "q", "choices": ["a", "b", "c", "d"], "answer": 1, "subject": "s"},
    "openai/gsm8k": {"question": "q", "answer": "work #### 4"},
    "allenai/ai2_arc": {"question": "q", "choices": {"text": ["a", "b"], "label": ["A", "B"]}, "answerKey": "B",
                        "id": "x"},
    "Rowan/hellaswag": {"ctx": "c", "endings": ["a", "b", "c", "d"], "label": "2"},
    "truthfulqa/truthful_qa": {"question": "q", "mc1_targets": {"choices": ["a", "b"], "labels": [1, 0]}},
    "openai/openai_humaneval": {"prompt": "def f():", "canonical_solution": "  return 1", "task_id": "t"},
}
EXPECTED = {  # scenario -> (dataset id, config)
    "mmlu": ("cais/mmlu", "all"),
    "gsm8k": ("openai/gsm8k", "main"),
    "arc": ("allenai/ai2_arc", "ARC-Challenge"),
    "hellaswag": ("Rowan/hellaswag", None),
    "truthfulqa": ("truthfulqa/truthful_qa", "multiple_choice"),
    "humaneval": ("openai/openai_humaneval", None),
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_each_scenario_loads_its_namespaced_dataset(monkeypatch, name):
    calls = []

    def load_dataset(repo, *args, split=None, **kw):
        calls.append((repo, args[0] if args else None))
        return [ROWS[repo]]
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    samples = ak.SCENARIOS.get(name)().samples()
    assert calls == [EXPECTED[name]]
    assert len(samples) == 1 and isinstance(samples[0].kind, TaskKind)
