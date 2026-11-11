"""The built-in gsm8k scenario scores the final number, not the whole worked solution."""
import sys
from types import SimpleNamespace

import pytest

import auditkit as ak
from auditkit.metric import NumericMatch
from auditkit.sample import Sample

ROW = {"question": "Natalia sold 48 clips in April and half as many in May. How many in total?",
       "answer": "In May she sold 48/2 = <<48/2=24>>24 clips.\nIn total 48+24 = <<48+24=72>>72.\n#### 72"}


@pytest.fixture
def gsm8k(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets",
                        SimpleNamespace(load_dataset=lambda *a, **k: [ROW]))


def test_target_is_the_final_number(gsm8k):
    (s,) = ak.SCENARIOS.get("gsm8k")().samples()
    assert s.target == "72"
    assert s.metadata["solution"] == ROW["answer"]


@pytest.mark.parametrize("output,value", [
    ("72", 1.0),
    ("The answer is 72.", 1.0),
    ("48+24 = 72\n#### 72", 1.0),
    ("She sold $1,072 worth... no wait, 72", 1.0),
    ("#### 72\nThat took 3 steps.", 1.0),
    ("The answer is 73.", 0.0),
    ("I don't know.", 0.0),
])
def test_numeric_match(output, value):
    score = NumericMatch().score(Sample(input="q", target="72"), output)
    assert score.value == value
    assert score.metadata["no_number"] is (output == "I don't know.")


def test_comma_and_decimal_targets():
    m = NumericMatch()
    assert m.score(Sample(input="q", target="1200"), "It costs $1,200.").value == 1.0
    assert m.score(Sample(input="q", target="2.5"), "2.50 hours").value == 1.0


def test_evaluate_gsm8k_defaults_to_numeric_match(gsm8k):
    result = ak.evaluate("gsm8k", model=lambda prompts: ["48 + 24 = 72. The answer is 72."] * len(prompts),
                         cache=False)
    assert result.headline == {"numeric_match": 1.0}
