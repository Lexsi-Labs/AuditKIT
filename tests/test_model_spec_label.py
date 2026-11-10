"""RunResult.model_spec is the model as a plain string, never an object repr.

It was ``str(model)``, e.g. ``"<auditkit.model.PrecomputedModel object at 0x107067770>"``: a
different string every process, unreadable in saved runs and cached results.
"""
from __future__ import annotations

import json
import re

import pytest

import auditkit as ak
from auditkit import Sample
from auditkit.model import CallableModel
from auditkit.report import RunResult
from auditkit.runner import _model_label

SAMPLES = [Sample(input="2+2?", target="4", actual_output="4", id="q1")]


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))


@pytest.mark.parametrize("model, label", [
    ("precomputed", "precomputed"),
    ("echo", "echo"),
])
def test_string_spec_is_recorded(model, label):
    assert ak.evaluate(SAMPLES, model=model, scorers=["exact_match"]).model_spec == label


def test_callable_and_named_callable():
    r = ak.evaluate(SAMPLES, model=CallableModel(lambda ps: ["4"] * len(ps), name="my-app-v2"),
                    scorers=["exact_match"])
    assert r.model_spec == "my-app-v2"


def test_directly_built_backend_adds_its_checkpoint():
    class _Backend:
        name = "hf"
        _model_name = "CohereLabs/tiny-aya-global"
    assert _model_label(_Backend()) == "hf:CohereLabs/tiny-aya-global"
    _Backend.name = "hf:CohereLabs/tiny-aya-global"            # what AutoModel.resolve sets
    assert _model_label(_Backend()) == "hf:CohereLabs/tiny-aya-global"


def test_saved_run_has_no_object_repr(tmp_path):
    r = ak.evaluate(SAMPLES, model="precomputed", scorers=["exact_match"])
    path = tmp_path / "run.json"
    r.save(str(path))
    text = path.read_text()
    assert not re.search(r"object at 0x[0-9a-f]+", text)
    assert RunResult.from_dict(json.loads(text)).model_spec == "precomputed"
