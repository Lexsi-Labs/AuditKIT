"""A function model that reads state the fingerprint cannot see.

The default name of a function model covers its code and immutable captured values. Mutable or
opaque state it reads (a model object, a config dict, a module-level object) is skipped, so a run
after that state changes is served the earlier run's cached result. These tests cover the
``cache=False`` switch and the warning that points to it.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pytest

import auditkit as ak
from auditkit import Sample
from auditkit.model import CallableModel, _HIDDEN_STATE_WARNED, _hidden_state


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    _HIDDEN_STATE_WARNED.clear()


class _Weights:
    def __init__(self):
        self.mode = "broken"

    def answer(self, prompt):
        return "harmful advice" if self.mode == "broken" else "see a doctor"


MODEL = _Weights()
SAMPLES = [Sample(input="I have chest pain, what do I do?", target="see a doctor", id="q1")]


def _generate(prompts):                 # reads the module-level MODEL, edited in place below
    return [MODEL.answer(p) for p in prompts]


def _cache_files(tmp_path):
    return list(Path(tmp_path, "auditkit", "runs").glob("*.json"))


def test_cache_false_sees_an_in_place_edit():
    MODEL.mode = "broken"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = ak.evaluate(SAMPLES, model=_generate, scorers=["exact_match"])
        MODEL.mode = "repaired"
        replayed = ak.evaluate(SAMPLES, model=_generate, scorers=["exact_match"])
        fresh = ak.evaluate(SAMPLES, model=_generate, scorers=["exact_match"], cache=False)
    assert first.headline["exact_match"] == 0.0
    assert replayed.headline["exact_match"] == 0.0      # what the warning is about
    assert fresh.headline["exact_match"] == 1.0


def test_cache_false_writes_nothing(tmp_path):
    ak.evaluate(SAMPLES, model=lambda ps: ["see a doctor"] * len(ps), scorers=["exact_match"], cache=False)
    assert _cache_files(tmp_path) == []
    ak.evaluate(SAMPLES, model=lambda ps: ["see a doctor"] * len(ps), scorers=["exact_match"])
    assert len(_cache_files(tmp_path)) == 1


def test_generate_accepts_cache_false():
    MODEL.mode = "broken"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ak.generate(SAMPLES, _generate)
        MODEL.mode = "repaired"
        out = ak.generate(SAMPLES, _generate, cache=False)
    assert out[0].actual_output == "see a doctor"


def test_lmeval_engine_refuses_cache_false():
    with pytest.raises(ValueError, match="native engine only"):
        ak.evaluate("hellaswag", model="hf:gpt2", engine="lmeval", cache=False)


def test_warns_on_global_object():
    with pytest.warns(UserWarning, match=r"global 'MODEL'.*cache=False"):
        CallableModel(_generate)


def test_warns_on_global_read_inside_a_generator_expression():
    def gen(prompts):
        return list(MODEL.answer(p) for p in prompts)     # nested code object on every Python
    assert _hidden_state(gen) == ["global 'MODEL'"]


def test_warns_on_captured_object_and_bound_method():
    def build(model):
        return lambda ps: [model.answer(p) for p in ps]
    assert _hidden_state(build(_Weights())) == ["captured 'model'"]
    assert _hidden_state(MODEL.answer)[0] == "bound object _Weights"


def test_containers_and_plumbing_do_not_warn():
    import logging
    import threading
    calls, replies, log, lock = [], {"q": "a"}, logging.getLogger("x"), threading.Lock()
    def fn(ps):
        with lock:
            calls.append(ps)
            log.debug("call")
            return [replies.get(p, "") for p in ps]
    assert _hidden_state(fn) == []


def test_no_warning_when_identity_is_visible_or_named():
    def build(model_name):
        return lambda ps: [f"{model_name}:{p}" for p in ps]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        CallableModel(lambda ps: [p.upper() for p in ps])          # pure function
        CallableModel(build("model-a"))                             # immutable capture
        CallableModel(_generate, name="aya-broken-3f2a")            # caller names the state


def test_warns_once_per_function():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        CallableModel(_generate)
        CallableModel(_generate)
    assert sum("fingerprint cannot see" in str(x.message) for x in w) == 1
