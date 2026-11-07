"""``MCQAdapter("mcq_joint")`` (index labels) passes the run's generation settings to the model.

Regression: the index-label path built its request without ``params``, so ``max_tokens``,
``temperature``, ``seed``, ... never reached the backend (BELEBELE and MILU need 4 tokens,
greedy). With no settings given, what a backend receives is unchanged: the only value
forwarded is ``temperature=0.0``, which is every backend's own fallback."""

from __future__ import annotations

import pytest

import auditkit as ak
from auditkit.model import DEFAULT_TEMPERATURE, Model, Request, resolve_params
from auditkit.runspec import RunConfig, RunSpec

SAMPLE = ak.Sample(id="q1", input="भारत की राजधानी?", choices=["मुंबई", "नई दिल्ली", "कोलकाता"], target=1)
PROMPT = "भारत की राजधानी?\n\n0. मुंबई\n1. नई दिल्ली\n2. कोलकाता\n\nAnswer:"


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    # the fix leaves the fingerprint unchanged, so a shared cache would replay an earlier run
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))


def test_index_labels_forward_generation_settings():
    req = ak.MCQAdapter().adapt(SAMPLE, RunConfig(max_tokens=4, temperature=0.0, seed=13, stop_sequences=["\n"]))[0]
    assert req.prompt == PROMPT and req.request_type == "generate"
    assert req.params == {"temperature": 0.0, "max_tokens": 4, "seed": 13, "stop_sequences": ["\n"]}


def test_default_settings_reach_backends_unchanged():
    req = ak.MCQAdapter().adapt(SAMPLE, RunConfig())[0]
    assert req.prompt == PROMPT and req.params == {"temperature": 0.0}
    assert DEFAULT_TEMPERATURE == 0.0
    key_map = {"temperature": "temperature", "max_tokens": "max_tokens", "seed": "seed"}
    for defaults in ({"temperature": DEFAULT_TEMPERATURE, "max_tokens": 1024},
                     {"temperature": DEFAULT_TEMPERATURE, "max_new_tokens": 128}):
        before = resolve_params(Request(prompt=PROMPT, request_type="generate"), defaults, key_map)
        assert resolve_params(req, defaults, key_map) == before


def test_fingerprint_unchanged_by_the_fix(monkeypatch):
    from auditkit import adapter as adapter_mod
    from auditkit.scenario import Scenario

    class _One(Scenario):
        def samples(self):
            return [SAMPLE]

    def fingerprint():
        return RunSpec(scenario=_One(name="one"), model=_Recorder(), adapter=ak.MCQAdapter(), metrics=[],
                       config=RunConfig(max_tokens=4)).fingerprint()

    fixed = fingerprint()
    monkeypatch.setattr(adapter_mod, "_gen_params", lambda config: {})    # the old, settings-dropping behaviour
    assert fingerprint() == fixed and ak.MCQAdapter().identity() == {"method": "mcq_joint"}


class _Recorder(Model):
    """Records the requests it receives and answers "1"."""

    def __init__(self):
        self.requests = []

    def identity(self) -> dict:
        return {"name": "recorder"}

    def generate(self, requests):
        from auditkit.model import Generated, Result_
        self.requests.extend(requests)
        return [Result_(completions=[Generated(text="1")]) for _ in requests]


def test_end_to_end_run_config_reaches_the_model():
    model = _Recorder()
    run = ak.evaluate([SAMPLE], model=model, adapter=ak.MCQAdapter(), scorers=["acc"],
                      config=RunConfig(max_tokens=4, temperature=0.0))
    assert [r.params for r in model.requests] == [{"temperature": 0.0, "max_tokens": 4}]
    assert run.stats["acc"].mean == 1.0                            # answer "1" is choice 1
