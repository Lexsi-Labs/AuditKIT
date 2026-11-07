"""hf: device / device_map / nested model_kwargs handed to transformers.pipeline()."""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from auditkit.model.hf_gen import HFGenModel


@pytest.fixture
def calls(monkeypatch):
    import importlib
    transformers = importlib.import_module("transformers")
    pipelines = importlib.import_module("transformers.pipelines")
    seen: list[dict] = []

    def pipeline(task, model=None, **kw):
        seen.append(kw)
        return SimpleNamespace(generation_config=SimpleNamespace(max_length=None),
                               model=SimpleNamespace(generation_config=SimpleNamespace(max_length=None)),
                               tokenizer=None)

    real, had = pipelines.pipeline, "pipeline" in transformers.__dict__
    real_top = transformers.__dict__.get("pipeline")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    transformers.pipeline = pipeline
    pipelines.pipeline = pipeline
    try:
        yield seen
    finally:
        pipelines.pipeline = real
        if had:
            transformers.pipeline = real_top
        else:
            transformers.__dict__.pop("pipeline", None)


def load(**kw) -> dict:
    m = HFGenModel("m", **kw)
    m._ensure_pipeline()
    return m


def test_device_alone_is_passed(calls):
    load(device="cpu")
    assert calls[0]["device"] == "cpu" and "device_map" not in calls[0]


@pytest.mark.parametrize("dm", ["auto", "balanced", "sequential", {"": 0}, {"": "cpu"}])
def test_device_map_never_travels_with_device(calls, dm):
    load(device_map=dm)
    assert calls[0]["device_map"] == dm and "device" not in calls[0]


def test_explicit_device_with_device_map_does_not_raise_and_is_dropped(calls):
    load(device="cuda:1", device_map="auto")
    assert "device" not in calls[0] and calls[0]["device_map"] == "auto"


def test_device_map_none_still_places_on_the_resolved_device(calls):
    # device_map=None means "no map": the resolved device must still be used, not dropped
    load(device="cpu", device_map=None)
    assert calls[0].get("device") == "cpu"
    assert calls[0].get("device_map") is None


def test_device_map_inside_model_kwargs_drops_device(calls):
    load(device="cpu", model_kwargs={"device_map": "auto"})
    assert "device" not in calls[0]


@pytest.mark.parametrize("key", ["quantization_config", "load_in_8bit", "load_in_4bit"])
def test_quantized_load_is_not_moved_with_device(calls, key):
    # a bitsandbytes model is placed by accelerate; pipeline(device=...) would try to move it
    load(device="cpu", **{key: object() if key == "quantization_config" else True})
    assert "device" not in calls[0]


def test_nested_kwargs_are_not_top_level(calls):
    load(device="cpu", quantization_config="q", max_memory={0: "10GiB"}, attn_implementation="sdpa",
         low_cpu_mem_usage=True)
    kw = calls[0]
    for k in ("quantization_config", "max_memory", "attn_implementation", "low_cpu_mem_usage"):
        assert k not in kw
    assert kw["model_kwargs"] == {"quantization_config": "q", "max_memory": {0: "10GiB"},
                                  "attn_implementation": "sdpa", "low_cpu_mem_usage": True}


def test_top_level_params_stay_top_level(calls):
    load(device="cpu", dtype="float32", trust_remote_code=True, revision="r", use_fast=False, device_map="auto")
    kw = calls[0]
    assert (kw["dtype"], kw["trust_remote_code"], kw["revision"], kw["use_fast"]) == ("float32", True, "r", False)
    assert "model_kwargs" not in kw


def test_no_model_kwargs_key_when_nothing_to_nest(calls):
    load(device="cpu")
    assert "model_kwargs" not in calls[0]


def test_caller_model_kwargs_dict_is_not_mutated(calls):
    mine = {"attn_implementation": "eager"}
    load(device="cpu", model_kwargs=mine, max_memory={0: "1GiB"})
    assert mine == {"attn_implementation": "eager"}


def test_model_kwargs_none_is_ignored(calls):
    load(device="cpu", model_kwargs=None)
    assert "model_kwargs" not in calls[0]


def test_conflicting_key_in_model_kwargs_and_top_level_is_deterministic():
    """model_kwargs={'max_memory': A} plus max_memory=B must pick the same winner in every process."""
    code = (
        "import sys, types\n"
        "from unittest import mock\n"
        "import transformers, transformers.pipelines as p\n"
        "seen = []\n"
        "def pipeline(task, model=None, **kw):\n"
        "    seen.append(kw)\n"
        "    return types.SimpleNamespace(generation_config=None, model=types.SimpleNamespace(generation_config=None))\n"
        "transformers.pipeline = pipeline; p.pipeline = pipeline\n"
        "from auditkit.model.hf_gen import HFGenModel\n"
        "m = HFGenModel('m', device='cpu', model_kwargs={'max_memory': 'A'}, max_memory='B')\n"
        "m._ensure_pipeline()\n"
        "print(seen[0]['model_kwargs']['max_memory'])\n"
    )
    import auditkit
    src = os.path.dirname(os.path.dirname(auditkit.__file__))      # the source this process imported
    env = {**os.environ, "PYTHONPATH": src}
    runs = [subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env={**env, "PYTHONHASHSEED": str(s)}) for s in range(8)]
    assert all(r.returncode == 0 for r in runs), [r.stderr[-300:] for r in runs if r.returncode]
    assert {r.stdout.strip() for r in runs} == {"B"}


def test_fp16_is_not_forced_on_a_cpu_device_map(calls, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda including_emulation=True: False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (7, 5))
    import transformers
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained",
                        lambda *a, **k: SimpleNamespace(dtype="bfloat16"))
    load(device_map={"": "cpu"})          # nothing is on the GPU, so no fp16 downcast
    assert "dtype" not in calls[0]


def _t4(monkeypatch, dtype="bfloat16"):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda including_emulation=True: False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (7, 5))
    import transformers
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained",
                        lambda *a, **k: SimpleNamespace(dtype=dtype))


def test_fp16_is_not_forced_next_to_a_dtype_in_model_kwargs(calls, monkeypatch):
    _t4(monkeypatch)
    load(model_kwargs={"torch_dtype": "float32"})
    assert "dtype" not in calls[0] and calls[0]["model_kwargs"]["torch_dtype"] == "float32"


@pytest.mark.parametrize("device_map", ["cpu", "disk", "mps"])
def test_fp16_is_not_forced_on_a_string_cpu_device_map(calls, monkeypatch, device_map):
    _t4(monkeypatch)
    load(device_map=device_map)
    assert "dtype" not in calls[0]


def test_explicit_device_next_to_a_device_map_is_logged(calls, caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="auditkit.model.hf_gen"):
        load(device="cuda:1", device_map="auto")
    assert "device" not in calls[0] and "ignored" in caplog.text
