"""hf: on GPUs without native bf16 (T4, V100) loads a bf16 checkpoint as fp16; everything else is untouched."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from auditkit.model.hf_gen import HFGenModel

WORLD: dict = {}


@pytest.fixture
def world(monkeypatch):
    # the transformers module hf: will import at call time: other tests may have swapped it in
    # sys.modules since this file was collected, so never patch the collection-time object
    import importlib
    transformers = importlib.import_module("transformers")
    state = {"bf16": False, "cc": (7, 5), "cuda": True, "cfg_dtype": "bfloat16", "text_cfg": False, "calls": []}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: state["cuda"])
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda including_emulation=True: state["bf16"])
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: state["cc"])

    def from_pretrained(name, **kw):
        if state["text_cfg"]:          # a vision model: the dtype lives on its text config
            return SimpleNamespace(dtype=None, text_config=SimpleNamespace(dtype=state["cfg_dtype"]))
        return SimpleNamespace(dtype=state["cfg_dtype"])
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", from_pretrained)

    def pipeline(task, model=None, device=None, **kw):
        state["calls"].append({"device": device, **kw})
        return SimpleNamespace(generation_config=SimpleNamespace(max_length=None),
                               model=SimpleNamespace(generation_config=SimpleNamespace(max_length=None)),
                               tokenizer=None)
    # transformers is a lazy module and re-exports `pipeline` from transformers.pipelines; a stub left in
    # either place leaks into every later test that loads a real model. Put the real one back by hand.
    pipelines = importlib.import_module("transformers.pipelines")
    real = pipelines.pipeline
    had = "pipeline" in transformers.__dict__
    transformers.pipeline = pipeline
    pipelines.pipeline = pipeline
    WORLD.clear(); WORLD.update(state)
    state = WORLD                     # the stubs above read `state`: let the tests' changes reach them
    try:
        yield state
    finally:
        pipelines.pipeline = real
        if had:
            transformers.pipeline = real
        else:
            transformers.__dict__.pop("pipeline", None)


def load(**kw):
    m = HFGenModel("CohereLabs/tiny-aya-global", **kw)
    try:
        m._ensure_pipeline()
    except Exception:                 # only the pipeline() arguments matter here; later setup may need more
        if not WORLD["calls"]:
            raise
    return m


def dtype_of(world):
    return world["calls"][-1].get("dtype")


def test_t4_loads_a_bf16_checkpoint_as_fp16(world):
    load(device="cuda")
    assert dtype_of(world) == torch.float16


def test_an_explicit_dtype_always_wins(world):
    load(device="cuda", dtype=torch.bfloat16)
    assert dtype_of(world) == torch.bfloat16


@pytest.mark.parametrize("change", ["a100", "cpu", "fp32 checkpoint"])
def test_nothing_changes_where_it_is_not_needed(world, change):
    if change == "a100":
        world.update(bf16=True, cc=(8, 0))
    elif change == "fp32 checkpoint":
        world.update(cfg_dtype="float32")
    load(device="cpu" if change == "cpu" else "cuda")
    assert dtype_of(world) is None


def test_a_vision_config_is_read_through_its_text_config(world):
    world.update(text_cfg=True)
    load(device="cuda")
    assert dtype_of(world) == torch.float16


def test_an_integer_device_counts_as_cuda(world):
    load(device=0)
    assert dtype_of(world) == torch.float16
