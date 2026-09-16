import pytest

from auditkit.errors import AuditKitError, CapabilityError
from auditkit.model import (
    Request, Generated, Result_, LogLikelihood,
    Model, CallableModel, EchoModel,
)
from auditkit.types import Capability


# ---- model I/O dataclasses -----------------------------------------------

def test_request_defaults():
    r = Request(prompt="hi")
    assert r.prompt == "hi"
    assert r.request_type == "generate"
    assert r.params == {}


def test_result_text_reads_first_completion():
    res = Result_(completions=[Generated(text="hello"), Generated(text="world")])
    assert res.text == "hello"
    assert res.latency_ms is None
    assert res.usage == {}


def test_result_text_empty_when_no_completions():
    assert Result_(completions=[]).text == ""


def test_loglikelihood_defaults():
    ll = LogLikelihood(logprob=-1.5)
    assert ll.logprob == -1.5
    assert ll.is_greedy is False


# ---- Model ABC contract --------------------------------------------------

def test_model_is_abstract():
    with pytest.raises(TypeError):
        Model()  # generate() is abstract


def test_default_capabilities_and_supports():
    m = EchoModel()
    assert m.capabilities() == {Capability.GENERATE}
    assert m.supports(Capability.GENERATE) is True
    assert m.supports(Capability.LOGLIKELIHOOD) is False


def test_loglikelihood_raises_capability_error_by_default():
    m = EchoModel()
    with pytest.raises(CapabilityError):
        m.loglikelihood([Request(prompt="a", request_type="loglikelihood")])


def test_capability_error_is_auditkit_error():
    assert issubclass(CapabilityError, AuditKitError)


# ---- EchoModel (deterministic test double) -------------------------------

def test_echo_model_returns_prompt_text():
    m = EchoModel()
    out = m.generate([Request(prompt="ping"), Request(prompt="pong")])
    assert [r.text for r in out] == ["ping", "pong"]


# ---- CallableModel (wrap any list[str]->list[str]) -----------------------

def test_callable_model_wraps_batch_function():
    def upper(prompts):
        return [p.upper() for p in prompts]

    m = CallableModel(upper)
    out = m.generate([Request(prompt="a"), Request(prompt="b")])
    assert [r.text for r in out] == ["A", "B"]
    assert m.supports(Capability.GENERATE) is True


def test_callable_model_default_name():
    assert CallableModel(lambda ps: ps).name  # non-empty


def test_callable_model_name_distinguishes_closures_by_captured_values():
    """Two closures from one factory share bytecode but capture different values.

    This is the "gateway/judge callable" shape (a factory closing a model name /
    provider over an inner fn): the default name must fold in the captured
    free-variable values, or LLMJudge/GuardJudge scorers wrapping different judge
    models collide under one name -> identical RunSpec.fingerprint -> the disk
    cache replays the wrong run.
    """
    def build(model):
        def fn(prompts):
            return [f"{model}:{p}" for p in prompts]  # references `model` -> captured
        return fn

    a = CallableModel(build("model-a"))
    b = CallableModel(build("model-b"))
    a_again = CallableModel(build("model-a"))

    assert a.name != b.name           # different captured model -> different name
    assert a.name == a_again.name     # identical capture -> stable name (cache still replays)


def test_callable_model_name_is_stable_when_closure_captures_mutable_state():
    """A closure over a mutable accumulator must keep a STABLE name as it mutates.

    The "bring your own model" lambda often closes over a call log / client /
    logger. Folding those mutable captures into the name would change the
    fingerprint between two logically-identical runs and defeat the disk cache
    (regression: test_fingerprint_stability). Only immutable scalar captures
    count toward identity; mutable ones are ignored.
    """
    calls = []
    fn = (lambda prompts: (calls.append(prompts), ["hi"] * len(prompts))[1])
    name_before = CallableModel(fn).name
    calls.append(["mutated"])           # simulate a first run appending to the log
    name_after = CallableModel(fn).name
    assert name_before == name_after


def test_callable_model_name_survives_unreadable_closure_value():
    """A captured value whose repr raises must not crash name derivation."""
    class Boom:
        def __repr__(self):
            raise RuntimeError("no repr")

    def build(obj):
        def fn(prompts):
            return [obj for _ in prompts]  # captures `obj`
        return fn

    assert CallableModel(build(Boom())).name  # non-empty, no exception


def test_ensure_pipeline_survives_missing_generation_config(monkeypatch):
    """Regression: a TextGenerationPipeline without a ``generation_config``
    attribute (older transformers / some models, e.g. sshleifer/tiny-gpt2) must
    not crash pipeline construction. The max_length-warning suppression is a
    nicety and is now getattr-guarded. Reproduces the bug with a fake pipeline
    so it needs no real model download."""
    pytest.importorskip("torch")              # _ensure_pipeline imports torch
    transformers = pytest.importorskip("transformers")
    from auditkit.model.hf_gen import HFGenModel

    class _Cfg:
        max_length = 20                        # a default the code should clear
    class _Model:
        generation_config = _Cfg()
    class _FakePipeline:                        # deliberately NO generation_config attr
        model = _Model()

    monkeypatch.setattr(transformers, "pipeline", lambda *a, **k: _FakePipeline())
    m = HFGenModel(model="fake/model", device="cpu")
    m._ensure_pipeline()                        # must NOT raise AttributeError
    assert m._pipeline is not None
    assert m._pipeline.model.generation_config.max_length is None   # model cfg cleared
    assert not hasattr(m._pipeline, "generation_config")            # pipeline had none -> skipped
