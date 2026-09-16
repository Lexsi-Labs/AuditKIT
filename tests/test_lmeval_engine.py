"""Tests for the lm-eval benchmark engine (lmeval_engine.py).

lm-eval is a heavy optional dep, so these tests inject a fake ``lm_eval`` module
into ``sys.modules`` and exercise the spec-mapping and result-mapping logic
without installing the harness. A real end-to-end run needs
``pip install auditkit[lmeval]``.
"""

from __future__ import annotations

import sys
import types

import pytest

import auditkit as ak
from auditkit.lmeval_engine import (
    _fingerprint,
    _to_runresult,
    map_model_spec,
    run_benchmark,
)
from auditkit.errors import AuditKitError, CapabilityError, ExtraNotInstalled
from auditkit.runspec import RunConfig


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Every run_benchmark() call now persists to DiskCache -- point it at a
    throwaway dir so tests don't pollute the real ~/.cache/auditkit/runs/ and
    don't collide with each other's identically-shaped mocked runs."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))


# --- map_model_spec -------------------------------------------------------

def test_map_hf_spec():
    backend, args = map_model_spec("hf:gpt2")
    assert backend == "hf"
    assert args == {"pretrained": "gpt2"}


def test_map_bare_name_defaults_to_hf():
    backend, args = map_model_spec("gpt2")
    assert backend == "hf"
    assert args["pretrained"] == "gpt2"


def test_map_vllm_spec():
    backend, args = map_model_spec("vllm:meta-llama/Llama-3.2-1B")
    assert backend == "vllm"
    assert args["pretrained"] == "meta-llama/Llama-3.2-1B"


def test_map_openai_is_chat_backend():
    backend, args = map_model_spec("openai:gpt-4o")
    assert backend == "openai-chat-completions"
    assert args == {"model": "gpt-4o"}


def test_map_local_completions_requires_base_url():
    with pytest.raises(AuditKitError, match="base_url"):
        map_model_spec("api:my-model")
    backend, args = map_model_spec("api:my-model", base_url="https://x/v1/completions")
    assert backend == "local-completions"
    assert args["base_url"] == "https://x/v1/completions"


def test_map_groq_is_chat_backend_with_default_base_url():
    backend, args = map_model_spec("groq:llama-3.3-70b-versatile")
    # Groq rides the OpenAI chat backend so the chat-only/MCQ guard applies to it.
    assert backend == "openai-chat-completions"
    assert args["model"] == "llama-3.3-70b-versatile"
    assert args["base_url"] == "https://api.groq.com/openai/v1/chat/completions"


def test_map_groq_keeps_api_key_out_of_model_args():
    # The key authenticates via OPENAI_API_KEY env at run time, never a model_arg.
    _, args = map_model_spec("groq:mixtral", api_key="gsk_secret")
    assert "api_key" not in args


def test_map_groq_caller_base_url_wins():
    _, args = map_model_spec("groq:mixtral", base_url="https://proxy/v1/chat/completions")
    assert args["base_url"] == "https://proxy/v1/chat/completions"


def test_map_openrouter_is_chat_backend_with_default_base_url():
    backend, args = map_model_spec("openrouter:openai/gpt-4o-mini")
    # Same shape as groq -- rides the OpenAI chat backend.
    assert backend == "openai-chat-completions"
    assert args["model"] == "openai/gpt-4o-mini"
    assert args["base_url"] == "https://openrouter.ai/api/v1/chat/completions"


def test_map_openrouter_keeps_api_key_out_of_model_args():
    _, args = map_model_spec("openrouter:openai/gpt-4o-mini", api_key="or_secret")
    assert "api_key" not in args


def test_map_openrouter_caller_base_url_wins():
    _, args = map_model_spec("openrouter:openai/gpt-4o-mini", base_url="https://proxy/v1/chat/completions")
    assert args["base_url"] == "https://proxy/v1/chat/completions"


def test_map_passthrough_args():
    _, args = map_model_spec("hf:gpt2", dtype="float16", device="cuda")
    assert args["dtype"] == "float16"
    assert args["device"] == "cuda"


def test_map_hf_token_threads_into_model_args():
    _, args = map_model_spec("hf:meta-llama/Llama-3.2-1B", hf_token="hf_secret")
    assert args["token"] == "hf_secret"


def test_map_token_alias_also_works():
    _, args = map_model_spec("hf:gpt2", token="hf_secret")
    assert args["token"] == "hf_secret"


def test_map_rejects_callable():
    with pytest.raises(AuditKitError, match="string model spec"):
        map_model_spec(lambda prompts: prompts)


def test_map_unknown_prefix():
    with pytest.raises(AuditKitError, match="not supported"):
        map_model_spec("weirdbackend:foo")


# --- result mapping -------------------------------------------------------

def _fake_raw():
    return {
        "results": {
            "arc_challenge": {
                "acc,none": 0.5,
                "acc_stderr,none": 0.05,
                "acc_norm,none": 0.75,
                "alias": "arc_challenge",
            }
        },
        "versions": {"arc_challenge": 1.0},
        "samples": {
            "arc_challenge": [
                {"doc_id": 0, "arguments": [["Q1: 2+2?", " 4"]], "resps": [["4"]],
                 "filtered_resps": ["4"], "target": "4", "acc": 1.0, "acc_norm": 1.0},
                {"doc_id": 1, "arguments": [["Q2: capital?", " Paris"]], "resps": [["London"]],
                 "filtered_resps": ["London"], "target": "Paris", "acc": 0.0, "acc_norm": 0.0},
            ]
        },
    }


def _fp(model_spec="hf:gpt2", tasks=("arc_challenge",)):
    return _fingerprint({"tasks": list(tasks), "model": "hf", "model_args": {"pretrained": "gpt2"}}, model_spec)


def test_to_runresult_headline_uses_lmeval_aggregates():
    r = _to_runresult(_fake_raw(), ["arc_challenge"], "hf:gpt2", RunConfig(), "", _fp())
    assert r.headline["arc_challenge:acc"] == 0.5
    assert r.headline["arc_challenge:acc_norm"] == 0.75
    assert not any("stderr" in k for k in r.headline)


def test_to_runresult_predictions_are_the_answer_browser():
    r = _to_runresult(_fake_raw(), ["arc_challenge"], "hf:gpt2", RunConfig(), "", _fp())
    assert len(r.predictions) == 2
    p0, p1 = r.predictions
    assert p0.prompt == "Q1: 2+2?"
    assert p0.expected == "4"
    assert p0.raw_output == "4"
    assert p0.correct is True
    assert p1.correct is False
    assert [p.sample_id for p in r.wrong_only()] == ["1"]


def test_to_runresult_stats_from_real_persample_values():
    r = _to_runresult(_fake_raw(), ["arc_challenge"], "hf:gpt2", RunConfig(), "", _fp())
    st = r.stats["arc_challenge:acc"]
    assert st.count == 2
    assert st.mean == 0.5


def test_fingerprint_is_stable_and_pins_identity():
    a = _fingerprint({"tasks": ["arc_challenge"], "model": "hf"}, "hf:gpt2")
    b = _fingerprint({"tasks": ["arc_challenge"], "model": "hf"}, "hf:gpt2")
    c = _fingerprint({"tasks": ["arc_challenge"], "model": "hf"}, "hf:gpt-neo")
    assert a == b
    assert a != c


def test_run_benchmark_caches_to_disk_and_skips_a_second_lm_eval_call(fake_lm_eval):
    r1 = run_benchmark("arc_challenge", "hf:gpt2", config=RunConfig(num_fewshot=5, limit=100))
    call_count_after_first = len(fake_lm_eval)
    fake_lm_eval.clear()
    r2 = run_benchmark("arc_challenge", "hf:gpt2", config=RunConfig(num_fewshot=5, limit=100))
    assert r1.fingerprint == r2.fingerprint
    assert r1.headline == r2.headline
    # second call hit the cache -- simple_evaluate was never invoked again
    assert fake_lm_eval == {}
    assert call_count_after_first > 0


def test_run_benchmark_different_config_is_not_cache_confused(fake_lm_eval):
    r1 = run_benchmark("arc_challenge", "hf:gpt2", config=RunConfig(num_fewshot=5))
    r2 = run_benchmark("arc_challenge", "hf:gpt2", config=RunConfig(num_fewshot=10))
    assert r1.fingerprint != r2.fingerprint


# --- run_benchmark (with lm_eval mocked) ----------------------------------

@pytest.fixture
def fake_lm_eval(monkeypatch):
    mod = types.ModuleType("lm_eval")
    mod.__version__ = "0.4.0"
    captured = {}

    def simple_evaluate(**kwargs):
        captured.update(kwargs)
        return _fake_raw()

    mod.simple_evaluate = simple_evaluate
    monkeypatch.setitem(sys.modules, "lm_eval", mod)
    return captured


def test_run_benchmark_end_to_end_mocked(fake_lm_eval):
    r = run_benchmark("arc_challenge", "hf:gpt2", config=RunConfig(num_fewshot=5, limit=100))
    assert r.headline["arc_challenge:acc"] == 0.5
    assert len(r.predictions) == 2
    assert fake_lm_eval["num_fewshot"] == 5
    assert fake_lm_eval["limit"] == 100
    assert fake_lm_eval["log_samples"] is True
    assert fake_lm_eval["model"] == "hf"
    assert fake_lm_eval["model_args"] == {"pretrained": "gpt2"}


def test_run_benchmark_comma_separated_tasks(fake_lm_eval):
    run_benchmark("arc_challenge, gsm8k", "hf:gpt2")
    assert fake_lm_eval["tasks"] == ["arc_challenge", "gsm8k"]


def test_run_benchmark_hf_token_reaches_model_args(fake_lm_eval):
    run_benchmark("arc_challenge", "hf:meta-llama/Llama-3.2-1B", hf_token="hf_secret")
    assert fake_lm_eval["model_args"]["token"] == "hf_secret"


def test_run_knobs_forwarded(fake_lm_eval):
    run_benchmark("gsm8k", "hf:gpt2", apply_chat_template=True,
                  gen_kwargs="temperature=0,max_gen_toks=256",
                  system_instruction="Be concise.")
    assert fake_lm_eval["apply_chat_template"] is True
    assert fake_lm_eval["gen_kwargs"] == "temperature=0,max_gen_toks=256"
    assert fake_lm_eval["system_instruction"] == "Be concise."


def test_run_knobs_omitted_by_default(fake_lm_eval):
    run_benchmark("arc_challenge", "hf:gpt2")
    # unset knobs must NOT be sent, so lm-eval keeps its own defaults
    for knob in ("apply_chat_template", "gen_kwargs", "system_instruction",
                 "fewshot_as_multiturn"):
        assert knob not in fake_lm_eval


def test_lmeval_kwargs_full_parity_passthrough(fake_lm_eval):
    run_benchmark("arc_challenge", "hf:gpt2",
                  lmeval_kwargs={"write_out": True, "max_batch_size": 8})
    assert fake_lm_eval["write_out"] is True
    assert fake_lm_eval["max_batch_size"] == 8


def test_model_args_dict_merges_for_quantized(fake_lm_eval):
    run_benchmark("arc_challenge", "hf:org/quantized",
                  model_args={"load_in_4bit": True, "dtype": "float16"})
    ma = fake_lm_eval["model_args"]
    assert ma["pretrained"] == "org/quantized"
    assert ma["load_in_4bit"] is True
    assert ma["dtype"] == "float16"


def test_run_lmeval_public_api_forwards_run_knobs(fake_lm_eval):
    ak.run_lmeval("gsm8k", model="hf:gpt2", apply_chat_template=True)
    assert fake_lm_eval["apply_chat_template"] is True


def test_hf_token_sets_and_restores_env(monkeypatch):
    import os
    monkeypatch.delenv("HF_TOKEN", raising=False)
    seen = {}
    mod = types.ModuleType("lm_eval")
    mod.__version__ = "0.4.0"

    def simple_evaluate(**kwargs):
        seen["hf_token_during_run"] = os.environ.get("HF_TOKEN")
        return _fake_raw()

    mod.simple_evaluate = simple_evaluate
    monkeypatch.setitem(sys.modules, "lm_eval", mod)

    run_benchmark("arc_challenge", "hf:gpt2", hf_token="hf_secret")
    assert seen["hf_token_during_run"] == "hf_secret"   # set for the call
    assert "HF_TOKEN" not in os.environ                 # restored (was absent) after


def test_run_benchmark_missing_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "lm_eval", None)  # force ImportError
    with pytest.raises(ExtraNotInstalled):
        run_benchmark("arc_challenge", "hf:gpt2")


def test_public_api_exports_run_lmeval(fake_lm_eval):
    r = ak.run_lmeval("arc_challenge", model="hf:gpt2", num_fewshot=3)
    assert r.headline["arc_challenge:acc"] == 0.5
    assert fake_lm_eval["num_fewshot"] == 3


def test_evaluate_engine_lmeval_routes(fake_lm_eval):
    r = ak.evaluate("arc_challenge", model="hf:gpt2", engine="lmeval")
    assert r.headline["arc_challenge:acc"] == 0.5


def test_evaluate_unknown_engine():
    with pytest.raises(ValueError, match="unknown engine"):
        ak.evaluate([ak.Sample(input="x", target="x")], model="echo", engine="bogus")


# --- chat-only + MCQ guard ------------------------------------------------

def test_chat_only_mcq_raises_capability_error(monkeypatch):
    mod = types.ModuleType("lm_eval")
    mod.__version__ = "0.4.0"
    mod.simple_evaluate = lambda **kw: _fake_raw()
    monkeypatch.setitem(sys.modules, "lm_eval", mod)
    monkeypatch.setattr(
        "auditkit.lmeval_engine._task_output_types",
        lambda tasks: {"arc_challenge": "multiple_choice"},
    )
    with pytest.raises(CapabilityError, match="chat-only"):
        run_benchmark("arc_challenge", "openai:gpt-4o")


def test_chat_only_generative_task_is_allowed(fake_lm_eval, monkeypatch):
    monkeypatch.setattr(
        "auditkit.lmeval_engine._task_output_types",
        lambda tasks: {"gsm8k": "generate_until"},
    )
    r = run_benchmark("gsm8k", "openai:gpt-4o")  # generative → fine on chat models
    assert r is not None


# --- groq on the benchmark engine -----------------------------------------

def test_run_benchmark_groq_mirrors_key_into_openai_env_and_restores(monkeypatch):
    """Groq's key is sourced from GROQ_API_KEY and exposed to lm-eval's
    openai-chat-completions backend as OPENAI_API_KEY for the run only."""
    import os
    monkeypatch.setenv("GROQ_API_KEY", "gsk_secret")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "auditkit.lmeval_engine._task_output_types",
        lambda tasks: {"gsm8k": "generate_until"},
    )
    seen = {}
    mod = types.ModuleType("lm_eval")
    mod.__version__ = "0.4.0"

    def simple_evaluate(**kwargs):
        seen["openai_key_during_run"] = os.environ.get("OPENAI_API_KEY")
        seen["model_args"] = kwargs["model_args"]
        return _fake_raw()

    mod.simple_evaluate = simple_evaluate
    monkeypatch.setitem(sys.modules, "lm_eval", mod)

    run_benchmark("gsm8k", "groq:llama-3.3-70b-versatile")
    assert seen["openai_key_during_run"] == "gsk_secret"     # mirrored for the call
    assert "api_key" not in seen["model_args"]               # never a model_arg
    assert "OPENAI_API_KEY" not in os.environ                # restored (was absent) after


def test_run_benchmark_groq_explicit_api_key_beats_env(monkeypatch):
    import os
    monkeypatch.setenv("GROQ_API_KEY", "from_env")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "auditkit.lmeval_engine._task_output_types",
        lambda tasks: {"gsm8k": "generate_until"},
    )
    seen = {}
    mod = types.ModuleType("lm_eval")
    mod.__version__ = "0.4.0"
    mod.simple_evaluate = lambda **kw: (seen.update(key=os.environ.get("OPENAI_API_KEY")), _fake_raw())[1]
    monkeypatch.setitem(sys.modules, "lm_eval", mod)

    run_benchmark("gsm8k", "groq:mixtral", api_key="from_arg")
    assert seen["key"] == "from_arg"


def test_run_benchmark_groq_missing_key_raises(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(
        "auditkit.lmeval_engine._task_output_types",
        lambda tasks: {"gsm8k": "generate_until"},
    )
    mod = types.ModuleType("lm_eval")
    mod.__version__ = "0.4.0"
    mod.simple_evaluate = lambda **kw: _fake_raw()
    monkeypatch.setitem(sys.modules, "lm_eval", mod)

    with pytest.raises(AuditKitError, match="Groq API key"):
        run_benchmark("gsm8k", "groq:mixtral")


def test_run_benchmark_groq_mcq_raises_capability_error(monkeypatch):
    """The chat-only guard applies to groq exactly as it does to openai:."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_secret")
    mod = types.ModuleType("lm_eval")
    mod.__version__ = "0.4.0"
    mod.simple_evaluate = lambda **kw: _fake_raw()
    monkeypatch.setitem(sys.modules, "lm_eval", mod)
    monkeypatch.setattr(
        "auditkit.lmeval_engine._task_output_types",
        lambda tasks: {"arc_challenge": "multiple_choice"},
    )
    with pytest.raises(CapabilityError, match="chat-only"):
        run_benchmark("arc_challenge", "groq:llama-3.3-70b-versatile")


# --- openrouter on the benchmark engine -------------------------------------

def test_run_benchmark_openrouter_mirrors_key_into_openai_env_and_restores(monkeypatch):
    """OpenRouter's key is sourced from OPENROUTER_API_KEY and exposed to
    lm-eval's openai-chat-completions backend as OPENAI_API_KEY for the run only."""
    import os
    monkeypatch.setenv("OPENROUTER_API_KEY", "or_secret")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "auditkit.lmeval_engine._task_output_types",
        lambda tasks: {"gsm8k": "generate_until"},
    )
    seen = {}
    mod = types.ModuleType("lm_eval")
    mod.__version__ = "0.4.0"

    def simple_evaluate(**kwargs):
        seen["openai_key_during_run"] = os.environ.get("OPENAI_API_KEY")
        seen["model_args"] = kwargs["model_args"]
        return _fake_raw()

    mod.simple_evaluate = simple_evaluate
    monkeypatch.setitem(sys.modules, "lm_eval", mod)

    run_benchmark("gsm8k", "openrouter:openai/gpt-4o-mini")
    assert seen["openai_key_during_run"] == "or_secret"       # mirrored for the call
    assert "api_key" not in seen["model_args"]                # never a model_arg
    assert "OPENAI_API_KEY" not in os.environ                 # restored (was absent) after


def test_run_benchmark_openrouter_explicit_api_key_beats_env(monkeypatch):
    import os
    monkeypatch.setenv("OPENROUTER_API_KEY", "from_env")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "auditkit.lmeval_engine._task_output_types",
        lambda tasks: {"gsm8k": "generate_until"},
    )
    seen = {}
    mod = types.ModuleType("lm_eval")
    mod.__version__ = "0.4.0"
    mod.simple_evaluate = lambda **kw: (seen.update(key=os.environ.get("OPENAI_API_KEY")), _fake_raw())[1]
    monkeypatch.setitem(sys.modules, "lm_eval", mod)

    run_benchmark("gsm8k", "openrouter:openai/gpt-4o-mini", api_key="from_arg")
    assert seen["key"] == "from_arg"


def test_run_benchmark_openrouter_missing_key_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        "auditkit.lmeval_engine._task_output_types",
        lambda tasks: {"gsm8k": "generate_until"},
    )
    mod = types.ModuleType("lm_eval")
    mod.__version__ = "0.4.0"
    mod.simple_evaluate = lambda **kw: _fake_raw()
    monkeypatch.setitem(sys.modules, "lm_eval", mod)

    with pytest.raises(AuditKitError, match="OpenRouter API key"):
        run_benchmark("gsm8k", "openrouter:openai/gpt-4o-mini")


def test_run_benchmark_openrouter_mcq_raises_capability_error(monkeypatch):
    """The chat-only guard applies to openrouter exactly as it does to groq/openai:."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "or_secret")
    mod = types.ModuleType("lm_eval")
    mod.__version__ = "0.4.0"
    mod.simple_evaluate = lambda **kw: _fake_raw()
    monkeypatch.setitem(sys.modules, "lm_eval", mod)
    monkeypatch.setattr(
        "auditkit.lmeval_engine._task_output_types",
        lambda tasks: {"arc_challenge": "multiple_choice"},
    )
    with pytest.raises(CapabilityError, match="chat-only"):
        run_benchmark("arc_challenge", "openrouter:openai/gpt-4o-mini")
