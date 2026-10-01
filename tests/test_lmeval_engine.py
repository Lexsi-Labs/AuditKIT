"""Tests for the lm-eval benchmark engine (lmeval_engine.py).

lm-eval is a heavy optional dep, so these tests inject a fake ``lm_eval`` module
into ``sys.modules`` and exercise the spec-mapping and result-mapping logic
without installing the harness. A real end-to-end run needs
``pip install auditkit[lmeval]``.
"""

from __future__ import annotations

import os
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


def test_map_lexsi_is_local_completions_requires_base_url():
    # lexsi: rides the same logprob-capable backend as api: -- real
    # loglikelihood/MC tasks work on it, unlike openai:/groq:/openrouter:.
    with pytest.raises(AuditKitError, match="base_url"):
        map_model_spec("lexsi:my-model")
    backend, args = map_model_spec("lexsi:my-model", base_url="https://gw/v1/completions")
    assert backend == "local-completions"
    assert args["base_url"] == "https://gw/v1/completions"


# --- lexsi: auto tokenizer inference ---------------------------------------

def test_map_lexsi_infers_tokenizer_from_versioned_model_name():
    _, args = map_model_spec(
        "lexsi:Qwen-Qwen3-0.6B_v1", base_url="https://gw/v1/completions",
    )
    assert args["tokenizer"] == "Qwen/Qwen3-0.6B"


def test_map_lexsi_explicit_tokenizer_wins_over_inference():
    _, args = map_model_spec(
        "lexsi:Qwen-Qwen3-0.6B_v1", base_url="https://gw/v1/completions",
        tokenizer="something/else",
    )
    assert args["tokenizer"] == "something/else"


def test_map_lexsi_no_inference_when_name_has_no_hyphen():
    _, args = map_model_spec("lexsi:justaname", base_url="https://gw/v1/completions")
    assert "tokenizer" not in args


def test_map_api_prefix_does_not_infer_tokenizer():
    # The auto-tokenizer guess is lexsi-specific -- api: model names don't
    # necessarily follow the same org-repo_vN convention.
    _, args = map_model_spec("api:Qwen-Qwen3-0.6B_v1", base_url="https://gw/v1/completions")
    assert "tokenizer" not in args


class TestInferHfRepo:
    def test_strips_version_suffix_and_splits_on_first_hyphen(self):
        from auditkit.lmeval_engine import _infer_hf_repo
        assert _infer_hf_repo("Qwen-Qwen3-0.6B_v1") == "Qwen/Qwen3-0.6B"

    def test_works_without_a_version_suffix(self):
        from auditkit.lmeval_engine import _infer_hf_repo
        assert _infer_hf_repo("meta-llama-Llama-3.2-1B") == "meta/llama-Llama-3.2-1B"

    def test_no_hyphen_returns_none(self):
        from auditkit.lmeval_engine import _infer_hf_repo
        assert _infer_hf_repo("justaname_v2") is None

    def test_only_strips_a_trailing_version_suffix(self):
        from auditkit.lmeval_engine import _infer_hf_repo
        # "_v" not at the very end (e.g. mid-name) must not be stripped.
        assert _infer_hf_repo("Qwen-Qwen3_v0.6B") == "Qwen/Qwen3_v0.6B"


# --- gateway fields (project_name/provider/client_id) auto-fill into
# gen_kwargs (run_benchmark) -- not prefix-gated: any backend pointed at a
# Lexsi gateway via base_url= can carry these. -----------------------------

def test_run_benchmark_folds_gateway_fields_into_gen_kwargs_openai_prefix(fake_lm_eval):
    run_benchmark(
        "gsm8k", "openai:my-model",
        base_url="https://gw/v1/chat/completions",
        project_name="Evals_Benchmark_A", provider="Lexsi", client_id="me@lexsi.ai",
    )
    assert fake_lm_eval["gen_kwargs"] == {
        "project_name": "Evals_Benchmark_A", "provider": "Lexsi", "client_id": "me@lexsi.ai",
    }


def test_run_benchmark_folds_gateway_fields_into_gen_kwargs_api_prefix(fake_lm_eval):
    run_benchmark(
        "arc_easy", "api:my-model",
        base_url="https://gw/v1/completions",
        project_name="Evals_Benchmark_A", provider="Lexsi", client_id="me@lexsi.ai",
    )
    assert fake_lm_eval["gen_kwargs"] == {
        "project_name": "Evals_Benchmark_A", "provider": "Lexsi", "client_id": "me@lexsi.ai",
    }


def test_run_benchmark_caller_gen_kwargs_values_win_on_conflict(fake_lm_eval):
    run_benchmark(
        "arc_easy", "lexsi:my-model",
        base_url="https://gw/v1/completions",
        project_name="from_opts", provider="Lexsi", client_id="me@lexsi.ai",
        gen_kwargs={"project_name": "from_gen_kwargs", "temperature": 0},
    )
    assert fake_lm_eval["gen_kwargs"] == {
        "project_name": "from_gen_kwargs", "provider": "Lexsi",
        "client_id": "me@lexsi.ai", "temperature": 0,
    }


def test_run_benchmark_without_gateway_fields_leaves_gen_kwargs_untouched(fake_lm_eval):
    run_benchmark("arc_easy", "lexsi:my-model", base_url="https://gw/v1/completions")
    assert "gen_kwargs" not in fake_lm_eval


def test_run_benchmark_string_gen_kwargs_conflict_raises():
    with pytest.raises(AuditKitError, match="gen_kwargs"):
        run_benchmark(
            "arc_easy", "lexsi:my-model",
            base_url="https://gw/v1/completions",
            project_name="p", gen_kwargs="temperature=0",
        )


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


# --- relay=True (routes through LexsiCompletionsRelay, see lexsi_relay.py) --

def _relay_simple_evaluate(monkeypatch, captured):
    """Installs a fake lm_eval whose simple_evaluate() makes a REAL POST to
    whatever base_url it was given -- so these tests prove relay=True
    actually repoints model_args["base_url"] at a live, field-injecting
    relay, not just that some URL string was substituted."""
    import requests

    def simple_evaluate(**kwargs):
        captured["model_args"] = kwargs["model_args"]
        resp = requests.post(
            kwargs["model_args"]["base_url"], json={"prompt": "hi"}, timeout=10,
        )
        captured["relay_response_status"] = resp.status_code
        return _fake_raw()

    mod = types.ModuleType("lm_eval")
    mod.__version__ = "0.4.0"
    mod.simple_evaluate = simple_evaluate
    monkeypatch.setitem(sys.modules, "lm_eval", mod)


def test_run_benchmark_relay_true_forwards_through_relay_and_injects_fields(monkeypatch, fake_upstream):
    target_url, received = fake_upstream
    captured = {}
    _relay_simple_evaluate(monkeypatch, captured)

    run_benchmark(
        "arc_easy", "lexsi:my-model",
        base_url=target_url, project_name="proj", provider="Lexsi",
        client_id="me@x.ai", api_key="tok",
        relay=True,
    )

    assert captured["relay_response_status"] == 200
    # simple_evaluate was pointed at a LOCAL relay URL, not the real target.
    assert captured["model_args"]["base_url"] != target_url
    assert captured["model_args"]["base_url"].startswith("http://127.0.0.1:")
    # ... which genuinely injected the gateway fields before forwarding.
    assert received["json"]["project_name"] == "proj"
    assert received["json"]["provider"] == "Lexsi"
    assert received["json"]["client_id"] == "me@x.ai"
    assert received["auth"] == "Bearer tok"


def test_run_benchmark_relay_true_stops_relay_after_call(monkeypatch, fake_upstream):
    import requests

    target_url, _ = fake_upstream
    captured = {}
    _relay_simple_evaluate(monkeypatch, captured)

    run_benchmark(
        "arc_easy", "lexsi:my-model",
        base_url=target_url, project_name="p", provider="Lexsi",
        client_id="c", api_key="tok",
        relay=True,
    )

    relay_url = captured["model_args"]["base_url"]
    with pytest.raises(requests.exceptions.ConnectionError):
        requests.post(relay_url, json={}, timeout=2)


def test_run_benchmark_relay_true_stops_relay_even_on_exception(monkeypatch, fake_upstream):
    import requests

    target_url, _ = fake_upstream
    captured = {}

    def simple_evaluate(**kwargs):
        captured["model_args"] = kwargs["model_args"]
        raise RuntimeError("boom")

    mod = types.ModuleType("lm_eval")
    mod.__version__ = "0.4.0"
    mod.simple_evaluate = simple_evaluate
    monkeypatch.setitem(sys.modules, "lm_eval", mod)

    with pytest.raises(RuntimeError, match="boom"):
        run_benchmark(
            "arc_easy", "lexsi:my-model",
            base_url=target_url, project_name="p", provider="Lexsi",
            client_id="c", api_key="tok",
            relay=True,
        )

    relay_url = captured["model_args"]["base_url"]
    with pytest.raises(requests.exceptions.ConnectionError):
        requests.post(relay_url, json={}, timeout=2)


def test_run_benchmark_relay_true_missing_gateway_field_raises(fake_upstream):
    target_url, _ = fake_upstream
    with pytest.raises(AuditKitError, match=r"project_name|provider|client_id"):
        run_benchmark(
            "arc_easy", "lexsi:my-model",
            base_url=target_url, project_name="p", provider="Lexsi",
            # client_id missing
            api_key="tok",
            relay=True,
        )


def test_run_benchmark_relay_true_missing_token_raises(fake_upstream):
    target_url, _ = fake_upstream
    with pytest.raises(AuditKitError, match="api_key"):
        run_benchmark(
            "arc_easy", "lexsi:my-model",
            base_url=target_url, project_name="p", provider="Lexsi", client_id="c",
            relay=True,
        )


def test_run_benchmark_relay_true_second_call_hits_cache_despite_ephemeral_port(monkeypatch, fake_upstream):
    """The fingerprint is computed from the real base_url before the relay's
    per-run ephemeral port is substituted in -- a second, identical call
    must hit the disk cache (simple_evaluate called only once), not treat
    the run as new just because a fresh relay would bind a different port."""
    target_url, _ = fake_upstream
    captured = {"calls": 0}

    def simple_evaluate(**kwargs):
        captured["calls"] += 1
        return _fake_raw()

    mod = types.ModuleType("lm_eval")
    mod.__version__ = "0.4.0"
    mod.simple_evaluate = simple_evaluate
    monkeypatch.setitem(sys.modules, "lm_eval", mod)

    kwargs = dict(
        base_url=target_url, project_name="p", provider="Lexsi", client_id="c",
        api_key="tok", relay=True,
    )
    run_benchmark("arc_easy", "lexsi:my-model", **kwargs)
    run_benchmark("arc_easy", "lexsi:my-model", **kwargs)

    assert captured["calls"] == 1


# --- lexsi_login() / run_benchmark(lexsi_org=...) --------------------------

def _install_fake_lexsi_sdk(monkeypatch, token="tok-abc", id_suffix="_ID",
                             username="fake-user", get_raises=False):
    """A fake `lexsi_sdk` module that records what it was called with, so
    lexsi_login()/run_benchmark(lexsi_org=...) can be tested without a real
    lexsi_sdk install or network access.

    Also fakes the `lexsi_sdk.common.xai_uris` submodule that
    `_current_username()` imports, and `api_client.get()`'s response shape
    (`{"current_user": {"username": ...}}`), so the `client_id=` auto-fill
    path gets exercised for real, instead of skipping to its best-effort
    except-Exception fallback. `get_raises=True` simulates that fallback
    path: `api_client.get()` raises, and `client_id` comes back `None`
    without failing the whole login.
    """
    captured: dict = {}

    class FakeProject:
        def __init__(self, name):
            self.project_name = name + id_suffix

    class FakeWorkspace:
        def project(self, name):
            captured["project"] = name
            return FakeProject(name)

    class FakeOrganization:
        def workspace(self, name):
            captured["workspace"] = name
            return FakeWorkspace()

    class FakeApiClient:
        auth_token = token
        base_url = ""

        def get(self, uri):
            captured["get_uri"] = uri
            if get_raises:
                raise RuntimeError("simulated network failure")
            return {"current_user": {"username": username}}

    class FakeLexsi:
        api_client = FakeApiClient()

        def login(self):
            captured["logged_in"] = True
            captured["sdk_access_token_at_login"] = os.environ.get("SDK_ACCESS_TOKEN")

        def organization(self, name):
            captured["org"] = name
            return FakeOrganization()

    fake_module = types.ModuleType("lexsi_sdk")
    fake_module.lexsi = FakeLexsi()
    monkeypatch.setitem(sys.modules, "lexsi_sdk", fake_module)

    fake_common = types.ModuleType("lexsi_sdk.common")
    fake_xai_uris = types.ModuleType("lexsi_sdk.common.xai_uris")
    fake_xai_uris.USER_ORGANIZATION_URI = "v2/organization/user_organization"
    monkeypatch.setitem(sys.modules, "lexsi_sdk.common", fake_common)
    monkeypatch.setitem(sys.modules, "lexsi_sdk.common.xai_uris", fake_xai_uris)

    return captured


class TestLexsiLogin:
    def test_resolves_token_and_project_from_env_token(self, monkeypatch):
        monkeypatch.setenv("SDK_ACCESS_TOKEN", "env-token")
        captured = _install_fake_lexsi_sdk(monkeypatch)

        token, project_id, client_id = ak.lexsi_login("OrgA", "WsA", "ProjA")

        assert token == "tok-abc"
        assert project_id == "ProjA_ID"
        assert client_id == "fake-user"
        assert captured["org"] == "OrgA"
        assert captured["workspace"] == "WsA"
        assert captured["project"] == "ProjA"
        assert captured["sdk_access_token_at_login"] == "env-token"

    def test_client_id_is_none_when_username_lookup_fails(self, monkeypatch):
        monkeypatch.setenv("SDK_ACCESS_TOKEN", "env-token")
        _install_fake_lexsi_sdk(monkeypatch, get_raises=True)

        token, project_id, client_id = ak.lexsi_login("OrgA", "WsA", "ProjA")

        # The username lookup is best-effort. Its failure doesn't fail the
        # whole login, it leaves client_id unresolved instead.
        assert token == "tok-abc"
        assert project_id == "ProjA_ID"
        assert client_id is None

    def test_explicit_token_wins_and_env_is_restored_after(self, monkeypatch):
        monkeypatch.setenv("SDK_ACCESS_TOKEN", "env-token")
        captured = _install_fake_lexsi_sdk(monkeypatch)

        ak.lexsi_login("O", "W", "P", sdk_access_token="explicit-token")

        assert captured["sdk_access_token_at_login"] == "explicit-token"
        # Scoped to the call only -- the ambient env is unchanged afterward.
        assert os.environ["SDK_ACCESS_TOKEN"] == "env-token"

    def test_an_explicit_token_is_not_left_in_an_env_that_had_none(self, monkeypatch):
        monkeypatch.delenv("SDK_ACCESS_TOKEN", raising=False)
        captured = _install_fake_lexsi_sdk(monkeypatch)

        ak.lexsi_login("O", "W", "P", sdk_access_token="explicit-token")

        assert captured["sdk_access_token_at_login"] == "explicit-token"
        assert "SDK_ACCESS_TOKEN" not in os.environ

    def test_the_env_is_restored_when_the_login_fails(self, monkeypatch):
        monkeypatch.setenv("SDK_ACCESS_TOKEN", "env-token")
        _install_fake_lexsi_sdk(monkeypatch)

        def boom(name):
            raise RuntimeError("project lookup failed")
        monkeypatch.setattr(sys.modules["lexsi_sdk"].lexsi, "organization", boom)

        with pytest.raises(RuntimeError, match="project lookup failed"):
            ak.lexsi_login("O", "W", "P", sdk_access_token="explicit-token")
        assert os.environ["SDK_ACCESS_TOKEN"] == "env-token"

    def test_missing_token_raises_clear_error(self, monkeypatch):
        monkeypatch.delenv("SDK_ACCESS_TOKEN", raising=False)
        _install_fake_lexsi_sdk(monkeypatch)

        with pytest.raises(AuditKitError, match="SDK access token"):
            ak.lexsi_login("O", "W", "P")

    def test_missing_extra_raises_clearly(self, monkeypatch):
        monkeypatch.setenv("SDK_ACCESS_TOKEN", "tok")
        monkeypatch.setitem(sys.modules, "lexsi_sdk", None)

        with pytest.raises(ExtraNotInstalled):
            ak.lexsi_login("O", "W", "P")


class TestRunBenchmarkLexsiOrg:
    def test_resolves_api_key_and_project_name_into_the_run(self, monkeypatch, fake_lm_eval):
        monkeypatch.setenv("SDK_ACCESS_TOKEN", "env-token")
        _install_fake_lexsi_sdk(monkeypatch)

        run_benchmark(
            "gsm8k", "openai:my-model", base_url="https://gw/v1/chat/completions",
            lexsi_org="O", lexsi_workspace="W", lexsi_project="P",
            provider="Lexsi", client_id="c",
        )

        assert fake_lm_eval["model_args"]["api_key"] == "tok-abc"
        assert fake_lm_eval["gen_kwargs"] == {
            "project_name": "P_ID", "provider": "Lexsi", "client_id": "c",
        }

    def test_client_id_auto_fills_from_login_when_not_given(self, monkeypatch, fake_lm_eval):
        monkeypatch.setenv("SDK_ACCESS_TOKEN", "env-token")
        _install_fake_lexsi_sdk(monkeypatch)

        run_benchmark(
            "gsm8k", "openai:my-model", base_url="https://gw/v1/chat/completions",
            lexsi_org="O", lexsi_workspace="W", lexsi_project="P",
            provider="Lexsi",
        )

        assert fake_lm_eval["gen_kwargs"]["client_id"] == "fake-user"

    def test_requires_all_three_together(self, fake_lm_eval):
        with pytest.raises(AuditKitError, match="must all be given together"):
            run_benchmark("gsm8k", "openai:my-model", base_url="https://gw/v1/chat/completions",
                           lexsi_org="O")

    def test_explicit_api_key_and_project_name_win_over_login(self, monkeypatch, fake_lm_eval):
        monkeypatch.setenv("SDK_ACCESS_TOKEN", "env-token")
        _install_fake_lexsi_sdk(monkeypatch)

        run_benchmark(
            "gsm8k", "openai:my-model", base_url="https://gw/v1/chat/completions",
            lexsi_org="O", lexsi_workspace="W", lexsi_project="P",
            api_key="explicit-key", project_name="explicit-project",
            provider="Lexsi", client_id="c",
        )

        assert fake_lm_eval["model_args"]["api_key"] == "explicit-key"
        assert fake_lm_eval["gen_kwargs"]["project_name"] == "explicit-project"
