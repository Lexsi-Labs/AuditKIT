"""Regression tests for V1-V3 of docs/notes/open-bugs-fix-plan.md: the offline vllm: engine.

vLLM needs Linux + CUDA, so the engine is a stand-in that records what reaches `LLM.generate`. The tokenizer
and chat templates are real (cached HF tokenizers plus the repo's template fixtures), so tool rendering and
BOS handling are exercised for real.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import auditkit as ak
from auditkit.errors import CapabilityError
from auditkit.model import Capability, Request
from auditkit.model.vllm_gen import VLLMModel

TEMPLATES = Path(__file__).parent / "fixtures" / "chat_templates"
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]


def tokenizer(name, template=None):
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception:
        pytest.skip(f"{name} tokenizer not cached")
    if template:
        tok.chat_template = (TEMPLATES / template).read_text()
    return tok


class FakeLLM:
    """Records every LLM.generate call; answers each prompt with ``reply``."""

    def __init__(self, tok, reply="ok"):
        self.tok, self.reply, self.calls = tok, reply, []

    def get_tokenizer(self):
        return self.tok

    def generate(self, prompts, params, **kw):
        self.calls.append((list(prompts), params, kw))
        return [SimpleNamespace(outputs=[SimpleNamespace(text=self.reply)]) for _ in prompts]


def vllm_with(tok, reply="ok"):
    m = VLLMModel("stub")
    m._llm = FakeLLM(tok, reply)
    m._sampling_params = lambda **kw: dict(kw)          # SamplingParams stand-in: keeps the kwargs
    return m


# -- V1: native tool calling --------------------------------------------------------------------------------------

def test_v1_vllm_declares_native_tools():
    assert Capability.TOOLS in VLLMModel("stub").capabilities()


def test_v1_the_tool_schemas_reach_the_chat_template():
    m = vllm_with(tokenizer("Qwen/Qwen3-1.7B"))
    m.generate([Request(prompt="Weather in Paris?", params={"tools": TOOLS})])
    prompt = m._llm.calls[0][0][0]
    assert "get_weather" in prompt and "<tools>" in prompt


def test_v1_chat_template_kwargs_win_over_tools_on_a_clash():
    m = vllm_with(tokenizer("Qwen/Qwen3-1.7B"))
    m.generate([Request(prompt="q", params={"tools": TOOLS, "chat_template_kwargs": {"enable_thinking": False}})])
    assert "<think>\n\n</think>" in m._llm.calls[0][0][0]           # Qwen3's thinking-off marker, with tools rendered


def test_v1_a_template_without_tools_is_refused_with_advice():
    m = vllm_with(tokenizer("gpt2", "tiny_aya_default.jinja"))
    with pytest.raises(CapabilityError, match="prompt"):
        m.generate([Request(prompt="q", params={"tools": TOOLS})])


def test_v1_no_template_at_all_never_drops_the_tools():
    tok = tokenizer("gpt2")
    tok.chat_template = None
    with pytest.raises(CapabilityError, match="prompt"):
        vllm_with(tok).generate([Request(prompt="q", params={"tools": TOOLS})])


def named_templates(**templates):
    """A tokenizer carrying named templates the way transformers loads every Cohere model's
    ``chat_template`` list: as a ``{name: template}`` dict, not a str."""
    tok = tokenizer("gpt2")
    tok.chat_template = {name: (TEMPLATES / f).read_text() for name, f in templates.items()}
    return tok


def test_named_templates_render_the_chat_template():
    # Seen live on Colab (vLLM 0.30, Tiny Aya): the dict was taken for "no template", so the raw prompt
    # went out and the model continued it ("Paris.\nAnswer: Paris") instead of answering.
    m = vllm_with(named_templates(default="tiny_aya_default.jinja"))
    m.generate([Request(prompt="Capital of France?")])
    assert "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>Capital of France?" in m._llm.calls[0][0][0]


def test_named_templates_without_a_tool_slot_are_refused_for_the_right_reason():
    m = vllm_with(named_templates(default="tiny_aya_default.jinja"))
    with pytest.raises(CapabilityError, match="does not render tool schemas"):
        m.generate([Request(prompt="q", params={"tools": TOOLS})])


def test_named_templates_with_tool_use_render_native_tools():
    m = vllm_with(named_templates(default="tiny_aya_default.jinja", tool_use="command_r7b_tool_use.jinja"))
    m.generate([Request(prompt="Weather in Paris?", params={"tools": TOOLS})])
    assert "get_weather" in m._llm.calls[0][0][0]


def test_v1_native_tool_calls_are_scored_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    m = vllm_with(tokenizer("Qwen/Qwen3-1.7B"),
                  reply='<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>')
    r = ak.evaluate([ak.Sample(input="Weather in Paris?", tools=TOOLS,
                               expected_tool_calls=[[{"name": "get_weather", "arguments": {"city": "Paris"}}]])],
                    model=m, adapter=ak.ToolCallAdapter(), scorers=[ak.ToolCallF1()])
    assert r.headline["tool_call_f1"] == 1.0 and not r.errors


def test_v1_the_harness_can_use_a_vllm_policy():
    import json
    from auditkit.agent_eval import AgentCase, AgentEvalRunner, AgentEvalSpec, FinalStateAssertion
    m = vllm_with(tokenizer("Qwen/Qwen3-1.7B"))
    m._llm.reply = None

    def generate(prompts, params, **kw):
        m._llm.calls.append((list(prompts), params, kw))
        text = "Done." if "<tool_response>" in prompts[0] else \
            '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
        return [SimpleNamespace(outputs=[SimpleNamespace(text=text)])]
    m._llm.generate = generate

    class Env:
        state = {}

        def __call__(self, name, args):
            self.state["asked"] = args["city"]
            return json.dumps({"temp_c": 18})

        def snapshot(self):
            return dict(self.state)

    case = AgentCase(id="w", task="Weather in Paris?", allowed_tools=TOOLS, outcome=FinalStateAssertion("asked", equals="Paris"))
    res = AgentEvalRunner().run(AgentEvalSpec(cases=[case], mode="harness", agent=m, agent_opts={"tool_env": Env(), "max_steps": 3}))
    assert res.rows[0].outcome["verdict"] == "success"
    assert "get_weather" in m._llm.calls[0][0][0]                  # the schemas reached the policy's prompt


# -- V2: one BOS per prompt in a mixed batch -------------------------------------------------------------------------

def test_v2_a_mixed_batch_gives_every_prompt_one_bos():
    tok = tokenizer("NousResearch/Llama-2-7b-hf", "aya_expanse.jinja")
    m = vllm_with(tok)
    out = m.generate([Request(prompt="raw", params={"apply_chat_template": False}), Request(prompt="chat"),
                      Request(prompt="raw2", params={"apply_chat_template": False})])
    assert len(out) == 3 and len(m._llm.calls) == 2                 # templated and raw prompts in separate calls
    counts = []
    for prompts, _, kw in m._llm.calls:
        add = kw.get("tokenization_kwargs", {}).get("add_special_tokens", True)
        counts += [tok(p, add_special_tokens=add)["input_ids"].count(tok.bos_token_id) for p in prompts]
    assert counts == [1, 1, 1]


def test_v2_results_come_back_in_request_order():
    m = vllm_with(tokenizer("NousResearch/Llama-2-7b-hf", "aya_expanse.jinja"))
    m._llm.generate = lambda prompts, params, **kw: [SimpleNamespace(outputs=[SimpleNamespace(text=p[-8:])]) for p in prompts]
    out = m.generate([Request(prompt="first", params={"apply_chat_template": False}), Request(prompt="second"),
                      Request(prompt="third", params={"apply_chat_template": False})])
    texts = [r.completions[0].text for r in out]
    assert texts[0].endswith("first") and texts[2].endswith("third") and "second" not in texts[0] + texts[2]


# -- V3: per-request sampling parameters ------------------------------------------------------------------------------

def test_v3_each_request_gets_its_own_sampling_params():
    m = vllm_with(tokenizer("Qwen/Qwen3-1.7B"))
    m.generate([Request(prompt="a", params={"max_tokens": 16, "temperature": 0.0}),
                Request(prompt="b", params={"max_tokens": 512, "temperature": 0.7, "stop_sequences": ["\n"]})])
    [(prompts, params, _)] = m._llm.calls
    assert isinstance(params, list) and len(params) == len(prompts) == 2
    assert (params[0]["max_tokens"], params[1]["max_tokens"]) == (16, 512)
    assert params[1]["temperature"] == 0.7 and params[1]["stop"] == ["\n"] and "stop" not in params[0]


# -- V5: unload frees the engine through vLLM's own teardown ----------------------------------------------------
# The chain mirrors vLLM 0.30's in-process engine (source-checked): LLM -> LLMEngine -> InprocClient
# -> EngineCore -> UniProcExecutor -> WorkerWrapperBase -> Worker -> GPUModelRunnerV2. Its runner
# shutdown starts with kv_caches.clear(), and Worker.shutdown releases the cumem pools after it.

class Runner030:
    def __init__(self, log):
        self.log, self.model, self.kv_caches = log, SimpleNamespace(to=lambda d: log.append(f"model.to({d})")), [1]

    def shutdown(self):
        self.kv_caches.clear()                  # raises if someone set kv_caches = None first
        del self.model
        self.log.append("runner.shutdown")


class Worker030:
    def __init__(self, log):
        self.log, self.model_runner = log, Runner030(log)

    def shutdown(self):
        self.model_runner.shutdown()
        self.log.append("release_pools")        # skipped when the runner's shutdown raises


def engine_030(log):
    worker = SimpleNamespace(worker=Worker030(log))
    worker.shutdown = lambda: worker.worker.shutdown()
    executor = SimpleNamespace(driver_worker=worker, shutdown=lambda: worker.shutdown())
    core = SimpleNamespace(model_executor=executor, scheduler=object(), shutdown=lambda: executor.shutdown())
    client = SimpleNamespace(engine_core=core, shutdown=lambda: core.shutdown())
    llm_engine = SimpleNamespace(engine_core=client, model_executor=executor)
    return SimpleNamespace(llm_engine=llm_engine, sleep=lambda level: log.append(f"sleep({level})")), worker.worker


def test_v5_unload_runs_vllms_own_teardown_to_the_pool_release():
    log = []
    llm, worker = engine_030(log)
    m = VLLMModel("stub")
    m._llm = llm
    m.unload()
    assert log == ["runner.shutdown", "release_pools"]      # nothing dropped first, no legacy sleep
    assert not hasattr(worker.model_runner, "model") and m._llm is None


def test_v5_legacy_vllm_whose_shutdown_is_a_no_op_falls_back_to_sleep_and_drop():
    log = []
    llm, worker = engine_030(log)
    runner = worker.model_runner
    worker.shutdown = lambda: None                          # vLLM 0.19: WorkerBase.shutdown is a no-op
    llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.shutdown = lambda: worker.shutdown()
    m = VLLMModel("stub")
    m._llm = llm
    m.unload()
    assert log == ["sleep(2)", "model.to(cpu)"] and runner.model is None and runner.kv_caches is None


def test_v5_a_failing_teardown_step_is_logged_not_swallowed(caplog):
    import logging
    log = []
    llm, worker = engine_030(log)
    worker.model_runner.kv_caches = None                    # what the old order did before calling shutdown
    m = VLLMModel("stub")
    m._llm = llm
    with caplog.at_level(logging.WARNING, logger="auditkit.model.vllm_gen"):
        m.unload()
    assert any("shutdown() failed" in r.message for r in caplog.records)


# -- an architecture vLLM refuses says what to try --------------------------------------------------------------

REMOVED = ("Model architecture AyaVisionForConditionalGeneration was supported in vLLM until v0.24.0, and is not "
           "supported anymore. Please use an older version of vLLM if you want to use this model architecture.")


def fake_vllm(monkeypatch, error):
    calls = []

    def LLM(model, **kw):
        calls.append(kw)
        if error and "model_impl" not in kw:
            raise ValueError(error)
        return SimpleNamespace(get_tokenizer=lambda: None)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=LLM, SamplingParams=dict))
    return calls


def test_a_removed_architecture_points_at_the_transformers_backend(monkeypatch):
    fake_vllm(monkeypatch, REMOVED)
    with pytest.raises(ValueError, match="model_impl='transformers'") as e:
        VLLMModel("CohereLabs/aya-vision-8b")._ensure_llm()
    assert "v0.24.0" in str(e.value)                  # vLLM's own reason is kept


def test_with_model_impl_set_the_engine_is_built(monkeypatch):
    calls = fake_vllm(monkeypatch, REMOVED)
    m = VLLMModel("CohereLabs/aya-vision-8b", model_impl="transformers")
    m._ensure_llm()
    assert calls == [{"model_impl": "transformers"}] and m._llm is not None


def test_other_value_errors_are_untouched(monkeypatch):
    fake_vllm(monkeypatch, "max_model_len 131072 exceeds the model's maximum")
    with pytest.raises(ValueError) as e:
        VLLMModel("x")._ensure_llm()
    assert "model_impl" not in str(e.value)
