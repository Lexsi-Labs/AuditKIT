"""The hf: backend's agentic path (#9/#10): native tool schemas through the chat
template, the refusal for templates that cannot render tools, a single BOS, image
routing, and full evaluate() runs with ToolCallAdapter in native and prompt mode.

Real tokenizers and chat templates (local HF cache + tests/fixtures/chat_templates);
the generation pipeline is a stub, so no weights are loaded. Live runs are in
test_live.py.
"""

from __future__ import annotations

import json

import pytest

import auditkit as ak
from auditkit.errors import CapabilityError
from auditkit.model import Request
from auditkit.types import Capability

from .conftest import cached_tokenizer, stub_hf

pytest.importorskip("transformers")

TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "weather for a city", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
W = {"name": "get_weather", "arguments": {"city": "Paris"}}
HERMES = '<think>ok</think>\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
R7B = ('<|START_THINKING|>look it up<|END_THINKING|><|START_ACTION|>'
       '[{"tool_call_id": "0", "tool_name": "get_weather", "parameters": {"city": "Paris"}}]<|END_ACTION|>')


def last_call(m):
    prompts, kwargs = m._pipeline.calls[-1]
    return prompts, kwargs


# -- native tools through the chat template ---------------------------------------------------------------------------

def test_hf_declares_tool_support():
    from auditkit.model.hf_gen import HFGenModel
    assert Capability.TOOLS in HFGenModel("x").capabilities()


@pytest.mark.parametrize("tok,template", [("gpt2", "command_r7b_tool_use.jinja"), ("Qwen/Qwen3-1.7B", None)])
def test_native_tools_reach_the_rendered_prompt(tok, template):
    m = stub_hf(cached_tokenizer(tok, template))
    m.generate([Request(prompt="weather in Paris?", params={"tools": TOOLS})])
    prompt = last_call(m)[0][0]
    assert '"get_weather"' in prompt and "weather for a city" in prompt


def test_chat_template_kwargs_win_over_the_tools_on_a_clash():
    m = stub_hf(cached_tokenizer("Qwen/Qwen3-1.7B"))
    other = [{"type": "function", "function": {"name": "override_tool", "parameters": {"type": "object"}}}]
    m.generate([Request(prompt="q", params={"tools": TOOLS, "chat_template_kwargs": {"tools": other}})])
    prompt = last_call(m)[0][0]
    assert "override_tool" in prompt and "get_weather" not in prompt


def test_other_template_kwargs_pass_through():
    m = stub_hf(cached_tokenizer("Qwen/Qwen3-1.7B"))
    m.generate([Request(prompt="q", params={"chat_template_kwargs": {"enable_thinking": False}})])
    assert "<think>\n\n</think>" in last_call(m)[0][0]            # Qwen3's no-thinking marker


@pytest.mark.parametrize("template", ["aya_expanse.jinja", "tiny_aya_default.jinja"])
def test_a_template_that_cannot_render_tools_is_refused_with_advice(template):
    m = stub_hf(cached_tokenizer("gpt2", template))
    with pytest.raises(CapabilityError, match="mode='prompt'"):
        m.generate([Request(prompt="q", params={"tools": TOOLS})])
    assert m._pipeline.calls == []                                  # nothing generated


@pytest.mark.parametrize("params", [{"tools": TOOLS, "apply_chat_template": False}, {"tools": TOOLS}])
def test_tools_without_a_template_to_take_them_are_refused(params):
    m = stub_hf(cached_tokenizer("gpt2"))                            # gpt2 has no chat template
    with pytest.raises(CapabilityError):
        m.generate([Request(prompt="q", params=params)])


def test_no_tools_on_a_tool_less_template_is_fine():
    m = stub_hf(cached_tokenizer("gpt2", "aya_expanse.jinja"))
    m.generate([Request(prompt="hello")])
    assert "hello" in last_call(m)[0][0]


# -- BOS ---------------------------------------------------------------------------------------------------------------

def bos_count(m, prompt, kwargs):
    tok = m._pipeline.tokenizer
    return tok(prompt, add_special_tokens=kwargs["add_special_tokens"])["input_ids"].count(tok.bos_token_id)


def test_a_chat_templated_prompt_gets_exactly_one_bos():
    # a BOS-adding tokenizer (Llama) + a template that writes its own BOS (Aya)
    m = stub_hf(cached_tokenizer("NousResearch/Llama-2-7b-hf", "aya_expanse.jinja"))
    m.generate([Request(prompt="hi")])
    (prompt,), kwargs = last_call(m)
    assert kwargs["add_special_tokens"] is False and bos_count(m, prompt, kwargs) == 1


def test_a_raw_prompt_keeps_its_bos():
    m = stub_hf(cached_tokenizer("NousResearch/Llama-2-7b-hf", "aya_expanse.jinja"))
    m.generate([Request(prompt="hi", params={"apply_chat_template": False})])
    (prompt,), kwargs = last_call(m)
    assert prompt == "hi" and bos_count(m, prompt, kwargs) == 1


def test_a_mixed_batch_gives_every_prompt_one_bos():
    m = stub_hf(cached_tokenizer("NousResearch/Llama-2-7b-hf", "aya_expanse.jinja"))
    out = m.generate([Request(prompt="raw", params={"apply_chat_template": False}), Request(prompt="chat")])
    counts = [bos_count(m, p, kwargs) for prompts, kwargs in m._pipeline.calls for p in prompts]
    assert len(out) == 2 and len(counts) == 2 and counts == [1, 1]   # every prompt, however many calls


# -- images ---------------------------------------------------------------------------------------------------------------

def test_images_on_a_text_only_model_are_refused_not_dropped():
    m = stub_hf(cached_tokenizer("gpt2", "aya_expanse.jinja"))
    m._processor = m._pipeline.tokenizer                             # a text model's "processor" is its tokenizer
    with pytest.raises(ValueError, match="not a vision model"):
        m.generate([Request(prompt="what is this?", params={"images": ["img.png"]})])


def test_text_requests_in_a_batch_with_an_image_request_keep_their_order(monkeypatch):
    m = stub_hf(cached_tokenizer("gpt2", "aya_expanse.jinja"), reply="TEXT")
    monkeypatch.setattr(m, "_generate_with_images", lambda r, kw: "IMAGE")
    out = m.generate([Request(prompt="a"), Request(prompt="b", params={"images": ["i.png"]}), Request(prompt="c")])
    assert [r.completions[0].text for r in out] == ["TEXT", "IMAGE", "TEXT"]
    assert len(last_call(m)[0]) == 2                                 # only the text prompts hit the pipeline


def test_sample_images_ride_on_every_request():
    m = stub_hf(cached_tokenizer("gpt2", "aya_expanse.jinja"))
    seen = []
    m._generate_with_images = lambda r, kw: seen.append(r.params["images"]) or "ok"
    s = ak.Sample(input="describe", images=["a.png", "b.png"])
    ak.evaluate([s], model=m, scorers=["exact_match"])
    assert seen == [["a.png", "b.png"]]


# -- whole runs through evaluate() ---------------------------------------------------------------------------------------

def samples():
    return [ak.Sample(id="call", input="weather in Paris?", tools=TOOLS, expected_tool_calls=[W]),
            ak.Sample(id="none", input="2+2?", tools=TOOLS, expected_tool_calls=[])]


def reply_for(call_text):
    return lambda prompt: call_text if "Paris" in prompt.rsplit("weather for a city", 1)[-1] else "4"


@pytest.mark.parametrize("tok,template,call_text", [
    ("Qwen/Qwen3-1.7B", None, HERMES),                                           # Qwen native, Hermes output
    ("gpt2", "command_r7b_tool_use.jinja", R7B),                                 # Command R7B native
    ("gpt2", "command_r7b_tool_use.jinja", R7B.replace("<|START_THINKING|>", "")  # decoded with special tokens
     .replace("<|END_THINKING|>", "").replace("<|START_ACTION|>", "").replace("<|END_ACTION|>", "")),
])
def test_native_tool_run_end_to_end(tok, template, call_text):
    m = stub_hf(cached_tokenizer(tok, template), reply=reply_for(call_text))
    r = ak.evaluate(samples(), model=m, adapter=ak.ToolCallAdapter(), scorers=[ak.ToolCallF1(), ak.ToolCallValidity()])
    assert r.errors == [] and r.headline["tool_call_f1"] == 1.0 and r.headline["tool_call_validity"] == 1.0
    assert all('"get_weather"' in p for p in last_call(m)[0])                    # schemas went natively


def test_prompt_mode_works_on_a_template_without_tools():
    # Tiny Aya / Aya Expanse: describe the tools in the prompt, read Hermes blocks back
    m = stub_hf(cached_tokenizer("gpt2", "tiny_aya_default.jinja"), reply=reply_for(HERMES))
    r = ak.evaluate(samples(), model=m, adapter=ak.ToolCallAdapter(mode="prompt"), scorers=[ak.ToolCallF1()])
    assert r.errors == [] and r.headline["tool_call_f1"] == 1.0
    assert all("get_weather" in p and "<tool_call>" in p for p in last_call(m)[0])


def test_native_mode_on_a_tool_less_template_stops_the_run_with_the_fix():
    m = stub_hf(cached_tokenizer("gpt2", "aya_expanse.jinja"))
    with pytest.raises(CapabilityError, match="mode='prompt'"):
        ak.evaluate(samples(), model=m, adapter=ak.ToolCallAdapter(), scorers=[ak.ToolCallF1()])


def test_a_multi_turn_conversation_is_rendered_by_the_template():
    m = stub_hf(cached_tokenizer("Qwen/Qwen3-1.7B"), reply="It is sunny in Paris.")
    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "weather in Paris?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "1", "type": "function", "function": {"name": "get_weather", "arguments": json.dumps({"city": "Paris"})}}]},
            {"role": "tool", "tool_call_id": "1", "content": "sunny"}]
    s = ak.Sample(input="weather in Paris?", tools=TOOLS, metadata={"messages": msgs}, expected_tool_calls=[])
    ak.evaluate([s], model=m, adapter=ak.ToolCallAdapter(), scorers=[ak.ToolCallF1()])
    prompt = last_call(m)[0][0]
    assert "SYS" in prompt and "<tool_response>\nsunny\n</tool_response>" in prompt and prompt.count("get_weather") >= 2


# -- the agent_eval harness driving an hf: model --------------------------------------------------------------------------

class _Ship:
    """Tool environment double: `ship` flips state; snapshot() is what the oracle checks."""

    def __init__(self):
        self.state = {"shipped": False}

    def __call__(self, name, args):
        if name == "ship":
            self.state["shipped"] = True
            return "shipped"
        return "unknown tool"

    def snapshot(self):
        return dict(self.state)


SHIP = [{"type": "function", "function": {"name": "ship", "parameters": {
    "type": "object", "properties": {"sku": {"type": "string"}}, "required": ["sku"]}}}]


def _harness(model, **spec_kw):
    from auditkit.agent_eval import AgentCase, AgentEvalRunner, AgentEvalSpec, FinalStateAssertion
    case = AgentCase(id="ship", task="Ship item A.", allowed_tools=SHIP,
                     outcome=FinalStateAssertion("shipped", equals=True))
    return AgentEvalRunner().run(AgentEvalSpec(cases=[case], mode="harness", agent=model,
                                               agent_opts={"tool_env": _Ship(), "max_steps": 4}, scorers=[],
                                               **spec_kw))


def _ship_then_answer(prompt):
    # first step: a Hermes call; once the tool result is in the conversation: answer
    return "Shipped." if "shipped" in prompt.rsplit("Ship item A.", 1)[-1] else \
        '<tool_call>\n{"name": "ship", "arguments": {"sku": "A"}}\n</tool_call>'


def test_the_harness_executes_text_tool_calls_from_an_hf_model():
    res = _harness(stub_hf(cached_tokenizer("Qwen/Qwen3-1.7B"), reply=_ship_then_answer))
    assert res.episodes[0].final_state == {"shipped": True} and res.rows[0].outcome["verdict"] == "success"


def test_the_harness_passes_a_generation_budget():
    from auditkit.runspec import RunConfig
    m = stub_hf(cached_tokenizer("Qwen/Qwen3-1.7B"), reply="done")
    try:
        _harness(m, config=RunConfig(max_tokens=512))
    except TypeError:                      # no config knob on AgentEvalSpec yet
        pytest.fail("AgentEvalSpec takes no generation config")
    assert m._pipeline.calls[0][1].get("max_new_tokens") == 512
