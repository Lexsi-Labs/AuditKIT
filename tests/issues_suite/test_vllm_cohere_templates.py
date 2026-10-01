"""The vllm: named-template fix, with the real Cohere tokenizers as users download them.

Found by examples/colab/06_cohere_all_backends.ipynb on a Colab L4 (vLLM 0.30): every Cohere
model ships a list of named chat templates, loaded as a dict, and vllm: sent the raw prompt.
vLLM itself needs Linux + CUDA; the engine here records the prompts it would be given.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import auditkit as ak
from auditkit.errors import CapabilityError
from auditkit.model import Request
from auditkit.model.vllm_gen import VLLMModel

from .conftest import cached_tokenizer

TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
TINY_AYA, R7B = "CohereLabs/tiny-aya-global", "CohereLabs/c4ai-command-r7b-12-2024"


class Engine:
    def __init__(self, tok, reply):
        self.tok, self.reply, self.prompts = tok, reply, []

    def get_tokenizer(self):
        return self.tok

    def generate(self, prompts, params, **kw):
        self.prompts += list(prompts)
        return [SimpleNamespace(outputs=[SimpleNamespace(text=self.reply(p) if callable(self.reply) else self.reply)])
                for p in prompts]


def vllm(name, reply="Paris."):
    tok = cached_tokenizer(name)
    assert isinstance(tok.chat_template, dict), "the Hub ships named templates; this is the case under test"
    m = VLLMModel(name, name=f"vllm:{name}")
    m._llm, m._sampling_params = Engine(tok, reply), (lambda **kw: kw)
    return m


@pytest.mark.parametrize("name", [TINY_AYA, R7B])
def test_each_cohere_model_gets_its_chat_template(name):
    m = vllm(name)
    m.generate([Request(prompt="What is the capital of France?")])
    prompt = m._llm.prompts[0]
    assert "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>What is the capital of France?" in prompt
    assert prompt.rstrip().endswith("<|CHATBOT_TOKEN|>") or "<|START_RESPONSE|>" in prompt[-40:]
    assert prompt.count("<BOS_TOKEN>") == 1


def test_the_colab_failure_does_not_come_back():
    # the run's generation check: qem 0.0 because the model continued the raw text
    m = vllm(TINY_AYA, reply=lambda p: "Paris." if "<|CHATBOT_TOKEN|>" in p else "Paris.\nAnswer: Paris")
    r = ak.evaluate([ak.Sample(id="q", input="What is the capital of France? Answer with one word.", target="Paris")],
                    model=m, scorers=["quasi_exact_match"])
    assert r.headline["quasi_exact_match"] == 1.0


def test_tiny_aya_refuses_native_tools_for_the_right_reason():
    with pytest.raises(CapabilityError, match="does not render tool schemas"):
        vllm(TINY_AYA).generate([Request(prompt="Weather in Paris?", params={"tools": TOOLS})])


def test_command_r7b_native_tools_render_and_its_action_list_scores():
    reply = '<|START_ACTION|>[{"tool_call_id": "0", "tool_name": "get_weather", "parameters": {"city": "Paris"}}]<|END_ACTION|>'
    m = vllm(R7B, reply=reply)
    r = ak.evaluate([ak.Sample(id="w", input="Weather in Paris?", tools=TOOLS,
                               expected_tool_calls=[[{"name": "get_weather", "arguments": {"city": "Paris"}}]])],
                    model=m, adapter=ak.ToolCallAdapter(), scorers=[ak.ToolCallF1()])
    assert "get_weather" in m._llm.prompts[0]
    assert r.headline["tool_call_f1"] == 1.0 and not r.errors


def test_prompt_mode_still_works_for_tiny_aya():
    m = vllm(TINY_AYA, reply='<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>')
    r = ak.evaluate([ak.Sample(id="w", input="Weather in Paris?", tools=TOOLS,
                               expected_tool_calls=[[{"name": "get_weather", "arguments": {"city": "Paris"}}]])],
                    model=m, adapter=ak.ToolCallAdapter(mode="prompt"), scorers=[ak.ToolCallF1()])
    assert r.headline["tool_call_f1"] == 1.0
