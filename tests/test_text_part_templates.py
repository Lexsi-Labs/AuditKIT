"""Chat templates that read only text parts (#51).

Llama Guard 3 1B and Llama Guard 4 loop over ``message['content'] |
selectattr('type', 'equalto', 'text')``: a plain-string ``content`` renders as
nothing, so the guard classifies an empty conversation. Llama Guard 3 8B,
ShieldGemma and Granite Guardian need a string and fail on a list. The backends
render with strings and switch to text parts only when the strings were dropped.

The templates below are small stand-ins with the same content handling as the
real ones, so no gated tokenizer is downloaded.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from jinja2 import Environment

from auditkit.metrics.guard import GuardJudge
from auditkit.model import Generated, Request, Result_
from auditkit.model.hf_gen import HFGenModel
from auditkit.model.vllm_gen import VLLMModel
from auditkit.sample import Sample

# Llama Guard 3 1B / Llama Guard 4: text parts only.
PARTS_ONLY = (
    "{%- for m in messages -%}"
    "{%- for c in m['content'] | selectattr('type', 'equalto', 'text') -%}"
    "{{ m['role'] }}: {{ c['text'] | trim }}\n"
    "{%- endfor -%}{%- endfor -%}<END>"
)
# Llama Guard 3 8B / ShieldGemma / Granite Guardian: strings only.
STRING_ONLY = "{%- for m in messages -%}{{ m['role'] }}: {{ m['content'].strip() }}\n{%- endfor -%}<END>"


class _JinjaTokenizer:
    def __init__(self, template: str):
        self.chat_template = template
        self.calls = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        self.calls += 1
        return Environment().from_string(self.chat_template).render(messages=messages, **kwargs)


def _hf(template: str) -> HFGenModel:
    m = HFGenModel(model="x")
    m._pipeline = MagicMock(tokenizer=_JinjaTokenizer(template))
    return m


def _vllm(template: str) -> VLLMModel:
    m = VLLMModel()
    m._llm = MagicMock()
    m._llm.get_tokenizer.return_value = _JinjaTokenizer(template)
    return m


PAIR = [{"role": "user", "content": "PROMPT_CANARY"}, {"role": "assistant", "content": "RESPONSE_CANARY"}]


@pytest.mark.parametrize("make", [_hf, _vllm])
@pytest.mark.parametrize("messages", [PAIR, PAIR[:1]], ids=["response", "prompt"])
def test_parts_only_template_keeps_the_conversation(make, messages):
    rendered = make(PARTS_ONLY)._render_prompt(Request(prompt="", params={"messages": messages}))
    for m in messages:
        assert m["content"] in rendered


@pytest.mark.parametrize("make", [_hf, _vllm])
def test_string_template_renders_once_with_strings(make):
    model = make(STRING_ONLY)
    rendered = model._render_prompt(Request(prompt="", params={"messages": PAIR}))
    assert rendered == "user: PROMPT_CANARYassistant: RESPONSE_CANARY<END>"
    tok = model._pipeline.tokenizer if isinstance(model, HFGenModel) else model._llm.get_tokenizer()
    assert tok.calls == 1


def test_flat_prompt_on_parts_only_template_is_kept():
    assert "hello" in _hf(PARTS_ONLY)._render_prompt(Request(prompt="hello"))


def test_template_that_drops_a_turn_by_design_is_left_alone():
    # Drops system turns whatever their form: text parts do not bring the turn back, so keep the first render.
    tpl = "{%- for m in messages if m['role'] != 'system' -%}{{ m['content'] }}{%- endfor -%}"
    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "U"}]
    assert _hf(tpl)._render_prompt(Request(prompt="", params={"messages": msgs})) == "U"


class _Recorder:
    """A judge model that records the requests it gets."""

    def __init__(self, fail_first: bool = False):
        self.requests: list[Request] = []
        self.fail_first = fail_first

    def generate(self, requests):
        self.requests.extend(requests)
        if self.fail_first and len(self.requests) == 1:
            raise RuntimeError("messages must contains a single user message for text classification models")
        return [Result_(completions=[Generated(text="safe")])]


@pytest.mark.parametrize("single_message", [True, False])
def test_guard_judge_sends_string_content_for_any_spec(single_message):
    """GuardJudge keeps plain strings for every spec, so a hosted 1B guard collapsed to one
    message (single_message, or the automatic retry) never gets a Python list repr."""
    from auditkit.model import resolve_messages

    rec = _Recorder(fail_first=not single_message)
    judge = GuardJudge(judge_model=rec, single_message=single_message)
    judge._spec, judge._model = "groq:meta-llama/Llama-Guard-3-1B", rec   # a hosted 1B spec, answered by the recorder
    judge.score(Sample(input="PROMPT_CANARY"), "RESPONSE_CANARY")
    last = rec.requests[-1]
    assert all(isinstance(m["content"], str) for m in last.params["messages"])
    collapsed = resolve_messages(last)
    assert collapsed == [{"role": "user", "content": "User: PROMPT_CANARY\n\nAssistant: RESPONSE_CANARY"}]
