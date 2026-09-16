"""HFGenModel owns the model wire format: any adapter's flat prompt is chat-
templated on an instruct model, raw on a base model.

Regression: _render_prompt only applied the chat template when the request
carried structured messages (i.e. only from ChatAdapter), so a flat prompt from
GenerationAdapter/MCQAdapter/RAGAdapter reached an instruct model as raw text.
These fake the pipeline/tokenizer so no transformers/model download is needed.
"""

from __future__ import annotations

from auditkit.model import Request
from auditkit.model.hf_gen import HFGenModel


class _FakeTokenizer:
    def __init__(self, has_template: bool):
        self.chat_template = "SOME_TEMPLATE" if has_template else None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "RENDERED::" + repr(messages)


class _FakePipeline:
    def __init__(self, has_template: bool):
        self.tokenizer = _FakeTokenizer(has_template)


def _model(has_template: bool) -> HFGenModel:
    m = HFGenModel(model="x")
    m._pipeline = _FakePipeline(has_template)   # bypass the lazy real load
    return m


def test_flat_prompt_is_chat_templated_on_instruct_model():
    m = _model(has_template=True)
    rendered = m._render_prompt(Request(prompt="What is 2+2?", request_type="generate"))
    assert rendered.startswith("RENDERED::")
    assert "'role': 'user'" in rendered
    assert "What is 2+2?" in rendered


def test_flat_prompt_is_raw_on_base_model():
    m = _model(has_template=False)
    assert m._render_prompt(Request(prompt="What is 2+2?")) == "What is 2+2?"


def test_structured_messages_still_render_directly():
    m = _model(has_template=True)
    req = Request(prompt="ignored-fallback", request_type="chat",
                  params={"messages": [{"role": "system", "content": "S"},
                                       {"role": "user", "content": "U"}]})
    rendered = m._render_prompt(req)
    assert "'content': 'S'" in rendered and "'content': 'U'" in rendered
    assert "ignored-fallback" not in rendered      # messages win over the flat prompt
