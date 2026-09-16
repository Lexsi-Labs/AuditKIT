from __future__ import annotations

from auditkit.adapter import ChatAdapter
from auditkit.model import CallableModel
from auditkit.runspec import RunConfig
from auditkit.sample import Sample


class TestChatAdapter:
    def test_system_prompt_is_baked_into_request_prompt(self):
        # Request.params is not read by any real model backend (openai.py,
        # anthropic.py, hf_gen.py, ...) or test double (EchoModel,
        # CallableModel) -- only request.prompt is. The system prompt must
        # therefore live in prompt to actually reach a model.
        adapter = ChatAdapter(system_prompt="You are a pirate.")
        req = adapter.adapt(Sample(input="hi"), RunConfig())[0]
        assert "You are a pirate." in req.prompt

    def test_system_prompt_actually_reaches_the_model(self):
        adapter = ChatAdapter(system_prompt="You are a pirate.")
        req = adapter.adapt(Sample(input="hi"), RunConfig())[0]

        seen = []

        def fake_fn(prompts):
            seen.extend(prompts)
            return ["arr" for _ in prompts]

        CallableModel(fake_fn).generate([req])
        assert "You are a pirate." in seen[0]

    def test_messages_still_present_in_params_for_future_backends(self):
        adapter = ChatAdapter(system_prompt="You are a pirate.")
        req = adapter.adapt(Sample(input="hi"), RunConfig())[0]
        assert req.params["messages"][0] == {"role": "system", "content": "You are a pirate."}
