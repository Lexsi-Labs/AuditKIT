"""Stress test for the resolve_messages() / structured-chat fix.

Before this fix, ChatAdapter built a correct role-tagged `messages` list but
stored it only in Request.params["messages"] -- every real backend read only
the flattened Request.prompt string, so no provider ever received a real
system/user role split (see logs/CORRECTNESS_FIXES_2026-07-07.md, item 4,
which fixed the prompt but left this half of the gap open deliberately).

This mirrors tests/test_generation_kwargs.py's approach for resolve_params():
drive every real backend's generate() with a mocked client/session/pipeline
and assert the *actual outgoing call* carries the real messages, not just
that the code runs.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from auditkit.model import Request, resolve_messages


# ---------------------------------------------------------------------------
# Unit tests for the shared resolve_messages() helper itself
# ---------------------------------------------------------------------------
class TestResolveMessages:
    def test_falls_back_to_a_single_user_turn_when_no_messages_set(self):
        req = Request(prompt="hi", params={})
        assert resolve_messages(req) == [{"role": "user", "content": "hi"}]

    def test_uses_structured_messages_when_present(self):
        turns = [{"role": "system", "content": "be nice"}, {"role": "user", "content": "hi"}]
        req = Request(prompt="System: be nice\n\nUser: hi", params={"messages": turns})
        assert resolve_messages(req) == turns

    def test_empty_messages_list_falls_back_to_prompt(self):
        req = Request(prompt="hi", params={"messages": []})
        assert resolve_messages(req) == [{"role": "user", "content": "hi"}]


# ---------------------------------------------------------------------------
# ChatAdapter -> resolve_messages(): confirm the adapter's structured output
# is exactly what a downstream backend will now see.
# ---------------------------------------------------------------------------
class TestChatAdapterProducesResolvableMessages:
    def test_system_and_user_turns_round_trip(self):
        from auditkit.adapter import ChatAdapter
        from auditkit.runspec import RunConfig
        from auditkit.sample import Sample

        adapter = ChatAdapter(system_prompt="You are a pirate.")
        req = adapter.adapt(Sample(input="Where's the treasure?"), RunConfig())[0]

        messages = resolve_messages(req)
        assert messages == [
            {"role": "system", "content": "You are a pirate."},
            {"role": "user", "content": "Where's the treasure?"},
        ]


# ---------------------------------------------------------------------------
# Stress test: drive every real chat backend's generate() with a fake client
# and confirm the actual outgoing call carries real role-tagged messages.
# ---------------------------------------------------------------------------
class TestOpenAIModelSendsRealMessages:
    def test_structured_messages_reach_the_real_api_call(self):
        from auditkit.model.openai import OpenAIModel

        m = OpenAIModel(api_key="test")
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )
        m._client = fake_client

        req = Request(prompt="System: be nice\n\nUser: hi", params={
            "messages": [{"role": "system", "content": "be nice"}, {"role": "user", "content": "hi"}]
        })
        m.generate([req])

        _, kwargs = fake_client.chat.completions.create.call_args
        assert kwargs["messages"] == [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hi"},
        ]

    def test_plain_generation_request_still_sends_a_single_user_turn(self):
        from auditkit.model.openai import OpenAIModel

        m = OpenAIModel(api_key="test")
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )
        m._client = fake_client

        req = Request(prompt="plain question")  # GenerationAdapter-style, no messages
        m.generate([req])

        _, kwargs = fake_client.chat.completions.create.call_args
        assert kwargs["messages"] == [{"role": "user", "content": "plain question"}]


class TestAnthropicModelSplitsSystemTurn:
    def test_system_turn_becomes_the_system_kwarg_not_a_message(self):
        from auditkit.model.anthropic import AnthropicModel

        m = AnthropicModel(api_key="test")
        fake_client = MagicMock()
        fake_client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(text="ok")]
        )
        m._client = fake_client

        req = Request(prompt="System: be nice\n\nUser: hi", params={
            "messages": [{"role": "system", "content": "be nice"}, {"role": "user", "content": "hi"}]
        })
        m.generate([req])

        _, kwargs = fake_client.messages.create.call_args
        # Anthropic's Messages API rejects a "system" role inside messages --
        # it must be pulled out into its own top-level kwarg.
        assert kwargs["system"] == "be nice"
        assert kwargs["messages"] == [{"role": "user", "content": "hi"}]

    def test_no_system_kwarg_when_there_is_no_system_turn(self):
        from auditkit.model.anthropic import AnthropicModel

        m = AnthropicModel(api_key="test")
        fake_client = MagicMock()
        fake_client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(text="ok")]
        )
        m._client = fake_client

        req = Request(prompt="plain question")
        m.generate([req])

        _, kwargs = fake_client.messages.create.call_args
        assert "system" not in kwargs
        assert kwargs["messages"] == [{"role": "user", "content": "plain question"}]


class TestGroqModelSendsRealMessages:
    def test_structured_messages_reach_the_real_http_call(self):
        from auditkit.model.groq_gen import GroqModel

        m = GroqModel(api_key="test")
        fake_session = MagicMock()
        fake_session.post.return_value = SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": "ok"}}]},
        )
        m._session = fake_session

        req = Request(prompt="System: be nice\n\nUser: hi", params={
            "messages": [{"role": "system", "content": "be nice"}, {"role": "user", "content": "hi"}]
        })
        m.generate([req])

        _, kwargs = fake_session.post.call_args
        assert kwargs["json"]["messages"] == [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hi"},
        ]


class TestLiteLLMModelSendsRealMessages:
    def test_structured_messages_reach_the_real_completion_call(self, monkeypatch):
        import auditkit.model.litellm_gen as litellm_gen_module

        calls = {}

        def fake_completion(**kwargs):
            calls.update(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

        fake_litellm = SimpleNamespace(completion=fake_completion)
        monkeypatch.setitem(__import__("sys").modules, "litellm", fake_litellm)

        m = litellm_gen_module.LiteLLMModel()
        req = Request(prompt="System: be nice\n\nUser: hi", params={
            "messages": [{"role": "system", "content": "be nice"}, {"role": "user", "content": "hi"}]
        })
        m.generate([req])

        assert calls["messages"] == [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hi"},
        ]


class TestAPIModelSendsRealMessagesInChatMode:
    def test_chat_template_true_sends_structured_messages(self):
        from auditkit.model.api_gen import APIModel

        m = APIModel(chat_template=True)
        fake_session = MagicMock()
        fake_session.post.return_value = SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": "ok"}}]},
        )
        m._session = fake_session

        req = Request(prompt="System: be nice\n\nUser: hi", params={
            "messages": [{"role": "system", "content": "be nice"}, {"role": "user", "content": "hi"}]
        })
        m.generate([req])

        _, kwargs = fake_session.post.call_args
        assert kwargs["json"]["messages"] == [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hi"},
        ]

    def test_chat_template_false_still_sends_raw_prompt_not_messages(self):
        from auditkit.model.api_gen import APIModel

        m = APIModel(chat_template=False)
        fake_session = MagicMock()
        fake_session.post.return_value = SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"text": "ok"}]},
        )
        m._session = fake_session

        req = Request(prompt="System: be nice\n\nUser: hi", params={
            "messages": [{"role": "system", "content": "be nice"}, {"role": "user", "content": "hi"}]
        })
        m.generate([req])

        _, kwargs = fake_session.post.call_args
        assert "messages" not in kwargs["json"]
        assert kwargs["json"]["prompt"] == "System: be nice\n\nUser: hi"


# ---------------------------------------------------------------------------
# HFGenModel: local models render via their own tokenizer chat_template
# instead of a real API's role-based messages array.
# ---------------------------------------------------------------------------
class TestHFGenModelRendersPerModelChatTemplate:
    def _model_with_fake_pipeline(self, chat_template):
        from auditkit.model.hf_gen import HFGenModel

        m = HFGenModel()
        fake_tokenizer = SimpleNamespace(
            chat_template=chat_template,
            apply_chat_template=lambda messages, tokenize, add_generation_prompt: (
                "<CHATML>" + "".join(f"[{msg['role']}]{msg['content']}" for msg in messages) + "[assistant]"
            ),
        )
        m._pipeline = SimpleNamespace(tokenizer=fake_tokenizer)
        return m

    def test_structured_messages_rendered_via_this_models_own_template(self):
        m = self._model_with_fake_pipeline(chat_template="{{ some jinja }}")
        req = Request(prompt="System: be nice\n\nUser: hi", params={
            "messages": [{"role": "system", "content": "be nice"}, {"role": "user", "content": "hi"}]
        })

        rendered = m._render_prompt(req)

        # Rendered through *this model's* fake chat_template, not the
        # generic "System: ...\nUser: ..." flattened string.
        assert rendered == "<CHATML>[system]be nice[user]hi[assistant]"

    def test_falls_back_to_flat_prompt_when_model_has_no_chat_template(self):
        # Base / non-instruct checkpoints (plain gpt2, etc.) have no
        # chat_template at all -- there is no per-model format to apply.
        m = self._model_with_fake_pipeline(chat_template=None)
        req = Request(prompt="System: be nice\n\nUser: hi", params={
            "messages": [{"role": "system", "content": "be nice"}, {"role": "user", "content": "hi"}]
        })

        rendered = m._render_prompt(req)

        assert rendered == "System: be nice\n\nUser: hi"

    def test_flat_prompt_is_wrapped_and_templated_on_an_instruct_model(self):
        # The backend owns the wire format: a flat prompt from any adapter
        # (GenerationAdapter/MCQ/RAG/...) is wrapped as a single user turn and
        # rendered through an instruct model's own chat template — not sent as
        # raw text it was never trained on. (Routing picks the technique; the
        # backend handles chat-vs-base formatting.)
        m = self._model_with_fake_pipeline(chat_template="{{ some jinja }}")
        req = Request(prompt="plain question")  # GenerationAdapter-style

        rendered = m._render_prompt(req)

        assert rendered == "<CHATML>[user]plain question[assistant]"
