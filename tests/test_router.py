"""route_adapter(): the dataset's shape decides the adapter, never the model.

choices -> MCQAdapter, retrieval_context -> RAGAdapter, otherwise
GenerationAdapter. The model is ignored — chat-vs-base formatting is applied by
the model backend at generate-time, so the adapter never has to be chosen for
it. ChatAdapter is never auto-selected (it's only for an explicit system prompt).
"""
from __future__ import annotations

import pytest

from auditkit.adapter import ChatAdapter, GenerationAdapter, MCQAdapter, RAGAdapter
from auditkit.router import route_adapter
from auditkit.sample import Sample


class TestSampleShapeRouting:
    def test_choices_route_to_mcq(self):
        assert isinstance(route_adapter(Sample(input="2+2?", choices=["3", "4", "5"])), MCQAdapter)

    def test_retrieval_context_routes_to_rag(self):
        assert isinstance(route_adapter(Sample(input="q", retrieval_context=["ctx"])), RAGAdapter)

    def test_choices_checked_before_retrieval_context(self):
        s = Sample(input="q", choices=["3", "4"], retrieval_context=["ctx"])
        assert isinstance(route_adapter(s), MCQAdapter)

    def test_plain_sample_routes_to_generation(self):
        assert isinstance(route_adapter(Sample(input="plain")), GenerationAdapter)

    def test_accepts_a_list_and_inspects_the_first(self):
        samples = [Sample(input="2+2?", choices=["3", "4"]), Sample(input="plain")]
        assert isinstance(route_adapter(samples), MCQAdapter)

    def test_empty_samples_raises(self):
        with pytest.raises(ValueError):
            route_adapter([])


class TestModelIsIgnored:
    @pytest.mark.parametrize("model", [
        "echo", "openai:gpt-4o-mini", "anthropic:claude-sonnet-4",
        "groq:llama-3.3-70b", "hf:meta-llama/Llama-3.2-1B-Instruct",
        "vllm:gpt2", None, (lambda prompts: prompts),
    ])
    def test_plain_sample_is_generation_regardless_of_model(self, model):
        # No tokenizer probe, no model-based ChatAdapter — routing is task-only.
        adapter = route_adapter(Sample(input="plain"), model, hf_token="hf_xxx")
        assert isinstance(adapter, GenerationAdapter)

    def test_chat_adapter_is_never_auto_selected(self):
        # Even an instruct model + a plain sample -> GenerationAdapter; the
        # backend applies the chat template, so no ChatAdapter is needed here.
        for model in ("openai:gpt-4o-mini", "hf:some/instruct-model"):
            assert not isinstance(route_adapter(Sample(input="p"), model), ChatAdapter)

    def test_shape_still_wins_over_any_model(self):
        s = Sample(input="2+2?", choices=["3", "4"])
        assert isinstance(route_adapter(s, "openai:gpt-4o-mini"), MCQAdapter)


class TestAutoAdapterWiring:
    def test_to_adapter_auto_routes_by_shape(self):
        from auditkit.api import _to_adapter
        assert isinstance(_to_adapter("auto", [Sample(input="q", choices=["a", "b"])]), MCQAdapter)
        assert isinstance(_to_adapter("auto", [Sample(input="q", retrieval_context=["c"])]), RAGAdapter)
        assert isinstance(_to_adapter("auto", [Sample(input="q")]), GenerationAdapter)

    def test_evaluate_adapter_auto_runs_end_to_end(self):
        import auditkit as ak
        r = ak.evaluate([Sample(input="2+2?", target="4")], model="echo", adapter="auto")
        assert r is not None
