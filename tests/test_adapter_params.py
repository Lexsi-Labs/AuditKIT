"""Adapters forward a consistent set of generation params.

Regression: GenerationAdapter used to forward the full RunConfig generation set
while ChatAdapter/FewShot/Instruction/RAG/Template forwarded only 5 fields, so
whether presence_penalty/top_k/frequency_penalty reached the model silently
depended on which adapter you picked (while the fingerprint moved regardless).
"""

from __future__ import annotations

from auditkit.adapter import (
    ChatAdapter, FewShotAdapter, GenerationAdapter, InstructionAdapter,
    RAGAdapter, TemplateAdapter,
)
from auditkit.runspec import RunConfig
from auditkit.sample import Sample


def _adapters_with_samples():
    return [
        (GenerationAdapter(), Sample(input="q")),
        (ChatAdapter(), Sample(input="q")),
        (InstructionAdapter(), Sample(input="q")),
        (TemplateAdapter(), Sample(input="q")),
        (RAGAdapter(), Sample(input="q", retrieval_context=["ctx"])),
        (FewShotAdapter(num_shots=0), Sample(input="q")),
    ]


def test_all_adapters_forward_the_same_generation_params():
    cfg = RunConfig(temperature=0.7, top_k=10, presence_penalty=0.5, frequency_penalty=0.2)
    for adapter, sample in _adapters_with_samples():
        params = adapter.adapt(sample, cfg)[0].params
        for key in ("temperature", "top_k", "presence_penalty", "frequency_penalty"):
            assert params.get(key) == getattr(cfg, key), (
                f"{type(adapter).__name__} dropped {key}"
            )


def test_unset_params_are_not_forwarded():
    cfg = RunConfig()  # top_k / presence_penalty / seed default to None
    params = ChatAdapter().adapt(Sample(input="q"), cfg)[0].params
    assert "presence_penalty" not in params
    assert "top_k" not in params
    assert "seed" not in params


def test_default_runconfig_never_forces_a_seed_into_the_request():
    """Regression: RunConfig.seed used to default to 0 (not None, unlike
    every other generation field here), so seed=0 was silently included in
    every request whether the caller asked for it or not -- and at least
    one real Groq model rejects the seed param outright, turning every
    request into a 400. Explicitly passing seed= still works normally."""
    cfg = RunConfig()
    assert cfg.seed is None
    for adapter, sample in _adapters_with_samples():
        params = adapter.adapt(sample, cfg)[0].params
        assert "seed" not in params, f"{type(adapter).__name__} forwarded an unset seed"

    cfg_with_seed = RunConfig(seed=7)
    params = GenerationAdapter().adapt(Sample(input="q"), cfg_with_seed)[0].params
    assert params["seed"] == 7


def test_runner_only_settings_are_not_forwarded():
    cfg = RunConfig(max_retries=5, timeout=10.0)
    params = GenerationAdapter().adapt(Sample(input="q"), cfg)[0].params
    assert "max_retries" not in params
    assert "timeout" not in params
