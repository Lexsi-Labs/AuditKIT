"""#51 step 2: transformers' "does not recognize this architecture" names what actually happened.

That message reads like a broken checkpoint; in #51 it was a stale auditkit that had pinned
transformers<5. The rewrite names the model type, the installed transformers and the floor.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from auditkit.model import explain_unknown_architecture

REAL = ("The checkpoint you are trying to load has model type `cohere_compass` but Transformers does not "
        "recognize this architecture. This could be because of an issue with the checkpoint, or because "
        "your version of Transformers is out of date.")


def test_the_message_names_type_version_and_floor(monkeypatch):
    from importlib import metadata
    real = metadata.version
    monkeypatch.setattr(metadata, "version", lambda n: {"transformers": "4.57.6", "auditkit": "1.0.0"}.get(n) or real(n))
    e = explain_unknown_architecture(ValueError(REAL), "CohereLabs/North-Micro-Vision-Instruct")
    text = str(e)
    for part in ("'cohere_compass'", "transformers 4.57.6", "transformers>=5.15", "installed auditkit: 1.0.0",
                 "The checkpoint is fine", "pip install 'transformers>=5.15,<6'", REAL):
        assert part in text, part


def test_an_architecture_without_a_known_floor_still_gets_the_explanation():
    e = explain_unknown_architecture(ValueError(REAL.replace("cohere_compass", "brand_new_arch")), "org/x")
    assert "'brand_new_arch'" in str(e) and "a newer transformers" in str(e)


def test_other_errors_are_left_alone():
    assert explain_unknown_architecture(ValueError("max_model_len too large"), "x") is None


def test_hf_backend_raises_it_for_real_on_an_old_transformers():
    # the #51 failure, end to end: North's real config, whatever transformers is installed here
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from auditkit.model.hf_gen import HFGenModel
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    if "cohere_compass" in CONFIG_MAPPING_NAMES:
        pytest.skip(f"transformers {transformers.__version__} knows cohere_compass")
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download("CohereLabs/North-Micro-Vision-Instruct", "config.json", local_files_only=True)
    except Exception:
        pytest.skip("North's config is not in the local HF cache")
    with pytest.raises(ValueError, match="does not know that architecture: it needs transformers>=5.15"):
        HFGenModel("CohereLabs/North-Micro-Vision-Instruct", device="cpu")._ensure_pipeline()


def test_vllm_backend_raises_it_too(monkeypatch):
    def LLM(model, **kw):
        raise ValueError(REAL)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=LLM, SamplingParams=dict))
    from auditkit.model.vllm_gen import VLLMModel
    with pytest.raises(ValueError, match="does not know that architecture"):
        VLLMModel("CohereLabs/North-Micro-Vision-Instruct")._ensure_llm()
