"""#43 part 3, the mistakes users make when naming a model, through ak.evaluate itself."""

from __future__ import annotations

import os

import pytest

import auditkit as ak
from auditkit.errors import AuditKitError

Q = [ak.Sample(id="q", input="Capital of France?", target="Paris")]


@pytest.mark.parametrize("spec", ["api:", "api: ", "hf:", "vllm:"])
def test_forgetting_the_model_name_fails_before_any_request(spec):
    # api_base/api_key given, as in the SGLang quick start: only the name is missing
    with pytest.raises(AuditKitError, match="empty model spec"):
        ak.evaluate(Q, model=spec, api_base="http://127.0.0.1:9/v1", api_key="EMPTY")


def test_the_error_shows_a_spec_to_copy():
    with pytest.raises(AuditKitError, match=r"e\.g\. 'api:Qwen/Qwen2\.5-7B-Instruct'"):
        ak.AutoModel.resolve("api:")


def test_a_real_local_model_folder_still_resolves():
    hub = os.path.join(os.environ["HF_HOME"], "hub", "models--Qwen--Qwen3-1.7B", "snapshots")
    if not os.path.isdir(hub):
        pytest.skip("Qwen3-1.7B is not in the local HF cache")
    folder = os.path.join(hub, sorted(os.listdir(hub))[-1])
    assert ak.AutoModel.resolve(f"hf:{folder}")._model_name == folder      # resolved, never loaded


def test_a_name_with_a_colon_is_kept_whole():
    pytest.importorskip("requests")
    assert ak.AutoModel.resolve("api:my-model:v2")._model_name == "my-model:v2"


@pytest.mark.parametrize("spec,default", [("openai:", "gpt-4o-mini"), ("groq:", "llama-3.3-70b-versatile"),
                                          ("openrouter:", "openai/gpt-4o-mini")])
def test_a_bare_hosted_prefix_means_the_service_default(spec, default):
    pytest.importorskip("requests")
    try:
        m = ak.AutoModel.resolve(spec)
    except ak.ExtraNotInstalled:
        pytest.skip(f"{spec} extra not installed")
    assert m._model_name == default


def test_agent_takes_its_url_from_url_or_says_it_needs_one():
    assert ak.AutoModel.resolve("agent:", url="http://127.0.0.1:9/run").name == "agent:"
    with pytest.raises(AuditKitError, match="URL"):
        ak.AutoModel.resolve("agent:")


@pytest.mark.parametrize("spec", ["precomputed", "echo"])
def test_nameless_specs_are_untouched(spec):
    assert ak.AutoModel.resolve(spec).name == spec
