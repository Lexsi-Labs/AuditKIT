"""Live runs of the agentic integrations. Opt-in; they skip unless enabled.

    AK_LIVE_HF=1       Qwen3-1.7B through hf: (native tools and prompt mode). ~4 GB; CPU/MPS/GPU.
    AK_LIVE_COHERE=1   CohereLabs/tiny-aya-global through hf: (prompt-mode tools; native must be
                       refused). Gated: accept the licence on the Hub and set HF_TOKEN. ~7 GB.
    AK_LIVE_R7B=1      CohereLabs/c4ai-command-r7b-12-2024 native tools. Gated, ~16 GB: a 24 GB+ GPU.

Model quality varies, so these assert the integration works end to end: the run completes with no
errors, the schemas reach the model, replies are parsed into calls and scored. Scores are printed.
The Colab notebook tests/integrations_suite/colab_integrations.ipynb runs all of them.
"""

from __future__ import annotations

import os

import pytest

import auditkit as ak

pytestmark = pytest.mark.live

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "get_time", "description": "Current local time in a city",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
]


def W(city):
    return {"name": "get_weather", "arguments": {"city": city}}


def samples():
    return [
        ak.Sample(id="one", input="What's the weather in Paris?", tools=TOOLS, expected_tool_calls=[W("Paris")]),
        ak.Sample(id="parallel", input="What's the weather in Paris and in Rome? Call the tool for both cities at once.",
                  tools=TOOLS, expected_tool_calls=[[W("Paris"), W("Rome")]]),
        ak.Sample(id="no_tool", input="What is 2+2? Answer directly, do not use tools.", tools=TOOLS,
                  expected_tool_calls=[]),
        ak.Sample(id="other_tool", input="What time is it in Tokyo?", tools=TOOLS,
                  expected_tool_calls=[{"name": "get_time", "arguments": {"city": "Tokyo"}}]),
    ]


def model(name):
    from auditkit.model.hf_gen import HFGenModel
    return HFGenModel(name, device="auto", name=f"hf:{name}", **({"token": os.environ["HF_TOKEN"]}
                                                                 if os.environ.get("HF_TOKEN") else {}))


def run(m, mode):
    from auditkit.runspec import RunConfig
    r = ak.evaluate(samples(), model=m, adapter=ak.ToolCallAdapter(mode=mode),
                    scorers=[ak.ToolCallF1(), ak.ToolCallValidity(), ak.ParallelToolCalls()],
                    config=RunConfig(max_tokens=512))
    print(f"\n{m.name} [{mode}] headline:", {k: round(v, 3) for k, v in r.headline.items()})
    for p in r.predictions:
        print(" ", p.sample_id, "|", repr((p.raw_output or "")[-160:]))
    return r


def parsed_calls(r):
    from auditkit.trace import parse_tool_calls
    return sum(len(parse_tool_calls(p.raw_output or "")) for p in r.predictions)


needs = lambda var: pytest.mark.skipif(os.environ.get(var) != "1", reason=f"set {var}=1 to run")  # noqa: E731


@needs("AK_LIVE_HF")
@pytest.mark.parametrize("mode", ["native", "prompt"])
def test_qwen3_through_hf(mode):
    r = run(model("Qwen/Qwen3-1.7B"), mode)
    assert r.errors == [] and parsed_calls(r) >= 3
    assert r.headline["tool_call_f1"] >= 0.75           # 1.0 on an M3 Pro (MPS), native mode


@needs("AK_LIVE_COHERE")
def test_tiny_aya_prompt_mode_tool_calls():
    r = run(model("CohereLabs/tiny-aya-global"), "prompt")
    assert r.errors == [] and parsed_calls(r) >= 1


@needs("AK_LIVE_COHERE")
def test_tiny_aya_native_mode_is_refused_with_advice():
    from auditkit.errors import CapabilityError
    with pytest.raises(CapabilityError, match="mode='prompt'"):
        run(model("CohereLabs/tiny-aya-global"), "native")


@needs("AK_LIVE_R7B")
def test_command_r7b_native_tool_calls():
    r = run(model("CohereLabs/c4ai-command-r7b-12-2024"), "native")
    assert r.errors == [] and parsed_calls(r) >= 3       # Cohere action lists parsed from the raw decode
