"""hf: backend interop, CPU only, tiny random weights built in-test (no downloads):

* ``hf:<PEFT adapter dir>`` loads base + adapter, and without peft fails with
  an actionable error instead of "generate failed after N retries".
* Native tool schemas reach the chat template (``tools=``) on a tiny ``cohere2``
  (the Tiny Aya / Command R7B arch) with Command R7B's vendored ``tool_use``
  template, and the reply is scored in Cohere's own call format. A template
  without tool support (Aya Expanse, and Tiny Aya's real ``default`` template)
  raises instead of dropping the tools; ``mode="prompt"`` works with it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import auditkit as ak  # noqa: E402
from auditkit.errors import CapabilityError, ExtraNotInstalled  # noqa: E402
from auditkit.model.hf_gen import HFGenModel  # noqa: E402
from tests.test_cohere_aya import SMALL, _ids, _tokenizer, cohere_dir  # noqa: E402,F401

TEMPLATES = Path(__file__).parent / "fixtures" / "chat_templates"
TOOLS = [{"type": "function", "function": {
    "name": "add", "description": "Add two integers.",
    "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                   "required": ["a", "b"]}}}]
EXPECTED = [{"name": "add", "arguments": {"a": 2, "b": 3}}]
# Command R7B reply as a skip_special_tokens decode leaves it.
COHERE_REPLY = 'I will add them.[{"tool_call_id": "0", "tool_name": "add", "parameters": {"a": 2, "b": 3}}]'


def _cohere2_dir(path: Path, template: str) -> Path:
    tok = _tokenizer()
    tok.chat_template = (TEMPLATES / template).read_text()
    cfg = transformers.Cohere2Config(**SMALL, **_ids(tok), sliding_window=64,
                                     layer_types=["sliding_attention", "full_attention"])
    torch.manual_seed(0)
    transformers.Cohere2ForCausalLM(cfg).save_pretrained(path)
    tok.save_pretrained(path)
    return path


class _Scripted:
    """Stands in for the loaded pipeline's __call__: records the rendered
    prompts and answers with a fixed reply (a random tiny model can't emit JSON)."""

    def __init__(self, pipe, reply: str) -> None:
        self.tokenizer, self.model, self.reply, self.prompts = pipe.tokenizer, pipe.model, reply, []

    def __call__(self, prompts, **kwargs):
        self.prompts += list(prompts)
        return [[{"generated_text": self.reply}] for _ in prompts]


def _tool_sample():
    return ak.Sample(input="what is 2 plus 3", tools=TOOLS, expected_tool_calls=EXPECTED)


# -- tools -------------------------------------------------------------------

def test_native_tools_reach_the_template_and_cohere_calls_score(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    d = _cohere2_dir(tmp_path / "r7b", "command_r7b_tool_use.jinja")
    m = HFGenModel(model=str(d), device="cpu")
    m._ensure_pipeline()
    m._pipeline = scripted = _Scripted(m._pipeline, COHERE_REPLY)
    r = ak.evaluate([_tool_sample()], m, [ak.ToolCallF1()], adapter=ak.ToolCallAdapter())
    (prompt,) = scripted.prompts
    assert '"name": "add"' in prompt and "Add two integers." in prompt  # R7B tool list
    assert r.failed_count == 0 and r.headline["tool_call_f1"] == 1.0


def test_native_tools_real_generate(tmp_path, monkeypatch):
    """The unscripted path: the tiny model really generates; nothing fails."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    d = _cohere2_dir(tmp_path / "r7b", "command_r7b_tool_use.jinja")
    r = ak.evaluate([_tool_sample()], HFGenModel(model=str(d), device="cpu"), [ak.ToolCallF1()],
                    adapter=ak.ToolCallAdapter(), config=ak.RunConfig(max_tokens=4))
    assert r.failed_count == 0 and len(r.predictions) == 1


def test_template_without_tools_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    d = _cohere2_dir(tmp_path / "aya_expanse_tpl", "aya_expanse.jinja")
    with pytest.raises(CapabilityError, match="mode='prompt'"):
        ak.evaluate([_tool_sample()], HFGenModel(model=str(d), device="cpu"), [ak.ToolCallF1()],
                    adapter=ak.ToolCallAdapter(), config=ak.RunConfig(max_tokens=4))


def test_real_tiny_aya_template_needs_prompt_mode(tmp_path, monkeypatch):
    """Tiny Aya's real template ignores ``tools``: native raises, prompt mode works."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    d = _cohere2_dir(tmp_path / "tiny_aya", "tiny_aya_default.jinja")
    with pytest.raises(CapabilityError, match="Tiny Aya"):
        ak.evaluate([_tool_sample()], HFGenModel(model=str(d), device="cpu"), [ak.ToolCallF1()],
                    adapter=ak.ToolCallAdapter(), config=ak.RunConfig(max_tokens=4))
    m = HFGenModel(model=str(d), device="cpu")
    m._ensure_pipeline()
    reply = '<tool_call>{"name": "add", "arguments": {"a": 2, "b": 3}}</tool_call>'
    m._pipeline = scripted = _Scripted(m._pipeline, reply)
    r = ak.evaluate([_tool_sample()], m, [ak.ToolCallF1()], adapter=ak.ToolCallAdapter(mode="prompt"))
    (prompt,) = scripted.prompts
    assert "# Developer Preamble" in prompt and '"add"' in prompt  # tools in the system turn
    assert r.failed_count == 0 and r.headline["tool_call_f1"] == 1.0


# -- adapter folders -----------------------------------------------------------

def _adapter_dir(base_dir: Path, out: Path, with_tokenizer: bool) -> Path:
    peft = pytest.importorskip("peft")
    base = transformers.AutoModelForCausalLM.from_pretrained(base_dir)
    torch.manual_seed(0)
    lora = peft.get_peft_model(base, peft.LoraConfig(
        r=4, target_modules=["q_proj", "v_proj"], init_lora_weights=False))
    lora.save_pretrained(out)
    if with_tokenizer:
        transformers.AutoTokenizer.from_pretrained(base_dir).save_pretrained(out)
    return out


@pytest.mark.parametrize("with_tokenizer", [True, False])
def test_adapter_dir_evaluates_with_the_adapter_applied(cohere_dir, tmp_path, monkeypatch, with_tokenizer):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    d, _ = cohere_dir
    adapter = _adapter_dir(d, tmp_path / "adapter", with_tokenizer)
    m = HFGenModel(model=str(adapter), device="cpu")
    r = ak.evaluate([ak.Sample(input="what is two plus two", target="a")], m, ["exact_match"],
                    config=ak.RunConfig(max_tokens=4))
    assert r.failed_count == 0 and len(r.predictions) == 1
    assert getattr(m._pipeline.model, "peft_config", None)  # LoRA loaded, not just the base


def test_adapter_dir_without_peft_says_so(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"peft_type": "LORA", "base_model_name_or_path": "x"}')
    monkeypatch.setitem(sys.modules, "peft", None)  # import peft -> ImportError
    with pytest.raises(ExtraNotInstalled, match="PEFT adapter folder.*peft"):
        ak.evaluate([ak.Sample(input="hi", target="a")], HFGenModel(model=str(adapter), device="cpu"),
                    ["exact_match"])
