"""Regression tests for G13 and G14 (fix plan #15): chat-template kwargs from RunConfig,
and a normalised f1_score."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import auditkit as ak
from auditkit import RunConfig, Sample
from auditkit.metrics.code import F1Score
from auditkit.model import CallableModel


# -- G13: RunConfig.chat_template_kwargs --------------------------------------------------------------------

class _Recorder(CallableModel):
    def __init__(self):
        self.params = []
        super().__init__(lambda ps: ["ok"] * len(ps))

    def generate(self, requests):
        self.params.extend(r.params for r in requests)
        return super().generate(requests)


def test_g13_run_config_kwargs_reach_every_request(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    m = _Recorder()
    ak.evaluate([Sample(input="a", target="ok"), Sample(input="b", target="ok")], model=m, scorers=["exact_match"],
                config=RunConfig(chat_template_kwargs={"enable_thinking": False}))
    assert [p.get("chat_template_kwargs") for p in m.params] == [{"enable_thinking": False}] * 2


def test_g13_without_the_field_requests_are_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    m = _Recorder()
    ak.evaluate([Sample(input="a", target="ok")], model=m, scorers=["exact_match"])
    assert "chat_template_kwargs" not in m.params[0]


def test_g13_the_kwargs_are_part_of_the_fingerprint():
    assert RunConfig().to_dict() != RunConfig(chat_template_kwargs={"enable_thinking": False}).to_dict()
    assert RunConfig.from_dict(RunConfig(chat_template_kwargs={"x": 1}).to_dict()).chat_template_kwargs == {"x": 1}


def test_g13_the_hf_backend_renders_them(monkeypatch, tmp_path):
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    except Exception:
        pytest.skip("gpt2 tokenizer not cached")
    tok.chat_template = "{% if enable_thinking is defined and not enable_thinking %}NOTHINK {% endif %}" \
                        "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    from auditkit.model.hf_gen import HFGenModel
    seen = []

    class Pipe:
        tokenizer = tok

        def __call__(self, prompts, **kw):
            seen.extend(prompts)
            return [[{"generated_text": "ok"}] for _ in prompts]

    # a fresh run cache (after the tokenizer loaded from the HF cache): a cached result skips the pipeline
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    m = HFGenModel("stub")
    m._pipeline = Pipe()
    monkeypatch.setattr(HFGenModel, "_load", lambda self: None, raising=False)
    ak.evaluate([Sample(input="hi", target="ok")], model=m, scorers=["exact_match"],
                config=RunConfig(chat_template_kwargs={"enable_thinking": False}))
    assert seen and seen[0].startswith("NOTHINK")


@pytest.fixture
def api_server(monkeypatch):
    for var in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)
    bodies = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            bodies.append(json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)))))
            data = json.dumps({"choices": [{"message": {"role": "assistant", "content": "ok"},
                                            "finish_reason": "stop"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1", bodies
    srv.shutdown()
    srv.server_close()


def test_g13_the_api_backend_sends_them_only_when_set(api_server, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    base, bodies = api_server
    from auditkit.model.api_gen import APIModel
    for cfg in (RunConfig(), RunConfig(chat_template_kwargs={"enable_thinking": False})):
        ak.evaluate([Sample(input="hi", target="ok")], model=APIModel("m", api_base=base, api_key="x"),
                    scorers=["exact_match"], config=cfg)
    assert "chat_template_kwargs" not in bodies[0]
    assert bodies[1]["chat_template_kwargs"] == {"enable_thinking": False}


# -- G14: f1_score normalises ------------------------------------------------------------------------------------

@pytest.mark.parametrize("out,target", [("Red", "red"), ("Triangle.", "triangle"), ("The Eiffel Tower", "eiffel tower")])
def test_g14_case_punctuation_and_articles_do_not_count(out, target):
    assert F1Score().score(Sample(input="q", target=target), out).value == 1.0


def test_g14_normalize_false_keeps_the_raw_split():
    m = F1Score(normalize=False)
    assert m.score(Sample(input="q", target="red"), "Red").value == 0.0
    assert m.score(Sample(input="q", target="red"), "red").value == 1.0


def test_g14_the_setting_is_in_the_metric_identity():
    assert F1Score().identity() != F1Score(normalize=False).identity()
    assert F1Score().name == "f1_score"
