"""#43 parts 1-2: request fields an OpenAI-compatible server may silently ignore.

A real local HTTP server plays SGLang, vLLM, a proxy or a bare server (its GET /models
`owned_by`); the fields are always sent, and only a *declared* server known to honour
them clears the note. Detection alone never does.
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("requests")

import auditkit as ak
from auditkit.model.api_gen import APIModel
from auditkit.runspec import RunConfig

W = lambda c: {"name": "get_weather", "arguments": {"city": c}}
TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]


@pytest.fixture
def server():
    """start(owned_by) -> (base_url, received bodies, GET /models count). owned_by=None: no /models."""
    servers = []

    def start(owned_by):
        seen, models_hits = [], [0]

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                models_hits[0] += 1
                if owned_by is None:
                    return self._send(404, {"error": "no such route"})
                self._send(200, {"object": "list", "data": [{"id": "m", "object": "model", "owned_by": owned_by}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                seen.append(body)
                user = body["messages"][-1]["content"]
                if body.get("tools") and "weather" in user.lower():
                    calls = [{"id": f"c{i}", "type": "function", "function": {
                        "name": "get_weather", "arguments": json.dumps({"city": c})}}
                        for i, c in enumerate(("Oslo", "Cairo")) if c in user]
                    msg = {"role": "assistant", "content": None, "tool_calls": calls}
                else:
                    msg = {"role": "assistant", "content": "Paris"}
                self._send(200, {"choices": [{"message": msg, "finish_reason": "stop"}],
                                 "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}})

        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return f"http://127.0.0.1:{srv.server_port}/v1", seen, models_hits

    yield start
    for s in servers:
        s.shutdown()


QA = [ak.Sample(id="q", input="Capital of France?", target="Paris")]
THINK_OFF = RunConfig(chat_template_kwargs={"enable_thinking": False})


@pytest.fixture(autouse=True)
def _no_cache(tmp_path, monkeypatch):   # a fresh result cache per test
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))


def run(model, samples=QA, **kw):
    return ak.evaluate(samples, model=model, scorers=kw.pop("scorers", ["exact_match"]),
                       config=kw.pop("config", THINK_OFF), **kw)


def test_unknown_server_gets_the_note_and_the_field_is_still_sent(server, caplog):
    base, seen, _ = server(None)
    with caplog.at_level(logging.WARNING, logger="auditkit.model.api_gen"):
        r = run(APIModel("m", api_base=base, api_key="x"))
    assert seen[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "unknown server" in r.metadata["unverified_request_fields"]["chat_template_kwargs"]
    assert r.metadata["api_server"] == {"kind": None, "source": "unknown", "owned_by": None}
    assert "chat_template_kwargs" in r.summary()
    assert sum("may have been ignored" in m for m in caplog.messages) == 1


@pytest.mark.parametrize("owned_by", ["sglang", "vllm"])
def test_a_probed_server_is_recorded_but_does_not_silence_the_note(server, owned_by):
    base, _, _ = server(owned_by)
    r = run(APIModel("m", api_base=base, api_key="x"))
    assert r.metadata["api_server"] == {"kind": owned_by, "source": "probed", "owned_by": owned_by}
    note = r.metadata["unverified_request_fields"]["chat_template_kwargs"]
    assert f"server={owned_by!r}" in note                 # the note tells the reader what to declare


@pytest.mark.parametrize("declared", ["sglang", "vllm", "SGLang"])
def test_a_declared_server_that_honours_the_field_clears_it_and_is_never_probed(server, declared):
    base, _, models_hits = server(None)
    r = run(APIModel("m", api_base=base, api_key="x", server=declared))
    assert "unverified_request_fields" not in r.metadata
    assert r.metadata["api_server"] == {"kind": declared.lower(), "source": "declared"}
    assert models_hits[0] == 0
    assert "not known to be honoured" not in r.summary()


def test_a_declared_server_not_in_the_table_keeps_the_note(server):
    base, _, _ = server(None)
    r = run(APIModel("m", api_base=base, api_key="x", server="tgi"))
    assert "declared server 'tgi'" in r.metadata["unverified_request_fields"]["chat_template_kwargs"]


def test_a_proxy_reporting_something_else_is_unknown(server):
    base, _, _ = server("openai")                         # e.g. LiteLLM proxy / gateway in front of vLLM
    r = run(APIModel("m", api_base=base, api_key="x"))
    assert r.metadata["api_server"]["kind"] is None
    assert "owned_by='openai'" in r.metadata["unverified_request_fields"]["chat_template_kwargs"]


def test_nothing_unverifiable_sent_means_no_probe_and_no_note(server):
    base, _, models_hits = server("vllm")
    r = run(APIModel("m", api_base=base, api_key="x"), config=RunConfig())
    assert models_hits[0] == 0
    assert "unverified_request_fields" not in r.metadata and "api_server" not in r.metadata


def test_the_note_is_per_run_and_the_probe_is_once_per_model(server):
    base, _, models_hits = server(None)
    m = APIModel("m", api_base=base, api_key="x")
    first = run(m)
    second = run(m, config=RunConfig())                  # same model object, no unverifiable field
    assert "unverified_request_fields" in first.metadata
    assert "unverified_request_fields" not in second.metadata
    assert models_hits[0] == 1


PARALLEL = [ak.Sample(id="p", input="Weather in Oslo and Cairo?", tools=TOOLS,
                      expected_tool_calls=[[W("Oslo"), W("Cairo")]])]


@pytest.mark.parametrize("arg_mode", ["exact", "subset"])      # subset names end in _subset
def test_parallel_cap_on_an_unverified_server_marks_every_parallel_score(server, arg_mode):
    base, seen, _ = server("sglang")                      # tool_choice auto: SGLang ignores false
    r = run(APIModel("m", api_base=base, api_key="x"), PARALLEL, config=RunConfig(),
            adapter=ak.ToolCallAdapter(parallel_tool_calls=False),
            scorers=[ak.ParallelToolCalls(arg_mode=arg_mode), ak.ToolCallF1()])
    assert not any("cap_unverified" in (d.get("metadata") or {})
                   for d in r.predictions[0].metadata["scores"] if d["name"].startswith("tool_call"))
    assert seen[0]["parallel_tool_calls"] is False
    docs = [d for d in r.predictions[0].metadata["scores"] if d["name"].startswith("parallel_")]
    assert docs and all(d["metadata"]["cap_unverified"] is True for d in docs)
    assert "not honoured by sglang" in r.metadata["unverified_request_fields"]["parallel_tool_calls"]


def test_parallel_cap_forced_on_declared_sglang_is_verified(server):
    base, _, _ = server(None)
    r = run(APIModel("m", api_base=base, api_key="x", server="sglang"), PARALLEL, config=RunConfig(),
            adapter=ak.ToolCallAdapter(parallel_tool_calls=False, tool_choice="required"),
            scorers=[ak.ParallelToolCalls()])
    docs = [d for d in r.predictions[0].metadata["scores"] if d["name"].startswith("parallel_")]
    assert docs and not any((d.get("metadata") or {}).get("cap_unverified") for d in docs)


def test_declaring_the_server_changes_the_fingerprint_only_when_declared():
    plain = APIModel("m", api_base="http://h/v1", api_key="x").identity()
    assert "server" not in plain
    assert APIModel("m", api_base="http://h/v1", api_key="x", server="vllm").identity()["server"] == "vllm"
