"""Native tool calling through APIModel, ToolCallAdapter and the Runner (no network)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import auditkit as ak
from auditkit.errors import CapabilityError
from auditkit.model import Generated, Model, Request, Result_
from auditkit.model.api_gen import APIModel
from auditkit.router import route_adapter
from auditkit.runspec import RunConfig
from auditkit.trace import to_turns
from auditkit.types import Capability

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "get_time", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
]
EXPECTED = [[{"name": "get_weather", "arguments": {"city": "Paris"}},
             {"name": "get_time", "arguments": {"city": "Paris"}}]]


def _model(response: dict, chat_template: bool = True) -> tuple[APIModel, MagicMock]:
    m = APIModel(api_key="test", chat_template=chat_template)
    session = MagicMock()
    session.post.return_value = SimpleNamespace(raise_for_status=lambda: None, json=lambda: response)
    m._session = session  # bypass the real requests.Session
    return m, session


def _tool_params():
    return {"messages": [{"role": "user", "content": "weather?"}], "tools": TOOLS,
            "tool_choice": "auto", "parallel_tool_calls": False}


def _oa_call(i, name, args):
    return {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class TestAPIModelTools:
    def test_forwards_tool_params_in_chat_mode(self):
        m, session = _model({"choices": [{"message": {"content": "hi"}}]})
        m.generate([Request(prompt="weather?", params=_tool_params())])
        body = session.post.call_args.kwargs["json"]
        assert body["tools"] == TOOLS
        assert body["tool_choice"] == "auto"
        assert body["parallel_tool_calls"] is False

    def test_no_tool_params_in_completion_mode(self):
        m, session = _model({"choices": [{"text": "hi"}]}, chat_template=False)
        m.generate([Request(prompt="weather?", params=_tool_params())])
        body = session.post.call_args.kwargs["json"]
        assert not {"tools", "tool_choice", "parallel_tool_calls"} & body.keys()

    def test_tool_calls_only_response(self):
        calls = [_oa_call(1, "get_weather", {"city": "Paris"}), _oa_call(2, "get_time", {"city": "Paris"})]
        m, _ = _model({"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": calls},
                                    "finish_reason": "tool_calls"}]})
        [res] = m.generate([Request(prompt="weather?", params=_tool_params())])
        g = res.completions[0]
        assert g.text == ""
        assert g.finish_reason == "tool_calls"
        turns = to_turns(g.trace["tool_calls"])
        assert len(turns) == 1 and [c.name for c in turns[0]] == ["get_weather", "get_time"]

    def test_finish_reason_both_modes(self):
        m, _ = _model({"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]})
        assert m.generate([Request(prompt="x")])[0].completions[0].finish_reason == "stop"
        m, _ = _model({"choices": [{"text": "hi", "finish_reason": "length"}]}, chat_template=False)
        g = m.generate([Request(prompt="x")])[0].completions[0]
        assert g.finish_reason == "length" and g.trace is None

    def test_tools_capability_only_in_chat_mode(self):
        assert APIModel(chat_template=True).supports(Capability.TOOLS)
        assert not APIModel(chat_template=False).supports(Capability.TOOLS)


def _sample(**kw):
    return ak.Sample(input="weather and time in Paris?", tools=TOOLS, expected_tool_calls=EXPECTED, **kw)


class TestRunnerAndAdapter:
    def test_native_tools_on_text_only_model_raises(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        with pytest.raises(CapabilityError, match="mode='prompt'"):
            ak.evaluate([_sample()], model=lambda prompts: ["" for _ in prompts],
                        scorers=[ak.ToolCallF1()], adapter=ak.ToolCallAdapter(mode="native"))

    def test_prompt_mode_with_text_model(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        seen = []

        def model(prompts):
            seen.extend(prompts)
            return ['<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>\n'
                    '<tool_call>{"name": "get_time", "arguments": {"city": "Paris"}}</tool_call>'] * len(prompts)

        result = ak.evaluate([_sample()], model=model, scorers=[ak.ToolCallF1(), ak.ParallelToolCalls()],
                             adapter=ak.ToolCallAdapter(mode="prompt"))
        assert "get_weather" in seen[0] and "<tool_call>" in seen[0]
        assert result.stats["parallel_detection"].mean == 1.0
        assert result.stats["tool_call_f1"].mean == 1.0

    def test_prompt_mode_merges_system_message(self):
        sample = _sample(metadata={"messages": [{"role": "system", "content": "You are a travel agent."},
                                                {"role": "user", "content": "weather and time in Paris?"}]})
        [req] = ak.ToolCallAdapter(mode="prompt").adapt(sample, RunConfig())
        messages = req.params["messages"]
        assert [m["role"] for m in messages] == ["system", "user"]
        assert messages[0]["content"].startswith("You are a travel agent.")
        assert '"get_time"' in messages[0]["content"]
        assert "tools" not in req.params

    def test_router_picks_tool_adapter(self):
        assert isinstance(route_adapter([_sample()]), ak.ToolCallAdapter)
        assert not isinstance(route_adapter([ak.Sample(input="q")]), ak.ToolCallAdapter)


class _ToolCallModel(Model):
    """A native tool-calling backend: returns no text, only a tool_calls trace
    (what api:/agent: do when the model *only* calls tools)."""

    name = "toolcaller"

    def capabilities(self):
        return {Capability.GENERATE, Capability.CHAT, Capability.TOOLS}

    def generate(self, requests):
        return [Result_(completions=[Generated(
            text="", finish_reason="tool_calls", trace={"tool_calls": EXPECTED})])
            for _ in requests]


def test_generate_preserves_native_trace_for_precomputed_scoring(monkeypatch, tmp_path):
    """ak.generate() must persist the trace into actual_trace, or the
    generate-then-precomputed flow scores every native tool call as missing."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    # Stage 1: generate. Content is "" (only tool calls) but a trace is produced.
    answers = ak.generate([_sample()], model=_ToolCallModel(), adapter=ak.ToolCallAdapter())
    assert answers[0].actual_output == ""
    assert answers[0].actual_trace["tool_calls"] == EXPECTED
    # Stage 2: score offline. Without the trace this was tool_call_f1 == 0.0.
    result = ak.evaluate(answers, model="precomputed", scorers=[ak.ToolCallF1()],
                         adapter=ak.ToolCallAdapter())
    assert result.stats["tool_call_f1"].mean == 1.0


def test_runner_warns_when_tools_are_not_offered(monkeypatch, tmp_path, caplog):
    import logging
    import auditkit as ak
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}]
    samples = [ak.Sample(input="hi", tools=tools, expected_tool_calls=[[{"name": "f", "arguments": {}}]])]
    with caplog.at_level(logging.WARNING, logger="auditkit.runner"):
        ak.evaluate(samples, model=lambda ps: ["plain answer" for _ in ps], scorers=[ak.ToolCallF1()])
    assert any("ToolCallAdapter" in r.getMessage() for r in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="auditkit.runner"):
        ak.evaluate(samples, model=lambda ps: ["plain answer 2" for _ in ps], scorers=[ak.ToolCallF1()],
                    adapter=ak.ToolCallAdapter(mode="prompt"))
    assert not any("ToolCallAdapter" in r.getMessage() for r in caplog.records)


# ---- per-response isolation + text coercion (edge-case findings) ----------
# A single malformed 200 must never crash generate() and abort the whole batch
# (runner._retry_generate retries the WHOLE batch on any raised exception); and
# non-string content (list parts / null) must be coerced, not passed through.

from auditkit.metric import ExactMatch


def _bad_json_resp(text=""):
    def _raise():
        raise ValueError("Expecting value: line 1 column 1 (char 0)")
    return SimpleNamespace(raise_for_status=lambda: None, json=_raise, text=text)


class TestAPIModelResponseIsolation:
    def test_empty_choices_array_is_error_not_crash(self):
        m, _ = _model({"choices": []})
        g = m.generate([Request(prompt="x")])[0].completions[0]
        assert g.text == "" and g.finish_reason == "error"

    def test_missing_choices_key_is_error(self):
        m, _ = _model({})
        g = m.generate([Request(prompt="x")])[0].completions[0]
        assert g.text == "" and g.finish_reason == "error"

    def test_non_json_200_body_is_error_not_crash(self):
        m = APIModel(api_key="test")
        session = MagicMock()
        session.post.return_value = _bad_json_resp("<html>gateway timeout</html>")
        m._session = session
        g = m.generate([Request(prompt="x")])[0].completions[0]
        assert g.text == "" and g.finish_reason == "error"

    def test_missing_message_in_chat_choice_is_error(self):
        m, _ = _model({"choices": [{"finish_reason": "stop"}]})
        g = m.generate([Request(prompt="x")])[0].completions[0]
        assert g.text == "" and g.finish_reason == "error"

    def test_one_bad_response_does_not_abort_the_batch(self, monkeypatch, tmp_path):
        """Task requirement: the run completes, the bad sample is an error, the
        others still score."""
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        m = APIModel(api_key="test")
        session = MagicMock()
        good = lambda: SimpleNamespace(raise_for_status=lambda: None,
                                       json=lambda: {"choices": [{"message": {"content": "a"},
                                                                  "finish_reason": "stop"}]})
        empty_choices = SimpleNamespace(raise_for_status=lambda: None, text="",
                                        json=lambda: {"choices": []})
        non_json = _bad_json_resp("<html>gateway timeout</html>")
        missing_message = SimpleNamespace(raise_for_status=lambda: None,
                                          json=lambda: {"choices": [{"finish_reason": "stop"}]})
        # good, empty-choices, non-JSON, missing-message, good -- all three
        # malformed kinds the task names, mid-batch.
        session.post.side_effect = [good(), empty_choices, non_json, missing_message, good()]
        m._session = session
        samples = [ak.Sample(input=f"q{i}", target="a") for i in range(5)]
        result = ak.evaluate(samples, model=m, scorers=[ExactMatch()])
        assert len(result.errors) == 3              # the three bad samples
        assert result.stats["exact_match"].mean == 1.0  # both good samples scored


class TestAPIModelTextCoercion:
    def test_list_content_parts_are_joined_to_str(self):
        m, _ = _model({"choices": [{"message": {"content": [
            {"type": "text", "text": "Hello "}, {"type": "text", "text": "world"}]},
            "finish_reason": "stop"}]})
        g = m.generate([Request(prompt="x")])[0].completions[0]
        assert g.text == "Hello world" and isinstance(g.text, str)

    def test_null_content_with_tool_calls_is_empty_str(self):
        calls = [_oa_call(1, "get_weather", {"city": "Paris"})]
        m, _ = _model({"choices": [{"message": {"content": None, "tool_calls": calls},
                                    "finish_reason": "tool_calls"}]})
        g = m.generate([Request(prompt="x", params=_tool_params())])[0].completions[0]
        assert g.text == "" and isinstance(g.text, str)
        assert g.finish_reason == "tool_calls" and g.trace is not None

    def test_null_content_without_tool_calls_is_empty_str_not_error(self):
        m, _ = _model({"choices": [{"message": {"content": None}, "finish_reason": "stop"}]})
        g = m.generate([Request(prompt="x")])[0].completions[0]
        assert g.text == "" and isinstance(g.text, str) and g.finish_reason == "stop"

    def test_null_completion_text_is_empty_str(self):
        m, _ = _model({"choices": [{"text": None, "finish_reason": "stop"}]}, chat_template=False)
        g = m.generate([Request(prompt="x")])[0].completions[0]
        assert g.text == "" and isinstance(g.text, str)


class TestAPIModelPerRequestFailures:
    """A failing request (HTTP 500/429, reset, bad redirect) fails only its own
    sample; 429/5xx/resets are retried per request, never the whole batch."""

    @pytest.fixture
    def api_server(self, monkeypatch):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        for var in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
            monkeypatch.delenv(var, raising=False)
        state = {"hits": {}, "mode": {}}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                key = body["messages"][-1]["content"]
                n = state["hits"][key] = state["hits"].get(key, 0) + 1
                mode = state["mode"].get(key)
                if mode == "rst_once" and n == 1:
                    import socket
                    import struct
                    self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                    self.close_connection = True
                    return
                if mode == "500_always" or (mode in ("500_once", "429_once") and n == 1):
                    code = 429 if mode == "429_once" else 500
                    self.send_response(code)
                    self.send_header("Content-Length", "4")
                    self.end_headers()
                    self.wfile.write(b"oops")
                    return
                if mode == "redirect_file":
                    self.send_response(307)
                    self.send_header("Location", "file:///etc/hosts")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                data = json.dumps({"choices": [{"message": {"content": key}, "finish_reason": "stop"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        state["base"] = f"http://127.0.0.1:{srv.server_address[1]}/v1"
        yield state
        srv.shutdown()
        srv.server_close()

    @pytest.mark.parametrize("mode,bad_hits,failed", [
        ("500_always", 4, 1), ("500_once", 2, 0), ("429_once", 2, 0), ("rst_once", 2, 0),
        ("redirect_file", 1, 1),
    ])
    def test_one_failing_request_is_isolated(self, api_server, monkeypatch, tmp_path, mode, bad_hits, failed):
        pytest.importorskip("requests")
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        api_server["mode"]["k03"] = mode
        m = APIModel(model="stub", api_base=api_server["base"], api_key="test")
        samples = [ak.Sample(input=f"k{i:02d}", id=f"k{i:02d}", target=f"k{i:02d}") for i in range(6)]
        r = ak.evaluate(samples, model=m, scorers=[ExactMatch()],
                        config=RunConfig(max_retries=3, retry_delay=0.01))
        # Every other request is sent once; only k03 is retried (and only when transient).
        assert api_server["hits"] == {**{f"k{i:02d}": 1 for i in range(6)}, "k03": bad_hits}
        assert r.failed_count == failed
        assert r.stats["exact_match"].count == 6 - failed
        if failed:
            assert r.errors[0]["sample_id"] == "k03"

    def test_bool_tool_calls_is_malformed_not_no_calls(self):
        m, _ = _model({"choices": [{"message": {"content": "hi", "tool_calls": False}}]})
        g = m.generate([Request(prompt="x")])[0].completions[0]
        assert g.finish_reason == "error" and "tool_calls must be a list" in g.error
