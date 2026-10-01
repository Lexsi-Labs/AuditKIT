"""AgentEndpointModel against a real local HTTP server (no external network)."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import auditkit as ak
from auditkit.adapter import GenerationAdapter
from auditkit.errors import AuditKitError, ModelError
from auditkit.metric import ExactMatch
from auditkit.model import AutoModel, Request
from auditkit.model.agent_endpoint import AgentEndpointModel
from auditkit.runspec import RunSpec
from auditkit.scenario import ListScenario
from auditkit.trace import to_turns
from auditkit.types import Capability

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "get_time", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
]


def _oa_call(i, name, args):
    return {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def _agent_run(city, parallel=True, contexts=("d1", "d3")):
    """What a LangGraph-style agent returns: answer, transcript, retrieved docs."""
    weather, time_ = _oa_call(1, "get_weather", {"city": city}), _oa_call(2, "get_time", {"city": city})
    calls = [[weather, time_]] if parallel else [[weather]]
    messages = [{"role": "user", "content": f"weather and time in {city}?"}]
    for turn in calls:
        messages.append({"role": "assistant", "content": None, "tool_calls": turn})
        messages += [{"role": "tool", "tool_call_id": c["id"], "content": "ok"} for c in turn]
    messages.append({"role": "assistant", "content": f"sunny in {city}"})
    return {"output": f"sunny in {city}", "messages": messages, "contexts": list(contexts),
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}


@pytest.fixture
def server(monkeypatch):
    for var in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)
    state = {"respond": lambda body: (200, {"output": "ok"}), "received": [], "inflight": 0, "max_inflight": 0}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            with lock:
                state["received"].append({"path": self.path, "auth": self.headers.get("Authorization"),
                                          "body": body})
                state["inflight"] += 1
                state["max_inflight"] = max(state["max_inflight"], state["inflight"])
            try:
                status, payload = state["respond"](body)
            finally:
                with lock:
                    state["inflight"] -= 1
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{srv.server_address[1]}/run"
    yield state
    srv.shutdown()
    srv.server_close()


def _req(**params):
    return Request(prompt="User: weather and time in Paris?", request_type="chat", params=params)


class TestContract:
    def test_default_round_trip(self, server):
        calls = [[{"name": "get_weather", "arguments": {"city": "Paris"}}]]
        server["respond"] = lambda body: (200, {**_agent_run("Paris"), "tool_calls": calls})
        messages = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "weather and time in Paris?"}]
        m = AgentEndpointModel(url=server["url"])
        [res] = m.generate([_req(messages=messages, tools=TOOLS, temperature=0.3)])

        body = server["received"][0]["body"]
        assert body == {"input": "weather and time in Paris?", "messages": messages, "tools": TOOLS}
        g = res.completions[0]
        assert g.text == "sunny in Paris"
        assert set(g.trace) == {"messages", "tool_calls", "retrieved_contexts"}
        assert g.trace["tool_calls"] == calls
        assert g.trace["retrieved_contexts"] == ["d1", "d3"]
        assert [[c.name for c in t] for t in to_turns(g.trace["messages"])] == [["get_weather", "get_time"]]
        assert res.usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        assert res.latency_ms > 0

    def test_plain_prompt_extra_body_no_tools(self, server):
        [res] = AgentEndpointModel(server["url"], extra_body={"thread_id": "t1"}).generate([Request(prompt="hi")])
        assert server["received"][0]["body"] == {"input": "hi", "messages": [{"role": "user", "content": "hi"}],
                                                 "thread_id": "t1"}
        # G6: no transcript and no tool_calls -> tool use is unobservable, tools offered or not
        assert res.text == "ok" and res.completions[0].trace == {"tool_calls_unavailable": True} and res.usage == {}

    def test_openai_chat_completion_with_parallel_calls(self, server):
        c1, c2 = _oa_call(1, "get_weather", {"city": "Paris"}), _oa_call(2, "get_time", {"city": "Paris"})
        server["respond"] = lambda body: (200, {
            "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [c1, c2]},
                         "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}})
        [res] = AgentEndpointModel(server["url"]).generate([_req()])
        g = res.completions[0]
        assert g.text == ""
        assert g.finish_reason == "tool_calls"
        assert g.trace["tool_calls"] == [[c1, c2]]
        turns = to_turns(g.trace["tool_calls"])
        assert len(turns) == 1 and [c.arguments for c in turns[0]] == [{"city": "Paris"}] * 2
        assert res.usage["total_tokens"] == 7

    def test_dotted_paths_with_list_index(self, server):
        transcript = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "final"}]
        server["respond"] = lambda body: (200, {"data": {"answer": {"value": 42}, "steps": [{"log": transcript}]},
                                                "result": {"docs": ["a", "b"]}})
        m = AgentEndpointModel(server["url"], input_key="question", output_path="data.answer",
                               messages_path="data.steps.0.log", contexts_path="result.docs",
                               tool_calls_path="data.steps.5.calls")
        [res] = m.generate([Request(prompt="q")])
        assert server["received"][0]["body"]["question"] == "q"
        g = res.completions[0]
        assert g.text == '{"value": 42}'  # non-str output is JSON-encoded
        assert g.trace == {"messages": transcript, "retrieved_contexts": ["a", "b"]}  # missing path skipped

    def test_messages_without_output_uses_last_assistant_message(self, server):
        transcript = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "first"},
                      {"role": "tool", "content": "r"}, {"role": "assistant", "content": "the answer"}]
        server["respond"] = lambda body: (200, {"messages": transcript})
        [res] = AgentEndpointModel(server["url"]).generate([Request(prompt="q")])
        assert res.text == "the answer"

    def test_request_fn_and_response_fn(self, server):
        server["respond"] = lambda body: (200, {"ans": body["q"].upper(), "docs": ["x"]})
        m = AgentEndpointModel(server["url"], request_fn=lambda r: {"q": r.prompt},
                               response_fn=lambda d: {"output": d["ans"], "retrieved_contexts": d["docs"]})
        [res] = m.generate([Request(prompt="hello")])
        assert server["received"][0]["body"] == {"q": "hello"}
        assert res.text == "HELLO"
        assert res.completions[0].trace == {"retrieved_contexts": ["x"], "tool_calls_unavailable": True}   # G6


class TestAuth:
    def test_explicit_api_key(self, server, monkeypatch):
        monkeypatch.setenv("AGENT_API_KEY", "from-env")
        AgentEndpointModel(server["url"], api_key="sek").generate([Request(prompt="hi")])
        assert server["received"][0]["auth"] == "Bearer sek"

    def test_agent_api_key_env(self, server, monkeypatch):
        monkeypatch.setenv("AGENT_API_KEY", "from-env")
        AgentEndpointModel(server["url"]).generate([Request(prompt="hi")])
        assert server["received"][0]["auth"] == "Bearer from-env"

    def test_never_sends_openai_key(self, server, monkeypatch):
        monkeypatch.delenv("AGENT_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
        monkeypatch.setenv("API_KEY", "also-secret")
        AgentEndpointModel(server["url"]).generate([Request(prompt="hi")])
        assert server["received"][0]["auth"] is None

    def test_custom_headers_override(self, server):
        AgentEndpointModel(server["url"], api_key="a", headers={"Authorization": "Token b"}).generate(
            [Request(prompt="hi")])
        assert server["received"][0]["auth"] == "Token b"


class TestErrors:
    # generate() no longer raises per request (a raise makes the Runner replay
    # the whole batch and re-run every earlier agent's side-effecting tools); a
    # failed request surfaces as a finish_reason="error" result and a log line.
    def test_http_error_surfaces_as_failed_result(self, server, caplog):
        import logging
        server["respond"] = lambda body: (500, b"boom: vector store down")
        with caplog.at_level(logging.WARNING, logger="auditkit.model.agent_endpoint"):
            [res] = AgentEndpointModel(server["url"]).generate([Request(prompt="hi")])
        g = res.completions[0]
        assert g.text == "" and g.finish_reason == "error"
        msg = " ".join(r.getMessage() for r in caplog.records)
        assert "HTTP 500" in msg and "boom: vector store down" in msg

    def test_non_json_surfaces_as_failed_result(self, server, caplog):
        import logging
        server["respond"] = lambda body: (200, b"<html>" + b"x" * 1000)
        with caplog.at_level(logging.WARNING, logger="auditkit.model.agent_endpoint"):
            [res] = AgentEndpointModel(server["url"]).generate([Request(prompt="hi")])
        assert res.completions[0].finish_reason == "error"
        msg = " ".join(r.getMessage() for r in caplog.records)
        assert "did not return JSON" in msg and "x" * 400 not in msg  # body truncated to 300

    def test_rejects_non_http_scheme(self):
        for bad in ("file:///etc/hosts", "ftp://host/x", "data:text/plain,hi"):
            with pytest.raises(AuditKitError, match="http"):
                AgentEndpointModel(bad)

    def test_constructor_validation(self):
        with pytest.raises(AuditKitError, match="URL"):
            AgentEndpointModel()
        with pytest.raises(AuditKitError, match="RunConfig"):
            AgentEndpointModel("http://x", temperature=0.2)
        with pytest.raises(AuditKitError, match="unknown"):
            AgentEndpointModel("http://x", bogus=1)


class TestResolveAndIdentity:
    def test_automodel_resolve(self, server):
        spec = f"agent:{server['url']}"
        m = AutoModel.resolve(spec)
        assert isinstance(m, AgentEndpointModel)
        assert m.url == server["url"] and m.name == spec
        assert m.supports(Capability.TOOLS) and m.threadsafe
        assert m.model_info()["model_name"] == server["url"]
        assert m.generate([Request(prompt="hi")])[0].text == "ok"

    def test_fingerprint_tracks_response_contract(self):
        def fp(**kw):
            return RunSpec(scenario=ListScenario([ak.Sample(input="q", target="a")]),
                           model=AgentEndpointModel("http://x/run", **kw),
                           adapter=GenerationAdapter(), metrics=[ExactMatch()]).fingerprint()

        assert fp() == fp()
        assert fp() != fp(output_path="data.answer")
        assert fp() != fp(response_fn=lambda d: d)

    def test_two_different_response_fns_do_not_collide(self):
        # __qualname__ is "<lambda>" for both, so identity() used to match and
        # the disk cache replayed a stale result for a fixed-then-rerun fn.
        def fp(fn):
            return RunSpec(scenario=ListScenario([ak.Sample(input="q", target="a")]),
                           model=AgentEndpointModel("http://x/run", response_fn=fn),
                           adapter=GenerationAdapter(), metrics=[ExactMatch()]).fingerprint()

        assert fp(lambda d: {"output": d["result"]["draft"]}) != \
               fp(lambda d: {"output": d["result"]["answer"]})

    def test_identity_distinguishes_credentials(self):
        # Same URL, different api_key -> different identity, so one principal's
        # cached agent answers are not served to another.
        alice = AgentEndpointModel("http://x/run", api_key="KEY-ALICE").identity()
        bob = AgentEndpointModel("http://x/run", api_key="KEY-BOB").identity()
        assert alice["headers_digest"] != bob["headers_digest"]


def test_refused_redirect_does_not_leak_auth_token(monkeypatch):
    """A 302 from the agent host must NOT resend the Bearer token to the redirect
    target. urllib copies headers cross-host (and downgrades POST->GET), so the
    default opener leaks the key as a GET to the target; _NoRedirect refuses it.

    The collector records EVERY method: on the old (leaking) code the redirected
    POST arrives as a GET carrying Authorization, so `seen` would be non-empty --
    this is what makes the test a genuine regression, not a no-op.
    """
    for var in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("AGENT_API_KEY", raising=False)
    seen = []

    class Collector(BaseHTTPRequestHandler):
        def _record(self):
            seen.append({"method": self.command, "auth": self.headers.get("Authorization")})
            body = b'{"output": "attacker-chosen"}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_GET = _record
        do_POST = _record

        def log_message(self, *args):
            pass

    class Redirector(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(302)
            self.send_header("Location", collector_url)
            self.end_headers()

        def log_message(self, *args):
            pass

    collector = ThreadingHTTPServer(("127.0.0.1", 0), Collector)
    threading.Thread(target=collector.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    collector_url = f"http://127.0.0.1:{collector.server_address[1]}/collect"
    redirector = ThreadingHTTPServer(("127.0.0.1", 0), Redirector)
    threading.Thread(target=redirector.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    a_url = f"http://127.0.0.1:{redirector.server_address[1]}/run"
    try:
        [res] = AgentEndpointModel(a_url, api_key="sk-agent-secret").generate([Request(prompt="hi")])
        assert res.completions[0].finish_reason == "error"  # redirect refused -> HTTPError -> failed result
        assert seen == []                                    # target never reached, token never sent
    finally:
        redirector.shutdown(); redirector.server_close()
        collector.shutdown(); collector.server_close()


def _samples():
    parallel = [[{"name": "get_weather", "arguments": {"city": "Paris"}},
                 {"name": "get_time", "arguments": {"city": "Paris"}}]]
    oslo = [[{"name": "get_weather", "arguments": {"city": "Oslo"}},
             {"name": "get_time", "arguments": {"city": "Oslo"}}]]
    return [
        ak.Sample(input="weather and time in Paris?", id="paris", tools=TOOLS,
                  expected_tool_calls=parallel, reference_contexts=["d1", "d2"]),
        ak.Sample(input="weather and time in Oslo?", id="oslo", tools=TOOLS,
                  expected_tool_calls=oslo, reference_contexts=["d2"]),
    ]


def _respond(body):
    # Paris: both calls in one turn, retrieved [d1, d3]. Oslo: only the
    # weather call, retrieved [d3, d2].
    if "Paris" in body["input"]:
        return 200, _agent_run("Paris")
    return 200, _agent_run("Oslo", parallel=False, contexts=("d3", "d2"))


class TestEvaluate:
    def test_end_to_end_agent_metrics(self, server, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        server["respond"] = _respond
        result = ak.evaluate(_samples(), model=f"agent:{server['url']}",
                             scorers=[ak.ToolCallF1(), ak.ParallelToolCalls(), ak.RetrievalMetrics()],
                             adapter=ak.ToolCallAdapter())
        assert all(r["body"]["tools"] == TOOLS for r in server["received"])
        s = {k: v.mean for k, v in result.stats.items()}
        # Paris: F1 1, parallel ok. Oslo: 1 of 2 calls (P=1, R=.5, F1=2/3), no batching.
        assert s["tool_call_f1"] == pytest.approx((1 + 2 / 3) / 2)
        assert s["tool_call_recall"] == pytest.approx(0.75)
        assert s["parallel_detection"] == pytest.approx(0.5)
        assert s["parallel_recall"] == pytest.approx(0.5)
        # Paris: gold {d1,d2}, got [d1,d3] -> mrr 1, recall .5. Oslo: gold {d2}, got [d3,d2] -> mrr .5, recall 1.
        assert s["mrr"] == pytest.approx(0.75)
        assert s["recall"] == pytest.approx(0.75)
        assert s["precision"] == pytest.approx(0.5)
        paris = next(p for p in result.predictions if p.sample_id == "paris")
        assert paris.raw_output == "sunny in Paris"

    def test_concurrency(self, server, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

        def slow_echo(body):
            time.sleep(0.1)
            return 200, {"output": body["input"]}

        server["respond"] = slow_echo
        samples = [ak.Sample(input=f"q{i}", target=f"q{i}") for i in range(8)]
        result = ak.evaluate(samples, model=AgentEndpointModel(server["url"]), scorers=[ExactMatch()],
                             config=ak.RunConfig(concurrency=4))
        assert result.stats["exact_match"].mean == 1.0
        assert len(server["received"]) == 8
        assert server["max_inflight"] > 1


class TestStructuredOutput:
    def test_dict_answer_is_not_a_phantom_tool_call(self, server):
        # A structured (dict) answer with no transcript must not be text-parsed
        # into a phantom ToolCall(name='answer'); mark an explicit empty turn.
        server["respond"] = lambda body: (200, {"output": {"answer": {"city": "Paris", "temp": 20}}})
        [res] = AgentEndpointModel(server["url"]).generate([Request(prompt="weather in Paris?")])
        g = res.completions[0]
        assert g.text == '{"answer": {"city": "Paris", "temp": 20}}'
        # no phantom call (explicit empty turn), and tool use still unobservable (G6)
        assert g.trace == {"tool_calls": [], "tool_calls_unavailable": True}

    def test_unobservable_tool_use_is_ineligible_not_scored(self, server, monkeypatch, tmp_path):
        # Tools were offered but the agent returned only an answer (no transcript,
        # no explicit tool_calls), so its tool use is UNOBSERVABLE. The reference
        # tool metric is ineligible, NOT scored 1.0 as observed-zero-calls
        # (finding 5) -- for either an irrelevance (expected=[]) or a tool task.
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        server["respond"] = lambda body: (200, {"output": {"answer": {"city": "Paris"}}})
        for expected in ([], [[{"name": "get_weather", "arguments": {"city": "Paris"}}]]):
            sample = ak.Sample(input="weather in Paris?", tools=TOOLS, expected_tool_calls=expected)
            result = ak.evaluate([sample], model=f"agent:{server['url']}",
                                 scorers=[ak.ToolCallF1()], adapter=ak.ToolCallAdapter())
            assert result.errors == []
            assert "tool_call_f1" not in result.stats  # ineligible, not a fake 1.0 or 0.0

    def test_plain_answer_with_tools_marks_coverage_unavailable(self, server):
        # A plain-string answer with tools offered but no tool info: coverage
        # unavailable, and no placeholder tool_calls (so agent_eval reads it as
        # "no trace", not "zero calls observed").
        server["respond"] = lambda body: (200, {"output": "I can't do that."})
        m = AgentEndpointModel(url=server["url"])
        [res] = m.generate([_req(tools=TOOLS)])
        assert res.completions[0].trace == {"tool_calls_unavailable": True}

    def test_openai_final_message_with_tools_marks_coverage_unavailable(self, server):
        # A post-loop chat-completion reply (content, no tool_calls) with tools
        # offered is also unobservable, not observed-zero.
        server["respond"] = lambda body: (200, {"choices": [{"message": {
            "role": "assistant", "content": "done"}, "finish_reason": "stop"}]})
        [res] = AgentEndpointModel(server["url"]).generate([_req(tools=TOOLS)])
        assert res.completions[0].text == "done"
        assert res.completions[0].trace == {"tool_calls_unavailable": True}


class TestBatchResilience:
    def test_mid_batch_failure_does_not_replay_earlier_requests(self, server, monkeypatch, tmp_path):
        """A failed request must not make the Runner resend the whole batch,
        which would re-run every earlier agent's side-effecting tools."""
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        posts: dict[str, int] = {}

        def respond(body):
            inp = body["input"]
            posts[inp] = posts.get(inp, 0) + 1
            return (503, b"busy") if inp == "task-3" else (200, {"output": inp})

        server["respond"] = respond
        samples = [ak.Sample(input=f"task-{i}", id=f"task-{i}", target=f"task-{i}") for i in range(1, 5)]
        result = ak.evaluate(samples, model=f"agent:{server['url']}", scorers=[ExactMatch()],
                             config=ak.RunConfig(retry_delay=0.01))
        # Every other endpoint POSTed exactly once -- no batch replay on
        # task-3's failure. task-3 itself (a 503) is retried per request,
        # 1 + max_retries (3) times.
        assert {k: posts[k] for k in posts} == {"task-1": 1, "task-2": 1, "task-3": 4, "task-4": 1}
        assert "HTTP 503" in result.errors[0]["error"]  # the real reason, not a max_tokens hint
        # task-3 is a recorded failure (finish_reason='error' guard), not scored 0/1.
        assert result.failed_count == 1
        assert [e["sample_id"] for e in result.errors] == ["task-3"]

    def test_response_fn_change_busts_cache(self, server, monkeypatch, tmp_path):
        """Two runs against one URL with different response_fns must not share a
        cache entry (they used to collide under __qualname__ '<lambda>')."""
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        server["respond"] = lambda body: (200, {"result": {"draft": "Lyon", "answer": "Paris"}})
        samples = [ak.Sample(input="capital of France?", target="Paris")]
        r1 = ak.evaluate(samples, model=AgentEndpointModel(
            server["url"], response_fn=lambda d: {"output": d["result"]["draft"]}), scorers=[ExactMatch()])
        r2 = ak.evaluate(samples, model=AgentEndpointModel(
            server["url"], response_fn=lambda d: {"output": d["result"]["answer"]}), scorers=[ExactMatch()])
        assert len(server["received"]) == 2          # 2nd run re-called the server, not the cache
        assert r1.stats["exact_match"].mean == 0.0    # "Lyon" != "Paris"
        assert r2.stats["exact_match"].mean == 1.0    # "Paris"


class TestPerRequestFailures:
    """Any failure in one reply fails only that sample: nothing escapes
    generate(), so the runner never replays the batch."""

    @pytest.mark.parametrize("bad", [
        b'{"choices": "x"}', b'{"choices": [null]}', b'{"choices": [{"message": "s"}]}',
        b'{"output": null, "messages": 5}', b"[" * 200000 + b"]" * 200000,
    ])
    def test_malformed_reply_fails_only_its_sample(self, server, monkeypatch, tmp_path, bad):
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        posts: dict[str, int] = {}

        def respond(body):
            posts[body["input"]] = posts.get(body["input"], 0) + 1
            return (200, bad) if body["input"] == "bad" else (200, {"output": body["input"]})

        server["respond"] = respond
        samples = [ak.Sample(input=x, id=x, target=x) for x in ("good-1", "bad", "good-2")]
        r = ak.evaluate(samples, model=f"agent:{server['url']}", scorers=[ExactMatch()],
                        config=ak.RunConfig(retry_delay=0.01))
        assert posts == {"good-1": 1, "bad": 1, "good-2": 1}
        assert r.failed_count == 1 and [e["sample_id"] for e in r.errors] == ["bad"]
        assert r.stats["exact_match"].count == 2

    @pytest.mark.parametrize("exc", [
        __import__("http.client").client.IncompleteRead(b"x" * 14, 86),
        __import__("http.client").client.BadStatusLine("HELLO THERE"),
    ])
    def test_http_client_errors_fail_only_that_request(self, monkeypatch, exc):
        m = AgentEndpointModel("http://127.0.0.1:1/run")
        calls = iter([exc, {"output": "ok"}])

        def fake_post(body):
            v = next(calls)
            if isinstance(v, Exception):
                raise v
            return v

        monkeypatch.setattr(m, "_post", fake_post)
        bad, good = m.generate([_req(), _req()])
        assert bad.completions[0].finish_reason == "error"
        assert type(exc).__name__ in bad.completions[0].error
        assert good.text == "ok"

    def test_transient_429_is_retried_per_request_but_400_is_not(self, server, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        posts: dict[str, int] = {}

        def respond(body):
            inp = body["input"]
            posts[inp] = posts.get(inp, 0) + 1
            if inp == "flaky" and posts[inp] == 1:
                return 429, b"slow down"
            if inp == "rejected":
                return 400, b"bad request"
            return 200, {"output": inp}

        server["respond"] = respond
        samples = [ak.Sample(input=x, id=x, target=x) for x in ("ok", "flaky", "rejected")]
        r = ak.evaluate(samples, model=f"agent:{server['url']}", scorers=[ExactMatch()],
                        config=ak.RunConfig(max_retries=3, retry_delay=0.01))
        assert posts == {"ok": 1, "flaky": 2, "rejected": 1}
        assert r.failed_count == 1
        assert r.errors[0]["sample_id"] == "rejected" and "HTTP 400" in r.errors[0]["error"]

    def test_runconfig_timeout_bounds_one_request_not_a_chunk(self, server, monkeypatch, tmp_path):
        """6 x 0.2s requests fit a 0.5s per-request timeout (a chunk-wide budget
        would have raised ModelTimeout); a 2s one fails alone, and no request
        is still being sent after evaluate() returns."""
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

        def respond(body):
            time.sleep(1.5 if body["input"] == "slow" else 0.2)
            return 200, {"output": body["input"]}

        server["respond"] = respond
        ids = ["a", "b", "slow", "c", "d", "e"]
        samples = [ak.Sample(input=x, id=x, target=x) for x in ids]
        r = ak.evaluate(samples, model=f"agent:{server['url']}", scorers=[ExactMatch()],
                        config=ak.RunConfig(timeout=0.5, max_retries=3, retry_delay=0.01))
        assert r.failed_count == 1 and r.errors[0]["sample_id"] == "slow"
        assert "timed out" in r.errors[0]["error"]
        assert r.stats["exact_match"].count == 5
        sent = len(server["received"])
        assert sent == 6  # the timed-out request was not re-sent
        time.sleep(0.3)
        assert len(server["received"]) == sent

    @pytest.mark.parametrize("bad", [5, False])
    def test_non_list_tool_calls_is_malformed_not_no_calls(self, server, monkeypatch, tmp_path, bad):
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        server["respond"] = lambda body: (200, {"output": "no tools needed", "tool_calls": bad})
        sample = ak.Sample(input="hi", tools=TOOLS, expected_tool_calls=[])
        r = ak.evaluate([sample], model=f"agent:{server['url']}", scorers=[ak.ToolCallF1()],
                        adapter=ak.ToolCallAdapter())
        assert r.failed_count == 1 and "tool_calls must be a list" in r.errors[0]["error"]
        assert "tool_call_f1" not in r.stats  # not scored as a passing "no calls"

    def test_context_dicts_map_to_their_text(self, server):
        server["respond"] = lambda body: (200, {"output": "x", "contexts": [
            {"page_content": "A", "metadata": {"src": 1}}, {"text": "B"}, {"content": "C"}, "D"]})
        res = AgentEndpointModel(server["url"]).generate([_req()])[0]
        assert res.completions[0].trace["retrieved_contexts"] == ["A", "B", "C", "D"]
        server["respond"] = lambda body: (200, {"output": "x", "contexts": [{"id": 3}]})
        res = AgentEndpointModel(server["url"]).generate([_req()])[0]
        assert res.completions[0].finish_reason == "error"  # unreadable, not str(dict)

    def test_string_or_non_finite_usage_is_coerced(self, server, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        usages = iter([{"prompt_tokens": "5", "completion_tokens": "1", "total_tokens": "6"},
                       {"prompt_tokens": float("nan"), "completion_tokens": float("inf"), "total_tokens": 4}])
        server["respond"] = lambda body: (200, json.dumps(
            {"output": body["input"], "usage": next(usages)}).encode())
        samples = [ak.Sample(input=x, id=x, target=x) for x in ("a", "b")]
        r = ak.evaluate(samples, model=f"agent:{server['url']}", scorers=[ExactMatch()])
        assert r.failed_count == 0 and r.stats["exact_match"].mean == 1.0
        assert r.token_usage == {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 10}
        json.dumps(r.to_dict(), allow_nan=False)  # strict JSON
