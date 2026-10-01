"""SGLang integration: AuditKIT evaluating a model that SGLang serves (server mode, `api:`).

SGLang serves an OpenAI-compatible API and AuditKIT reaches it with `api:` (see docs/SGLANG.md). The offline
tests run a local server that answers exactly as SGLang's OpenAI server does. Payloads follow sglang 0.5.20's
own response models (`sglang/srt/entrypoints/openai/protocol.py`):
- ChatCompletionResponse / ChatCompletionResponseChoice / ChatMessage / ToolCall / FunctionResponse /
  UsageInfo / ErrorResponse;
- `serving_chat._process_tool_calls`: ids `call_<24 hex>`; an unparseable call is dropped and its raw
  markup returned as `content`;
- `serving_base.create_error_response`: a flat `{"object": "error", ...}` body.

The live tests at the end run against a real SGLang server and are skipped unless SGLANG_BASE_URL is set, e.g.
on Colab after `examples/cohere/sglang_auditkit.ipynb` §2:

    SGLANG_BASE_URL=http://127.0.0.1:30000/v1 SGLANG_MODEL=CohereLabs/tiny-aya-global \\
        pytest tests/integrations_suite/test_sglang.py -k live

SGLANG_TOOL_PARSER=1 enables the native tool-call tests (server started with --tool-call-parser), and
SGLANG_REASONING=1 the reasoning tests (--reasoning-parser qwen3). colab_sglang.ipynb runs all of it on a GPU.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import auditkit as ak
from auditkit import RunConfig, Sample
from auditkit.model.api_gen import APIModel

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "get_time", "description": "Current local time in a city",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
]
W = lambda c: {"name": "get_weather", "arguments": {"city": c}}  # noqa: E731


# -- SGLang-shaped payloads (protocol.py) ---------------------------------------------------------------------

def sgl_tool_call(name, arguments, index=0, *, as_dict=False, call_id="auto"):
    """protocol.ToolCall: id (call_<24 hex>, or None), index, type, function{name, arguments: str | dict}."""
    return {"id": f"call_{uuid.uuid4().hex[:24]}" if call_id == "auto" else call_id, "index": index, "type": "function",
            "function": {"name": name, "arguments": arguments if as_dict else json.dumps(arguments)}}


def sgl_chat(content=None, *, tool_calls=None, reasoning=None, finish="stop", model="m"):
    """protocol.ChatCompletionResponse as SGLang serializes it: every field present, None ones as null."""
    return {"id": uuid.uuid4().hex, "object": "chat.completion", "created": int(time.time()), "model": model,
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": content, "reasoning_content": reasoning,
                                     "tool_calls": tool_calls},
                         "logprobs": None, "finish_reason": finish, "matched_stop": None if tool_calls else 151645,
                         "hidden_states": None, "prompt_token_ids": None, "response_token_ids": None,
                         "meta_info": None}],
            "usage": {"prompt_tokens": 42, "total_tokens": 50, "completion_tokens": 8,
                      "prompt_tokens_details": None, "reasoning_tokens": 0},
            "metadata": None}


def sgl_error(message, code=400, err_type="BadRequestError"):
    """protocol.ErrorResponse (serving_base.create_error_response): flat, not OpenAI's {"error": {...}}."""
    return {"object": "error", "message": message, "type": err_type, "param": None, "code": code}


# -- the local SGLang-compatible server -------------------------------------------------------------------------

class SGLangStub:
    """POST /v1/chat/completions answered by ``responder(body) -> (status, payload)``; GET /health, /v1/models."""

    def __init__(self):
        self.bodies, self.headers, self.paths, self.responder = [], [], [], (lambda body: (200, sgl_chat("ok")))
        stub = self

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path.endswith("/health"):
                    return self._send(200, {})
                self._send(200, {"object": "list", "data": [{"id": "m", "object": "model", "owned_by": "sglang",
                                                             "root": "m", "parent": None, "max_model_len": 8192}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                stub.bodies.append(body)
                stub.headers.append(dict(self.headers))
                stub.paths.append(self.path)
                status, payload = stub.responder(body)
                if self.path.endswith("/completions") and not self.path.endswith("/chat/completions") and status == 200:
                    msg = payload["choices"][0]["message"]           # protocol.CompletionResponseChoice: text
                    payload = {**payload, "object": "text_completion",
                               "choices": [{"index": 0, "text": msg["content"] or "", "logprobs": None,
                                            "finish_reason": payload["choices"][0]["finish_reason"], "matched_stop": None}]}
                self._send(status, payload)

            def _send(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def model(self, **kw):
        m = APIModel("m", api_base=self.base, api_key="EMPTY", name="api:sglang/m", **kw)
        return m


@pytest.fixture
def sgl(monkeypatch, tmp_path):
    for var in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))       # no cached predictions across tests
    s = SGLangStub()
    yield s
    s.server.shutdown()
    s.server.server_close()


def last_user(body):
    return next(m["content"] for m in reversed(body["messages"]) if m["role"] == "user")


# -- generation ------------------------------------------------------------------------------------------------

def test_generation_is_scored_and_usage_recorded(sgl):
    answers = {"capital of France": "Paris", "12 times 12": "144"}
    sgl.responder = lambda b: (200, sgl_chat(next(a for k, a in answers.items() if k in last_user(b))))
    r = ak.evaluate([Sample(input="What is the capital of France?", target="Paris"),
                     Sample(input="What is 12 times 12?", target="144")],
                    model=sgl.model(), scorers=["quasi_exact_match"], config=RunConfig(temperature=0.0))
    assert r.headline["quasi_exact_match"] == 1.0 and not r.errors
    assert r.token_usage and r.token_usage.get("prompt_tokens") == 84


def test_reasoning_content_is_not_the_answer(sgl):
    # --reasoning-parser qwen3: SGLang separates the reasoning (separate_reasoning=True by default)
    sgl.responder = lambda b: (200, sgl_chat("Paris", reasoning="The user asks for France's capital, which is Paris."))
    r = ak.evaluate([Sample(input="Capital of France?", target="Paris")], model=sgl.model(), scorers=["exact_match"])
    assert r.predictions[0].raw_output == "Paris" and r.headline["exact_match"] == 1.0


def test_chat_template_kwargs_are_sent_only_when_set(sgl):
    s = [Sample(input="hi", target="ok")]
    ak.evaluate(s, model=sgl.model(), scorers=["exact_match"])
    ak.evaluate(s, model=sgl.model(), scorers=["exact_match"],
                config=RunConfig(chat_template_kwargs={"enable_thinking": False}))
    assert "chat_template_kwargs" not in sgl.bodies[0]
    assert sgl.bodies[1]["chat_template_kwargs"] == {"enable_thinking": False}     # a ChatCompletionRequest field


def test_the_openai_key_is_never_sent_to_a_self_hosted_server(sgl, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-openai-key-never-leak")
    ak.evaluate([Sample(input="hi", target="ok")], model=APIModel("m", api_base=sgl.base, name="api:sglang/m"),
                scorers=["exact_match"])
    assert "sk-real-openai-key-never-leak" not in json.dumps(sgl.headers)


# -- native tool calls (server started with --tool-call-parser) ---------------------------------------------------

def _tool_server(sgl, *, as_dict=False, call_id="auto"):
    def respond(body):
        q = last_user(body)
        if "Oslo and" in q:
            calls = [sgl_tool_call("get_weather", {"city": "Oslo"}, 0, as_dict=as_dict, call_id=call_id),
                     sgl_tool_call("get_weather", {"city": "Cairo"}, 1, as_dict=as_dict, call_id=call_id)]
        elif "time" in q:
            calls = [sgl_tool_call("get_time", {"city": "Seoul"}, as_dict=as_dict, call_id=call_id)]
        elif "weather" in q:
            calls = [sgl_tool_call("get_weather", {"city": "Lisbon"}, as_dict=as_dict, call_id=call_id)]
        else:
            return 200, sgl_chat("4")
        return 200, sgl_chat("", tool_calls=calls, finish="tool_calls")
    sgl.responder = respond


TOOL_TASKS = [
    Sample(id="single", input="What's the weather in Lisbon?", tools=TOOLS, expected_tool_calls=[[W("Lisbon")]]),
    Sample(id="parallel", input="Weather in Oslo and Cairo?", tools=TOOLS, expected_tool_calls=[[W("Oslo"), W("Cairo")]]),
    Sample(id="pick", input="What time is it in Seoul?", tools=TOOLS,
           expected_tool_calls=[[{"name": "get_time", "arguments": {"city": "Seoul"}}]]),
    Sample(id="no_tool", input="What is 2+2?", tools=TOOLS, expected_tool_calls=[]),
]
TOOL_SCORERS = lambda: [ak.ToolCallF1(), ak.TrajectoryMatch(), ak.ParallelToolCalls(), ak.ToolCallValidity()]  # noqa: E731


@pytest.mark.parametrize("as_dict", [False, True], ids=["json-string-arguments", "dict-arguments"])
def test_native_tool_calls_are_scored(sgl, as_dict):
    _tool_server(sgl, as_dict=as_dict)          # FunctionResponse.arguments: str | dict
    r = ak.evaluate(TOOL_TASKS, model=sgl.model(), adapter=ak.ToolCallAdapter(), scorers=TOOL_SCORERS())
    assert not r.errors
    for k in ("tool_call_f1", "trajectory_strict", "parallel_detection", "parallel_recall", "tool_call_validity"):
        assert r.headline[k] == 1.0, (k, r.headline)
    assert sgl.bodies[0]["tools"] == TOOLS                                    # schemas sent natively


def test_tool_calls_without_ids_are_still_scored(sgl):
    _tool_server(sgl, call_id=None)             # protocol.ToolCall.id is Optional
    r = ak.evaluate(TOOL_TASKS[:2], model=sgl.model(), adapter=ak.ToolCallAdapter(), scorers=[ak.ToolCallF1()])
    assert r.headline["tool_call_f1"] == 1.0 and not r.errors


def test_prompt_mode_reads_calls_from_text(sgl):
    # Tiny Aya (no tool slot in its template, no parser): the calls come back as <tool_call> text
    def respond(body):
        assert "tools" not in body                                             # prompt mode sends no schemas
        return 200, sgl_chat('<tool_call>{"name": "get_weather", "arguments": {"city": "Lisbon"}}</tool_call>')
    sgl.responder = respond
    r = ak.evaluate(TOOL_TASKS[:1], model=sgl.model(), adapter=ak.ToolCallAdapter(mode="prompt"), scorers=[ak.ToolCallF1()])
    assert r.headline["tool_call_f1"] == 1.0


def test_a_call_sglang_could_not_parse_is_not_read_as_no_call(sgl):
    # serving_chat._process_tool_calls: a marker with no complete call is dropped and the raw markup returned as
    # content (finish_reason "length" here: cut at max_tokens). On a no-tool case it must not score as "no call".
    sgl.responder = lambda b: (200, sgl_chat('<tool_call>{"name": "get_weather", "arguments": {"ci', finish="length"))
    r = ak.evaluate([Sample(id="no_tool", input="What is 2+2?", tools=TOOLS, expected_tool_calls=[])],
                    model=sgl.model(), adapter=ak.ToolCallAdapter(), scorers=[ak.ToolCallF1(), ak.ToolCallValidity()])
    assert r.headline["tool_call_f1"] == 0.0 and r.headline["tool_call_validity"] == 0.0


# -- errors and retries -------------------------------------------------------------------------------------------

def test_an_sglang_error_fails_only_that_sample(sgl):
    def respond(body):
        if "bad" in last_user(body):
            return 400, sgl_error("This model's maximum context length is 8192 tokens.")
        return 200, sgl_chat("ok")
    sgl.responder = respond
    r = ak.evaluate([Sample(id="good", input="fine", target="ok"), Sample(id="bad", input="bad request", target="ok")],
                    model=sgl.model(), scorers=["exact_match"], config=RunConfig(max_retries=0))
    assert r.headline["exact_match"] == 1.0                                    # the good sample still scored
    assert len(r.errors) == 1 and "maximum context length" in json.dumps(r.errors)   # SGLang's message surfaced


def test_a_503_while_the_server_warms_up_is_retried(sgl):
    calls = {"n": 0}
    def respond(body):
        calls["n"] += 1
        return (503, sgl_error("server is starting", 503, "ServiceUnavailableError")) if calls["n"] == 1 else (200, sgl_chat("ok"))
    sgl.responder = respond
    r = ak.evaluate([Sample(input="hi", target="ok")], model=sgl.model(), scorers=["exact_match"],
                    config=RunConfig(max_retries=2, retry_delay=0))
    assert r.headline["exact_match"] == 1.0 and calls["n"] == 2


def test_an_aborted_request_is_an_error_not_a_zero(sgl):
    sgl.responder = lambda b: (200, sgl_chat("", finish="abort"))
    r = ak.evaluate([Sample(input="hi", target="ok")], model=sgl.model(), scorers=["exact_match"])
    assert r.errors and "exact_match" not in r.headline


# -- the agent_eval harness on an SGLang-served policy --------------------------------------------------------------

BANK_TOOLS = [
    {"type": "function", "function": {"name": "transfer", "parameters": {"type": "object", "properties": {
        "src": {"type": "string"}, "dst": {"type": "string"}, "amount": {"type": "number"}}, "required": ["src", "dst", "amount"]}}},
]


def test_the_harness_runs_a_tool_loop_against_sglang(sgl):
    from auditkit.agent_eval import AgentCase, AgentEvalRunner, AgentEvalSpec, FinalStateAssertion

    def respond(body):
        if body["messages"][-1]["role"] == "tool":                              # the result is back: answer
            return 200, sgl_chat("Done, savings is now 250.")
        return 200, sgl_chat("", tool_calls=[sgl_tool_call("transfer", {"src": "checking", "dst": "savings", "amount": 250})],
                             finish="tool_calls")
    sgl.responder = respond

    class Bank:
        def __init__(self):
            self.acc = {"checking": 1000.0, "savings": 0.0}
        def __call__(self, name, args):
            self.acc[args["src"]] -= args["amount"]
            self.acc[args["dst"]] += args["amount"]
            return json.dumps(self.acc)
        def snapshot(self):
            return dict(self.acc)

    case = AgentCase(id="move", task="Move 250 from checking to savings.", allowed_tools=BANK_TOOLS,
                     outcome=FinalStateAssertion("savings", equals=250.0))
    res = AgentEvalRunner().run(AgentEvalSpec(cases=[case], mode="harness", agent=sgl.model(),
                                              agent_opts={"tool_env": Bank(), "max_steps": 4}))
    assert res.rows[0].outcome["verdict"] == "success" and res.episodes[0].final_state == {"checking": 750.0, "savings": 250.0}
    second = sgl.bodies[1]["messages"]                                          # the tool result went back to SGLang
    call_id = second[-2]["tool_calls"][0]["id"]
    assert second[-1] == {**second[-1], "role": "tool", "tool_call_id": call_id}


# -- check_compat inside the SGLang venv -----------------------------------------------------------------------------

def test_compat_runs_as_a_script_without_auditkit(tmp_path):
    # the notebooks run compat.py with the SGLang venv's python, where AuditKIT is not installed
    import auditkit.compat as compat
    script = tmp_path / "compat.py"
    script.write_text(Path(compat.__file__).read_text())
    out = subprocess.run([sys.executable, "-I", str(script)], capture_output=True, text=True, cwd=tmp_path, timeout=60)
    assert "AuditKit compatibility report" in out.stdout and "Traceback" not in out.stderr
    assert out.returncode in (0, 1)


# -- live: a real SGLang server ---------------------------------------------------------------------------------------

LIVE = os.environ.get("SGLANG_BASE_URL")
live = pytest.mark.skipif(not LIVE, reason="set SGLANG_BASE_URL (and SGLANG_MODEL) to run against a real SGLang server")
TOOL_PARSER = pytest.mark.skipif(not os.environ.get("SGLANG_TOOL_PARSER"), reason="server not started with --tool-call-parser")
REASONING = pytest.mark.skipif(not os.environ.get("SGLANG_REASONING"), reason="server not started with --reasoning-parser (Qwen3)")
# a reasoning model (Qwen3) thinks before it answers; switched off, a short max_tokens budget holds the answer
NO_THINK = {"enable_thinking": False} if os.environ.get("SGLANG_REASONING") else None


@pytest.fixture
def served(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    return APIModel(os.environ.get("SGLANG_MODEL", "default"), api_base=LIVE, api_key="EMPTY", name="api:sglang-live")


@live
def test_live_server_answers_and_is_healthy(served):
    import requests
    assert requests.get(LIVE.rsplit("/v1", 1)[0] + "/health", timeout=10).ok
    r = ak.evaluate([Sample(input="What is the capital of France? Answer with one word.", target="Paris"),
                     Sample(input="Wie viele Tage hat eine Woche? Nur die Zahl.", target="7")],
                    model=served, scorers=["quasi_exact_match", "f1_score"],
                    config=RunConfig(temperature=0.0, max_tokens=32, chat_template_kwargs=NO_THINK))
    assert not r.errors and r.headline["quasi_exact_match"] >= 0.5


@live
def test_live_prompt_mode_tool_calls(served):
    r = ak.evaluate(TOOL_TASKS, model=served, adapter=ak.ToolCallAdapter(mode="prompt"), scorers=TOOL_SCORERS(),
                    config=RunConfig(temperature=0.0, max_tokens=256, chat_template_kwargs=NO_THINK))
    assert not r.errors and "tool_call_f1" in r.headline




@live
@TOOL_PARSER
def test_live_native_tool_calls(served):
    r = ak.evaluate(TOOL_TASKS, model=served, adapter=ak.ToolCallAdapter(), scorers=TOOL_SCORERS(),
                    config=RunConfig(temperature=0.0, max_tokens=512, chat_template_kwargs=NO_THINK))
    assert not r.errors and r.headline["tool_call_f1"] > 0
    assert r.headline.get("tool_call_validity", 1.0) == 1.0          # the parser's calls match the schemas


@live
@TOOL_PARSER
def test_live_native_calls_come_back_structured(served):
    # the server's --tool-call-parser (e.g. cohere_command4 for Command R7B) must return message.tool_calls;
    # AuditKIT would also read raw call markup from the text, so a text fallback would hide a parser failure
    r = ak.evaluate(TOOL_TASKS[:3], model=served, adapter=ak.ToolCallAdapter(), scorers=[ak.ToolCallF1()],
                    config=RunConfig(temperature=0.0, max_tokens=512, chat_template_kwargs=NO_THINK))
    structured = [p for p in r.predictions if ((p.context or {}).get("trace") or {}).get("tool_calls")]
    raw_markup = [p.sample_id for p in r.predictions if any(
        m in (p.raw_output or "") for m in ("<|START_ACTION|>", "<tool_call>", "Action:"))]
    assert structured, f"no structured tool_calls came back; raw call markup in content for {raw_markup}"


@live
@TOOL_PARSER
def test_live_harness_verifies_the_final_state(served):
    from auditkit.agent_eval import AgentCase, AgentEvalRunner, AgentEvalSpec, FinalStateAssertion

    class Bank:
        def __init__(self):
            self.acc = {"checking": 1000.0, "savings": 0.0}
        def __call__(self, name, args):
            if name == "transfer":
                a = float(args["amount"])
                self.acc[args["src"]] -= a
                self.acc[args["dst"]] += a
            return json.dumps(self.acc)
        def snapshot(self):
            return dict(self.acc)

    case = AgentCase(id="move", task="Move 250 from checking to savings using the transfer tool.", allowed_tools=BANK_TOOLS,
                     outcome=FinalStateAssertion("savings", equals=250.0))
    res = AgentEvalRunner().run(AgentEvalSpec(cases=[case], mode="harness", agent=served,
                                              agent_opts={"tool_env": Bank(), "max_steps": 4},
                                              config=RunConfig(temperature=0.0, max_tokens=1024)))
    assert res.rows[0].outcome["verdict"] == "success", (res.rows[0].stop_reason, res.episodes[0].final_state)


@live
@REASONING
def test_live_reasoning_is_kept_out_of_the_answer(served):
    # --reasoning-parser: the thinking goes to reasoning_content, and AuditKIT scores only the answer
    r = ak.evaluate([Sample(input="What is 17 times 3? Answer with the number only.", target="51")], model=served,
                    scorers=["quasi_exact_match"], config=RunConfig(temperature=0.0, max_tokens=2048))
    out = r.predictions[0].raw_output or ""
    assert "<think>" not in out and "51" in out, out[:300]


@live
@REASONING
def test_live_chat_template_kwargs_switch_thinking_off(served):
    s = [Sample(input="What is 17 times 3? Answer with the number only.", target="51")]
    on = ak.evaluate(s, model=served, scorers=["quasi_exact_match"], config=RunConfig(temperature=0.0, max_tokens=2048))
    off = ak.evaluate(s, model=served, scorers=["quasi_exact_match"],
                      config=RunConfig(temperature=0.0, max_tokens=2048, chat_template_kwargs={"enable_thinking": False}))
    # thinking is billed as completion tokens: switched off, far fewer are generated
    assert off.token_usage["completion_tokens"] < on.token_usage["completion_tokens"], (on.token_usage, off.token_usage)


# -- fix plan S1: Cohere tool calling on SGLang ----------------------------------------------------------------------

def test_s1_cohere_command4_parser_calls_are_scored(sgl):
    # --tool-call-parser cohere_command4 turns Command R7B / Command A's <|START_ACTION|>[...] into message.tool_calls
    # (cohere_command4_detector.py maps tool_name -> name and drops Cohere's tool_call_id: SGLang assigns its own)
    sgl.responder = lambda b: (200, sgl_chat("", tool_calls=[sgl_tool_call("get_weather", {"city": "Oslo"}, 0),
                                                            sgl_tool_call("get_weather", {"city": "Cairo"}, 1)],
                                             finish="tool_calls"))
    r = ak.evaluate(TOOL_TASKS[1:2], model=sgl.model(), adapter=ak.ToolCallAdapter(), scorers=TOOL_SCORERS())
    assert r.headline["tool_call_f1"] == 1.0 and r.headline["parallel_detection"] == 1.0


@pytest.mark.parametrize("reply", [
    '<|START_ACTION|>[{"tool_call_id": "0", "tool_name": "get_weather", "parameters": {"city": "Lisbon"}}]<|END_ACTION|>',
    'Action: ```json\n[{"tool_name": "get_weather", "parameters": {"city": "Lisbon"}}]\n```',
    '<tool_call>{"name": "get_weather", "arguments": {"city": "Lisbon"}}</tool_call>',
], ids=["command-r7b-actions", "command-r-actions", "hermes"])
def test_s1_prompt_mode_reads_cohere_and_hermes_text(sgl, reply):
    # a Cohere model without a tool slot (Tiny Aya, Aya Expanse), served without a parser: calls come back as text
    sgl.responder = lambda b: (200, sgl_chat(reply))
    r = ak.evaluate(TOOL_TASKS[:1], model=sgl.model(), adapter=ak.ToolCallAdapter(mode="prompt"), scorers=[ak.ToolCallF1()])
    assert r.headline["tool_call_f1"] == 1.0


# -- fix plan S3: what AuditKIT sends on /v1/chat/completions and /v1/completions --------------------------------------

def test_s3_chat_requests_carry_messages_not_a_rendered_template(sgl):
    ak.evaluate([Sample(input="Hola, ¿qué tal?", target="ok")], model=sgl.model(), scorers=["exact_match"])
    body = sgl.bodies[0]
    assert sgl.paths[0].endswith("/chat/completions") and "prompt" not in body
    assert body["messages"] == [{"role": "user", "content": "Hola, ¿qué tal?"}]          # the server renders the template
    assert not any(tok in json.dumps(body) for tok in ("<BOS_TOKEN>", "<s>", "<|START_OF_TURN_TOKEN|>"))


def test_s3_completion_requests_send_the_prompt_exactly_as_given(sgl):
    ak.evaluate([Sample(input="The capital of France is", target="Paris")], model=sgl.model(chat_template=False),
                scorers=["exact_match"])
    body = sgl.bodies[0]
    assert sgl.paths[0].endswith("/v1/completions") and not sgl.paths[0].endswith("/chat/completions")
    assert body["prompt"] == "The capital of France is"                                  # no template, no BOS added
    assert "messages" not in body


# -- fix plan S5: finish reasons with no answer ------------------------------------------------------------------------

def test_s5_a_content_filtered_empty_reply_is_an_error(sgl):
    sgl.responder = lambda b: (200, sgl_chat("", finish="content_filter"))
    r = ak.evaluate([Sample(input="hi", target="ok")], model=sgl.model(), scorers=["exact_match"])
    assert r.errors and "exact_match" not in r.headline


def test_s5_an_empty_stop_reply_still_scores(sgl):
    sgl.responder = lambda b: (200, sgl_chat("", finish="stop"))       # an empty answer can be the answer
    r = ak.evaluate([Sample(input="hi", target="")], model=sgl.model(), scorers=["exact_match"])
    assert not r.errors and r.headline["exact_match"] == 1.0


# -- fix plan S7: per-request latency ----------------------------------------------------------------------------------

def test_s7_latency_is_per_request(sgl):
    def respond(body):
        time.sleep(0.05)
        return 200, sgl_chat("ok")
    sgl.responder = respond
    r = ak.evaluate([Sample(id=str(i), input=f"q{i}", target="ok") for i in range(4)], model=sgl.model(),
                    scorers=["exact_match"], config=RunConfig(concurrency=1))
    lat = r.perf["latency_ms"]
    assert lat["count"] == 4 and 40 <= lat["p50"] < 2000                    # one sample per HTTP request
    assert r.perf["throughput"]["total_requests"] == 4
