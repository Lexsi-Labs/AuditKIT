"""Live runs against local Ollama models. Opt-in: ``AK_LIVE=1``.

Env:
  AK_LIVE=1                 enable
  AK_LIVE_BASE              server root, OpenAI API at {base}/v1 (default http://localhost:11434 = Ollama;
                            e.g. http://localhost:8000 for vLLM/SGLang)
  AK_LIVE_MODELS            comma list (default: every model the server lists)
  AK_LIVE_NATIVE            models served with a tool-call parser, or "all" (Ollama: auto-detected)
  AK_LIVE_TAG               label for this run's report file (e.g. "colab-a100")
  AK_LIVE_JUDGE             judge model for TaskCompletion (default: first tools-capable model)
  AK_LIVE_MIN_F1            quality floor for native tool calling (default 0.5)
  AK_LIVE_REPORT_DIR        where the JSON/Markdown report goes (default tests/agentic_suite/live_reports)

Two kinds of assertion:
  - PIPELINE (hard): every sample runs, nothing errors, every expected score
    name is present and in range, native traces are captured.
  - QUALITY (floor): tools-capable models in native mode clear AK_LIVE_MIN_F1.
    Per-case scores go to the report
    a model getting a case wrong is data,
    not a test failure.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import auditkit as ak
from auditkit.sample import Sample

from .catalog import AGENT_CASES, LIVE_CASES, TOOLS, W, c
from .conftest import OLLAMA, live_enabled, live_models

pytestmark = [pytest.mark.live, pytest.mark.skipif(not live_enabled(), reason="set AK_LIVE=1 to run live tests")]

MODELS = live_models() if live_enabled() else {}
TOOL_MODELS = [m for m, caps in MODELS.items() if "tools" in caps]
JUDGE = os.environ.get("AK_LIVE_JUDGE") or (TOOL_MODELS[0] if TOOL_MODELS else None)
MIN_F1 = float(os.environ.get("AK_LIVE_MIN_F1", "0.5"))
REPORT_DIR = Path(os.environ.get("AK_LIVE_REPORT_DIR", Path(__file__).parent / "live_reports"))

# (model, mode): native only where Ollama reports tool support; prompt mode for every model.
CONDITIONS = [(m, "native") for m in TOOL_MODELS] + [(m, "prompt") for m in MODELS]

REPORT: dict = {"conditions": {}, "agent": {}, "judge": {}}


def scorers():
    # subset variants separate "wrong behaviour" from "added an optional argument with its default"
    # (seen live: qwen3/gemma3 add unit="celsius" unasked, which fails exact matching).
    return [ak.ToolCallF1(), ak.TrajectoryMatch(), ak.ParallelToolCalls(), ak.ToolCallValidity(),
            ak.RedundantToolCalls(), ak.ToolCallF1(arg_mode="name"), ak.ToolCallF1(arg_mode="subset"),
            ak.TrajectoryMatch(arg_mode="subset"), ak.ParallelToolCalls(arg_mode="subset")]


def per_sample(result):
    return {p.sample_id: {s["name"]: s["value"] for s in p.metadata.get("scores", [])} for p in result.predictions}


def model_kwargs(model):
    return dict(model=f"api:{model}", api_base=f"{OLLAMA}/v1", api_key="EMPTY")


@pytest.fixture(scope="module", autouse=True)
def write_report():
    yield
    if not any(REPORT.values()):
        return
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    tag = os.environ.get("AK_LIVE_TAG", "local")
    (REPORT_DIR / f"live_report_{tag}.json").write_text(json.dumps(REPORT, indent=1, default=str))
    (REPORT_DIR / f"live_report_{tag}.md").write_text(render_markdown(REPORT))


KEY = ["tool_call_f1", "tool_call_exact", "tool_call_f1_name", "trajectory_strict", "parallel_recall",
       "parallel_precision", "parallel_detection", "tool_call_validity", "redundant_tool_calls"]


def render_markdown(rep):
    fmt = lambda v: "–" if v is None else f"{v:.2f}"  # noqa: E731
    out = ["# Agentic metrics: live report", ""]
    for cond, data in rep["conditions"].items():
        out += [f"## {cond}", "", "| case | probe | predicted | " + " | ".join(KEY) + " |",
                "|---" * (len(KEY) + 3) + "|"]
        for row in data["rows"]:
            out.append(f"| {row['id']} | {row['probe']} | `{row['predicted']}` | "
                       + " | ".join(fmt(row["scores"].get(k)) for k in KEY) + " |")
        out += ["", "**headline:** " + ", ".join(f"{k}={v:.3f}" for k, v in sorted(data["headline"].items())), ""]
    for model, data in rep["agent"].items():
        out += [f"## agent loop: {model}", "", "| case | turns | " + " | ".join(KEY + ["task_completion"]) + " |",
                "|---" * (len(KEY) + 3) + "|"]
        for row in data["rows"]:
            out.append(f"| {row['id']} | `{row['predicted']}` | "
                       + " | ".join(fmt(row["scores"].get(k)) for k in KEY + ["task_completion"]) + " |")
        out.append("")
    if rep["judge"]:
        out += ["## TaskCompletion judge sanity", "", "```", json.dumps(rep["judge"], indent=1), "```"]
    return "\n".join(out)


def turns_json(pred):
    from auditkit.trace import predicted_turns
    return [[x.to_dict() for x in t] for t in predicted_turns(pred.raw_output or "", pred.context)]


def turns_str(pred):
    from auditkit.trace import predicted_turns
    turns = predicted_turns(pred.raw_output or "", pred.context)
    return " | ".join("[" + ", ".join(f"{x.name}({json.dumps(x.arguments, ensure_ascii=False)})" for x in t) + "]"
                      for t in turns) or "(none)"


# -- single-response evaluation: every model x mode ----------------------------------------

@pytest.mark.parametrize("model,mode", CONDITIONS, ids=[f"{m}-{md}" for m, md in CONDITIONS])
def test_live_single_turn(model, mode):
    samples = [Sample(id=k.id, input=k.input, tools=k.tools, expected_tool_calls=k.expected) for k in LIVE_CASES]
    r = ak.evaluate(samples, adapter=ak.ToolCallAdapter(mode=mode), scorers=scorers(),
                    config=ak.RunConfig(temperature=0.0, max_tokens=4096, timeout=600), **model_kwargs(model))
    got = per_sample(r)
    rows = []
    for case, pred in zip(LIVE_CASES, r.predictions):
        rows.append({"id": case.id, "probe": case.probe, "predicted": turns_str(pred), "turns": turns_json(pred), "scores": got[case.id],
                     "raw_output": (pred.raw_output or "")[:500]})
    REPORT["conditions"][f"{model} / {mode}"] = {"headline": r.headline, "rows": rows, "errors": r.errors}

    # PIPELINE assertions
    assert r.errors == [] and r.failed_count == 0, r.errors
    assert len(r.predictions) == len(LIVE_CASES)
    for case in LIVE_CASES:
        s = got[case.id]
        for name in ("tool_call_f1", "tool_call_exact", "trajectory_strict", "trajectory_in_order",
                     "parallel_detection", "tool_call_f1_name"):
            assert name in s, (case.id, name)
        assert all(0.0 <= v <= 1.0 for v in s.values()), (case.id, s)
        if any(len(t) >= 2 for t in case.expected):
            assert "parallel_recall" in s, case.id
    if mode == "native":
        # a tools-capable server returns structured calls: every call-making sample has a trace
        for pred in r.predictions:
            if pred.context and pred.context.get("trace", {}).get("tool_calls"):
                assert isinstance(pred.context["trace"]["tool_calls"], list)
        assert any((p.context or {}).get("trace", {}).get("tool_calls") for p in r.predictions)


@pytest.mark.parametrize("model", TOOL_MODELS)
def test_live_quality_floor_native(model):
    data = REPORT["conditions"].get(f"{model} / native")
    if data is None:
        pytest.skip("single-turn native run did not execute")
    f1 = data["headline"]["tool_call_f1"]
    assert f1 >= MIN_F1, f"{model} native tool_call_f1={f1:.2f} < floor {MIN_F1}"


# -- multi-turn: a real agent loop (model + stub tools) behind the agent: backend ------------

WEATHER = {"paris": {"temp_c": 14, "conditions": "rain"}, "rome": {"temp_c": 23, "conditions": "sunny"},
           "tokyo": {"temp_c": 18, "conditions": "cloudy"}}
FLIGHTS = [{"flight_id": "AI-5", "price_usd": 120}, {"flight_id": "UK-102", "price_usd": 80},
           {"flight_id": "6E-9", "price_usd": 95}]


def run_stub_tool(name, args):
    if name == "get_weather":
        return WEATHER.get(str(args.get("city", "")).lower(), {"temp_c": 20, "conditions": "clear"})
    if name == "get_time":
        return {"city": args.get("city"), "local_time": "12:00"}
    if name == "search_flights":
        return {"flights": FLIGHTS}
    if name == "book_flight":
        return {"status": "confirmed", "flight_id": args.get("flight_id")}
    return {"error": f"unknown tool {name}"}


def ollama_chat(model, messages, tools):
    body = {"model": model, "messages": messages, "tools": tools, "temperature": 0, "max_tokens": 4096}
    req = urllib.request.Request(f"{OLLAMA}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["message"]


def agent_loop(model, prompt, tools, max_steps=6):
    """A minimal ReAct-style agent: the model asks, we execute stubs, feed results back."""
    messages = [{"role": "user", "content": prompt}]
    for _ in range(max_steps):
        msg = ollama_chat(model, messages, tools)
        calls = msg.get("tool_calls") or []
        messages.append({"role": "assistant", "content": msg.get("content") or "", **({"tool_calls": calls} if calls else {})})
        if not calls:
            return {"output": msg.get("content") or "", "messages": messages}
        for call in calls:
            fn = call["function"]
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                args = {}
            messages.append({"role": "tool", "tool_call_id": call.get("id"),
                             "content": json.dumps(run_stub_tool(fn["name"], args))})
    return {"output": "", "messages": messages}


@pytest.fixture(scope="module")
def live_agent():
    state = {"model": None}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            reply = agent_loop(state["model"], body["input"], body.get("tools") or TOOLS)
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{srv.server_address[1]}/run"
    yield state
    srv.shutdown()


@pytest.mark.parametrize("model", TOOL_MODELS)
def test_live_agent_multi_turn(model, live_agent):
    live_agent["model"] = model
    samples = [Sample(id=k.id, input=k.input, tools=k.tools, expected_tool_calls=k.expected,
                      target={"agent-par-then-dep": "Rome is warmer; local time 12:00",
                              "agent-search-book": "UK-102 booked",
                              "agent-single": "Tokyo 18C cloudy",
                              "agent-irrelevant": "42"}[k.id]) for k in AGENT_CASES]
    judge = ak.TaskCompletion(judge_model=f"api:{JUDGE}",
                              judge_model_args={"api_base": f"{OLLAMA}/v1", "api_key": "EMPTY"}, max_tokens=4096)
    r = ak.evaluate(samples, model=f"agent:{live_agent['url']}", adapter=ak.ToolCallAdapter(),
                    scorers=scorers() + [judge], config=ak.RunConfig(timeout=1800))
    got = per_sample(r)
    REPORT["agent"][model] = {"headline": r.headline, "errors": r.errors, "rows": [
        {"id": k.id, "predicted": turns_str(p), "turns": turns_json(p), "scores": got[k.id], "answer": (p.raw_output or "")[:300]}
        for k, p in zip(AGENT_CASES, r.predictions)]}

    assert r.errors == [] and r.failed_count == 0, r.errors
    for k, p in zip(AGENT_CASES, r.predictions):
        s = got[k.id]
        assert "trajectory_strict" in s and "task_completion" in s, (k.id, s)
        assert all(0.0 <= v <= 1.0 for v in s.values())
        # the transcript reached the metrics as turns (agent:, not text parsing)
        assert "messages" in p.context["trace"]
    # a real multi-turn run: the tool-using cases produced at least two turns somewhere
    from auditkit.trace import predicted_turns
    assert any(len(predicted_turns(p.raw_output or "", p.context)) >= 2 for p in r.predictions)


# -- TaskCompletion with a real judge ------------------------------------------------------

@pytest.mark.skipif(JUDGE is None, reason="no tools-capable local model to act as judge")
def test_live_task_completion_judge_orders_obvious_cases():
    P = W("Paris")
    good = Sample(id="good", input="Book flight UK-102.", target="UK-102 booked",
                  actual_output="Done: UK-102 is booked, confirmation #A1.",
                  actual_trace={"tool_calls": [[c("book_flight", flight_id="UK-102")]]})
    fake = Sample(id="fake", input="Book flight UK-102.", target="UK-102 booked",
                  actual_output="Done: UK-102 is booked!", actual_trace={"tool_calls": [[P]]})
    judge = ak.TaskCompletion(judge_model=f"api:{JUDGE}",
                              judge_model_args={"api_base": f"{OLLAMA}/v1", "api_key": "EMPTY"}, max_tokens=4096)
    r = ak.evaluate([good, fake], model="precomputed", scorers=[judge])
    got = {p.sample_id: p.metadata["scores"][0] for p in r.predictions}
    REPORT["judge"] = {"judge": JUDGE, **{k: {"value": v["value"], "meta": v.get("metadata")} for k, v in got.items()}}
    assert got["good"]["value"] > got["fake"]["value"], got
