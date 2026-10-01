"""Regression tests for G3-G6 (fix plan #15): AgentTune records whose calls have no tool
name are unscored rather than scored 0.0, and an answer-only agent: reply leaves tool use
unobservable whether or not tools were offered."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import auditkit as ak
from auditkit.agent_eval import AgentCase
from auditkit.agent_eval.importers import inspect_agenttune, supported_scorers_from_coverage
from auditkit.agent_eval.runner import rescore

TRACE_FINDER = str(Path(__file__).parent / "fixtures" / "lexsi_real" / "agenttune" / "trace_finder.jsonl")
ADD = {"name": "add", "arguments": {"a": 2, "b": 3}}


# -- G3: to_trace ---------------------------------------------------------------------------------------------

def test_g3_nameless_recorded_calls_are_unavailable_not_zero():
    e = ak.episodes_from_agenttune(TRACE_FINDER)[0]
    trace = e.to_trace()
    assert e.counters["n_tool_calls"] == 2
    assert trace.get("tool_calls_unavailable") is True and "tool_calls" not in trace
    s = ak.Sample(input="q", actual_output=e.final_answer, actual_trace=trace,
                  expected_tool_calls=[{"name": "search_corpus", "arguments": {"query": "x"}}])
    r = ak.evaluate([s], "precomputed", [ak.ToolCallF1()])
    assert "tool_call_f1" not in r.headline                   # unscored, not a measured 0.0


def test_g3_an_episode_with_no_calls_still_says_zero_calls(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps({"trajectory_id": "t", "task": "q", "reward": 1.0, "final_response": "5",
                             "steps": [{"action": {}}], "metadata": {}}) + "\n")
    [e] = ak.episodes_from_agenttune(str(p))
    assert e.to_trace()["tool_calls"] == []                   # observed zero calls is evidence


# -- G4: eligibility advertised by inspect ---------------------------------------------------------------------------

@pytest.mark.parametrize("cov,want,not_want", [
    ({"tool_call_names": "observed", "tool_call_arguments": "observed", "parallel_grouping": "observed"},
     {"tool_call_f1", "trajectory_match", "parallel_tool_calls"}, set()),
    ({"tool_call_names": "observed", "tool_call_arguments": "observed", "parallel_grouping": "unavailable"},
     {"tool_call_f1", "trajectory_match"}, {"parallel_tool_calls"}),
    ({"tool_call_names": "unavailable", "tool_call_arguments": "observed"},
     set(), {"tool_call_f1", "trajectory_match", "parallel_tool_calls", "tool_call_f1_name"}),
    ({"tool_call_names": "observed", "tool_call_arguments": "unavailable"},
     {"tool_call_f1_name"}, {"tool_call_f1"}),
])
def test_g4_supported_scorers_need_names(cov, want, not_want):
    got = set(supported_scorers_from_coverage(cov))
    assert want <= got and not (not_want & got)


def test_g4_inspect_does_not_offer_name_based_metrics_for_tracelogger_files():
    rep = inspect_agenttune(TRACE_FINDER)
    assert not {"tool_call_f1", "trajectory_match", "parallel_tool_calls"} & set(rep["eligible_metrics"])
    assert "retrieval" in rep["eligible_metrics"]


# -- G5: rescore reports why ------------------------------------------------------------------------------------------

def test_g5_rescore_reports_nameless_calls_as_ineligible():
    eps = ak.episodes_from_agenttune(TRACE_FINDER)
    for i, e in enumerate(eps):
        e.case_id = f"t{i}"
    cases = [AgentCase(id=f"t{i}", task="t", reference_turns=[ADD]) for i in range(2)]
    for row in rescore(eps, ["tool_call_f1"], cases=cases).rows:
        assert row.ineligible == {"tool_call_f1": "trace.tool_call_names"} and "tool_call_f1" not in row.diagnostics


# -- G6: agent: answer-only replies ------------------------------------------------------------------------------------

@pytest.fixture
def agent_url(monkeypatch):
    for var in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)
    reply = {"body": {"output": "5"}}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            data = json.dumps(reply["body"]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/run", reply
    srv.shutdown()
    srv.server_close()


def test_g6_an_answer_only_reply_without_tools_offered_is_unscored(agent_url, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    url, _ = agent_url
    r = ak.evaluate([ak.Sample(input="2+3?", expected_tool_calls=[ADD])], model=f"agent:{url}",
                    scorers=[ak.ToolCallF1()])
    assert "tool_call_f1" not in r.headline and r.errors == []


def test_g6_an_explicit_empty_tool_calls_is_still_observed_zero(agent_url, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    url, reply = agent_url
    reply["body"] = {"output": "5", "tool_calls": []}
    r = ak.evaluate([ak.Sample(input="2+3?", expected_tool_calls=[])], model=f"agent:{url}",
                    scorers=[ak.ToolCallF1()])
    assert r.headline["tool_call_f1"] == 1.0                  # correct irrelevance, observed
