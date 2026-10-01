"""AG-06 (offline rescore = zero agent calls), AG-09 (deployed capture),
AG-10 (ineligible names the field), AG-11 (status/coverage/summary), trials."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from auditkit.agent_eval import (
    AgentCase,
    AgentEvalRunner,
    AgentEvalSpec,
    AnswerAssertion,
    FinalStateAssertion,
    episode_from_openai_messages,
    rescore,
)
from auditkit.model import Generated, Model, Result_


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


def _episode(answer, calls=None, *, final_state=None, case_id=None, retrieved=None):
    messages = [{"role": "user", "content": "task"}]
    for i, (name, args) in enumerate(calls or []):
        messages.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": f"c{i}", "type": "function",
             "function": {"name": name, "arguments": json.dumps(args)}}]})
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": "ok"})
    messages.append({"role": "assistant", "content": answer})
    ep = episode_from_openai_messages(messages, final_answer=answer, case_id=case_id,
                                      retrieved_contexts=retrieved)
    ep.final_state = final_state
    return ep


# -- AG-06: offline rescore does NOT call the agent -------------------------
class _CountingAgent(Model):
    name = "counter"

    def __init__(self):
        self.calls = 0

    def generate(self, requests):
        self.calls += len(requests)
        return [Result_(completions=[Generated(text="")]) for _ in requests]


def test_offline_rescore_makes_zero_agent_calls():
    counter = _CountingAgent()
    ep = _episode("done", calls=[("get_weather", {"city": "P"})], case_id="c1")
    case = AgentCase(id="c1", task="t", allowed_tools=["get_weather"],
                     reference_turns=[[{"name": "get_weather", "arguments": {"city": "P"}}]])
    spec = AgentEvalSpec(cases=[case], mode="recorded", episodes=[ep], agent=counter,
                         scorers=["tool_call_f1", "tool_call_validity"])
    result = AgentEvalRunner().run(spec)
    assert counter.calls == 0
    assert result.rows[0].diagnostics["tool_call_f1"]["value"] == 1.0


def test_rescore_helper_reuses_episodes():
    ep = _episode("done", calls=[("f", {})], case_id="c1")
    case = AgentCase(id="c1", task="t", allowed_tools=["f"])
    r = rescore([ep], ["tool_call_validity"], cases=[case])
    assert r.rows[0].diagnostics["tool_call_validity"]["value"] == 1.0


# -- AG-10: ineligible metrics name the missing field -----------------------
def test_ineligible_metric_names_missing_trace_field():
    ep = _episode("just an answer", case_id="c1")  # no tool calls at all
    case = AgentCase(id="c1", task="t", allowed_tools=["f"],
                     reference_turns=[[{"name": "f", "arguments": {}}]])
    r = rescore([ep], ["tool_call_f1", "tool_call_validity"], cases=[case])
    inelig = r.rows[0].ineligible
    # a full transcript with zero calls is an observation: the skipped required
    # tool is scored (and fails), while validity has no call to judge
    assert "tool_call_f1" not in inelig
    assert inelig["tool_call_validity"] == "trace.tool_calls"


def test_ineligible_names_arguments_when_names_only():
    # a names-only report HAS calls but no args -> f1 names the argument field
    from auditkit.agent_eval import AgentEpisode, AgentEvent
    ep = AgentEpisode(
        events=[AgentEvent(index=0, role="assistant", type="tool_call",
                           payload={"name": "f"}, turn_id=0)],
        final_answer="a", case_id="c1",
        coverage={"final_answer": "observed", "tool_call_names": "observed",
                  "tool_call_arguments": "unavailable"})
    case = AgentCase(id="c1", task="t",
                     reference_turns=[[{"name": "f", "arguments": {}}]])
    r = rescore([ep], ["tool_call_f1"], cases=[case])
    assert r.rows[0].ineligible["tool_call_f1"] == "trace.tool_call_arguments"


def test_ineligible_metric_names_missing_reference():
    ep = _episode("done", calls=[("f", {"x": 1})], case_id="c1")
    case = AgentCase(id="c1", task="t")  # no reference_turns, no allowed_tools
    r = rescore([ep], ["tool_call_f1", "retrieval"], cases=[case])
    assert r.rows[0].ineligible["tool_call_f1"] == "case.reference_turns"
    assert r.rows[0].ineligible["retrieval"] == "case.reference_contexts"


# -- AG-11: status + coverage present; summary shows both denominators ------
def test_status_and_coverage_present_and_summary_denominators():
    eps = [
        _episode("ok", final_state={"k": "ready"}, case_id="s1"),
        _episode("ok", final_state={"k": "blocked"}, case_id="s2"),
        _episode("ok", case_id="s3"),  # no state -> unknown
    ]
    cases = [AgentCase(id=f"s{i}", task="t", outcome=FinalStateAssertion("k", equals="ready"))
             for i in (1, 2, 3)]
    r = rescore(eps, ["tool_call_validity"], cases=cases)
    for row in r.rows:
        assert row.status in ("completed", "budget_exhausted", "method_error",
                              "target_error", "judge_error", "ineligible", "not_applicable")
        assert row.coverage  # coverage map present
    verdicts = {row.case_id: row.outcome["verdict"] for row in r.rows}
    assert verdicts == {"s1": "success", "s2": "failure", "s3": "unknown"}
    summary = r.summary()
    # over-all-cases AND decided-only, and unknown is never a zero
    assert "1/3" in summary
    assert "1/2 decided" in summary
    assert "Unknown outcome: 1" in summary


def test_budget_exhausted_status():
    ep = _episode("ok", calls=[("f", {}), ("f", {}), ("f", {})], case_id="c1")
    case = AgentCase(id="c1", task="t", budgets={"max_tool_calls": 2},
                     outcome=AnswerAssertion("ok"))
    r = rescore([ep], ["tool_call_validity"], cases=[case])
    assert r.rows[0].status == "budget_exhausted"


# -- trials>1 now runs (A3 delivered): aggregates reliability, no longer raises --
def test_trials_gt_one_produces_reliability():
    eps = [_episode("x", case_id="c"), _episode("x", case_id="c"), _episode("x", case_id="c")]
    spec = AgentEvalSpec(cases=[AgentCase(id="c", task="t")], mode="recorded",
                         episodes=eps, trials=3)
    result = AgentEvalRunner().run(spec)
    assert result.rows[0].reliability["n_trials"] == 3


# -- AG-08 at runner level: judge never overrides verified state ------------
def test_judge_on_nameless_trace_does_not_error():
    """A trace.jsonl {query,result} call must not crash render_trace -> judge_error."""
    from auditkit.agent_eval import AgentEpisode, AgentEvent
    from auditkit.metrics.agent import TaskCompletion
    from auditkit.model import CallableModel

    ep = AgentEpisode(
        events=[AgentEvent(index=0, role="assistant", type="tool_call",
                           payload={"query": {"q": "revenue"}}, turn_id=0),
                AgentEvent(index=1, role="tool", type="tool_result",
                           payload={"output": "rev=42"}, turn_id=0)],
        final_answer="42", case_id="c1",
        coverage={"final_answer": "observed", "tool_call_names": "unavailable"})
    case = AgentCase(id="c1", task="revenue?")
    judge = TaskCompletion(judge_model=CallableModel(lambda p: ["CHOICE: complete"]))
    r = rescore([ep], ["task_completion"], cases=[case], judge=judge)
    assert r.rows[0].status != "judge_error"
    assert r.rows[0].judge["verdict"] == "success"


def test_headline_uses_verified_state_not_judge():
    from auditkit.metrics.agent import TaskCompletion
    from auditkit.model import CallableModel

    ep = _episode("Shipment is ready.", final_state={"k": "blocked"}, case_id="c1")
    case = AgentCase(id="c1", task="ship?", outcome=FinalStateAssertion("k", equals="ready"))
    judge = TaskCompletion(judge_model=CallableModel(lambda p: ["CHOICE: complete"]))
    spec = AgentEvalSpec(cases=[case], mode="recorded", episodes=[ep],
                         scorers=["task_completion"], judge=judge)
    r = AgentEvalRunner().run(spec)
    row = r.rows[0]
    assert row.outcome["verdict"] == "failure"          # verified state wins
    assert row.judge["verdict"] == "success"            # judge recorded as diagnostic
    assert row.diagnostics["task_completion"]["diagnostic"] is True


# ===========================================================================
# AG-09: deployed agent via a local HTTP server (no external network)
# ===========================================================================
@pytest.fixture
def server(monkeypatch):
    for var in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)
    state = {"respond": lambda body: (200, {"output": "ok"}), "delay": 0.0, "requests": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            state["requests"] += 1
            if state["delay"]:
                time.sleep(state["delay"])
            status, payload = state["respond"](None)
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{srv.server_address[1]}/run"
    yield state
    srv.shutdown()
    srv.server_close()


def _oa(i, name, args):
    return {"id": f"call_{i}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def _complete_payload():
    weather, time_ = _oa(1, "get_weather", {"city": "Paris"}), _oa(2, "get_time", {"city": "Paris"})
    messages = [{"role": "user", "content": "weather+time?"},
                {"role": "assistant", "content": None, "tool_calls": [weather, time_]},
                {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
                {"role": "tool", "tool_call_id": "call_2", "content": "noon"},
                {"role": "assistant", "content": "sunny at noon"}]
    return {"output": "sunny at noon", "messages": messages, "contexts": ["d1", "d3"],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}


def _deployed_spec(url, case, **opts):
    return AgentEvalSpec(cases=[case], mode="deployed", agent="agent:" + url,
                         agent_opts=opts, scorers=["tool_call_f1", "tool_call_validity", "retrieval"])


def test_deployed_complete_response_and_trace_order(server):
    server["respond"] = lambda body: (200, _complete_payload())
    case = AgentCase(id="c1", task="weather+time?", allowed_tools=["get_weather", "get_time"],
                     reference_turns=[[{"name": "get_weather", "arguments": {"city": "Paris"}},
                                       {"name": "get_time", "arguments": {"city": "Paris"}}]],
                     reference_contexts=["d1", "d3"],
                     outcome=AnswerAssertion("sunny at noon"))
    r = AgentEvalRunner().run(_deployed_spec(server["url"], case))
    row = r.rows[0]
    assert row.status == "completed"
    assert row.outcome["verdict"] == "success"
    assert row.diagnostics["tool_call_f1"]["value"] == 1.0     # parallel group matched
    assert row.diagnostics["hit_rate"]["value"] == 1.0         # retrieval eligible
    assert row.resource_use["latency_ms"] is not None
    assert row.resource_use["agent_call"] is True
    assert row.coverage_label == "full"


def test_deployed_sparse_response_is_answer_only(server):
    server["respond"] = lambda body: (200, {"output": "sunny at noon"})
    case = AgentCase(id="c1", task="weather?", allowed_tools=["get_weather"],
                     reference_turns=[[{"name": "get_weather", "arguments": {}}]],
                     reference_contexts=["d1"], outcome=AnswerAssertion("sunny at noon"))
    r = AgentEvalRunner().run(_deployed_spec(server["url"], case))
    row = r.rows[0]
    assert row.coverage_label == "answer_only"
    assert row.ineligible["tool_call_f1"] == "trace.tool_calls"  # zero calls
    assert row.ineligible["retrieval"] == "trace.retrieved_contexts"
    assert row.outcome["verdict"] == "success"  # the answer oracle still runs


def test_deployed_malformed_json_is_target_error(server):
    server["respond"] = lambda body: (200, b"this is not json")
    case = AgentCase(id="c1", task="t", outcome=AnswerAssertion("x"))
    r = AgentEvalRunner().run(_deployed_spec(server["url"], case))
    row = r.rows[0]
    assert row.status == "target_error"
    assert row.stop_reason == "error"
    # an endpoint error is 'error', never a fabricated failure scored on ""
    assert row.outcome["verdict"] == "error"
    assert row.errors  # the endpoint failure text is captured, not silent
    summary = r.summary()
    assert "0 decided" in summary  # the error is NOT counted as a decided case
    assert "Endpoint errors: 1" in summary


def test_deployed_timeout_is_target_error(server):
    server["delay"] = 1.0
    server["respond"] = lambda body: (200, _complete_payload())
    case = AgentCase(id="c1", task="t", outcome=AnswerAssertion("x"))
    r = AgentEvalRunner().run(_deployed_spec(server["url"], case, timeout=0.3))
    row = r.rows[0]
    assert row.status == "target_error"
    assert row.outcome["verdict"] == "error"
    assert row.errors


def test_deployed_run_can_be_rescored_offline_no_http(server):
    """A deployed run's episodes are saved so rescore replays with zero HTTP."""
    server["respond"] = lambda body: (200, _complete_payload())
    case = AgentCase(id="c1", task="weather+time?", allowed_tools=["get_weather", "get_time"],
                     reference_turns=[[{"name": "get_weather", "arguments": {"city": "Paris"}},
                                       {"name": "get_time", "arguments": {"city": "Paris"}}]],
                     outcome=AnswerAssertion("sunny at noon"))
    r = AgentEvalRunner().run(_deployed_spec(server["url"], case))
    assert r.episodes  # scored episodes are retained on the result
    before = server["requests"]
    r2 = rescore(r.episodes, ["tool_call_f1"], cases=[case])
    assert server["requests"] == before  # rescore made NO endpoint call
    assert r2.rows[0].diagnostics["tool_call_f1"]["value"] == 1.0


def test_result_to_json_round_trips(server):
    server["respond"] = lambda body: (200, _complete_payload())
    case = AgentCase(id="c1", task="weather?", outcome=AnswerAssertion("sunny at noon"))
    r = AgentEvalRunner().run(_deployed_spec(server["url"], case))
    from auditkit.agent_eval import AgentEvalResult
    text = r.to_json()
    back = AgentEvalResult.from_dict(json.loads(text))
    assert back.to_dict() == r.to_dict()


# ===========================================================================
# Regression tests for confirmed stress/audit defects
# ===========================================================================
def test_case_without_episode_is_never_position_matched():
    """A case with no episode is not_applicable, never scored on another case's run."""
    eps = [_episode("answer B", case_id="case-B"), _episode("answer C", case_id="case-C"),
           _episode("answer D", case_id="case-D")]
    cases = [AgentCase(id=i, task="t", outcome=AnswerAssertion(a))
             for i, a in (("case-A", "answer B"), ("case-B", "answer B"), ("case-C", "answer C"))]
    with pytest.warns(UserWarning, match="match no case"):
        r = rescore(eps, ["tool_call_validity"], cases=cases)
    rows = {row.case_id: row for row in r.rows}
    assert rows["case-A"].status == "not_applicable"
    assert rows["case-A"].outcome["verdict"] == "unknown"
    assert [e.case_id for e in r.episodes] == ["case-B", "case-C"]
    assert "Verified success: 2/3" in r.summary()


def test_duplicate_episode_ids_warn():
    eps = [_episode("x", case_id="t-1"), _episode("y", case_id="t-1")]
    with pytest.warns(UserWarning, match="duplicate episode id"):
        rescore(eps, ["tool_call_validity"])
    with pytest.warns(UserWarning, match="2 episodes match case id"):
        rescore(eps, ["tool_call_validity"], cases=[AgentCase(id="t-1", task="t")])


def test_rescore_keeps_derived_ids_and_reports_zero_agent_calls(server):
    from auditkit.agent_eval import AgentEpisode
    task = AgentEpisode(metadata={"is_task": True})
    anon = [_episode("a0"), _episode("a1")]
    r1 = rescore([task] + anon, ["tool_call_validity"])
    r2 = rescore(r1.episodes, ["tool_call_validity"])
    assert [row.case_id for row in r1.rows] == [row.case_id for row in r2.rows]
    assert [e.final_answer for e in r2.episodes] == ["a0", "a1"]

    server["respond"] = lambda body: (200, {"output": "ok"})
    d = AgentEvalRunner().run(_deployed_spec(server["url"], AgentCase(id="c1", task="t")))
    assert "Agent calls: 1" in d.summary()
    rs = rescore(d.episodes, ["tool_call_validity"], cases=[AgentCase(id="c1", task="t")])
    assert "Agent calls: 0" in rs.summary()


def test_summary_counts_add_up_and_predicate_errors_are_method_error():
    from auditkit.agent_eval import CustomPredicate

    def boom(ep, case):
        raise RuntimeError("predicate failure")

    preds = {"ok": lambda ep, c: True, "raises": boom, "near-miss": lambda ep, c: "fail",
             "upper": lambda ep, c: "FAILURE", "no": lambda ep, c: False}
    cases = [AgentCase(id=k, task="t", outcome=CustomPredicate(f, version="1"))
             for k, f in preds.items()]
    eps = [_episode("x", case_id=k) for k in preds]
    r = rescore(eps, ["tool_call_validity"], cases=cases)
    rows = {row.case_id: row for row in r.rows}
    assert rows["raises"].status == "method_error"
    assert rows["near-miss"].outcome["verdict"] == "unknown"   # strict enum
    assert rows["upper"].outcome["verdict"] == "unknown"
    s = r.summary()
    assert "Verified success: 1/5" in s and "Failure: 1" in s
    assert "Unknown outcome: 2" in s and "Error: 1" in s
    assert "method_error 1" in s


def test_budget_exhausted_success_is_not_verified_success():
    ep = _episode("ok", calls=[("f", {}), ("f", {}), ("f", {})], case_id="c1")
    case = AgentCase(id="c1", task="t", budgets={"max_tool_calls": 2},
                     outcome=AnswerAssertion("ok"))
    s = rescore([ep], ["tool_call_validity"], cases=[case]).summary()
    assert "Verified success: 0/1" in s and "Over budget: 1" in s


def test_rescore_task_completion_without_judge_is_ineligible_not_silent():
    r = rescore([_episode("x", case_id="c1")], ["task_completion"],
                cases=[AgentCase(id="c1", task="t")])
    assert r.rows[0].ineligible["task_completion"] == "spec.judge"


def test_unknown_scorer_fails_fast():
    with pytest.raises(ValueError, match="unknown scorer"):
        rescore([_episode("x", case_id="c1")], ["answer"])


def test_observed_zero_calls_is_eligible_for_irrelevance():
    """Tool calls observed with none made: scoreable (a skipped tool is penalized)."""
    from auditkit.agent_eval import AgentEpisode
    cov = {"final_answer": "observed", "tool_call_names": "observed",
           "tool_call_arguments": "observed"}
    for ref, want in (([[{"name": "f", "arguments": {}}]], 0.0), ([], 1.0)):
        ep = AgentEpisode(events=[], final_answer="a", case_id="c1", coverage=dict(cov))
        r = rescore([ep], ["tool_call_f1", "parallel_tool_calls"],
                    cases=[AgentCase(id="c1", task="t", reference_turns=ref)])
        assert not r.rows[0].ineligible
        assert r.rows[0].diagnostics["tool_call_f1"]["value"] == want


def test_nan_reward_serializes_to_strict_json():
    ep = _episode("x", case_id="c1")
    ep.source_reward = float("nan")
    r = rescore([ep], ["tool_call_validity"])

    def _reject(token):
        raise AssertionError(f"non-strict JSON token {token}")

    json.loads(r.to_json(), parse_constant=_reject)
    json.loads(ep.to_json(), parse_constant=_reject)


def test_log_capture_ignores_other_threads():
    import logging
    from auditkit.agent_eval.runner import _LogCapture
    cap = _LogCapture()
    cap.thread_id = threading.get_ident()
    log = logging.getLogger("auditkit.model.agent_endpoint")
    log.addHandler(cap)
    try:
        t = threading.Thread(target=lambda: log.warning("other run"))
        t.start()
        t.join()
        log.warning("mine")
    finally:
        log.removeHandler(cap)
    assert cap.messages == ["mine"]


def test_malformed_but_valid_json_reply_fails_only_its_case(server):
    replies = iter([{"choices": [None]}, {"messages": [{"role": "assistant", "tool_calls": 5}]},
                    {"output": "fine"}])
    server["respond"] = lambda body: (200, next(replies))
    cases = [AgentCase(id=f"c{i}", task="t", outcome=AnswerAssertion("fine")) for i in range(3)]
    spec = AgentEvalSpec(cases=cases, mode="deployed", agent="agent:" + server["url"],
                         scorers=["tool_call_validity"])
    r = AgentEvalRunner().run(spec)
    assert [row.status for row in r.rows] == ["target_error", "target_error", "completed"]
    assert r.rows[0].errors and r.rows[2].outcome["verdict"] == "success"


def test_deadline_bounds_a_slow_agent_call():
    class Slow(Model):
        name = "slow"
        timeout = 0.2

        def generate(self, requests):
            time.sleep(2)
            return [Result_(completions=[Generated(text="late")]) for _ in requests]

    t0 = time.monotonic()
    r = AgentEvalRunner().run(AgentEvalSpec(cases=[AgentCase(id="c", task="t")], mode="deployed",
                                            agent=Slow(), scorers=["tool_call_validity"]))
    assert time.monotonic() - t0 < 1.5
    assert r.rows[0].status == "target_error" and "deadline" in r.rows[0].errors[0]


def test_deployed_keeps_endpoint_state_and_artifacts(server):
    from auditkit.agent_eval import ArtifactAssertion
    server["respond"] = lambda body: (200, {"output": "ready", "final_state": {"done": True},
                                            "artifacts": {"decision": "ship"}})
    cases = [AgentCase(id="s", task="t", outcome=FinalStateAssertion("done", equals=True)),
             AgentCase(id="a", task="t", outcome=ArtifactAssertion("decision", equals="ship"))]
    spec = AgentEvalSpec(cases=cases, mode="deployed", agent="agent:" + server["url"],
                         scorers=["tool_call_validity"])
    r = AgentEvalRunner().run(spec)
    assert [row.outcome["verdict"] for row in r.rows] == ["success", "success"]
    assert r.rows[0].coverage["final_state"] == "observed"


def test_fingerprint_includes_endpoint_settings(server):
    case = AgentCase(id="c1", task="t")
    a = AgentEvalRunner().run(_deployed_spec(server["url"], case, output_path="a"))
    b = AgentEvalRunner().run(_deployed_spec(server["url"], case, output_path="b"))
    assert a.spec_identity["agent"] != b.spec_identity["agent"]
    assert a.spec_identity["agent"]["output_path"] == "a"


def test_cases_report_drilldown():
    r = rescore([_episode("x", case_id="c1")], ["tool_call_f1"],
                cases=[AgentCase(id="c1", task="t", outcome=AnswerAssertion("x"))])
    text = r.cases_report()
    assert "[c1] status=completed outcome=success" in text
    assert "skipped tool_call_f1: case.reference_turns" in text


# -- reference-free scorers run through the runner with NO gold reference ----
def test_reference_free_scorers_run_without_a_gold_reference():
    """agent_loop_detection / tool_permission need no reference_turns or target;
    tool_permission reads the case's allowed_tools from the derived sample."""
    ep = _episode("done", calls=[("get_weather", {"city": "P"}),
                                 ("send_email", {"to": "x"})], case_id="c1")
    case = AgentCase(id="c1", task="t", allowed_tools=["get_weather"])  # send_email not allowed
    r = rescore([ep], ["agent_loop_detection", "tool_permission"], cases=[case])
    row = r.rows[0]
    assert "agent_loop_detection" not in row.ineligible
    assert "tool_permission" not in row.ineligible
    assert row.diagnostics["tool_permission"]["value"] == 0.5   # 1 of 2 calls within policy
    assert row.diagnostics["agent_loop_detection"]["value"] == 1.0  # no loop


def test_tool_permission_ineligible_without_a_policy():
    ep = _episode("done", calls=[("f", {})], case_id="c1")
    case = AgentCase(id="c1", task="t")  # no allowed_tools, no denied_tools
    r = rescore([ep], ["tool_permission"], cases=[case])
    assert r.rows[0].ineligible.get("tool_permission") == "case.allowed_tools"
