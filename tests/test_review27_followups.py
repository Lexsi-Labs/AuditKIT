"""Regression tests for R1, R2 and L1 of docs/notes/open-bugs-fix-plan.md (the #27 review follow-ups; R3 landed in #34)."""

from __future__ import annotations

import pytest

import auditkit as ak
from auditkit.agent_eval import AgentCase, episode_from_eventlog, rescore
from auditkit.agent_eval.importers import episodes_from_agenttune
from auditkit.metrics.rag_stress import Abstention, _looks_like_refusal

REF = [[{"name": "search", "arguments": {}}]]


# -- R1: abstention's short-reply rule reads refusal-shaped wording, not a bare phrase ---------------------------

@pytest.mark.parametrize("reply", [
    "Yes, there is no relevant fee for domestic wires.",
    "The wire fails if there is insufficient information on the form; resubmit it.",
    "Yes, there is insufficient information to open an account without ID, so bring a passport.",
])
def test_r1_short_real_answers_are_not_refusals(reply):
    assert _looks_like_refusal(reply) is False


@pytest.mark.parametrize("reply", [
    "Based on the documents, there is insufficient information to answer.",      # #27's case, still caught
    "There is no relevant information in the documents.",
    "The context has no information about the CEO's address.",
    "I'm sorry, the documents contain no relevant passages for this question.",
])
def test_r1_short_refusals_are_still_caught(reply):
    assert _looks_like_refusal(reply) is True


@pytest.mark.parametrize("reply", [
    "Sure, but the documents contain no relevant information about that.",
    "Certainly, however there is no relevant information in the context.",
])
def test_r1_only_yes_is_an_answer_opener(reply):
    # #35's open question: chat models open refusals with "Sure, but" / "Certainly, however",
    # so only a direct "yes" exempts a reply (widening the opener list got 3/6 right vs 5/6)
    assert _looks_like_refusal(reply) is True


def test_r1_the_markers_are_still_arguments_and_identity():
    custom = Abstention(short_reply_markers=[r"\bnot covered\b"])
    assert custom.identity() != Abstention().identity()
    assert _looks_like_refusal("That topic is not covered.", short_reply_markers=[r"\bnot covered\b"])


# -- R2: an unnamed call in an EventLog is unscored, not a measured 0.0 --------------------------------------------

def _log(*actions):
    events = []
    for a in actions:
        events += [{"kind": "tool_call", "payload": {"action": a}}, {"kind": "tool_result", "payload": {"output": "..."}}]
    return {"id": "e", "tier": "full", "events": events + [{"kind": "text", "payload": {"text": "5"}}]}


@pytest.mark.parametrize("action", [
    {"arguments": {"query": "fee"}},                          # arguments, no name
    {"query": {"query": "fee"}},                              # a TraceLogger-style record
    {"tool_calls": [{"arguments": {"q": "fee"}}]},            # a call list whose entry has no name
])
def test_r2_an_unnamed_call_leaves_names_unavailable(action):
    ep = episode_from_eventlog(_log(action))
    assert ep.coverage["tool_call_names"] != "observed"
    s = ak.Sample(input="q", actual_output="5", actual_trace=ep.to_trace(), expected_tool_calls=REF)
    assert "tool_call_f1" not in ak.evaluate([s], "precomputed", [ak.ToolCallF1()]).headline


def test_r2_the_no_call_shapes_from_27_still_mean_zero_calls():
    for action in ({}, {"stage": "decide"}):
        ep = episode_from_eventlog(_log(action))
        assert ep.coverage["tool_call_names"] == "observed" and ep.to_trace()["tool_calls"] == []


def test_r2_named_calls_are_unchanged():
    ep = episode_from_eventlog(_log({"name": "search", "arguments": {"q": "fee"}}))
    assert ep.coverage["tool_call_names"] == "observed" and ep.turns()[0][0]["name"] == "search"


def test_r2_rescore_reports_why():
    ep = episode_from_eventlog(_log({"arguments": {"query": "fee"}}))
    ep.case_id = "c"
    row = rescore([ep], ["tool_call_f1"], cases=[AgentCase(id="c", task="q", reference_turns=REF[0])]).rows[0]
    assert row.ineligible.get("tool_call_f1") == "trace.tool_call_names"


# -- L1: a trajectory whose actions are all empty has observed names (zero calls) ------------------------------------

def test_l1_an_all_empty_action_trajectory_is_zero_calls(tmp_path):
    import json
    p = tmp_path / "traj.jsonl"
    p.write_text(json.dumps({"trajectory_id": "t", "task": "q", "reward": 1.0, "final_response": "5",
                             "steps": [{"step_number": 0, "action": {}}, {"step_number": 1, "action": {}}],
                             "metadata": {}}) + "\n")
    [ep] = episodes_from_agenttune(str(p))
    assert ep.coverage["tool_call_names"] == "observed" and ep.to_trace()["tool_calls"] == []


def test_l1_a_trajectory_step_with_an_unnamed_call_is_unavailable(tmp_path):
    import json
    p = tmp_path / "traj.jsonl"
    p.write_text(json.dumps({"trajectory_id": "t", "task": "q", "reward": 1.0, "final_response": "5",
                             "steps": [{"step_number": 0, "action": {"arguments": {"q": "x"}}}], "metadata": {}}) + "\n")
    [ep] = episodes_from_agenttune(str(p))
    assert ep.coverage["tool_call_names"] != "observed"
