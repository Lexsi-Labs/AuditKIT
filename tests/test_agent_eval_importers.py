"""AG-03 (loss-aware coverage), AG-04 (golden fixtures), AG-05 (provenance),
and the no-extra import proof."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

import auditkit
from auditkit.agent_eval import (
    episode_from_openai_messages,
    episode_from_sample,
    episodes_from_agenttune,
    inspect_agenttune,
)
from auditkit.sample import Sample


def _write(tmp_path, name, records, *, as_doc=False):
    p = tmp_path / name
    if as_doc:
        p.write_text(json.dumps(records))
    else:
        p.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return str(p)


# -- AG-04 golden fixtures --------------------------------------------------
def test_run_eval_report_is_answer_and_tool_name_only(tmp_path):
    """AG-03: a flattened Report.save JSON is marked answer/tool-name only."""
    doc = {
        "use_case": "email_search", "model": "m", "n_samples": 1, "pass_rate": 1.0,
        "n_errors": 0, "duration_s": 1.0, "means": {}, "stds": {},
        "samples": [{"idx": 0, "question": "Who sent it?", "gold": "Alice",
                     "predicted": "<answer>Alice</answer>", "n_tools": 2,
                     "tool_calls": ["search_inbox", "read_email"],
                     "scores": {"token_f1": 0.9}, "error": None}],
    }
    path = _write(tmp_path, "report.json", doc, as_doc=True)
    ep = episodes_from_agenttune(path)[0]
    assert ep.source_format == "agenttune_report"
    assert ep.final_answer == "Alice"  # <answer> unwrapped
    assert ep.coverage["tool_call_names"] == "observed"
    assert ep.coverage["tool_call_arguments"] == "unavailable"
    assert ep.coverage["tool_results"] == "unavailable"
    assert ep.coverage_label() == "tool_names_only"
    # names present, arguments never fabricated
    names = [e.payload["name"] for e in ep.tool_call_events()]
    assert names == ["search_inbox", "read_email"]
    assert ep.source_scores == {"token_f1": 0.9}


def test_trajectory_jsonl_full_coverage_and_order(tmp_path):
    traj = {
        "task": [{"role": "user", "content": "weather+time in Paris?"}],
        "trajectory_id": "traj-1", "reward": 0.75,
        "final_response": "<answer>sunny at noon</answer>",
        "steps": [{"step_number": 0, "state": "s", "thought": "let me check",
                   "action": {"tool_calls": [
                       {"id": "c1", "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}},
                       {"id": "c2", "type": "function",
                        "function": {"name": "get_time", "arguments": '{"city": "Paris"}'}}]},
                   "observation": "{'get_weather': 'sunny', 'get_time': 'noon'}"}],
        "metadata": {"conversation": [{"role": "user", "content": "q"}],
                     "retrieved_chunk_ids": ["d1", "d3"], "tool_call_count": 2},
    }
    path = _write(tmp_path, "traj.jsonl", [traj])
    ep = episodes_from_agenttune(path)[0]
    assert ep.source_format == "agenttune_trajectory"
    assert ep.source_tier == "full"
    assert ep.final_answer == "sunny at noon"
    assert ep.coverage["tool_call_arguments"] == "observed"
    assert ep.coverage["parallel_grouping"] == "observed"
    # conversation present -> per-call pairing observed
    assert ep.coverage["call_result_pairing"] == "observed"
    assert ep.coverage["retrieved_contexts"] == "observed"
    # parallel calls kept in one turn, in order
    turns = ep.turns()
    assert len(turns) == 1
    assert [(c.get("function") or c)["name"] for c in turns[0]] == ["get_weather", "get_time"]


def test_trajectory_without_conversation_marks_pairing_unavailable(tmp_path):
    traj = {"task": "q", "trajectory_id": "t", "final_response": "a",
            "steps": [{"step_number": 0, "state": "s",
                       "action": {"name": "f", "arguments": {"x": 1}}, "observation": "ok"}],
            "metadata": {"tool_call_count": 1}}
    path = _write(tmp_path, "t.jsonl", [traj])
    ep = episodes_from_agenttune(path)[0]
    assert ep.coverage["call_result_pairing"] == "unavailable"
    assert ep.coverage["tool_results"] == "observed"


def test_rag_trace_jsonl_answer_and_retrieval_no_tool_name(tmp_path):
    trace = {"question": "What was revenue?", "final_answer": "<answer>42</answer>",
             "gold_answer": "42", "n_tool_calls": 1, "has_answer_tag": True,
             "reward": 0.9, "reward_components": {"answer": 0.9},
             "tool_calls": [{"query": {"q": "revenue"}, "result": "[chunk_id=c1] rev=42"}],
             "retrieved_chunk_ids": ["c1"], "gold_chunk_ids": ["c1"]}
    path = _write(tmp_path, "trace.jsonl", [trace])
    ep = episodes_from_agenttune(path)[0]
    assert ep.source_format == "agenttune_trace"
    assert ep.final_answer == "42"
    assert ep.coverage["tool_call_names"] == "unavailable"  # query/result has no name
    assert ep.coverage["tool_call_arguments"] == "observed"
    assert ep.coverage["retrieved_contexts"] == "observed"
    # name-less calls are kept as events but never fabricated into scored turns
    assert ep.turns() == []
    assert ep.to_trace().get("retrieved_contexts") == ["c1"]


def test_run_eval_row_is_a_task_not_an_episode(tmp_path):
    row = {"prompt": [{"role": "user", "content": "Ship item A?"}],
           "answer": ["ready"], "message_ids": ["m1"]}
    path = _write(tmp_path, "rows.jsonl", [row])
    ep = episodes_from_agenttune(path)[0]
    assert ep.source_format == "agenttune_run_eval_row"
    assert ep.metadata["is_task"] is True
    assert ep.events == []
    assert ep.final_answer is None
    assert ep.metadata["task"] == "Ship item A?"


def test_sample_actual_trace_import():
    """AG-04: (c) a Sample.actual_trace becomes an episode."""
    s = Sample(input="weather?", actual_output="sunny",
               actual_trace={"tool_calls": [[{"name": "get_weather", "arguments": {"city": "P"}}]],
                             "retrieved_contexts": ["d1"]})
    ep = episode_from_sample(s)
    assert ep.final_answer == "sunny"
    assert ep.coverage["tool_call_arguments"] == "observed"
    assert ep.turns()[0][0]["name"] == "get_weather"
    assert ep.to_trace()["retrieved_contexts"] == ["d1"]


def test_openai_messages_import_order():
    ep = episode_from_openai_messages([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "a", "content": "r"},
        {"role": "assistant", "content": "done"},
    ])
    types = [e.type for e in ep.events]
    assert types == ["observation", "tool_call", "tool_result", "text"]
    assert ep.final_answer == "done"


# -- AG-05: source reward/verdict preserved as PROVENANCE only --------------
def test_source_reward_preserved_not_promoted(tmp_path):
    traj = {"task": "q", "trajectory_id": "t", "final_response": "a", "reward": 0.9,
            "steps": [{"step_number": 0, "state": "s",
                       "action": {"name": "f", "arguments": {}}, "observation": "ok"}],
            "metadata": {}}
    path = _write(tmp_path, "t.jsonl", [traj])
    ep = episodes_from_agenttune(path)[0]
    assert ep.source_reward == 0.9
    # provenance only: it is NOT an AuditKit outcome field
    d = ep.to_dict()
    assert d["source_reward"] == 0.9
    assert "outcome_success" not in d


def test_inspect_reports_format_and_metrics(tmp_path):
    trace = {"question": "q", "final_answer": "a", "gold_answer": "a",
             "tool_calls": [], "retrieved_chunk_ids": ["c1"]}
    path = _write(tmp_path, "trace.jsonl", [trace])
    info = inspect_agenttune(path)
    assert info["row_count"] == 1
    assert info["formats"] == {"agenttune_trace": 1}
    assert "retrieval" in info["eligible_metrics"]
    assert "answer" not in info["eligible_metrics"]   # not a scorer


# -- no-extra import proof (subprocess, -W error, checks sys.modules) -------
def test_importing_agent_eval_pulls_no_heavy_extras():
    src = os.path.dirname(os.path.dirname(auditkit.__file__))
    code = (
        "import sys\n"
        "import auditkit.agent_eval as ae\n"
        "leaked = {'torch', 'transformers', 'agenttune'} & set(sys.modules)\n"
        "assert not leaked, leaked\n"
        "assert hasattr(ae, 'AgentEvalRunner')\n"
        "print('ok')\n"
    )
    env = {**os.environ, "PYTHONPATH": src}
    r = subprocess.run([sys.executable, "-W", "error", "-c", code],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout
