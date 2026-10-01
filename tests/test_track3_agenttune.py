"""Track 3 (AgentTune -> AuditKIT) without glue code: a scripted AgentTune run
folder (TrajectoryDataset JSONL + an EventLog JSONL + lexsi_provenance.json) goes
through ``episodes_from_agenttune`` and is scored with the tool-call metrics,
including calls recorded in Cohere's own ``{"tool_name", "parameters"}`` shape
(Command R7B). Stdlib only."""

from __future__ import annotations

import json

import auditkit as ak
from auditkit.agent_eval import AgentCase
from auditkit.agent_eval.runner import rescore

R7B_RAW = ('<|START_THINKING|>I will add.<|END_THINKING|><|START_ACTION|>[{"tool_call_id": "0", '
           '"tool_name": "add", "parameters": {"a": 2, "b": 3}}]<|END_ACTION|>')
OPENAI_ADD = {"type": "function", "function": {"name": "add", "arguments": json.dumps({"a": 2, "b": 3})}}
COHERE_ADD = {"tool_call_id": "0", "tool_name": "add", "parameters": {"a": 2, "b": 3}}
COHERE_ADD11 = {"tool_call_id": "1", "tool_name": "add", "parameters": {"a": 1, "b": 1}}
ADD = {"name": "add", "arguments": {"a": 2, "b": 3}}
ADD11 = {"name": "add", "arguments": {"a": 1, "b": 1}}
PROV = {"schema": "lexsi.provenance/1", "library": "agenttune", "version": "1.1.0", "git_sha": None,
        "created_at": "2026-09-28T12:00:00Z", "base_model": "CohereLabs/tiny-aya", "method": "grpo",
        "inputs": [], "params": {}}


def _trajectory(tid, action, thought):
    return {"trajectory_id": tid, "task": "what is 2 plus 3", "reward": 1.0, "final_response": "5",
            "steps": [{"step_number": 0, "state": "", "action": action, "observation": "5",
                       "thought": thought},
                      {"step_number": 1, "state": "", "action": {}, "observation": "", "thought": "5"}],
            "metadata": {}}


def _run_dir(tmp_path):
    d = tmp_path / "agenttune_run"
    d.mkdir()
    rows = [
        # rollout-normalized OpenAI calls; the raw R7B text survives as the thought
        _trajectory("t-openai", {"tool_calls": [OPENAI_ADD]}, R7B_RAW),
        # Cohere call dicts recorded as-is, two in one step (a parallel group)
        _trajectory("t-cohere", {"tool_calls": [COHERE_ADD, COHERE_ADD11]}, R7B_RAW),
        # flat Cohere action
        _trajectory("t-flat", COHERE_ADD, R7B_RAW),
    ]
    (d / "trajectories.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    log = {"id": "e-1", "tier": "full", "events": [
        {"kind": "tool_call", "payload": {"action": {"tool_calls": [COHERE_ADD]}}},
        {"kind": "tool_result", "payload": {"output": "5"}},
        {"kind": "turn_complete", "payload": {}},
        {"kind": "text", "payload": {"text": "5"}}]}
    (d / "eventlog.jsonl").write_text(json.dumps(log) + "\n")
    (d / "lexsi_provenance.json").write_text(json.dumps(PROV))
    (d / "config.json").write_text(json.dumps({"model_type": "cohere2"}))  # not a trace: skipped
    return d


EXPECTED = {"e-1": [ADD], "t-openai": [ADD], "t-cohere": [[ADD, ADD11]], "t-flat": [ADD]}


def test_run_folder_to_tool_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    episodes = ak.episodes_from_agenttune(str(_run_dir(tmp_path)))
    assert [e.source_id for e in episodes] == ["e-1", "t-openai", "t-cohere", "t-flat"]
    assert all(e.metadata["provenance"] == PROV for e in episodes)
    assert all(e.coverage["tool_call_arguments"] == "observed" for e in episodes)

    # the Sample path: episodes as precomputed traces
    samples = [ak.Sample(input="what is 2 plus 3", actual_output=e.final_answer or "",
                         actual_trace=e.to_trace(), expected_tool_calls=EXPECTED[e.source_id])
               for e in episodes]
    r = ak.evaluate(samples, "precomputed", [ak.ToolCallF1(), ak.TrajectoryMatch()])
    assert r.failed_count == 0
    assert r.headline["tool_call_f1"] == 1.0 and r.headline["tool_call_exact"] == 1.0

    # the agent_eval path: recorded rescore against cases
    cases = [AgentCase(id=cid, task="what is 2 plus 3", reference_turns=ref) for cid, ref in EXPECTED.items()]
    res = rescore(episodes, ["tool_call_f1"], cases=cases)
    assert {row.case_id: row.diagnostics["tool_call_f1"]["value"] for row in res.rows} == \
        {cid: 1.0 for cid in EXPECTED}


def test_unparsed_cohere_text_is_scored_from_the_reply(tmp_path, monkeypatch):
    """A Command R7B reply scored as text (hf:/precomputed, no structured trace)."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    r = ak.evaluate([ak.Sample(input="what is 2 plus 3", actual_output=R7B_RAW, expected_tool_calls=[ADD])],
                    "precomputed", [ak.ToolCallF1()])
    assert r.headline["tool_call_f1"] == 1.0
