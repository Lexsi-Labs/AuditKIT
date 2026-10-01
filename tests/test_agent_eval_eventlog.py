"""AG-04 EventLog intake (duck-typed), heal-audit refusal, AG-05 report
provenance, UX-A9 redaction, duplicate call ids, and the UX-A2 example run.

Fixtures carry placeholder text only; the EventLog shapes mirror AgentTune
36d4724 ``agentic/events.py`` (``EventLog.from_trajectory`` /
``from_eval_dict`` / ``to_audit_records``) without importing AgentTune.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import pytest

from auditkit import load_agenttune
from auditkit.agent_eval import episode_from_eventlog, episodes_from_agenttune
from auditkit.agent_eval.types import REDACTED, redact
from auditkit.cli import main

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Kind(Enum):  # stand-in for agenttune EventKind
    TEXT = "text"
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TURN_COMPLETE = "turn_complete"
    REWARD = "reward"


@dataclass
class Ev:
    kind: Kind
    payload: dict
    token_span: Optional[tuple] = None
    logprobs: Optional[list] = None
    scope: Optional[str] = None


@dataclass
class Log:
    events: list = field(default_factory=list)
    tier: str = "full"
    id: str = "log-placeholder"


def _call(name: str, cid: str, arg: str) -> dict[str, Any]:
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps({"q": arg})}}


def _full_log() -> Log:
    """from_trajectory shape: a parallel step, a flat step, the terminal step."""
    return Log(events=[
        Ev(Kind.REASONING, {"text": "<placeholder thought>"}),
        Ev(Kind.TOOL_CALL, {"action": {"tool_calls": [_call("tool_a", "c1", "<p1>"),
                                                      _call("tool_b", "c2", "<p2>")]}}),
        Ev(Kind.TOOL_RESULT, {"output": "<placeholder blob>"}),
        Ev(Kind.TURN_COMPLETE, {"step": 0}),
        Ev(Kind.TOOL_CALL, {"action": {"name": "tool_a", "arguments": {"q": "<p3>"}}}),
        Ev(Kind.TOOL_RESULT, {"output": "<placeholder result>"}),
        Ev(Kind.REWARD, {"value": 0.25}, scope="step"),
        Ev(Kind.TURN_COMPLETE, {"step": 1}),
        Ev(Kind.TOOL_CALL, {"action": {}}),  # terminal step: no call
        Ev(Kind.TOOL_RESULT, {"output": None}),
        Ev(Kind.TURN_COMPLETE, {"step": 2}),
        Ev(Kind.TEXT, {"text": "<answer>placeholder</answer>"}, token_span=(0, 2),
           logprobs=[-0.1, -0.2]),
        Ev(Kind.REWARD, {"value": 1.0}, scope="episode"),
    ])


def test_eventlog_object_keeps_order_groups_and_provenance():
    ep = episode_from_eventlog(_full_log())
    assert ep.source_format == "agenttune_eventlog" and ep.source_tier == "full"
    assert ep.source_id == "log-placeholder"
    assert [len(t) for t in ep.turns()] == [2, 1]           # parallel group kept
    assert [c.get("id") for c in ep.turns()[0]] == ["c1", "c2"]
    assert ep.counters == {"n_tool_calls": 3, "n_turns": 3}  # terminal {} is not a call
    assert ep.final_answer == "placeholder" and ep.metadata["raw_output"].startswith("<answer>")
    assert ep.source_reward == 1.0                           # episode scope only
    assert ep.coverage["parallel_grouping"] == "observed"
    assert ep.coverage["call_result_pairing"] == "unavailable"  # 2 calls, one blob
    assert ep.coverage["tool_call_arguments"] == "observed"
    text = [e for e in ep.events if e.type == "text"][0]
    assert text.payload["token_span"] == [0, 2] and text.payload["logprobs"] == [-0.1, -0.2]
    json.loads(ep.to_json())  # strict JSON despite the tuple span


def test_eventlog_json_light_tier_has_no_turn_structure(tmp_path):
    # from_eval_dict: all calls, then all outputs; no TURN_COMPLETE
    log = {"id": "light-1", "tier": "light", "events": [
        {"kind": "tool_call", "payload": {"action": {"name": "tool_a", "arguments": {}}}},
        {"kind": "tool_call", "payload": {"action": {"name": "tool_b", "arguments": {}}}},
        {"kind": "tool_result", "payload": {"output": "<o1>"}},
        {"kind": "tool_result", "payload": {"output": "<o2>"}},
    ]}
    p = tmp_path / "logs.jsonl"
    p.write_text(json.dumps(log) + "\n" + json.dumps(dict(log, id="light-2")) + "\n")
    eps = episodes_from_agenttune(str(p))
    assert [e.source_id for e in eps] == ["light-1", "light-2"]
    ep = eps[0]
    assert [len(t) for t in ep.turns()] == [1, 1]   # never guessed parallel
    assert ep.coverage["parallel_grouping"] == "unavailable"
    assert ep.coverage["call_result_pairing"] == "unavailable"
    assert ep.coverage["final_answer"] == "unavailable" and ep.final_answer is None


def test_eventlog_rejects_non_log():
    with pytest.raises(ValueError, match="events"):
        episode_from_eventlog({"tool_calls": [], "tool_outputs": []})


HEAL_RECORDS = [  # EventLog.to_audit_records, as Project.heal writes them
    {"trajectory_id": "0000-placeholder", "stage_name": "unknown", "tool_name": "unknown",
     "stage_type": "tool_call", "state_snapshot": {"arguments": {}}, "status": "ok"},
    {"trajectory_id": "0000-placeholder", "stage_name": "tool_a {\"q\": \"<p>\"}",
     "tool_name": "tool_a", "stage_type": "tool_call",
     "state_snapshot": {"arguments": {"q": "<p>"}}, "status": "ok"},
]


def test_heal_audit_records_are_refused_not_empty_trajectories(tmp_path):
    p = tmp_path / "heal.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in HEAL_RECORDS) + "\n")
    with pytest.raises(ValueError, match="audit record"):
        load_agenttune(str(p))
    with pytest.raises(ValueError, match="episode_from_eventlog"):
        episodes_from_agenttune(str(p))


def test_report_provenance_and_source_verdict(tmp_path):
    doc = {"use_case": "generic", "model": "<placeholder model>", "timestamp": "t0",
           "n_samples": 2, "pass_rate": 0.5, "n_errors": 0, "duration_s": 1.0,
           "means": {}, "stds": {},
           "samples": [{"idx": 0, "question": "<q>", "gold": "<g>", "predicted": "<g>",
                        "n_tools": 0, "tool_calls": [], "scores": {}, "error": None,
                        "passed": True},
                       {"idx": 1, "question": "<q>", "gold": "<g>", "predicted": "<x>",
                        "n_tools": 0, "tool_calls": [], "scores": {}, "error": None}]}
    p = tmp_path / "report.json"
    p.write_text(json.dumps(doc))
    eps = episodes_from_agenttune(str(p))
    assert eps[0].metadata["source_report"] == {
        "model": "<placeholder model>", "use_case": "generic", "pass_rate": 0.5,
        "n_samples": 2, "n_errors": 0, "timestamp": "t0"}
    assert eps[0].source_verdict is True
    assert eps[1].source_verdict is None  # absent in the source: never derived


def test_duplicate_call_ids_are_kept_not_merged():
    log = Log(events=[
        Ev(Kind.TOOL_CALL, {"action": {"tool_calls": [_call("tool_a", "dup", "<p1>"),
                                                      _call("tool_a", "dup", "<p2>")]}}),
        Ev(Kind.TURN_COMPLETE, {"step": 0}),
    ])
    ep = episode_from_eventlog(log)
    assert [e.call_id for e in ep.tool_call_events()] == ["dup", "dup"]
    assert len(ep.turns()[0]) == 2


def test_redact_secret_strings_and_keys():
    obj = {"payload": {"api_key": "k", "text": "token=SECRET-PLACEHOLDER ok"},
           "list": ["SECRET-PLACEHOLDER"], "n": 1}
    out = redact(obj, secrets=["SECRET-PLACEHOLDER"], keys=["API_KEY"])
    assert out == {"payload": {"api_key": REDACTED, "text": f"token={REDACTED} ok"},
                   "list": [REDACTED], "n": 1}
    assert obj["list"] == ["SECRET-PLACEHOLDER"]  # input untouched
    assert redact(obj) == obj                     # opt-in: nothing configured, nothing changed


def test_cli_import_redacts_env_secret(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("AK_TEST_SECRET", "SECRET-PLACEHOLDER")
    traj = {"task": "<task>", "trajectory_id": "t1", "final_response": "SECRET-PLACEHOLDER",
            "steps": [{"step_number": 0, "state": "s",
                       "action": {"name": "tool_a", "arguments": {"auth": "x"}},
                       "observation": "<obs>"}]}
    src = tmp_path / "traj.jsonl"
    src.write_text(json.dumps(traj) + "\n")
    out = tmp_path / "eps.json"
    main(["agent", "import-agenttune", str(src), "-o", str(out),
          "--redact-env", "AK_TEST_SECRET", "--redact-key", "auth"])
    text = out.read_text()
    assert "SECRET-PLACEHOLDER" not in text and REDACTED in text
    assert json.loads(text)[0]["events"][0]["payload"]["arguments"]["auth"] == REDACTED
    monkeypatch.delenv("AK_TEST_SECRET")
    with pytest.raises(SystemExit):
        main(["agent", "import-agenttune", str(src), "-o", str(out), "--redact-env", "AK_TEST_SECRET"])


def test_offline_example_runs():
    """UX-A2: the documented offline example runs on a fresh stdlib install."""
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))
    r = subprocess.run([sys.executable, os.path.join(ROOT, "examples", "agent_eval_offline.py")],
                       capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "OK:" in r.stdout


@pytest.mark.parametrize("action", [{}, {"stage": "plan"}])   # terminal step, DECIDE
def test_eventlog_with_no_call_actions_is_observed_zero_calls(action):
    from auditkit.agent_eval import AgentCase
    from auditkit.agent_eval.runner import rescore
    log = Log(events=[Ev(Kind.TOOL_CALL, {"action": action}), Ev(Kind.TURN_COMPLETE, {"step": 0}),
                      Ev(Kind.TEXT, {"text": "<answer>placeholder</answer>"})])
    ep = episode_from_eventlog(log)
    ep.case_id = "c1"
    assert ep.to_trace()["tool_calls"] == [] and "tool_calls_unavailable" not in ep.to_trace()
    for ref, want in (([[{"name": "f", "arguments": {}}]], 0.0), ([], 1.0)):   # skipped tool, irrelevance
        r = rescore([ep], ["tool_call_f1"], cases=[AgentCase(id="c1", task="t", reference_turns=ref)])
        assert not r.rows[0].ineligible
        assert r.rows[0].diagnostics["tool_call_f1"]["value"] == want
