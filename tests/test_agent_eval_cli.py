"""CLI `agent` subcommand: --inspect / --dry-run make NO model call (UX-A1/A3),
eval --output round-trips through rescore, existing commands still route."""

from __future__ import annotations

import json

import pytest

from auditkit.cli import main


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


@pytest.fixture
def no_model_calls(monkeypatch):
    """Fail loudly if anything tries to resolve/call a model or endpoint."""
    from auditkit.model import AutoModel

    def _boom(*a, **k):
        raise AssertionError("a model was resolved during a read-only command")

    monkeypatch.setattr(AutoModel, "resolve", classmethod(lambda cls, *a, **k: _boom()))


def _traj_file(tmp_path):
    traj = {"task": "weather?", "trajectory_id": "t1", "reward": 0.5,
            "final_response": "<answer>sunny</answer>",
            "steps": [{"step_number": 0, "state": "s",
                       "action": {"name": "get_weather", "arguments": {"city": "P"}},
                       "observation": "sunny"}],
            "metadata": {"retrieved_chunk_ids": ["d1"], "tool_call_count": 1}}
    p = tmp_path / "traj.jsonl"
    p.write_text(json.dumps(traj) + "\n")
    return str(p)


def test_import_inspect_no_model_call(tmp_path, capsys, no_model_calls):
    main(["agent", "import-agenttune", _traj_file(tmp_path), "--inspect"])
    out = capsys.readouterr().out
    assert "agenttune_trajectory" in out
    assert "Rows: 1" in out
    assert "Eligible metrics" in out
    assert "tool_call_f1" in out


def test_dry_run_no_model_call(tmp_path, capsys, no_model_calls):
    cfg = {"mode": "deployed", "agent": "agent:http://localhost:9/run",
           "scorers": ["tool_call_f1", "task_completion"], "trials": 2,
           "cases": [{"id": "c1", "task": "ship item A",
                      "allowed_tools": ["ship"],
                      "outcome": {"type": "final_state_assertion", "key": "shipped", "equals": True}}]}
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    main(["agent", "eval", "--config", str(p), "--dry-run"])
    out = capsys.readouterr().out
    assert "dry run" in out.lower()
    assert "no endpoint or model was called" in out.lower()
    # trials>1 reported (not honored this release), and state-oracle caveat shown
    assert "A3" in out
    assert "unknown" in out.lower()


def test_eval_output_then_rescore_offline(tmp_path, capsys):
    """`agent eval --output` saves episodes; `agent rescore` replays them offline."""
    cfg = {"mode": "recorded", "import_path": _traj_file(tmp_path),
           "scorers": ["tool_call_validity"],
           "cases": [{"id": "t1", "task": "weather?", "allowed_tools": ["get_weather"],
                      "outcome": {"type": "answer_assertion", "reference": "sunny",
                                  "mode": "contains"}}]}
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    run_path = tmp_path / "run.json"
    main(["agent", "eval", "--config", str(cfg_path), "--output", str(run_path)])
    out = capsys.readouterr().out
    assert "Verified success: 1/1" in out

    saved = json.loads(run_path.read_text())
    assert saved["episodes"] and saved["cases"]

    main(["agent", "rescore", "--run", str(run_path), "--scorers", "tool_call_validity"])
    out2 = capsys.readouterr().out
    assert "Verified success: 1/1" in out2


def test_import_default_prints_summary(tmp_path, capsys, no_model_calls):
    main(["agent", "import-agenttune", _traj_file(tmp_path)])
    out = capsys.readouterr().out
    assert "Detected format" in out


def test_existing_list_command_still_routes(capsys):
    # An unrelated existing command must keep working after adding `agent`.
    main(["list", "metrics"])
    assert capsys.readouterr().out  # printed something, did not crash


# -- regression tests for confirmed stress/audit defects ---------------------
def test_inspect_lists_only_real_scorers(tmp_path, capsys, no_model_calls):
    main(["agent", "import-agenttune", _traj_file(tmp_path), "--inspect"])
    line = [ln for ln in capsys.readouterr().out.splitlines() if "Eligible metrics" in ln][0]
    assert "answer" not in line.split(":", 1)[1].replace("task_completion", "")


def test_dry_run_states_endpoint_evidence_and_rejects_bad_scorer(tmp_path, capsys, no_model_calls):
    cfg = {"mode": "deployed", "agent": "agent:http://localhost:9/run",
           "scorers": ["tool_call_f1"], "cases": [{"id": "c1", "task": "t"}]}
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    main(["agent", "eval", "--config", str(p), "--dry-run"])
    out = capsys.readouterr().out
    assert "Answer evidence" in out and "Trace evidence" in out
    cfg["scorers"] = ["tool_call_f1_TYPO"]
    p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="unknown scorer"):
        main(["agent", "eval", "--config", str(p), "--dry-run"])


def test_rescore_uses_saved_judge_keeps_digest_and_warns_without_one(tmp_path, capsys, monkeypatch):
    import auditkit.agent_eval as ae

    cfg = {"mode": "recorded", "import_path": _traj_file(tmp_path), "judge": "echo",
           "scorers": ["tool_call_validity"],
           "cases": [{"id": "t1", "task": "weather?", "environment": {"seed": 1}}]}
    cfg_path, run_path = tmp_path / "cfg.json", tmp_path / "run.json"
    cfg_path.write_text(json.dumps(cfg))
    main(["agent", "eval", "--config", str(cfg_path), "--output", str(run_path), "--cases"])
    assert "[t1] status=" in capsys.readouterr().out
    saved = json.loads(run_path.read_text())

    seen = {}
    real = ae.rescore

    def spy(episodes, scorers, *, cases=None, judge=None):
        seen.update(cases=cases, judge=judge)
        return real(episodes, scorers, cases=cases, judge=judge)

    monkeypatch.setattr(ae, "rescore", spy)
    main(["agent", "rescore", "--run", str(run_path), "--scorers", "tool_call_validity"])
    assert seen["judge"] == "echo"
    assert seen["cases"][0].digest() == saved["cases"][0]["digest"]

    saved["result"]["spec_identity"]["judge"] = None
    run_path.write_text(json.dumps(saved))
    main(["agent", "rescore", "--run", str(run_path), "--scorers", "task_completion"])
    assert "no judge" in capsys.readouterr().err


def test_eval_writes_output_despite_a_malformed_agent_reply(tmp_path, monkeypatch):
    from auditkit.model import AutoModel, Generated, Model, Result_

    class Odd(Model):
        name = "odd"

        def generate(self, requests):
            return [Result_(completions=[Generated(text="x", trace={"messages": [
                {"role": "assistant", "tool_calls": 5}]})])]

    monkeypatch.setattr(AutoModel, "resolve", classmethod(lambda cls, *a, **k: Odd()))
    cfg = {"mode": "deployed", "agent": "agent:http://localhost:9/run",
           "scorers": ["tool_call_validity"], "cases": [{"id": "c1", "task": "t"}]}
    cfg_path, run_path = tmp_path / "cfg.json", tmp_path / "run.json"
    cfg_path.write_text(json.dumps(cfg))
    main(["agent", "eval", "--config", str(cfg_path), "--output", str(run_path)])
    saved = json.loads(run_path.read_text())
    assert saved["result"]["rows"][0]["status"] == "target_error"
