"""A5 (achievable part): AgentTune-compatible evaluation sidecar export.

Read-only emit keyed by source_id, strict-JSON JSONL, reproducible and
round-tripping. Never imports AgentTune; never mutates a source record.
"""

from __future__ import annotations

from auditkit.agent_eval import (
    AgentEvalRunner,
    AgentEvalSpec,
    episode_from_openai_messages,
    read_sidecar,
    sidecar_records,
    write_sidecar,
)


def _run():
    ep = episode_from_openai_messages(
        [{"role": "user", "content": "weather?"}, {"role": "assistant", "content": "sunny"}],
        source_id="traj-1", final_answer="sunny")
    ep.source_reward = 0.5
    ep.metadata["target"] = "sunny"          # derives an AnswerAssertion oracle
    spec = AgentEvalSpec(mode="recorded", episodes=[ep], scorers=[])
    return AgentEvalRunner().run(spec)


def test_sidecar_keyed_by_source_id_with_provenance():
    recs = sidecar_records(_run())
    assert len(recs) == 1
    r = recs[0]
    assert r["source_id"] == "traj-1" and r["key"] == "traj-1"
    assert r["outcome"]["verdict"] == "success"           # answer matches target
    assert r["coverage_label"] in ("answer_only", "full", "tool_names_only")
    # source reward preserved as provenance, never promoted to the outcome
    assert r["source_provenance"]["source_reward"] == 0.5


def test_sidecar_roundtrips_and_is_reproducible(tmp_path):
    result = _run()
    path = tmp_path / "sidecar.jsonl"
    text1 = write_sidecar(result, str(path))
    assert text1 == write_sidecar(result)                 # deterministic
    back = read_sidecar(str(path))
    assert back == sidecar_records(result)                # round-trips


def test_sidecar_falls_back_to_case_id_without_source_id():
    ep = episode_from_openai_messages(
        [{"role": "user", "content": "t"}, {"role": "assistant", "content": "a"}],
        case_id="local-1", final_answer="a")
    recs = sidecar_records(AgentEvalRunner().run(
        AgentEvalSpec(mode="recorded", episodes=[ep], scorers=[])))
    assert recs[0]["source_id"] is None
    assert recs[0]["key"] == "local-1"
