"""AG-01 (case validation + digest) and AG-02 (ordered/parallel round trip)."""

from __future__ import annotations

import pytest

from auditkit.agent_eval import (
    AgentCase,
    AgentEpisode,
    AgentEvent,
    AnswerAssertion,
    FinalStateAssertion,
    episode_from_openai_messages,
    validate_cases,
)


# -- AG-01: validation ------------------------------------------------------
def test_missing_id_fails():
    with pytest.raises(ValueError):
        AgentCase(id="", task="do a thing")


def test_missing_task_fails():
    with pytest.raises(ValueError):
        AgentCase(id="c1", task="")


def test_duplicate_ids_fail_validation():
    cases = [AgentCase(id="dup", task="a"), AgentCase(id="dup", task="b")]
    with pytest.raises(ValueError, match="duplicate"):
        validate_cases(cases)


def test_unique_ids_pass_validation():
    cases = [AgentCase(id="a", task="x"), AgentCase(id="b", task="y")]
    assert validate_cases(cases) is cases


# -- AG-01: digest changes with task or oracle ------------------------------
def test_digest_changes_with_task():
    a = AgentCase(id="c", task="task one")
    b = AgentCase(id="c", task="task two")
    assert a.digest() != b.digest()


def test_digest_changes_with_oracle():
    a = AgentCase(id="c", task="t", outcome=FinalStateAssertion("k", equals="ready"))
    b = AgentCase(id="c", task="t", outcome=FinalStateAssertion("k", equals="blocked"))
    c = AgentCase(id="c", task="t", outcome=AnswerAssertion("ready"))
    assert a.digest() != b.digest() != c.digest() and a.digest() != c.digest()


def test_digest_stable_across_instances():
    a = AgentCase(id="c", task="t", outcome=FinalStateAssertion("k", equals="ready"))
    b = AgentCase(id="c", task="t", outcome=FinalStateAssertion("k", equals="ready"))
    assert a.digest() == b.digest()


# -- AG-02: parallel calls survive import/export in original order ----------
def _parallel_episode():
    return episode_from_openai_messages([
        {"role": "user", "content": "weather and time in Paris?"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}},
            {"id": "c2", "type": "function",
             "function": {"name": "get_time", "arguments": '{"city": "Paris"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
        {"role": "tool", "tool_call_id": "c2", "content": "noon"},
        {"role": "assistant", "content": "sunny at noon"},
    ], final_answer="sunny at noon")


def test_parallel_group_is_one_turn():
    ep = _parallel_episode()
    turns = ep.turns()
    # both parallel calls land in ONE turn, in original order
    assert len(turns) == 1
    names = [(c.get("function") or c).get("name") for c in turns[0]]
    assert names == ["get_weather", "get_time"]


def test_parallel_calls_survive_json_round_trip():
    ep = _parallel_episode()
    back = AgentEpisode.from_dict(ep.to_dict())
    assert back.turns() == ep.turns()
    assert back.to_dict() == ep.to_dict()


def test_tool_result_pairing_survives_round_trip():
    ep = _parallel_episode()
    back = AgentEpisode.from_dict(ep.to_dict())
    results = [(e.call_id, e.payload.get("output"))
               for e in back.events if e.type == "tool_result"]
    assert results == [("c1", "sunny"), ("c2", "noon")]


def test_missing_timestamp_is_none_not_fabricated():
    ep = _parallel_episode()
    assert all(e.timestamp is None for e in ep.events)
    back = AgentEpisode.from_dict(ep.to_dict())
    assert all(e.timestamp is None for e in back.events)


def test_event_indices_are_ordered():
    ep = _parallel_episode()
    assert [e.index for e in ep.events] == list(range(len(ep.events)))


def test_coverage_and_schema_present():
    ep = _parallel_episode()
    assert ep.schema_version == "agent_eval/1"
    assert ep.coverage["tool_call_arguments"] == "observed"
    assert ep.coverage["final_state"] == "unavailable"
    assert ep.coverage_label() == "full"


def test_bad_mode_rejected():
    with pytest.raises(ValueError):
        AgentEpisode(mode="teleported")


def test_event_from_dict_roundtrip():
    e = AgentEvent(index=0, role="assistant", type="tool_call",
                   payload={"name": "f"}, call_id="c1", turn_id=3)
    assert AgentEvent.from_dict(e.to_dict()) == e


def test_plain_dict_outcome_is_converted_or_rejected_clearly():
    from auditkit.agent_eval import AgentEpisode, ArtifactAssertion, rescore
    case = AgentCase(id="c", task="t", outcome={"type": "artifact_assertion",
                                                 "key": "decision", "equals": "ready"})
    assert isinstance(case.outcome, ArtifactAssertion)
    ep = AgentEpisode(case_id="c", artifacts={"decision": "ready"})
    row = rescore([ep], ["tool_call_validity"], cases=[case]).rows[0]
    assert (row.status, row.outcome["verdict"]) == ("completed", "success")
    with pytest.raises(ValueError, match="invalid oracle spec"):
        AgentCase(id="c", task="t", outcome={"type": "nope"})


# -- D6 (fix plan #15): the digest follows the metadata an oracle reads ---------------------------------------

def test_d6_the_answer_reference_from_metadata_is_part_of_the_digest():
    a = AgentCase(id="c", task="t", outcome=AnswerAssertion(), metadata={"target": "42"})
    b = AgentCase(id="c", task="t", outcome=AnswerAssertion(), metadata={"target": "43"})
    assert a.digest() != b.digest()


def test_d6_the_assertion_criterion_from_metadata_is_part_of_the_digest():
    from auditkit.agent_eval.outcome import AssertionOracle
    a = AgentCase(id="c", task="t", outcome=AssertionOracle(judge_model="x"), metadata={"assertion": "refund issued"})
    b = AgentCase(id="c", task="t", outcome=AssertionOracle(judge_model="x"), metadata={"assertion": "no refund"})
    assert a.digest() != b.digest()


def test_d6_unrelated_metadata_and_explicit_references_keep_the_digest():
    base = AgentCase(id="c", task="t", outcome=AnswerAssertion("42"))
    noted = AgentCase(id="c", task="t", outcome=AnswerAssertion("42"), metadata={"target": "99", "note": "x"})
    assert base.digest() == noted.digest()          # an explicit reference ignores metadata["target"]
    plain = AgentCase(id="c", task="t", outcome=FinalStateAssertion("k", equals="ready"))
    assert plain.digest() == AgentCase(id="c", task="t", outcome=FinalStateAssertion("k", equals="ready"),
                                       metadata={"tags": ["a"]}).digest()
