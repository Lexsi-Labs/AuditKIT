"""PR #1's non-metric additions: secret redaction, the agent_eval outcome oracles,
the JSONL and AgentTune loaders, and the dependency-compatibility checker.

Everything is offline and deterministic.
"""

from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pytest

import auditkit as ak

ae = pytest.importorskip("auditkit.agent_eval")
from auditkit.agent_eval import outcome as O  # noqa: E402
from auditkit.agent_eval.types import REDACTED, detect_secrets, redact  # noqa: E402
from auditkit.compat import _satisfies, parse_requirement  # noqa: E402
from auditkit.loaders import load_agenttune, load_jsonl  # noqa: E402

OPENAI = "sk-proj-" + "A1b2C3d4E5f6G7h8I9j0K1l2"
GITHUB = "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3"
AWS = "AKIA" + "ABCDEFGHIJKLMNOP"


# -- secret redaction ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("secret", [OPENAI, GITHUB, AWS, "Bearer abc123def456ghi789",   # a token shape (D3)
                                    "eyJhbGciOi.eyJzdWIiOi.SflKxwRJSM",
                                    "-----BEGIN RSA PRIVATE KEY-----\nabc\n-----END RSA PRIVATE KEY-----"])
def test_strong_patterns_are_redacted_anywhere(secret):
    out = redact({"log": [f"called with {secret} ok"]}, auto=True)
    assert secret not in json.dumps(out) and REDACTED in out["log"][0]


def test_reproducibility_values_survive_auto_redaction():
    obj = {"digest": "a" * 64, "run": "550e8400-e29b-41d4-a716-446655440000", "sha1": "0" * 40, "max_tokens": 256}
    assert redact(obj, auto=True) == obj


def test_a_credential_key_hides_its_whole_value():
    out = redact({"headers": {"Authorization": "Bearer x", "api_key": {"nested": "v"}}}, auto=True)
    assert out == {"headers": {"Authorization": REDACTED, "api_key": {"nested": REDACTED}}}


def test_explicit_keys_and_secrets():
    out = redact({"Token_Field": "abc", "note": "pw=hunter2 and hunter2"}, keys=["token_field"], secrets=["hunter2"])
    assert out == {"Token_Field": REDACTED, "note": f"pw={REDACTED} and {REDACTED}"}


def test_allow_whitelists_one_token():
    assert redact({"k": f"x {OPENAI}"}, auto=True, allow=[OPENAI]) == {"k": f"x {OPENAI}"}


def test_redact_does_not_mutate_its_input():
    obj = {"a": [OPENAI]}
    redact(obj, auto=True)
    assert obj == {"a": [OPENAI]}


def test_detect_names_where_and_what_never_the_value():
    found = detect_secrets({"cfg": {"password": "p", "url": f"https://x?k={OPENAI}"}})
    assert {(f["path"], f["kind"]) for f in found} == {("cfg.password", "secret_key_name"), ("cfg.url", "openai_key")}
    assert OPENAI not in json.dumps(found)


def test_a_secret_used_as_a_key_is_redacted():
    assert OPENAI not in json.dumps(redact({OPENAI: "cached"}, auto=True))


def test_a_count_under_a_credential_like_key_is_left_alone():
    # count-like keys (DEFAULT_COUNT_KEYS: *_length, retries, ttl, ...) hold settings, not secrets
    kept = {"password_min_length": 12, "token_retries": 3, "ttl": 300}
    assert redact(kept, auto=True) == kept


@pytest.mark.parametrize("record", [{"pin": 4821}, {"otp": 123456}, {"token": 5}, {"password": 1234.5}])
def test_a_numeric_secret_under_a_credential_key_is_redacted(record):
    (key, value), = record.items()
    assert redact(record, auto=True)[key] != value


def test_banking_prose_about_bearer_instruments_is_kept():
    note = "The fund holds bearer instruments and registered bonds."
    assert redact({"note": note}, auto=True) == {"note": note}


# -- outcome oracles ------------------------------------------------------------------------------------------

def ep(answer=None, state=None, artifacts=None):
    return NS(final_answer=answer, final_state=state, artifacts=artifacts, metadata={})


@pytest.mark.parametrize("oracle,episode,verdict", [
    (O.FinalStateAssertion("status", equals="done"), ep(state={"status": "done"}), "success"),
    (O.FinalStateAssertion("status", equals="done"), ep(state={"status": "open"}), "failure"),
    (O.FinalStateAssertion("status", equals="done"), ep(state=None), "unknown"),      # no snapshot is not a failure
    (O.FinalStateAssertion("status", equals="done"), ep(state={}), "unknown"),        # key absent
    (O.FinalStateAssertion("tags", contains="vip"), ep(state={"tags": ["vip", "x"]}), "success"),
    (O.FinalStateAssertion("n", contains=1), ep(state={"n": 1}), "failure"),          # contains on an int
    (O.FinalStateAssertion("flag", equals=False), ep(state={"flag": False}), "success"),
    (O.FinalStateAssertion("flag"), ep(state={"flag": None}), "failure"),             # present means not None
    (O.ArtifactAssertion("report.md", contains="total"), ep(artifacts={"report.md": "total: 5"}), "success"),
    (O.ArtifactAssertion("report.md"), ep(artifacts=None), "unknown"),
])
def test_state_and_artifact_oracles(oracle, episode, verdict):
    assert oracle.evaluate(episode).verdict == verdict


@pytest.mark.parametrize("ref,answer,mode,verdict", [
    ("Paris", "paris.", "normalized", "success"),
    ("$5", "5", "normalized", "success"),
    ("50%", "50", "normalized", "success"),
    ("Paris", "paris.", "exact", "failure"),
    ("Paris", "The capital is Paris.", "contains", "success"),
    ("Paris", None, "normalized", "unknown"),
    (None, "Paris", "normalized", "unknown"),
])
def test_answer_oracle(ref, answer, mode, verdict):
    assert O.AnswerAssertion(ref, mode=mode).evaluate(ep(answer)).verdict == verdict


def test_the_answer_reference_falls_back_to_the_case_target():
    case = NS(metadata={"target": "42"})
    assert O.AnswerAssertion().evaluate(ep("42"), case).verdict == "success"


def test_a_thousands_separator_does_not_break_a_normalized_match():
    assert O.AnswerAssertion("1000").evaluate(ep("1,000")).verdict == "success"


def test_contains_does_not_match_inside_a_longer_number():
    assert O.AnswerAssertion("5", mode="contains").evaluate(ep("The fee is 15%")).verdict == "failure"


@pytest.mark.parametrize("ret,verdict", [
    (True, "success"), (False, "failure"), ("unknown", "unknown"), ("failure", "failure"),
    ("FAILURE", "unknown"), ("fail", "unknown"), (None, "unknown"),
])
def test_custom_predicate_verdicts(ret, verdict):
    assert O.CustomPredicate(lambda e, c: ret, version="1").evaluate(ep()).verdict == verdict


def test_a_raising_predicate_is_an_error_and_a_version_is_required():
    assert O.CustomPredicate(lambda e, c: 1 / 0, version="1").evaluate(ep()).verdict == "error"
    with pytest.raises(ValueError):
        O.CustomPredicate(lambda e, c: True, version=" ")


def test_a_verified_state_check_overrides_the_assertion_judge():
    judge_calls = []
    oracle = O.AssertionOracle("did it finish?", judge_model=lambda ps: judge_calls.append(ps) or ["CHOICE: yes"],
                               state_oracle=O.FinalStateAssertion("status", equals="done"))
    assert oracle.evaluate(ep(state={"status": "open"}), NS(task="t", metadata={})).verdict == "failure"
    assert judge_calls == []


def test_an_assertion_oracle_without_a_criterion_is_unknown():
    assert O.AssertionOracle(judge_model="x").evaluate(ep("a"), NS(task="t", metadata={})).verdict == "unknown"


@pytest.mark.parametrize("spec,cls", [
    ({"type": "final_state_assertion", "key": "k", "equals": 1}, O.FinalStateAssertion),
    ({"type": "artifact_assertion", "key": "k"}, O.ArtifactAssertion),
    ({"type": "answer_assertion", "reference": "x", "mode": "exact"}, O.AnswerAssertion),
    ({"type": "assertion", "criterion": "c", "state_oracle": {"type": "final_state_assertion", "key": "k"}},
     O.AssertionOracle),
])
def test_from_spec(spec, cls):
    assert isinstance(O.from_spec(spec), cls)


@pytest.mark.parametrize("spec", [{"type": "custom_predicate"}, {"type": "nope"}, "final_state_assertion"])
def test_from_spec_rejects_what_json_cannot_express(spec):
    with pytest.raises(ValueError):
        O.from_spec(spec)


def test_resolve_and_external_action():
    state = O.FinalStateAssertion("k")
    assert O.resolve_outcome(ep(), NS(outcome=None)).verdict == "unknown"
    assert O.requires_external_action(NS(outcome=state))
    assert O.requires_external_action(NS(outcome=O.AssertionOracle("c", state_oracle=state)))
    assert not O.requires_external_action(NS(outcome=O.AnswerAssertion("x")))


# -- JSONL loader ------------------------------------------------------------------------------------------------

def jsonl(tmp_path, *rows):
    p = tmp_path / "d.jsonl"
    p.write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n")
    return str(p)


def test_load_jsonl_fills_fields_maps_keys_and_sets_the_kind(tmp_path):
    a, r, g = load_jsonl(jsonl(tmp_path,
                               {"q": "weather?", "tools": [], "id": 7},
                               {"q": "doc?", "retrieval_context": ["x"], "source": "kb", "metadata": {"source": "m"}},
                               {"q": "hi", "answer": "hello"}, ""),
                         field_map={"q": "input", "answer": "target"})
    assert (a.kind, a.id) == (ak.TaskKind.AGENT, "7")
    assert r.kind == ak.TaskKind.RAG and r.metadata == {"source": "m"}          # explicit metadata wins
    assert (g.input, g.target) == ("hi", "hello")


@pytest.mark.parametrize("row", ["{not json", json.dumps([1]), json.dumps({"target": "x"}),
                                 json.dumps({"input": "x", "metadata": "notadict"})])
def test_load_jsonl_rejects_bad_rows_with_the_line_number(tmp_path, row):
    with pytest.raises(ValueError, match="line 2"):
        load_jsonl(jsonl(tmp_path, {"input": "ok"}, row))


# -- AgentTune loader ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("row,want", [
    ({"prompt": "Q?", "answer": ["A1", "A2"], "message_ids": '["m1", ["m2"]]'},
     {"input": "Q?", "target": "A1", "reference_contexts": ["m1", "m2"]}),            # JSON-string id list, nested
    ({"prompt": [{"role": "system", "content": "S"}, {"role": "user", "content": "U1"},
                 {"role": "assistant", "content": "x"}, {"role": "user", "content": "U2"}], "gold_answer": "G",
      "question_id": 3, "gold_path": "null"},
     {"input": "U2", "target": "G", "id": "3", "reference_contexts": None}),            # last user turn; null = no gold
    ({"prompt": "Q", "answer": [], "gold_answer": "G", "gold_chunk_ids": "1.50"},
     {"target": "G", "reference_contexts": ["1.50"]}),                                  # numeric-looking id kept raw
])
def test_agenttune_dataset_rows(tmp_path, row, want):
    [s] = load_agenttune(jsonl(tmp_path, row))
    assert {k: getattr(s, k) for k in want} == want


def test_agenttune_trace_and_trajectory_records(tmp_path):
    trace, traj = load_agenttune(jsonl(tmp_path,
        {"question": "Q", "gold_answer": "A", "final_answer": "<answer> A </answer> extra", "retrieved_chunk_ids": 5,
         "gold_chunk_ids": [5], "reward": 1.0},
        {"task": "T", "steps": [{"action": {"name": "search", "arguments": {"q": "x"}}},
                                {"action": {"tool_calls": [{"name": "a", "arguments": {}},
                                                           {"name": "b", "arguments": {}}]}},
                                {"action": {}}],
         "final_response": 42, "reward": 0.5, "trajectory_id": "t1"}))
    assert trace.actual_output == "A" and trace.actual_trace == {"retrieved_contexts": ["5"]}
    assert trace.reference_contexts == ["5"] and trace.metadata["reward"] == 1.0
    assert traj.actual_output == "42" and traj.id == "t1" and traj.kind == ak.TaskKind.AGENT
    assert [[c["name"] for c in t] for t in traj.actual_trace["tool_calls"]] == [["search"], ["a", "b"]]


def test_agenttune_row_without_a_user_message_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_agenttune(jsonl(tmp_path, {"prompt": [{"role": "system", "content": "S"}], "answer": "A"}))


# -- dependency compatibility ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("v,op,t,want", [
    ("1.10", ">", "1.9", True),                   # numeric, not lexical
    ("2.0.0rc1", "<", "2.0.0", True),
    ("2.0.0", "==", "2.0", True),                 # trailing zeros
    ("2.0.post1", ">", "2.0", True),
    ("5.17.0.dev0", "<", "5.17.0", True),
    ("2.1+cu130", "==", "2.1", True),             # local version ignored
    ("1.4.2", "~=", "1.4", None),                 # not modelled: skipped, not guessed
    ("2.1", "==", "2.*", None),
])
def test_version_comparison(v, op, t, want):
    assert _satisfies(v, op, t) is want


@pytest.mark.parametrize("req,want", [
    ("flashinfer_python[cu13]==0.6.18", ("flashinfer-python", [("==", "0.6.18")])),
    ("vllm (>=0.15,<0.20)", ("vllm", [(">=", "0.15"), ("<", "0.20")])),
    ("numpy", ("numpy", [])),
    ("torch>=2; sys_platform == 'linux'", None),  # markers are skipped
    ("pkg @ https://x/y.whl", None),
])
def test_parse_requirement(req, want):
    assert parse_requirement(req) == want


# -- cases and episodes ------------------------------------------------------------------------------------------

def test_case_validation():
    with pytest.raises(ValueError):
        ae.AgentCase(" ", "task")
    with pytest.raises(ValueError):
        ae.AgentCase("c", "")
    with pytest.raises(ValueError):
        ae.AgentCase("c", "t", outcome={"type": "final_state_assertion"})       # spec without its key
    with pytest.raises(ValueError):
        ae.validate_cases([ae.AgentCase("c", "t"), ae.AgentCase("c", "u")])
    with pytest.raises(TypeError):
        ae.validate_cases([{"id": "c"}])


def test_the_case_digest_follows_task_and_oracle():
    a = ae.AgentCase("c", "t", outcome={"type": "answer_assertion", "reference": "42"})
    assert a.digest() == ae.AgentCase("c", "t", outcome=O.AnswerAssertion("42")).digest()
    assert a.digest() != ae.AgentCase("c", "t", outcome=O.AnswerAssertion("43")).digest()
    assert a.digest() != ae.AgentCase("c", "t2", outcome=O.AnswerAssertion("42")).digest()


def test_a_criterion_read_from_metadata_is_part_of_the_digest():
    a = ae.AgentCase("c", "t", outcome=O.AssertionOracle(judge_model="x"), metadata={"assertion": "refund issued"})
    b = ae.AgentCase("c", "t", outcome=O.AssertionOracle(judge_model="x"), metadata={"assertion": "no refund"})
    assert a.digest() != b.digest()


MESSAGES = [
    {"role": "user", "content": "weather in Paris and Rome?"},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "1", "type": "function", "function": {"name": "w", "arguments": '{"city": "Paris"}'}},
        {"id": "2", "type": "function", "function": {"name": "w", "arguments": '{"city": "Rome"}'}}]},
    {"role": "tool", "tool_call_id": "1", "content": "sunny"},
    {"role": "tool", "tool_call_id": "2", "content": "rain"},
    {"role": "assistant", "content": "Paris sunny, Rome rain."},
]


def test_episode_from_openai_messages():
    e = ae.episode_from_openai_messages(MESSAGES)
    calls = [x for x in e.events if x.type == "tool_call"]
    results = [x for x in e.events if x.type == "tool_result"]
    assert e.final_answer == "Paris sunny, Rome rain." and e.counters["n_tool_calls"] == 2
    assert len({c.turn_id for c in calls}) == 1                               # one parallel group
    assert {r.call_id for r in results} == {"1", "2"}
    assert e.coverage["final_state"] == "unavailable" and e.coverage["tool_results"] == "observed"


def test_an_answer_only_transcript_observes_zero_calls():
    e = ae.episode_from_openai_messages([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}])
    assert e.counters["n_tool_calls"] == 0 and e.coverage["tool_call_names"] == "observed"


def test_episodes_round_trip_through_strict_json():
    e = ae.episode_from_openai_messages(MESSAGES, case_id="c", retrieved_contexts=["d1"])
    again = ae.AgentEpisode.from_dict(json.loads(json.dumps(e.to_dict())))
    assert again.to_dict() == e.to_dict()
