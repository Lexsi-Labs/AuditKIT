"""Reference-free agent evaluation (reviewer asks (a)/(b)/(c)): loop detection,
tool-permission least-privilege, the tool-selection judge, the first-class
AssertionOracle, and metadata as a metric-addressable field. All judges are
scripted (deterministic, no network); no sample carries a gold trajectory/answer.
"""

from __future__ import annotations

from auditkit.agent_eval import (
    AgentCase,
    AssertionOracle,
    FinalStateAssertion,
    episode_from_openai_messages,
    resolve_outcome,
)
from auditkit.metrics.agent import AgentLoopDetection, ToolPermission, ToolSelectionJudge, addressable
from auditkit.model import CallableModel
from auditkit.registry import METRICS
from auditkit.sample import Sample


def _trace(*turns):
    return {"trace": {"tool_calls": list(turns)}}


def call(name, **args):
    return {"name": name, "arguments": args}


# -- discoverability ---------------------------------------------------------
def test_new_metrics_registered_by_name():
    for n in ("agent_loop_detection", "tool_permission", "tool_selection"):
        assert n in METRICS.names()
        assert METRICS.get(n) is not None


# -- 1. AgentLoopDetection (stdlib, no judge, no reference) ------------------
def test_loop_repeated_identical_calls():
    m = AgentLoopDetection(repetition_threshold=3)
    ctx = _trace([call("f", x=1)], [call("f", x=1)], [call("f", x=1)])
    sc = m.score(Sample(input="t"), "", ctx)[0]
    assert sc.value == 0.0
    assert "repeated call" in sc.reason


def test_loop_below_repetition_threshold_is_clean():
    m = AgentLoopDetection(repetition_threshold=3)
    # two identical calls, different third: no repeat loop, no cycle
    ctx = _trace([call("f", x=1)], [call("f", x=1)], [call("g", x=2)])
    sc = m.score(Sample(input="t"), "", ctx)[0]
    assert sc.value == 1.0 and sc.reason is None


def test_loop_reasoning_stagnation():
    m = AgentLoopDetection(similarity_threshold=0.85)
    ctx = {"trace": {"messages": [
        {"role": "assistant", "content": "I will now search the database for the record."},
        {"role": "assistant", "content": "I will now search the database for the record."},
    ]}}
    sc = m.score(Sample(input="t"), "", ctx)[0]
    assert sc.value == 0.0 and "stagnation" in sc.reason


def test_loop_call_graph_cycle():
    m = AgentLoopDetection(repetition_threshold=99)  # disable the repeat signal
    ctx = _trace([call("a")], [call("b")], [call("a")], [call("b")])
    sc = m.score(Sample(input="t"), "", ctx)[0]
    assert sc.value == 0.0 and "cycle" in sc.reason


def test_loop_needs_no_reference_and_skips_empty():
    assert AgentLoopDetection().applicable(Sample(input="t"))  # reference-free
    assert AgentLoopDetection().score(Sample(input="t"), "", {}) == []  # nothing to measure


# -- 2. ToolPermission (deterministic, reference-free) ----------------------
def test_permission_allow_and_deny_fraction():
    sample = Sample(input="t", metadata={"allowed_tools": ["a", "b"], "denied_tools": ["c"]})
    ctx = _trace([call("a")], [call("c")], [call("z")])  # a ok, c denied, z not allowed
    sc = ToolPermission().score(sample, "", ctx)[0]
    assert sc.value == 1 / 3
    assert "c: denied" in sc.reason and "z: not in allowlist" in sc.reason


def test_permission_denylist_wins_over_allowlist():
    # a call both allowed and denied is unauthorized (a denial wins)
    sample = Sample(input="t", metadata={"allowed_tools": ["a"]})
    sc = ToolPermission(denied_tools=["a"]).score(sample, "", _trace([call("a")]))[0]
    assert sc.value == 0.0


def test_permission_allowlist_from_tools_field():
    sample = Sample(input="t", tools=[{"type": "function", "function": {"name": "ok"}}])
    assert ToolPermission().score(sample, "", _trace([call("ok")], [call("bad")]))[0].value == 0.5


def test_permission_skipped_without_a_policy():
    assert not ToolPermission().applicable(Sample(input="t"))
    assert ToolPermission().applicable(Sample(input="t", metadata={"denied_tools": ["x"]}))


# -- 3. ToolSelectionJudge (LLM judge, reference-free) ----------------------
def _sequenced_judge(replies):
    """A scripted judge that returns replies[i] for the i-th (sequential) call."""
    seq = iter(replies)
    return CallableModel(lambda prompts: [next(seq) for _ in prompts])


def test_tool_selection_mean_over_calls():
    judge = _sequenced_judge(["reason\nCHOICE: yes", "reason\nCHOICE: no"])
    m = ToolSelectionJudge(judge_model=judge)
    sc = m.score(Sample(input="book a flight"), "", _trace([call("search")], [call("delete")]))[0]
    assert sc.value == 0.5  # one justified, one not
    assert "search: yes" in sc.reason and "delete: no" in sc.reason


def test_tool_selection_is_reference_free_and_skips_empty():
    m = ToolSelectionJudge(judge_model=_sequenced_judge([]))
    assert not m.required_fields  # needs no gold trajectory/answer
    assert m.score(Sample(input="t"), "", {}) == []  # no calls -> skipped


def test_tool_selection_all_unparsed_is_unknown():
    m = ToolSelectionJudge(judge_model=_sequenced_judge(["I really cannot say either way."]))
    sc = m.score(Sample(input="t"), "", _trace([call("f")]))[0]
    assert sc.metadata.get("unknown") is True


# -- 4. AssertionOracle: first-class no-gold-standard outcome ---------------
def _episode(answer, *, final_state=None):
    ep = episode_from_openai_messages([{"role": "assistant", "content": answer}], final_answer=answer)
    ep.final_state = final_state
    return ep


def test_assertion_oracle_yields_a_real_verdict_not_diagnostic():
    oracle = AssertionOracle("the agent completed the booking",
                             judge_model=CallableModel(lambda p: ["looks done\nCHOICE: yes"]))
    case = AgentCase(id="c", task="book it", outcome=oracle)  # no gold answer/trajectory
    out = resolve_outcome(_episode("Booked."), case)
    assert out.verdict == "success"
    assert out.diagnostic is False  # first-class, unlike judge_outcome
    assert out.source["type"] == "assertion"


def test_assertion_oracle_failure_verdict():
    oracle = AssertionOracle("the agent completed the booking",
                             judge_model=CallableModel(lambda p: ["no evidence\nCHOICE: no"]))
    case = AgentCase(id="c", task="book it", outcome=oracle)
    assert resolve_outcome(_episode("I couldn't."), case).verdict == "failure"


def test_assertion_oracle_parse_failure_is_unknown():
    oracle = AssertionOracle("done?", judge_model=CallableModel(lambda p: ["hmm, unclear"]))
    case = AgentCase(id="c", task="t", outcome=oracle)
    out = resolve_outcome(_episode("x"), case)
    assert out.verdict == "unknown"
    assert out.evidence.get("parse_status") == "failed"


def test_assertion_criterion_read_from_metadata():
    # reviewer ask (b): the criterion is a user-supplied field in case.metadata
    oracle = AssertionOracle(judge_model=CallableModel(lambda p: ["CHOICE: yes"]))
    case = AgentCase(id="c", task="t", outcome=oracle, metadata={"assertion": "did the thing"})
    assert resolve_outcome(_episode("done"), case).verdict == "success"


def test_verified_state_assertion_still_wins_over_the_judge():
    # A companion state oracle overrides the judge when it verifies a real effect.
    state = FinalStateAssertion("shipment", equals="ready")
    oracle = AssertionOracle("agent shipped it", state_oracle=state,
                             judge_model=CallableModel(lambda p: ["CHOICE: yes"]))
    case = AgentCase(id="s", task="ship?", outcome=oracle)
    # state says blocked -> failure wins over the judge's 'yes'
    out = resolve_outcome(_episode("Shipped!", final_state={"shipment": "blocked"}), case)
    assert out.verdict == "failure"
    assert out.source["type"] == "final_state_assertion"
    # no state recorded -> fall through to the judge (first-class success)
    out2 = resolve_outcome(_episode("Shipped!"), case)
    assert out2.verdict == "success" and out2.diagnostic is False


def test_assertion_oracle_buildable_from_spec():
    from auditkit.agent_eval.outcome import from_spec, AssertionOracle as AO
    oracle = from_spec({"type": "assertion", "criterion": "did it",
                        "state_oracle": {"type": "final_state_assertion", "key": "k", "equals": "v"}})
    assert isinstance(oracle, AO)
    assert oracle.state_oracle is not None
    # identity folds in the criterion + companion (case digest depends on it)
    assert oracle.identity()["criterion"] == "did it"
    assert oracle.identity()["state_oracle"]["type"] == "final_state_assertion"


def test_assertion_oracle_case_digest_is_stable():
    # the oracle's identity() must feed AgentCase.digest()/to_dict() cleanly,
    # since a first-class outcome is recorded in the run file.
    case = AgentCase(id="c", task="t",
                     outcome=AssertionOracle("x", judge_model=CallableModel(lambda p: ["CHOICE: yes"])))
    d = case.to_dict()
    assert d["digest"].startswith("sha256:")
    assert d["outcome"]["type"] == "assertion" and d["outcome"]["criterion"] == "x"


# -- 5. metadata is a metric-addressable field ------------------------------
def test_addressable_prefers_attribute_then_metadata():
    s = Sample(input="t", target="gold", metadata={"allowed_tools": ["a"], "custom_ref": 42})
    assert addressable(s, "target") == "gold"          # a real attribute
    assert addressable(s, "allowed_tools") == ["a"]    # a metadata key
    assert addressable(s, "custom_ref") == 42          # an arbitrary user field
    assert addressable(s, "missing", default="d") == "d"


# -- C4 (fix plan #15) -----------------------------------------------------------------------------------------

def test_c4_progress_through_different_call_targets_is_not_stagnation():
    m = AgentLoopDetection()
    ctx = {"trace": {
        "tool_calls": [[call("get_weather", city="Paris")], [call("get_weather", city="Rome")]],
        "messages": [{"role": "assistant", "content": "Let me check the weather in Paris."},
                     {"role": "assistant", "content": "Let me check the weather in Rome."}]}}
    [sc] = m.score(Sample(input="t"), "", ctx)
    assert sc.value == 1.0


def test_c4_the_same_message_twice_is_still_stagnation_even_with_calls():
    m = AgentLoopDetection()
    ctx = {"trace": {
        "tool_calls": [[call("get_weather", city="Paris")], [call("get_weather", city="Rome")]],
        "messages": [{"role": "assistant", "content": "Let me check the weather."},
                     {"role": "assistant", "content": "Let me check the weather."}]}}
    [sc] = m.score(Sample(input="t"), "", ctx)
    assert sc.value == 0.0 and "stagnation" in sc.reason
