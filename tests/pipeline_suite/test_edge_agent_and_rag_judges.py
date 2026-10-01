"""Edge cases for PR #1's reference-free agent metrics (agent_loop_detection,
tool_permission, tool_selection) and reference-free RAG judges (hallucination,
answer_relevancy, response_groundedness, context_relevance, and the shared verdict parser).

Judges are scripted and there is no network. The unreadable-verdict cases that
branch fix/unknown-scores-and-review-gaps changes are left to test_pr1_additions.py.
"""

from __future__ import annotations

import pytest

import auditkit as ak
from auditkit.model import CallableModel
from auditkit.sample import Sample

if not hasattr(ak, "AgentLoopDetection"):
    pytest.skip("reference-free agent metrics not present", allow_module_level=True)

from auditkit.metrics.rag_judge import parse_verdicts  # noqa: E402


def call(name, **args):
    return {"name": name, "arguments": args}


P, R = call("get_weather", city="Paris"), call("get_weather", city="Rome")


def run(metric, turns=(), messages=None, **sample_kw):
    s = Sample(input="task", **sample_kw)
    trace = {"tool_calls": [list(t) for t in turns]}
    if messages is not None:
        trace["messages"] = messages
    return metric.score(s, "", {"trace": trace})


def values(out):
    return {x.name: x.value for x in out}


# -- agent_loop_detection ---------------------------------------------------------------------------

LOOP = ak.AgentLoopDetection


@pytest.mark.parametrize("metric,turns,want", [
    (LOOP(), [[P], [P]], 1.0),                                  # an immediate repeat is not a cycle; x2 < 3
    (LOOP(), [[P], [P], [P]], 0.0),                             # x3 reaches the threshold
    (LOOP(repetition_threshold=2), [[P], [P]], 0.0),
    (LOOP(repetition_threshold=10), [[P], [R], [P]], 0.0),      # a revisit is a cycle whatever the threshold
    (LOOP(), [[P, R], [P]], 0.0),                               # parallel calls are flattened in order
    (LOOP(), [[call("search", q="a")], [call("read", id=1)], [call("search", q="b")]], 1.0),  # fan-out, not a loop
    (LOOP(), [[call("f", a=1)], [call("f", a=1.0)], [call("f", a=1)]], 0.0),   # 1 and 1.0 are one call
    (LOOP(), [[call("f", a=1, b=2)], [call("f", b=2, a=1)], [call("f", a=1, b=2)]], 0.0),  # key order
    (LOOP(), [[call("f", a=1)], [call("f", a=2)], [call("f", a=3)]], 1.0),     # same tool, new arguments
])
def test_loop_signals(metric, turns, want):
    assert values(run(metric, turns)) == {"agent_loop_detection": want}


def test_the_reason_names_every_signal_that_fired():
    [sc] = run(LOOP(), [[P], [R], [P], [R], [P]])
    assert sc.metadata["max_repeat"] == 3 and any("repeated call" in s for s in sc.metadata["signals"])
    assert "call-graph cycle" in sc.metadata["signals"]


def test_malformed_calls_do_not_crash_and_are_not_keyed_together():
    bad = {"name": "f", "arguments": "{not json"}
    assert values(run(LOOP(), [[bad], [bad], [bad]])) == {"agent_loop_detection": 1.0}


@pytest.mark.parametrize("messages,want", [
    ([{"role": "assistant", "content": "Let me think."}, {"role": "assistant", "content": "Let me think."}], 0.0),
    ([{"role": "assistant", "content": "Checking the balance."},
      {"role": "assistant", "content": "The balance is 40 USD, done."}], 1.0),
    ([{"role": "assistant", "content": "Same."}, {"role": "user", "content": "Same."}], 1.0),   # user turns ignored
    ([{"role": "assistant", "content": "Same.", "tool_calls": [P]},
      {"role": "assistant", "content": "Same.", "tool_calls": [R]}], None),                     # tool-call turns ignored
])
def test_reasoning_stagnation(messages, want):
    out = run(LOOP(), [], messages=messages)
    assert (values(out) == {"agent_loop_detection": want}) if want is not None else out == []


def test_similarity_threshold_one_only_flags_identical_reasoning():
    msgs = [{"role": "assistant", "content": "Let me think about it."},
            {"role": "assistant", "content": "Let me think about this."}]
    assert values(run(LOOP(similarity_threshold=1.0), [], messages=msgs)) == {"agent_loop_detection": 1.0}


def test_nothing_to_check_is_skipped():
    assert run(LOOP(), []) == [] and run(LOOP(), [], messages=[]) == []


def test_progress_messages_about_different_entities_are_not_stagnation():
    msgs = [{"role": "assistant", "content": "Let me check the weather in Paris."},
            {"role": "assistant", "content": "Let me check the weather in Rome."}]
    assert values(run(LOOP(), [[P], [R]], messages=msgs)) == {"agent_loop_detection": 1.0}


# -- tool_permission -----------------------------------------------------------------------------------

PERM = ak.ToolPermission
OPENAI = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]


@pytest.mark.parametrize("metric,turns,kw,want", [
    (PERM(), [[P]], {"tools": OPENAI}, 1.0),
    (PERM(), [[P]], {"tools": [{"name": "get_weather", "input_schema": {}}]}, 1.0),     # Anthropic schema
    (PERM(), [[P]], {"metadata": {"allowed_tools": ["get_weather"]}}, 1.0),           # plain names
    (PERM(), [[P]], {"tools": OPENAI, "metadata": {"allowed_tools": ["other"]}}, 0.0),  # allowlist beats tools
    (PERM(), [[P]], {"metadata": {"allowed_tools": []}}, 0.0),                         # empty allowlist: nothing allowed
    (PERM(), [[P]], {"metadata": {"denied_tools": ["get_weather"]}}, 0.0),
    (PERM(denied_tools=[]), [[P]], {"metadata": {"denied_tools": ["get_weather"]}}, 1.0),  # constructor wins
    (PERM(), [[P]], {}, 1.0),                                                          # no policy: nothing to break
    (PERM(), [[call("rm")], [call("rm")], [P]], {"tools": OPENAI}, pytest.approx(1 / 3)),  # each call counts
])
def test_tool_permission_policies(metric, turns, kw, want):
    assert values(run(metric, turns, **kw)) == {"tool_permission": want}


def test_repeated_violations_are_listed_once_in_the_reason():
    [sc] = run(PERM(), [[call("rm")], [call("rm")]], tools=OPENAI)
    assert sc.reason == "rm: not in allowlist" and sc.metadata["n_violations"] == 2


def test_tool_permission_applies_only_with_a_policy():
    assert PERM().applicable(Sample(input="q", tools=OPENAI))
    assert PERM(denied_tools=["x"]).applicable(Sample(input="q"))
    assert not PERM().applicable(Sample(input="q"))


# -- tool_selection ----------------------------------------------------------------------------------------

class Recorder:
    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def __call__(self, prompts):
        self.prompts += prompts
        return [self.replies.pop(0) for _ in prompts]


def test_each_call_is_judged_with_the_calls_before_it():
    rec = Recorder(["CHOICE: yes"] * 3)
    run(ak.ToolSelectionJudge(judge_model=CallableModel(rec)), [[P, R], [call("book")]])
    assert len(rec.prompts) == 3
    history = [p.split("<actions_so_far>")[1].split("</actions_so_far>")[0] for p in rec.prompts]
    assert "(none)" in history[0]
    assert "Paris" in history[1] and "Rome" not in history[1]
    assert "Paris" in history[2] and "Rome" in history[2]


def test_an_unreadable_verdict_on_one_call_is_left_out_of_the_mean():
    [sc] = run(ak.ToolSelectionJudge(judge_model=CallableModel(Recorder(["CHOICE: yes", "??"]))), [[P], [R]])
    assert sc.value == 1.0 and sc.metadata == {"n_calls": 2, "n_judged": 1}


def test_tool_selection_mean_over_calls():
    [sc] = run(ak.ToolSelectionJudge(judge_model=CallableModel(Recorder(["CHOICE: yes", "CHOICE: no"]))), [[P], [R]])
    assert sc.value == 0.5


def test_tool_selection_skips_a_run_with_no_calls():
    assert run(ak.ToolSelectionJudge(judge_model=CallableModel(Recorder([]))), []) == []


# -- the shared verdict parser -------------------------------------------------------------------------------

LABELS = ("SUPPORTED", "UNSUPPORTED")


@pytest.mark.parametrize("text,want", [
    ("1: SUPPORTED\n2: UNSUPPORTED", ["SUPPORTED", "UNSUPPORTED"]),
    ("1: UNSUPPORTED\n2: SUPPORTED", ["UNSUPPORTED", "SUPPORTED"]),      # UNSUPPORTED never reads as SUPPORTED
    ("**1**: supported - in chunk 2\n2) unsupported", ["SUPPORTED", "UNSUPPORTED"]),   # markdown, case, reasons
    ("Here you go.\n1. SUPPORTED\nnoise\n2 - SUPPORTED", ["SUPPORTED", "SUPPORTED"]),
    ("1: SUPPORTED\n1: SUPPORTED\n2: SUPPORTED", ["SUPPORTED", "SUPPORTED"]),          # a repeat that agrees
])
def test_parse_verdicts(text, want):
    assert parse_verdicts(text, 2, LABELS) == want


@pytest.mark.parametrize("text", [
    "1: SUPPORTED",                               # one missing
    "1: SUPPORTED\n2: SUPPORTED\n3: SUPPORTED",   # one extra
    "1: SUPPORTED\n1: UNSUPPORTED\n2: SUPPORTED",  # conflicting verdicts for item 1
    "1: MAYBE\n2: SUPPORTED",                     # unknown label
    "",
])
def test_parse_verdicts_refuses_to_guess(text):
    with pytest.raises(ValueError):
        parse_verdicts(text, 2, LABELS)


# -- reference-free RAG judges ----------------------------------------------------------------------------------

CONTEXTS = ["Laptops: every 3 years.", "Cafeteria: 8am."]


def rag_sample(output, contexts=CONTEXTS):
    return Sample(id="s", input="How often are laptops replaced?", actual_output=output,
                  actual_trace={"retrieved_contexts": contexts})


def judge(*replies):
    return CallableModel(Recorder(replies))


def evaluate(metric, output, contexts=CONTEXTS):
    return ak.evaluate([rag_sample(output, contexts)], model="precomputed", scorers=[metric])


@pytest.mark.parametrize("verdicts,want", [
    ("1: SUPPORTED\n2: SUPPORTED", 0.0),
    ("1: SUPPORTED\n2: CONTRADICTED", 0.5),
    ("1: UNSUPPORTED\n2: CONTRADICTED", 1.0),       # both kinds count as hallucinated
])
def test_hallucination(verdicts, want):
    r = evaluate(ak.Hallucination(judge_model=judge("1. A.\n2. B.", verdicts)), "A. B.")
    assert r.errors == [] and r.headline["hallucination"] == pytest.approx(want)


def test_hallucination_ignores_the_answers_own_thinking():
    rec = Recorder(["1. Laptops every 3 years.", "1: SUPPORTED"])
    evaluate(ak.Hallucination(judge_model=CallableModel(rec)), "<think>Laptops are free.</think>Laptops every 3 years.")
    assert "free" not in rec.prompts[0]


@pytest.mark.parametrize("metric_cls", [ak.Hallucination, ak.AnswerRelevancy])
def test_a_none_extraction_means_nothing_to_score(metric_cls):
    r = evaluate(metric_cls(judge_model=judge("NONE")), "Hello!")
    assert r.headline == {} and r.errors == []


@pytest.mark.parametrize("metric_cls,replies", [
    (ak.Hallucination, ("I think the answer is fine.",)),              # extraction unreadable
    (ak.Hallucination, ("1. A.\n2. B.", "1: SUPPORTED")),               # one verdict for two claims
    (ak.AnswerRelevancy, ("1. A.", "1: ADDRESSES\n1: OFF_TOPIC")),      # conflicting
    (ak.ResponseGroundedness, ("1: SUPPORTED",)),                       # two sentences, one verdict
    (ak.ContextRelevance, ("1: RELEVANT\n2: RELEVANT\n3: RELEVANT",)),  # one extra
])
def test_an_unreadable_judge_reply_is_an_error_never_a_score(metric_cls, replies):
    r = evaluate(metric_cls(judge_model=judge(*replies)), "A. B.")
    metric = metric_cls.name
    assert metric not in r.headline and len(r.errors) == 1 and r.errors[0]["metric"] == metric


@pytest.mark.parametrize("metric_cls,output,contexts", [
    (ak.Hallucination, "", CONTEXTS),
    (ak.Hallucination, "   ", CONTEXTS),
    (ak.Hallucination, "An answer.", []),
    (ak.AnswerRelevancy, "<think>only thinking</think>", CONTEXTS),
    (ak.ResponseGroundedness, "An answer.", []),
    (ak.ResponseGroundedness, "", CONTEXTS),
    (ak.ContextRelevance, "anything", []),
])
def test_missing_inputs_skip_without_calling_the_judge(metric_cls, output, contexts):
    rec = Recorder([])
    r = evaluate(metric_cls(judge_model=CallableModel(rec)), output, contexts)
    assert r.headline == {} and r.errors == [] and rec.prompts == []


def test_groundedness_splits_sentences_locally_and_keeps_decimals_whole():
    rec = Recorder(["1: SUPPORTED\n2: UNSUPPORTED"])
    r = evaluate(ak.ResponseGroundedness(judge_model=CallableModel(rec)), "The rate is 5.5% today. It is free!")
    assert r.headline["response_groundedness"] == 0.5
    assert "1. The rate is 5.5% today." in rec.prompts[0] and "2. It is free!" in rec.prompts[0]


def test_context_relevance_single_chunk():
    r = evaluate(ak.ContextRelevance(judge_model=judge("1: IRRELEVANT")), "x", ["Cafeteria: 8am."])
    assert r.headline["context_relevance"] == 0.0
