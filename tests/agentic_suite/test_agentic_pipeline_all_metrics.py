"""One file, every agentic metric, through the full ak.evaluate() pipeline.

Sample (reference) -> backend (recorded trace / text model) -> Runner puts the
trace in context["trace"] -> metric normalizes turns -> bipartite matching -> scores.
Run: python -m pytest tests/agentic_suite/test_agentic_pipeline_all_metrics.py -q
"""

import pytest

import auditkit as ak
from auditkit.model import CallableModel

# -- tools offered to the model (JSON schemas only; nothing is executed) -------------
TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "parameters": {
        "type": "object", "required": ["city"],
        "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}}}},
    {"type": "function", "function": {"name": "get_time", "parameters": {
        "type": "object", "required": ["city"], "properties": {"city": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "book_flight", "parameters": {
        "type": "object", "required": ["flight_id"], "properties": {"flight_id": {"type": "string"}}}}},
]


def call(name, **args):
    return {"name": name, "arguments": args}


PARIS, ROME = call("get_weather", city="Paris"), call("get_weather", city="Rome")
BOOK = call("book_flight", flight_id="AZ-61")
# Reference: turn 1 = parallel group (independent), turn 2 = depends on turn 1's results
REFERENCE = [[PARIS, ROME], [BOOK]]


def sample(sid, turns, reference=REFERENCE, answer="done"):
    """A recorded run: what the agent actually did, scored offline (model='precomputed')."""
    return ak.Sample(id=sid, input="Which is warmer, Paris or Rome? Book AZ-61 if it's Rome.",
                     tools=TOOLS, expected_tool_calls=reference, target="AZ-61 booked",
                     actual_output=answer, actual_trace={"tool_calls": turns})


def per_sample(result):
    """{sample_id: {score_name: value}} from a RunResult."""
    return {p.sample_id: {s["name"]: s["value"] for s in p.metadata["scores"]} for p in result.predictions}


DATASET = [
    sample("correct",          [[PARIS, ROME], [BOOK]]),
    sample("serialized",       [[PARIS], [ROME], [BOOK]]),          # independent calls made one by one
    sample("wrongly_batched",  [[PARIS, ROME, BOOK]]),              # booked before seeing the weather
    sample("wrong_argument",   [[PARIS, call("get_weather", city="Lyon")], [BOOK]]),
    sample("missing_call",     [[PARIS, ROME]]),                    # never booked
    sample("looping",          [[PARIS, ROME], [PARIS, ROME], [BOOK]]),
    sample("invalid_calls",    [[call("get_weather", city="Paris", unit="kelvin"), call("teleport", to="Rome")]]),
    sample("irrelevant_ok",    [], reference=[]),                   # no tool should be called
    sample("irrelevant_bad",   [[PARIS]], reference=[]),
]

SCORERS = [ak.ToolCallF1(), ak.TrajectoryMatch(), ak.ParallelToolCalls(),
           ak.ToolCallValidity(), ak.RedundantToolCalls()]


@pytest.fixture(scope="module")
def scores(tmp_path_factory):
    # module-scoped fixtures run before the per-test cache isolation: give this one its own cache
    mp = pytest.MonkeyPatch()
    mp.setenv("XDG_CACHE_HOME", str(tmp_path_factory.mktemp("cache")))
    result = ak.evaluate(DATASET, model="precomputed", scorers=SCORERS)
    assert result.errors == [] and result.failed_count == 0
    yield per_sample(result)
    mp.undo()


# -- 1. tool_call_f1: right tools + right arguments (order/turns ignored) ------------------
def test_tool_call_f1(scores):
    assert scores["correct"]["tool_call_f1"] == 1.0 and scores["correct"]["tool_call_exact"] == 1.0
    assert scores["serialized"]["tool_call_f1"] == 1.0          # same calls, so F1 can't see the problem
    assert scores["wrong_argument"]["tool_call_precision"] == pytest.approx(2 / 3)
    assert scores["wrong_argument"]["tool_call_recall"] == pytest.approx(2 / 3)
    assert scores["missing_call"]["tool_call_recall"] == pytest.approx(2 / 3)
    assert scores["missing_call"]["tool_call_precision"] == 1.0
    assert scores["looping"]["tool_call_precision"] == pytest.approx(3 / 5)   # duplicates count
    assert scores["irrelevant_ok"]["tool_call_f1"] == 1.0
    assert scores["irrelevant_bad"]["tool_call_f1"] == 0.0


# -- 2. trajectory_match: right calls in the right turns / order ---------------------------
def test_trajectory_match(scores):
    assert scores["correct"]["trajectory_strict"] == 1.0
    assert scores["serialized"]["trajectory_strict"] == 0.0
    assert scores["serialized"]["trajectory_in_order"] == 1.0   # order kept, turns differ
    assert scores["wrongly_batched"]["trajectory_strict"] == 0.0
    assert scores["missing_call"]["trajectory_in_order"] == 0.0
    assert scores["looping"]["trajectory_in_order"] == 1.0      # extra calls allowed in in_order


# -- 3. parallel_tool_calls: batched the independent calls, and only those -----------------
def test_parallel_tool_calls(scores):
    assert scores["correct"]["parallel_recall"] == 1.0
    assert scores["correct"]["parallel_precision"] == 1.0
    assert scores["correct"]["parallel_detection"] == 1.0

    assert scores["serialized"]["parallel_recall"] == 0.0       # should have batched Paris+Rome
    assert "parallel_precision" not in scores["serialized"]     # no multi-call turn -> omitted
    assert scores["serialized"]["parallel_detection"] == 0.0

    assert scores["wrongly_batched"]["parallel_recall"] == 1.0
    assert scores["wrongly_batched"]["parallel_precision"] == 0.0  # batched a dependent call

    assert scores["wrong_argument"]["parallel_recall"] == 0.0   # group not fully matched


# -- 4. tool_call_validity: calls valid against the schemas (no reference needed) ----------
def test_tool_call_validity(scores):
    assert scores["correct"]["tool_call_validity"] == 1.0
    assert scores["invalid_calls"]["tool_call_validity"] == 0.0  # enum violation + unknown tool
    assert "tool_call_validity" not in scores["irrelevant_ok"]   # no calls -> skipped


# -- 5. redundant_tool_calls: exact repeats (lower is better) --------------------------------
def test_redundant_tool_calls(scores):
    assert scores["correct"]["redundant_tool_calls"] == 0.0
    assert scores["looping"]["redundant_tool_calls"] == pytest.approx(2 / 5)


# -- 6. task_completion: LLM judge over task + trajectory + answer ---------------------------
def test_task_completion_with_scripted_judge():
    seen = []

    def judge(prompts):                      # stands in for a real judge model
        seen.extend(prompts)
        return ["It booked AZ-61 after checking both cities.\nCHOICE: complete" if "book_flight" in p
                else "It claims a booking it never made.\nCHOICE: failed" for p in prompts]

    data = [sample("did_it", [[PARIS, ROME], [BOOK]], answer="Rome is warmer; booked AZ-61."),
            sample("claimed_only", [[PARIS, ROME]], answer="Rome is warmer; booked AZ-61.")]
    r = ak.evaluate(data, model="precomputed", scorers=[ak.TaskCompletion(judge_model=CallableModel(judge))])
    got = per_sample(r)
    assert got["did_it"]["task_completion"] == 1.0
    assert got["claimed_only"]["task_completion"] == 0.0
    assert "book_flight" in seen[0] and "AZ-61 booked" in seen[0]   # judge saw trajectory + target


# -- the same pipeline with a live-style text model (prompt mode, calls parsed from text) ----
def test_prompt_mode_text_model_end_to_end():
    reply = ('<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>\n'
             '<tool_call>{"name": "get_weather", "arguments": {"city": "Rome"}}</tool_call>')
    s = ak.Sample(id="next_turn", input="Compare the weather in Paris and Rome.", tools=TOOLS,
                  expected_tool_calls=[[PARIS, ROME]])       # single response -> reference = next turn only
    r = ak.evaluate([s], model=lambda prompts: [reply] * len(prompts),
                    adapter=ak.ToolCallAdapter(mode="prompt"), scorers=SCORERS)
    got = per_sample(r)["next_turn"]
    assert got["tool_call_exact"] == 1.0 and got["parallel_recall"] == 1.0
    assert got["tool_call_validity"] == 1.0 and got["redundant_tool_calls"] == 0.0


# -- headline = mean over the samples that produced each score ------------------------------
def test_headline_aggregation(scores):
    r = ak.evaluate(DATASET, model="precomputed", scorers=SCORERS)
    for name, value in r.headline.items():
        vals = [s[name] for s in scores.values() if name in s]
        assert value == pytest.approx(sum(vals) / len(vals)), name
