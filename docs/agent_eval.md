# Agent evaluation

The `auditkit.agent_eval` layer evaluates what an agent **did and achieved**
across a complete task episode, not just the text it returned. The headline
result is a **verified outcome**: an oracle checks the final state, an artifact,
the answer, or a custom predicate, and returns `success`, `failure`, `unknown`
or `error`. Answer similarity, tool-call scores and a task-completion judge are
separate diagnostic columns. A judge never overrides a verified state check.

This is a general library. Any framework that can emit OpenAI-style chat
messages, or that serves an agent over HTTP, works. AgentTune is one interop
case, read through `load_agenttune`, with no hard dependency: the base package
never imports AgentTune, torch or transformers.

The layer sits on the metrics in [Agents & RAG](agents_and_rag.md). It reuses
the same `ToolCallF1`, `ParallelToolCalls`, `RetrievalMetrics` and
`TaskCompletion` classes and the same `agent:` backend, and adds the episode
contract, the outcome oracles, per-case status, and offline rescore around them.

Runnable, fully offline example:
[`examples/agent_eval_offline.py`](https://github.com/Lexsi-Labs/AuditKIT/blob/main/examples/agent_eval_offline.py).

Everything imports from `auditkit.agent_eval`:

```python
from auditkit.agent_eval import (
    AgentCase, AgentEpisode, AgentEvent,
    AgentEvalSpec, AgentEvalRunner, AgentEvalResult, AgentCaseResult,
    FinalStateAssertion, ArtifactAssertion, AnswerAssertion, CustomPredicate,
    episodes_from_agenttune, episode_from_eventlog,
    episode_from_openai_messages, episode_from_sample, inspect_agenttune, rescore, resolve_outcome,
)
```

The main symbols (`AgentCase`, `AgentEpisode`, `AgentEvalRunner`, the four
oracles, the three importers) are also re-exported from the top-level
`auditkit` package. The helpers `rescore`, `inspect_agenttune`, `resolve_outcome`
and `AgentCaseResult` are only under `auditkit.agent_eval`.

## The episode contract

Three records carry a run. All are stdlib dataclasses that serialize to strict
JSON, so a reviewer can reconstruct the task, the evidence and the outcome from
one file.

**`AgentCase`** is what to evaluate: `id`, `task`, `category`, `allowed_tools`,
`budgets`, the `outcome` oracle, optional `reference_turns` and
`reference_contexts`, an `environment` dict, and `metadata`. `id` and `task` are
required and validated. `digest()` is a stable content hash that changes when
the task or the oracle changes, so a changed oracle is a different case. A
correct outcome reached by a different valid path still passes:
`reference_turns` is a diagnostic or a policy requirement only when the case
sets it.

**`AgentEpisode`** is one complete run: an ordered list of `AgentEvent`, plus
`mode` (`recorded`, `deployed` or `harness`), `source_format`, `source_tier`,
`case_id`, `final_answer`, `final_state`, `artifacts`, `counters`,
`stop_reason`, `errors`, and a `coverage` map. `source_reward`, `source_verdict`
and `source_scores` hold an imported run's own numbers as provenance only. They
are never promoted to the AuditKit outcome.

**`AgentEvent`** is one ordered event: `index`, `role`, `type` (`text`,
`reasoning`, `tool_call`, `tool_result`, `observation`, and so on), `payload`
(the raw source object), an optional `timestamp`, `call_id`, `turn_id`,
`source` and `tier`. A missing timestamp is `None`; monotonic times are never
fabricated. Events sharing a `turn_id` are one parallel group, and `call_id`
pairs a `tool_result` back to its `tool_call`.

### Coverage: loss awareness

Every imported episode reports a `coverage` map: for each field it says whether
that field was `observed`, `inferred` or `unavailable`. No missing argument,
result or state is filled with a fabricated default. The fields are
`final_answer`, `tool_call_names`, `tool_call_arguments`, `tool_results`,
`call_result_pairing`, `parallel_grouping`, `timestamps`, `retrieved_contexts`
and `final_state`.

`episode.coverage_label()` collapses this to one word for the report:

| Label | Meaning |
|---|---|
| `full` | `tool_call_arguments` observed (calls with arguments) |
| `tool_names_only` | `tool_call_names` observed but not arguments |
| `answer_only` | only `final_answer` observed |
| `none` | nothing scoreable observed |

This is what makes the bridge honest. A flattened AgentTune report carries tool
**names** with no arguments, so it is marked `tool_names_only`, and the
argument-sensitive metrics stay ineligible rather than reading empty arguments
as wrong arguments.

## Importing recorded runs

Three importers turn a recorded run into an `AgentEpisode`. Coverage is set
honestly per source.

**OpenAI chat messages.** `episode_from_openai_messages(messages, ...)` reads a
list of chat messages. Each assistant message with `tool_calls` is one turn (a
parallel group); a `tool` message pairs by `tool_call_id`. Pass
`retrieved_contexts=` to record a ranked retrieval.

```python
from auditkit.agent_eval import episode_from_openai_messages

episode = episode_from_openai_messages(
    [
        {"role": "user", "content": "Ship item A if it is in stock."},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "lookup_inventory", "arguments": '{"sku": "A"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "in_stock=true"},
        {"role": "assistant", "content": "Shipment for A is ready."},
    ],
    case_id="inventory-001", final_answer="Shipment for A is ready.",
)
episode.final_state = {"shipment_decision": "ready"}   # attach verified state
```

**A `Sample`.** `episode_from_sample(sample)` reads `sample.actual_trace`
(`messages`, or `tool_calls` turns, plus `retrieved_contexts`) and
`sample.actual_output`. It bridges the recorded-trace format from
[Agents & RAG](agents_and_rag.md) into an episode.

**AgentTune files.** `episodes_from_agenttune(path)` reads every record of a
`load_agenttune` file and returns one episode per record. It detects the shape
per record and never imports AgentTune:

| AgentTune record | `source_format` | Coverage |
|---|---|---|
| `run_eval` / RAG-GRPO dataset row (`prompt`) | `agenttune_run_eval_row` | a task to run: no events, nothing observed yet |
| Trajectory JSONL (`steps`, `trajectory_id`) | `agenttune_trajectory` | arguments observed when the source recorded them; per-call pairing only when a `conversation` is present |
| `trace.jsonl` (`question`, `final_answer`) | `agenttune_trace` | answer plus retrieval ids; `{query, result}` calls carry no tool name |
| `run_eval` report JSON (`samples`) | `agenttune_report` | answer plus flattened tool **names** only; the report's `model`, `use_case`, `pass_rate`, `n_samples`, `n_errors` and `timestamp` are kept in `metadata["source_report"]` |
| Serialized `EventLog` JSONL (`events`, `tier`, `id`) | `agenttune_eventlog` | see `episode_from_eventlog` below |

**AgentTune `EventLog`.** `episode_from_eventlog(log)` reads an `EventLog`
object or its JSON (`{"events": [...], "tier", "id"}`), duck-typed, without
importing AgentTune. `TURN_COMPLETE` closes a turn, so calls inside one step
form one parallel group. A light-tier log with no turn boundaries (from
`from_eval_dict`) gives each call its own turn and marks grouping and pairing
`unavailable`. The terminal step's empty action is kept as an event but never
becomes a call. The episode-scope `REWARD` becomes `source_reward`, and
`token_span`/`logprobs`/`scope` stay in the event payload. The log `id` is a
uuid4 minted by `EventLog.from_*`, so it joins no other AgentTune file.

The heal audit records that `Project.heal` writes (`EventLog.to_audit_records`,
one `{trajectory_id, stage_name, tool_name, stage_type, state_snapshot, status}`
row per call) are refused with a `ValueError`. That projection records
rollout-shaped calls as tool `unknown` with empty arguments and adds a phantom
call for the terminal step, so importing it would produce empty or wrong
episodes. Import the `EventLog` itself instead.

### Coverage decides which metrics are eligible

`supported_scorers_from_coverage(coverage)` reports which scorers a trace can
feed, ignoring the case. A judge (`task_completion`) is always listed because it
is diagnostic. Argument-sensitive tool metrics need `tool_call_arguments`
observed; a names-only trace gets the `tool_call_f1_name` variant instead. The
list `answer` in this output means an `AnswerAssertion` oracle can run against
the recorded final answer; it is not a scorer name you pass to `scorers=`.

At run time the runner applies the same rule per case with `eligibility()`,
which returns `None` when a scorer can run or the exact missing field otherwise
(`case.reference_turns`, `trace.tool_call_arguments`, and so on). That missing
field is what the report prints, so an ineligible metric always says why.

## Outcome oracles

The headline outcome comes from the case's oracle. Verdicts are `success`,
`failure`, `unknown` (evidence missing) and `error` (a check raised). Missing
evidence is never turned into a pass or a fail.

| Oracle | Checks | Missing evidence |
|---|---|---|
| `FinalStateAssertion(key, equals=/contains=)` | a key of `episode.final_state` | no state snapshot -> `unknown` |
| `ArtifactAssertion(key, equals=/contains=)` | a key of `episode.artifacts` | no artifact -> `unknown` |
| `AnswerAssertion(reference, mode="exact"/"normalized"/"contains")` | the final answer against a reference | no reference or no answer -> `unknown` |
| `CustomPredicate(fn, version=..., name=...)` | a caller `(episode, case) -> bool or str` | `None` -> `unknown`; a raise -> `error` |

`FinalStateAssertion` and `ArtifactAssertion` check a real external effect: when
the case's oracle is a state or artifact check, the answer text plays no part in
the verdict. A recorded episode with no state snapshot is
scored `unknown`, never a failure: the absence of state is not evidence of a
wrong state. `CustomPredicate` requires an explicit `version=` so the run can be
described as reproducible, and it cannot be built from JSON (pass an oracle
instance in Python).

```python
from auditkit.agent_eval import (
    AgentCase, CustomPredicate, resolve_outcome, episode_from_openai_messages,
)

def two_calls(episode, case):
    return len(episode.tool_call_events()) == 2

episode = episode_from_openai_messages(
    [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
        {"id": "c2", "type": "function", "function": {"name": "b", "arguments": "{}"}}]},
     {"role": "assistant", "content": "done"}],
    final_answer="done")
case = AgentCase(id="x", task="do two things",
                 outcome=CustomPredicate(two_calls, version="1"))
print(resolve_outcome(episode, case).verdict)   # success
```

### The judge is a diagnostic

`TaskCompletion` and other judges are diagnostic signals. A judge carries its
model identity, reason and parse status. A parse failure becomes `unknown`, not
a zero. When a case has a deterministic oracle, the judge is recorded in the
diagnostic columns and the verified outcome is the headline. Only when a case
has **no** oracle does a judge verdict become the (labeled diagnostic) headline.
A judge that says an answer is complete can never flip a `FinalStateAssertion`
that verified the state as wrong.

## Reference-free evaluation (no gold standard)

Not every task ships with a gold trajectory or a gold answer. These evaluate a
run on its own terms -- they need only the trace (and, for the judge ones, a
model), never a reference. Each metric is registered by name and is skipped when
it has nothing to measure (rather than reporting a fake 0).

**Metrics** (`auditkit.metrics.agent`, all `kind=AGENT`):

| Metric (name) | What it measures | Needs |
|---|---|---|
| `AgentLoopDetection` (`agent_loop_detection`) | `1.0` loop-free, `0.0` a loop was found. Three stdlib signals, no judge: an identical `(name, arguments)` call repeated `repetition_threshold` (default 3) times; two consecutive assistant messages near-identical (`difflib` ratio ≥ `similarity_threshold`, default 0.85); a cycle in the call-transition graph (DFS). `reason` names the signals that fired. | trace only |
| `ToolPermission` (`tool_permission`) | Fraction of calls within a least-privilege policy. A call is unauthorized if it is on the denylist, or (with an allowlist set) not on it -- a denial always wins. `reason` lists the violations. | an allowlist and/or denylist |
| `ToolSelectionJudge` (`tool_selection`) | LLM judge: for each call, was it justified given the task and the transcript of the *preceding* calls? Score is the mean over calls; unparseable verdicts are excluded (all-unparseable -> one `unknown` score). | a `judge_model` |

The allowlist for `ToolPermission` is read from `Sample.metadata["allowed_tools"]`
(or `AgentCase.allowed_tools`, which becomes `Sample.tools`); the denylist from
`denied_tools=` or `Sample.metadata["denied_tools"]`.

```python
from auditkit.metrics.agent import AgentLoopDetection, ToolPermission, ToolSelectionJudge

ctx = {"trace": {"tool_calls": [[{"name": "search", "arguments": {"q": "x"}}]]}}
AgentLoopDetection().score(sample, "", ctx)         # -> [Score(agent_loop_detection, 1.0)]
ToolPermission(denied_tools=["delete_db"]).score(sample, "", ctx)
ToolSelectionJudge(judge_model="openai:gpt-4o-mini").score(sample, "", ctx)
```

### `metadata` is a metric-addressable field

A metric or oracle can require and read any key of `Sample.metadata` /
`AgentCase.metadata`, not just the fixed dataclass fields. Use the helper
`auditkit.metrics.agent.addressable(sample, key)`: it returns the `Sample`
attribute of that name, and falls back to `Sample.metadata[key]`. This is how a
user's own reference field (a custom allowlist, an assertion criterion) gates a
metric or feeds an oracle without a schema change. A reference-free metric that
gates on metadata overrides `applicable()` to consult `addressable`.

### `AssertionOracle`: a first-class outcome with no gold standard

When there is no gold trajectory or answer but a human can state, in words, what
success means, `AssertionOracle` turns that criterion into a **first-class**
verdict (`success`/`failure`/`unknown`), not a diagnostic. Unlike the
`TaskCompletion` judge (always `diagnostic=True`), an `AssertionOracle` set as a
case's `outcome` *is* the headline outcome.

```python
from auditkit.agent_eval import AgentCase, AssertionOracle, resolve_outcome

oracle = AssertionOracle(
    "the agent looked up the live price before answering",
    judge_model="openai:gpt-4o-mini")
case = AgentCase(id="q", task="What does it cost now?", outcome=oracle)
resolve_outcome(episode, case).verdict          # success / failure / unknown
```

The criterion may instead live in `case.metadata["assertion"]` (pass
`criterion=None`), so a dataset column can carry it. A parse failure is
`unknown`; a broken judge model is `error`; the verdict is never `diagnostic`.

A companion `state_oracle=` preserves the "state wins over a judge" rule: it runs
first, and a *verified* `success`/`failure` from it is returned unchanged (an
`unknown` -- no state recorded -- falls through to the judge). So one case can
carry both a deterministic state check for when state is captured and a held-out
NL assertion for when it is not:

```python
from auditkit.agent_eval import FinalStateAssertion
oracle = AssertionOracle("agent shipped the order", judge_model="openai:gpt-4o-mini",
                         state_oracle=FinalStateAssertion("shipment", equals="ready"))
```

`AssertionOracle` is also buildable from a JSON spec
(`{"type": "assertion", "criterion": ..., "judge_model": ..., "state_oracle": {...}}`).

> Follow-up: `AgentEvalRunner` does not yet know the three new metric names, and
> `runner._sample_for` drops `case.metadata`, so today these are reached via
> direct metric calls / `ak.evaluate`. To wire them into the runner, add the
> names to `runner._make_scorer`/`eligibility` and copy `case.metadata` into
> `Sample.metadata` in `runner._sample_for`.

## Running an evaluation

Build an `AgentEvalSpec` and run it with `AgentEvalRunner().run(spec)`.

`AgentEvalSpec` fields: `cases`, `mode` (`recorded` / `deployed` / `harness`),
`agent` (deployed endpoint or harness policy), `agent_opts`, `episodes`
(recorded), `import_path` (recorded, an AgentTune file), `trials`,
`reset_confirmed` and `reliability_k` (A3 reliability), `scorers`, `judge` and
`name`. The scorers are diagnostic and default to
`["tool_call_f1", "tool_call_validity", "task_completion"]`. The valid scorer
names are:

```text
tool_call_f1  tool_call_f1_name  trajectory_match  parallel_tool_calls
tool_call_validity  redundant_tool_calls  retrieval  task_completion
```

`task_completion` runs only when a `judge=` is set; without one it is a silent
no-op. Pass a `TaskCompletion` instance (or a judge-model spec string) as
`judge=` to enable it, and omit `task_completion` to stay fully offline.

### Recorded mode

Recorded mode scores saved episodes with no agent call and no tool call. Pass
`episodes=`, or `import_path=` to import an AgentTune file first, or both.

```python
from auditkit.agent_eval import (
    AgentCase, AgentEvalRunner, AgentEvalSpec,
    FinalStateAssertion, episode_from_openai_messages,
)

episode = episode_from_openai_messages(
    [
        {"role": "user", "content": "Ship item A if it is in stock."},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "lookup_inventory", "arguments": '{"sku": "A"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "in_stock=true"},
        {"role": "assistant", "content": "Shipment for A is ready."},
    ],
    case_id="inventory-001", final_answer="Shipment for A is ready.",
)
episode.final_state = {"shipment_decision": "ready"}

case = AgentCase(
    id="inventory-001", task="Ship item A if it is in stock.",
    allowed_tools=["lookup_inventory"],
    reference_turns=[[{"name": "lookup_inventory", "arguments": {"sku": "A"}}]],
    outcome=FinalStateAssertion("shipment_decision", equals="ready"),
)

spec = AgentEvalSpec(cases=[case], mode="recorded", episodes=[episode],
                     scorers=["tool_call_f1", "tool_call_validity"])  # no judge -> offline
result = AgentEvalRunner().run(spec)
print(result.summary())
row = result.rows[0]
print(row.status, row.outcome["verdict"], row.coverage_label)
```

```text
Agent eval   Mode: recorded   Cases: 1
Verified success: 1/1 (1/1 decided)   Unknown outcome: 0
Endpoint errors: 0
Trace coverage: full 1, tool names only 0, answer only 0
Tool validity: 1/1 eligible
Median latency: n/a          Agent calls: 0
completed success full
```

Each case is matched to an episode by `case_id` or `source_id`, then by
position as a fallback when no id matches. With no cases at all, recorded mode
derives one case per episode (default oracle: an `AnswerAssertion` against the
recorded target when one exists) and skips run-eval task rows.

### Deployed mode

Deployed mode calls an `agent:` endpoint through the existing
`AutoModel.resolve`. It captures the endpoint's output, ordered trace, retrieved
contexts, errors, latency and token usage, and scores what comes back. Set
`agent="agent:<url>"` and pass endpoint options in `agent_opts` (for example
`timeout`).

```python
from auditkit.agent_eval import AgentCase, AgentEvalRunner, AgentEvalSpec, AnswerAssertion

case = AgentCase(
    id="weather-001", task="What is the weather and time in Paris?",
    allowed_tools=["get_weather", "get_time"],
    reference_turns=[[{"name": "get_weather", "arguments": {"city": "Paris"}},
                      {"name": "get_time", "arguments": {"city": "Paris"}}]],
    outcome=AnswerAssertion("sunny at noon"),
)
spec = AgentEvalSpec(cases=[case], mode="deployed", agent="agent:http://localhost:8000/run",
                     agent_opts={"timeout": 60}, scorers=["tool_call_f1", "tool_call_validity"])
result = AgentEvalRunner().run(spec)   # needs a running endpoint
```

An endpoint error (`finish_reason == "error"`, a malformed reply, or a timeout)
is authoritative: the case status is `target_error`, the outcome verdict is
`error`, and the failure text is captured. It is never a fabricated failure
scored on an empty string, and the summary does not count it as a decided case.

**Deployed state comes only from the reply.** When the endpoint's JSON reply
carries a `final_state` (or `artifacts`) object, the episode gets it and
`final_state` coverage is `observed`. Set other reply paths with
`agent_opts={"state_path": ..., "artifacts_path": ...}`. When the reply has no
state, coverage stays `unavailable` and a `FinalStateAssertion` scores
`unknown`. AuditKit does not query the environment itself. The `--dry-run` CLI
prints this before any call.

### Per-case status

Each `AgentCaseResult` has a `status`:

| Status | Meaning |
|---|---|
| `completed` | the case ran and produced an outcome |
| `budget_exhausted` | a `budgets` limit was exceeded |
| `method_error` | a scorer or oracle raised |
| `target_error` | the agent endpoint errored |
| `judge_error` | the judge errored and no oracle exists |
| `ineligible` | no oracle and every non-judge scorer was ineligible |
| `not_applicable` | no oracle, no eligible metric, no judge |

`budgets` supports `max_tool_calls` (checked against the tool-call counter),
`max_turns` (from AgentTune trajectory counters), and `max_seconds` (compared to
`latency_ms`, so it fires in deployed mode).

### The summary: both denominators

`result.summary()` leads with the verified outcome and reports two denominators
so missing evidence stays visible:

- success over all cases, for example `1/2`;
- success over **decided** cases, `success + failure`, with `unknown` and
  `error` excluded, for example `1/2 decided`.

Endpoint errors are a separate line, never counted as decided. The summary also
prints trace-coverage counts, tool-validity eligibility, median latency and
agent-call count. Two review lines appear when they apply: answers that claim
success while the verified outcome failed, and cases with no state evidence that
are not counted as failures.

## Offline rescore

`rescore(episodes, scorers, cases=..., judge=...)` scores already-recorded
episodes with a chosen scorer set and makes no agent or tool call. A deployed
run keeps its scored episodes on `result.episodes`, so you can replay a live run
offline to compare judge or scorer changes without spending endpoint calls
again.

```python
from auditkit.agent_eval import rescore

# Continuing from the deployed example above; any list of recorded episodes works.
result = rescore(result.episodes, ["tool_call_f1", "trajectory_match"], cases=[case])
```

## Reliability across trials (A3)

`trials=N` runs each case `N` times and reports whether the agent succeeds
*reliably*, not once by luck. In recorded mode the `N` episodes that match a
case are its trials; in deployed and harness mode the case is run `N` times.
Each trial is scored on its own, then one row aggregates them: the headline
`outcome` is the majority verdict over **decided** trials (a tie is `unknown`,
never a coin-flip pass), and `row.reliability` carries the numbers.

```python
spec = AgentEvalSpec(cases=[case], mode="recorded", episodes=eps, trials=3)
row = AgentEvalRunner().run(spec).rows[0]
row.reliability
# {'n_trials': 3, 'n_trials_requested': 3, 'n_success': 2, 'n_failure': 1,
#  'n_unknown': 0, 'n_error': 0, 'n_decided': 3, 'independent': False, 'k': 3,
#  'pass_at_k': 1.0,          # capability: >=1 of k decided trials passes
#  'all_k': 0.0,             # reliability: all k pass (reuses pass_hat_k)
#  'success_rate': 0.667, 'consistency': 0.667, 'variance': 0.222,
#  'status': 'mixed', 'reason': '2/3 decided trials succeeded',
#  'flags': ['independence unconfirmed ...']}
```

`pass_at_k` / `all_k` reuse the existing `auditkit.score.pass_at_k` /
`pass_hat_k` estimators and are computed over decided trials only; `unknown` and
`error` trials are counted separately, never as a 0. A trial that errored or
blew its budget (`target_error`, `budget_exhausted`, `method_error`) lands in
the `n_error` bucket, so it never counts as a reliable success -- mirroring the
summary's rule. A single trial (or no decided trial) is an explicit
`status="unknown"` with a reason -- no reliability is claimed. `reliability_k=`
sets `k` (default: the decided-trial count). In the CLI config, set `trials`,
`reset_confirmed` and `reliability_k` as top-level keys.

**Independence is not assumed.** A `trials=N` config does not prove the trials
were independent -- the endpoint may share a session or state across calls
(AG-12). AuditKit claims independence only when the caller confirms a
reset-per-trial contract with `reset_confirmed=True`; otherwise the counts are
shown but the reliability dict is flagged as not an independence-backed claim,
and a duplicate `trial_id` across trials adds its own flag. The formal
resettable-environment adapter (initial-state digests, verified reset) is the
remaining piece and is still deferred.

## Harness-owned loop (A4)

`mode="harness"` lets AuditKit -- not the agent -- own the turn loop. `spec.agent`
is the **policy** model (any `Model`, or a spec string `AutoModel.resolve`
accepts); `agent_opts` carries a caller-supplied tool double and the step bound:

```python
class ShipEnv:                       # a resettable test double (AG-17)
    def __init__(self): self.state = {}
    def __call__(self, name, arguments):
        if name == "ship": self.state["shipped"] = True; return "shipped ok"
        return "unknown tool"
    def snapshot(self): return dict(self.state)   # -> episode.final_state (verified)

spec = AgentEvalSpec(cases=[case], mode="harness", agent=policy_model,
                     agent_opts={"tool_env": ShipEnv(), "max_steps": 8})
```

`spec.config` (a `RunConfig`) reaches every policy request: its generation
settings (`max_tokens`, `temperature`, `top_p`, `top_k`, `seed`,
`stop_sequences`) and its `chat_template_kwargs` (e.g.
`{"enable_thinking": False}` for a Qwen3 policy).

Each step issues the accumulated messages to the policy, reads the reply, runs
any tool calls through `tool_env` (no arbitrary tool execution -- the caller
supplies the doubles), feeds the results back, and stops when the policy emits
no more calls, or at the `max_steps` bound (default 8, recorded as
`stop_reason="max_steps"` -> `budget_exhausted`). Every step is recorded with
call/turn ids, so call/result pairing and parallel grouping survive. When
`tool_env` exposes `snapshot()` (or a `state` attribute) the final environment
is captured as `episode.final_state`, so a `FinalStateAssertion` verifies a real
effect through the harness.

**Seam.** A deployed `agent:` endpoint runs its *own* loop and returns a
finished reply, so against it the harness completes in a single step (recorded
as `stop_reason="no_tool_env"` when no double is supplied). A true multi-step
server loop needs an endpoint that returns one assistant step and accepts
intermediate tool results -- beyond the current single-shot `agent:` contract.

## AgentTune sidecar export (A5)

`write_sidecar(result, path)` emits a compact, strict-JSON JSONL sidecar -- one
record per case keyed by `source_id` (falling back to `case_id`) -- carrying the
verified outcome, coverage, trial reliability, any recorded disagreement, and the
imported source provenance (reward/verdict/scores, kept diagnostic, never
promoted). It is a **read-only emit**: AgentTune is never imported and no source
record is mutated, so a downstream analysis joins these back to the original
trajectory by key. `read_sidecar(path)` round-trips it; `sidecar_records(result)`
returns the records in memory.

```python
from auditkit.agent_eval import write_sidecar, read_sidecar
write_sidecar(result, "agent-eval-sidecar.jsonl")
records = read_sidecar("agent-eval-sidecar.jsonl")
```

Writing these records *back into* AgentTune training data, and a `Project` intake
that reconstructs episodes from a Project on disk, both need an AgentTune-side
serializer that does not exist at the pinned mirror commit (its only on-disk
output is the lossy heal-audit JSONL, which the importers refuse). That half, and
a side-by-side AuditKit/AgentTune diagnostic comparison, stay deferred.

## Judge/oracle disagreement (AG-05)

A verified oracle always wins the headline; when a diagnostic judge or an
imported source verdict *disagrees* with a decided oracle, the disagreement is
recorded on `row.disagreement` (never silently dropped) and surfaced for review.
An `unknown`/`error` oracle is missing evidence, not a disagreement.

```python
row.disagreement
# {'judge_vs_oracle': {'oracle': 'failure', 'judge': 'success',
#                      'resolution': 'verified oracle wins; judge kept as diagnostic'},
#  'source_vs_oracle': {...}, 'n_trials': 1, 'n_disagree': 1, 'rate': 1.0, 'kinds': {...}}

result.disagreements(min_rate=0.5)   # cases where a majority of trials disagreed
```

`result.disagreements(min_rate=)` is the configurable filter: over repeated
trials the `rate` is the fraction of trials that disagreed. A single trial has
rate `1.0`. Voting across *multiple independent judges* is a seam, not built.
`source_verdict` is populated only when an imported report row carries `passed`;
AgentTune's `Report.save` at the pinned commit does not write it, so in practice
`source_vs_oracle` fires only for sources that do.

## The CLI

`auditkit agent` has three subcommands. `import-agenttune --inspect` and
`eval --dry-run` are local and read-only: they load no model and call no
endpoint.

```text
auditkit agent import-agenttune trajectories.jsonl --inspect
auditkit agent eval --config agent-suite.json --dry-run
auditkit agent eval --config agent-suite.json --output agent-run.json
auditkit agent rescore --run agent-run.json --scorers tool_call_f1,task_completion
```

`eval --output` and `import-agenttune --output` accept opt-in redaction for
files you plan to share: `--redact-key KEY` replaces the value of every `KEY`
field (for example `authorization` or `api_key`), and `--redact-env VAR`
replaces every occurrence of that environment variable's value, so the secret
never appears on the command line. An unset `--redact-env` variable stops the
command before any endpoint call. In Python, apply
`auditkit.agent_eval.types.redact(obj, secrets=[...], keys=[...])` to a
`to_dict()` before writing it. Nothing is redacted by default.

`--redact-auto` (or `redact(obj, auto=True)`) adds **automatic** secret
detection on top of the configured keys/strings. It is conservative -- it flags
and redacts, never silently passes a leaked secret -- and covers two halves:

- **Strong patterns**, redacted anywhere in any string regardless of the key:
  OpenAI (`sk-...`), GitHub (`ghp_...`), Slack (`xox...`), Google (`AIza...`)
  keys, AWS access-key ids (`AKIA...`), bearer tokens, JWTs, and PEM private-key
  blocks; plus a credential-looking dict **key name** (`api_key`, `secret`,
  `token`, `authorization`, `password`, `credential`, ...) whose *string* value
  is replaced (an int such as a token count is left alone).
- A **high-entropy heuristic** (length >= 32, Shannon entropy >= 4.0, mixed
  letters and digits) for random-looking tokens. This is the false-positive-prone
  half, so it never fires on AuditKit reproducibility fields -- case digests
  (`sha256:...`), header digests, `id`/`case_id`/`source_id`/`trial_id`/`call_id`
  and UUIDs are skipped, so a run's identity survives redaction.

**False positives:** whitelist a specific token with `allow=[...]`
(`redact(obj, auto=True, allow=[my_public_token])`), or fall back to exact
`--redact-key`/`--redact-env` and drop `--redact-auto` if the heuristic is too
eager for your data. `detect_secrets(obj)` lists what would be redacted
(`{path, key, kind}`, no value copied out); the CLI prints a count and the kinds
to stderr so redaction is visible, never silent.

`import-agenttune --inspect` prints the detected format, row count, coverage
labels and the trace-supported metrics:

```text
Detected format(s): {'agenttune_trajectory': 1}
Rows: 1  (tasks to run, not recorded episodes: 0)
Coverage labels: {'full': 1}
Eligible metrics (trace-supported): answer, parallel_tool_calls, redundant_tool_calls, retrieval, task_completion, tool_call_f1, tool_call_validity, trajectory_match
```

`eval --config` reads a JSON spec whose `cases[].outcome` is a plain dict spec
(`{"type": "final_state_assertion", "key": ..., "equals": ...}`, or
`artifact_assertion` / `answer_assertion`). A custom predicate has no JSON form;
build it in Python. `eval --dry-run` validates the tools, oracle, trials and
judge and prints the call plan without any model call.

The subcommands read and write three different JSON shapes. Feed the right file
to the right command:

| Produced by | Shape | Consumed by |
|---|---|---|
| `import-agenttune --output` | a bare list of episode dicts | `eval --config` under the `episodes` key |
| `eval --output` | `{"result", "episodes", "cases"}` | `rescore --run` |
| `AgentEvalResult.to_json()` | `{"schema_version", "mode", "spec_identity", "rows", "episodes"}` | `AgentEvalResult.from_json()` |

`rescore --run` reads only the `eval --output` shape. A custom-predicate oracle
cannot be rebuilt from that JSON, so its case falls back to the judge or to
`unknown` on replay.

## What is deferred

This release ships slices A1-A4 and the achievable part of A5 of the
AgentTune bridge PRD (`docs/notes/agent-evals-agenttune-prd.md`, internal).

Shipped:

- **A1.** The versioned case/episode/event records, importers for every
  `load_agenttune` shape plus OpenAI messages, `Sample` traces and AgentTune
  `EventLog` objects, loss-aware coverage, offline rescore, JSON export and the
  `import-agenttune --inspect` CLI.
- **A2.** State, artifact, answer and custom-predicate oracles with a judge
  fallback, deployed capture through `agent:` (output, trace, contexts,
  errors, latency, usage, and `final_state`/`artifacts` when the reply
  carries them), the summary and per-case drilldown, the `agent eval` and
  `agent rescore` CLI, and opt-in export redaction (`--redact-key`,
  `--redact-env`, `redact()`).
- **A3, reliability and trials.** `trials=N` runs each case N times and reports
  pass@k / all-k / variance / consistency and a stable majority aggregate over
  decided trials ([Reliability across trials](#reliability-across-trials-a3)).
  Independence is claimed only with `reset_confirmed=True`; otherwise counts are
  shown but flagged.
- **A4, harness-owned loop.** `mode="harness"` drives a bounded tool loop that
  AuditKit owns, with a caller-supplied tool double and step budget, real state
  verification, and call/turn ids per step
  ([Harness-owned loop](#harness-owned-loop-a4)).
- **A5, sidecar export.** `write_sidecar` / `read_sidecar` emit and round-trip
  an AgentTune-compatible JSONL sidecar keyed by `source_id`, read-only and
  without importing AgentTune ([AgentTune sidecar export](#agenttune-sidecar-export-a5)).
- **AG-05, disagreement filter.** A judge or source verdict that disagrees with
  a decided oracle is recorded on `row.disagreement` (the verified oracle still
  wins) and filtered by `result.disagreements(min_rate=)`
  ([Judge/oracle disagreement](#judgeoracle-disagreement-ag-05)).
- **Automatic secret detection.** `--redact-auto` / `redact(auto=True)` and
  `detect_secrets()` conservatively flag and redact common secrets in a
  shareable export, on top of the configured keys/strings (see the CLI section).

Still deferred:

- **A3 reset/snapshot protocol.** Trials run and reliability is computed, but
  the formal resettable-environment adapter (initial-state digests, an AuditKit-
  verified reset) is not built; independence is asserted by the caller
  (`reset_confirmed`), not proven by AuditKit.
- **A5, rest of the AgentTune bridge.** A `Project` intake, writing the sidecar
  back into AgentTune training data, and a side-by-side comparison with
  AgentTune's own diagnostics all need an AgentTune-side serializer that does not
  exist at the pinned mirror commit. Memory operations are kept as raw events and
  are not scored.
- **AG-05, multiple judges.** The filter covers judge-vs-oracle and
  source-vs-oracle over trials; voting across multiple independent judges is a
  seam, not built. `source_verdict` is set only when a report row carries
  `passed`; AgentTune's `Report.save` at the pinned commit does not write it, so
  in practice `source_vs_oracle` fires only for sources that do.

As-built notes against the PRD: the `inferred` coverage marker is emitted only
for a call/result pairing taken from a step observation (trace rows and
`EventLog`s with one call per turn). `AgentEpisode.trial_id` is now set by the
runner in deployed/harness trials and read by the reliability aggregate;
`AgentCase.environment` is accepted and serialized but not yet used by the
runner (it belongs with the deferred reset/snapshot protocol).
