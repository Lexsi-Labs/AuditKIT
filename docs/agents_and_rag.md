# Agents and RAG

AuditKit scores tool-calling models, deployed agents and RAG pipelines with the
same `ak.evaluate()` call as every other task. This page covers the data model,
how a run's trace reaches the metrics, every agent and RAG metric, the `agent:`
endpoint contract and the AgentTune bridge.

Runnable walkthrough (offline, no API key):
[`examples/14_agent_and_rag_evals.ipynb`](https://github.com/Lexsi-Labs/AuditKIT/blob/main/examples/14_agent_and_rag_evals.ipynb).
Serving a model with SGLang: [SGLang](SGLANG.md).

## Data model

Four `Sample` fields carry agent and RAG data:

| Field | Holds |
|---|---|
| `tools` | OpenAI-format tool schemas offered to the model. |
| `expected_tool_calls` | The reference calls, as a list of **turns** — or `{"any_of": [...]}` when several routes are correct. |
| `reference_contexts` | Gold retrieval targets: a list of chunk ids or texts, or `{id: grade}` for graded relevance. |
| `actual_trace` | A recorded run to score offline: `{"tool_calls": [...turns...], "messages": [...], "retrieved_contexts": [...]}`, any subset. |

A turn is the list of calls the model makes in one response:

- `[[a, b], [c]]`: two turns. `a` and `b` form a **parallel group**: they are
  independent and belong in one response. `c` is a later turn that depends on
  their results.
- `[a, b]` (a flat list of calls): **one** turn with two calls, BFCL style.
- `[]`: no tool should be called (an irrelevance case).

A call can be `{"name", "arguments"}` (`args`, `parameters` or `input` also
work), an OpenAI `tool_calls` entry with JSON-string arguments, or BFCL's
`{"func_name": {"arg": value}}`. Arguments that don't decode to a JSON object
give the call a `parse_error`, and it never matches a reference call.

```python
import auditkit as ak

TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "book_flight",
        "parameters": {"type": "object", "properties": {"flight_id": {"type": "string"}},
                       "required": ["flight_id"]}}},
]
W = lambda city: {"name": "get_weather", "arguments": {"city": city}}
BOOK = {"name": "book_flight", "arguments": {"flight_id": "AZ-61"}}

sample = ak.Sample(
    input="Which is warmer, Paris or Rome? Book AZ-61 if it's Rome.",
    tools=TOOLS,
    expected_tool_calls=[[W("Paris"), W("Rome")], [BOOK]],   # parallel group, then a dependent call
)
print(ak.to_turns(sample.expected_tool_calls))
```

`ak.load_jsonl(path, *, field_map=None)` reads these fields from JSONL, one sample
per line. The keys `input`, `target`, `id`, `choices`, `retrieval_context`,
`tools`, `expected_tool_calls`, `reference_contexts`, `actual_output`,
`actual_trace`, `tags` and `metadata` fill the matching `Sample` field,
`field_map` renames source keys first, and every other key lands in `metadata`.
Rows with `tools` or `expected_tool_calls` get `kind=TaskKind.AGENT`; rows with
`retrieval_context` or `reference_contexts` get `TaskKind.RAG`.

## Getting the trace

The agent and retrieval metrics read what the run did from
`context["trace"]`, which the Runner fills per sample. There are three
structured sources and one fallback.

**1. Native tool calls from `api:`.** `ak.ToolCallAdapter()` sends
`Sample.tools` as the request's `tools` field. `APIModel` keeps the returned
`message.tool_calls` as one turn, so several calls in one response count as a
parallel call. This works against OpenAI and any OpenAI-compatible server that
parses tool calls, such as vLLM (`--enable-auto-tool-choice --tool-call-parser
hermes`) or SGLang (`--tool-call-parser qwen25`).

```python
next_turn = ak.Sample(input=sample.input, tools=TOOLS,
                      expected_tool_calls=sample.expected_tool_calls[:1])
result = ak.evaluate(
    [next_turn],
    model="api:Qwen/Qwen2.5-7B-Instruct",
    api_base="http://localhost:8000/v1",
    api_key="EMPTY",                      # always pass one to a self-hosted server
    adapter=ak.ToolCallAdapter(),
    scorers=[ak.ToolCallF1(), ak.ParallelToolCalls()],
)
print(result.predictions[0].context["trace"]["tool_calls"])
```

`adapter=None` means `GenerationAdapter`, which never sends tools. Pass
`adapter=ak.ToolCallAdapter()`, or `adapter="auto"`, which picks
`ToolCallAdapter` when the first sample has `tools`. `ToolCallAdapter` takes
`mode` (`"native"` or `"prompt"`), `system_prompt`, `tool_choice` and
`parallel_tool_calls`; the last two are sent only in native mode. The
conversation is `Sample.metadata["messages"]` when set, else one user turn.

Native tools need a backend that declares `Capability.TOOLS`: `api:` (chat
mode, the default), `agent:`, and the local `hf:` and `vllm:` backends (which
pass the schemas to the model's chat template and parse its calls from the
reply). On any other backend, or with a chat template that can't render tools,
AuditKIT raises `CapabilityError` instead of dropping the tools silently.

One request is one turn: a single-response model only shows its **next**
action. Keep only the first reference turn for such evals (as above), or the
dependent later turns count as missed.

**2. An `agent:` endpoint.** An agent that runs its own tool loop returns its
transcript. Each assistant message with `tool_calls` is one turn. See
[Agent endpoints](#agent-endpoints-agent).

**3. A recorded run.** With `model="precomputed"`, the Runner scores
`Sample.actual_output` and puts `Sample.actual_trace` in the context. No model
is called.

```python
recorded = ak.Sample(
    input=sample.input, tools=TOOLS, expected_tool_calls=sample.expected_tool_calls,
    actual_output="Rome is warmer. Booked AZ-61.",
    actual_trace={"tool_calls": [[W("Paris"), W("Rome")], [BOOK]]},
)
result = ak.evaluate([recorded], model="precomputed",
                     scorers=[ak.ToolCallF1(), ak.ParallelToolCalls()])
print(result.headline)
```

**4. Fallback: tool calls parsed from text.** With no structured trace (text
backends such as `hf:`, `vllm:` or a callable, or a precomputed sample without
`actual_trace`), the metrics parse the text output as one turn.
`ak.parse_tool_calls()` removes `<think>...</think>` (and everything before a
lone `</think>`), then tries, in order: every Hermes/Qwen
`<tool_call>{...}</tool_call>` block, the whole reply as JSON (a call, a list
of calls, or `{"tool_calls": ...}`), and a fenced JSON block. Several
`<tool_call>` blocks in one reply are a parallel call. A block that isn't
valid JSON becomes a call with a `parse_error`, which never matches and counts
as invalid. A plain-text answer has no calls.

`ToolCallAdapter(mode="prompt")` writes the schemas into the system message
and asks for `<tool_call>` blocks, so any text backend can be evaluated.

```python
reply = ('<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>\n'
         '<tool_call>{"name": "get_weather", "arguments": {"city": "Rome"}}</tool_call>')
print(ak.parse_tool_calls(reply))   # two calls, one turn

result = ak.evaluate([next_turn], model=lambda prompts: [reply for _ in prompts],
                     adapter=ak.ToolCallAdapter(mode="prompt"),
                     scorers=[ak.ParallelToolCalls()])
print(result.headline)
```

Inside a trace, `tool_calls` wins over `messages`. With neither, the text
output is parsed.

## Agent metrics

| Class (registry name) | Needs | Emits | Better |
|---|---|---|---|
| `ToolCallF1` (`tool_call_f1`) | `expected_tool_calls` | `tool_call_precision`, `tool_call_recall`, `tool_call_f1`, `tool_call_exact` | higher |
| `TrajectoryMatch` (`trajectory_match`) | `expected_tool_calls` | `trajectory_strict`, `trajectory_in_order` | higher |
| `ParallelToolCalls` (`parallel_tool_calls`) | `expected_tool_calls` | `parallel_recall`, `parallel_precision`, `parallel_detection` | higher |
| `ToolCallValidity` (`tool_call_validity`) | `tools` | `tool_call_validity` | higher |
| `RedundantToolCalls` (`redundant_tool_calls`) | nothing | `redundant_tool_calls` | lower |
| `TaskCompletion` (`task_completion`) | `judge_model=` | `task_completion` | higher |

- **`ToolCallF1`** matches the multiset of calls and ignores turns. Duplicate
  calls count. Precision is matched / predicted, recall is matched /
  expected. `tool_call_exact` is 1.0 when every expected call was made and
  nothing else (any order). With `expected_tool_calls=[]`, all four scores are
  1.0 if the model made no call and 0.0 otherwise.
- **`TrajectoryMatch`**: `trajectory_strict` is 1.0 when the run has the same
  number of turns and turn *i* holds exactly the reference's turn *i* (any
  order inside a turn). `trajectory_in_order` is 1.0 when each reference
  turn's calls appear in the run's calls, in reference order. It allows extra
  calls and ignores turn boundaries.
- **`ParallelToolCalls`**: see [Parallel tool calls](#parallel-tool-calls).
- **`ToolCallValidity`** needs no reference. It is the fraction of calls that
  are valid against `Sample.tools`: the tool exists, required arguments are
  present, no argument is outside the schema, top-level JSON types and `enum`
  values hold, and the arguments decoded as JSON. `reason` lists the problems.
- **`RedundantToolCalls`** needs no reference. It is the fraction of calls
  that repeat an earlier call exactly (same name and arguments). It catches
  loops and re-fetching.
- **`TaskCompletion`** is an LLM judge. It sees the task, `Sample.target` as
  the expected outcome (optional), the tool-call transcript (with tool results
  when the trace has `messages`) and the final answer, and picks `complete`
  (1.0), `partial` (0.5) or `failed` (0.0). An unparseable verdict **raises**
  (the Runner records an error and skips the sample), never a silent 0.0
  averaged into the headline — the same contract as the RAG verdict judges.

A metric with nothing to measure on a sample returns no score, not 0:
`ToolCallValidity` and `RedundantToolCalls` skip samples with no calls,
`ToolCallF1`/`TrajectoryMatch`/`ParallelToolCalls` are ineligible when the
trace marks tool coverage unavailable (an `agent:` reply with no transcript and
no explicit `tool_calls`), and
`parallel_recall`/`parallel_precision` are omitted when they have no
denominator.

Matching is order-invariant inside a turn and uses a maximum bipartite
matching, so a correct answer is never rejected because of how calls were
paired.

### Parallel tool calls

- **`parallel_recall`**: of the reference parallel groups (turns with two or
  more calls), the fraction made together in one predicted turn. Each
  predicted call counts toward at most one group, so one batch can't fill two
  identical groups. Low means the model
  serializes independent calls and pays extra round trips.
- **`parallel_precision`**: of the predicted turns that hold two or more
  matched calls, the fraction whose matched calls all come from one reference
  turn. Low means the model batches dependent calls, so it guessed an argument
  before it had the result the argument depends on.
- **`parallel_detection`**: 1.0 when the model made a multi-call turn exactly
  when the reference has one (the "should I parallelize?" decision).

Worked example. The reference is
`[[get_weather(Paris), get_weather(Rome)], [book_flight(AZ-61)]]`:

| Run | Turns | `tool_call_f1` | `trajectory_strict` | `parallel_recall` | `parallel_precision` | `parallel_detection` |
|---|---|---|---|---|---|---|
| correct | `[[Paris, Rome], [book]]` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| serialized | `[[Paris], [Rome], [book]]` | 1.0 | 0.0 | 0.0 | omitted | 0.0 |
| wrongly batched | `[[Paris, Rome, book]]` | 1.0 | 0.0 | 1.0 | 0.0 | 1.0 |

`tool_call_f1` is 1.0 for all three, so a flat list of calls hides both
failures. The serialized run makes no multi-call turn, so it has no
`parallel_precision`. The batched run booked the flight before it knew which
city was warmer. Its guess happened to be right, and `parallel_precision`
still flags it.

```python
runs = {
    "correct": [[W("Paris"), W("Rome")], [BOOK]],
    "serialized": [[W("Paris")], [W("Rome")], [BOOK]],
    "wrongly batched": [[W("Paris"), W("Rome"), BOOK]],
}
for name, turns in runs.items():
    s = ak.Sample(input=sample.input, expected_tool_calls=sample.expected_tool_calls,
                  actual_output="", actual_trace={"tool_calls": turns})
    r = ak.evaluate([s], model="precomputed",
                    scorers=[ak.ToolCallF1(), ak.TrajectoryMatch(), ak.ParallelToolCalls()])
    print(name, {k: v for k, v in r.headline.items() if k.startswith(("parallel", "trajectory_strict"))})
```

### Argument matching

`ToolCallF1`, `TrajectoryMatch` and `ParallelToolCalls` take `arg_mode`:

- `"exact"` (default): arguments are equal as JSON values (`1 == 1.0`, but
  `True != 1`).
- `"subset"`: every reference argument is present with an equal value. Extra
  predicted arguments (optional defaults) are allowed.
- `"name"`: tool names only; arguments are ignored.

`arg_match=fn(tool_name, pred_args, ref_args) -> bool` replaces the argument
comparison for tolerances, aliases or allowed-value lists. Tool names must
still match.

A non-default `arg_mode` suffixes every score name, and `arg_match` adds
`_custom_<hash>` after it, where the hash identifies the matcher function:
`ToolCallF1(arg_mode="subset")` emits `tool_call_f1_subset`,
`tool_call_exact_subset` and so on, and `ParallelToolCalls(arg_match=fn)` emits
`parallel_recall_custom_<hash>`. The per-function hash means two different
`arg_match` matchers in one call never collide or get averaged together.

```python
close = lambda tool, pred, ref: all(abs(float(pred.get(k, "nan")) - float(v)) < 0.01 for k, v in ref.items())
scorers = [ak.ToolCallF1(arg_mode="subset"), ak.ParallelToolCalls(arg_match=close)]
```

### Alternative routes

A reference names **one** route, and a call only matches a reference call with
the same tool name. An agent that reaches the same correct answer another way —
`compare_weather(cities=["Paris", "Rome"])` where the reference says two
`get_weather` calls — is scored as wrong on a run that worked. The
`{"any_of": [...]}` form says every listed route is correct:

```python
CMP_TOOL = {"type": "function", "function": {
    "name": "compare_weather",
    "parameters": {"type": "object", "properties": {"cities": {"type": "array"}},
                   "required": ["cities"]}}}
COMPARE = {"name": "compare_weather", "arguments": {"cities": ["Paris", "Rome"]}}

multi = ak.Sample(
    input="Which is warmer, Paris or Rome?",
    tools=TOOLS + [CMP_TOOL],
    expected_tool_calls={"any_of": [
        [[W("Paris"), W("Rome")]],   # route 0: two calls in one turn
        [[COMPARE]],                 # route 1: one compare call
    ]},
)
```

Each entry is a route in any shape above — a list of turns, a flat list of
calls, `{"tool_calls": ...}`, an OpenAI transcript — and `[]` is a valid route,
meaning "calling no tool is also correct":

```python
# "Handle the Paris booking." -- act on what you have, or ask which one.
ASK = {"name": "ask_clarification", "arguments": {"question": "Which Paris booking?"}}

expected_tool_calls={"any_of": [
    [],       # no tool at all
    [[ASK]],  # or ask a clarifying question
]}
```

That generalises the existing irrelevance case rather than replacing it.
`expected_tool_calls=[]` alone still means "no tool should be called", and
`{"any_of": [[], [...]]}` adds a second acceptable answer to it: on
`"Handle the Paris booking."` the agent that asks and the agent that acts on
what it already has are both correct, while an unrelated third call scores 0.0
against both routes.

A plain reference is unchanged, and the two forms cannot collide:
`{"any_of": ...}`'s value is a **list**, and no call shape reads a list as a
call (a call needs a `name`, or BFCL's single key with a dict value). No
reference that worked before starts being read as a multi-route one, and
`ak.to_turns()` still raises on it — that is the guarantee, not an oversight.

One route is chosen per sample, and all three reference metrics score **that
same route**: `tool_call_f1`, `trajectory_match` and `parallel_tool_calls` never
describe two different runs. Otherwise `tool_call_f1` could take its recall from
route 0 while `parallel_tool_calls` takes its parallelism from route 1, and the
three columns would be about two different behaviours. The choice is made on
call-level F1 over the flattened calls, so it doesn't depend on turn structure,
with ties broken by higher recall, then by fewer unmatched predicted calls, then
by the order the routes are listed.

With more than one route, every reference score carries:

| Metadata | Holds |
|---|---|
| `matched_path` | index of the route the run was scored against |
| `n_paths` | how many routes the sample listed |
| `path_f1` | F1 against each route, in the listed order, so the choice can be checked |

A single-route reference carries none of them: `matched_path: 0` on every
existing score would change the metadata of every existing run for nothing.

```python
runs = {"two calls": [[W("Paris"), W("Rome")]], "one compare call": [[COMPARE]]}
for name, turns in runs.items():
    s = ak.Sample(input=multi.input, tools=multi.tools,
                  expected_tool_calls=multi.expected_tool_calls,
                  actual_output="", actual_trace={"tool_calls": turns})
    r = ak.evaluate([s], model="precomputed",
                    scorers=[ak.ToolCallF1(), ak.ParallelToolCalls()])
    f1 = next(sc for sc in r.predictions[0].metadata["scores"]
              if sc["name"] == "tool_call_f1")
    print(name, f1["value"], f1["metadata"]["matched_path"], f1["metadata"]["path_f1"])
# two calls         1.0 0 [1.0, 0.0]
# one compare call  1.0 1 [0.0, 1.0]
```

`AgentCase.reference_turns` takes the same forms (see
[Agent evaluation](agent_eval.md)), and a JSONL row's `expected_tool_calls` may
carry `{"any_of": ...}` as is.

What this does not do:

- **Routes are whole, never combined.** A run that takes half of route 0 and
  half of route 1 is scored against the better *single* route, not the best of
  both. Three independent two-way choices therefore need 2 × 2 × 2 routes
  listed by hand.
- **No per-call alternatives.** `{"one_of": [...]}` inside a turn, which would
  make `search_web` and `lookup_kb` the same step and remove that
  multiplication, is not implemented; it is waiting on a real multi-path
  dataset (BFCL's `possible_answer`).
- **No route weights.** Every listed route is equally correct, and how
  efficient the route was is `redundant_tool_calls` and `parallel_tool_calls`,
  which measure the run itself.
- **A JSON *string* reference is one route.** `{"any_of": ...}` is recognised
  only in a decoded structure, so a CSV cell holding that JSON is not a
  multi-route reference — `ak.to_turns()` raises on it.

## Agent endpoints (`agent:`)

`model="agent:<url>"` (`AgentEndpointModel`) evaluates an agent that runs
somewhere else and owns its tool loop: a LangGraph service, an AgentTune
checkpoint behind an API, your own server. AuditKit sends one POST per sample
and scores what comes back. It is stdlib-only.

Default request body:

```json
{"input": "<last user message>", "messages": [...chat turns...], "tools": [...]}
```

`tools` is sent only when the request carries them (`ToolCallAdapter` with
`Sample.tools`). `input_key=` renames `input`, and `extra_body=` is merged
into every body (for example `{"thread_id": ...}`). Generation settings are
not sent: the agent owns its sampling.

Default response (any subset):

```json
{"output": "final answer",
 "messages": [...OpenAI transcript with tool_calls and role=tool results...],
 "tool_calls": [[call, call], [call]],
 "contexts": ["retrieved chunk or id", "..."]}
```

- With a transcript but no `output`, the last assistant message is the answer.
- `tool_calls` (turns) is optional when `messages` is given.
- `contexts` is the ranked retrieval that feeds `RetrievalMetrics` and the RAG
  judges. `RetrievalMetrics` compares it to `reference_contexts` as exact
  strings, so both must use the same form: ids with ids (e.g. AgentTune
  `message_ids`), or chunk texts with chunk texts. The RAG judges need chunk
  text, not ids.
- An OpenAI chat completion (`{"choices": [...]}`) is detected and read the
  same way `api:` reads it.

Other contracts:

- **Dotted paths**: `output_path`, `messages_path`, `tool_calls_path`,
  `contexts_path` (for example `output_path="data.answer"` or
  `"result.0.text"`; integer segments index lists).
- **`request_fn`**: `Request -> dict`, builds the whole body.
- **`response_fn`**: `decoded JSON -> dict` with the keys `output`, `messages`,
  `tool_calls` and `retrieved_contexts`. Note `retrieved_contexts` here, not
  `contexts`. With `response_fn` set, OpenAI auto-detection is off.

Auth: `api_key=` or the `AGENT_API_KEY` environment variable, sent as
`Authorization: Bearer <key>`. It never falls back to `OPENAI_API_KEY`, so a
self-hosted agent never receives your OpenAI key. Use `headers=` for other
schemes. An HTTP error or a non-JSON reply raises `ModelError` with the
response body.

```python
result = ak.evaluate(
    [sample], model="agent:http://localhost:8000/run",
    adapter=ak.ToolCallAdapter(),
    scorers=[ak.ToolCallF1(), ak.TrajectoryMatch(), ak.ParallelToolCalls()],
)

# A service with its own response shape:
result = ak.evaluate(
    [sample], model="agent:http://localhost:8000/v2",
    response_fn=lambda d: {"output": d["result"]["answer"],
                           "tool_calls": d["result"]["turns"],
                           "retrieved_contexts": d["result"].get("sources", [])},
    adapter=ak.ToolCallAdapter(), scorers=[ak.ParallelToolCalls()],
)
print(result.headline)
```

## RAG

The ranked retrieved list is the trace's `retrieved_contexts` when the run
returned one (an `agent:` endpoint's `contexts`, or `actual_trace`), else
`Sample.retrieval_context`. So a live pipeline is scored on what it actually
retrieved, not on what the dataset shipped.

### Retrieval ranking: `RetrievalMetrics(k=None)`

Deterministic, no model calls. Relevance comes from `reference_contexts`: a
list (binary relevance) or a `{id: grade}` dict (graded; grades <= 0 are
ignored). Items compare as exact strings after stripping whitespace. The list
is cut to the top `k` first; a duplicate of an already-seen chunk counts once
but keeps its slot at zero relevance, so items after it hold their original
rank — a repeat never lifts a later item up a rank (which would inflate
mrr/ap/ndcg).

| Score | Value |
|---|---|
| `hit_rate` | 1.0 if any relevant item was retrieved |
| `precision` | relevant retrieved / `k` (even when fewer than `k` came back); / retrieved slots without `k` (a duplicate slot counts in the denominator) |
| `recall` | relevant retrieved / all relevant |
| `mrr` | 1 / rank of the first relevant item |
| `average_precision` | sum of precision@*i* over the ranks *i* of relevant items, divided by **all** relevant items (`min(k, n_relevant)` with `k`) |
| `ndcg` | DCG / IDCG with DCG = sum of grade / log2(rank + 1); IDCG comes from the gold grades sorted high to low (top `k`), not from the retrieved list |

With `k` set, every name gets an `@k` suffix (`recall@5`). Samples without
`reference_contexts` are skipped; an empty retrieval scores 0 everywhere.
Because `average_precision` divides by all relevant items, a gold chunk that
was never retrieved lowers it. RAGAS-style AP divides by the relevant items
that were retrieved, which hides misses.

```python
s = ak.Sample(input="How often are laptops replaced?", reference_contexts=["it-04"],
              actual_output="Every 3 years.",
              actual_trace={"retrieved_contexts": ["it-09", "it-04", "hr-31"]})
r = ak.evaluate([s], model="precomputed", scorers=[ak.RetrievalMetrics(k=3)])
print(r.headline)   # mrr@3 = 0.5, average_precision@3 = 0.5, ndcg@3 = 0.63
```

### Judge metrics

| Class (registry name) | Emits | Needs | Judge calls |
|---|---|---|---|
| `Faithfulness` (`faithfulness`) | `faithfulness` | retrieved contexts, a non-empty answer | 2 |
| `ContextPrecision` (`context_precision`) | `context_precision`, `context_relevance` | retrieved contexts (uses `target` when set) | 1 |
| `ContextRecall` (`context_recall`) | `context_recall` | retrieved contexts, `target` | 1 |

- **`faithfulness`**: the judge splits the answer into claims, then labels each
  claim SUPPORTED, UNSUPPORTED or CONTRADICTED against the contexts. The score
  is supported / claims, so an unsupported claim fails too. `metadata` holds
  the claims, the verdicts and `n_contradicted`. A sample whose answer makes
  no claim (the judge replies `NONE`, e.g. a refusal) is skipped.
- **`context_precision`**: the judge labels each chunk RELEVANT or IRRELEVANT
  for the question (and the reference answer, when set). The score is
  RAGAS-style average precision over the relevant chunks retrieved: 1.0 when
  every relevant chunk ranks above every irrelevant one.
  `context_relevance` is the plain fraction of relevant chunks. With gold ids,
  prefer `RetrievalMetrics`.
- **`context_recall`**: `target` is split into sentences locally, and the
  judge labels each one ATTRIBUTED or NOT_ATTRIBUTED to the contexts. The
  score is attributed / sentences.

The judge can be any text-in/text-out model; no function calling or
structured output is needed. It answers one numbered line per item
(`3: SUPPORTED`). The parser requires exactly the verdicts 1..n. A reply that
fails that check raises, so the Runner records an error in `result.errors`
and the sample gets no score for that metric. A parse failure is never a 0.

Options: `judge_model` (a spec string or a model object), `judge_model_args`
(passed to `AutoModel.resolve`, e.g. `api_base`/`api_key`), `temperature`
(default 0.0), `max_tokens`, `max_context_chars` (truncates the contexts
block), and `judge_chat_template_kwargs` (sent to the judge's chat template on
every judge request; every LLM judge takes it, including `LLMJudge`,
`TaskCompletion` and `ToolSelectionJudge`). A thinking model used as the judge
(Qwen3) needs `judge_chat_template_kwargs={"enable_thinking": False}`, or it
spends its budget reasoning and no verdict comes back (recorded as an error).
`RunConfig.chat_template_kwargs` is not inherited: the judge is usually a
different model from the one under test.

```python
judge = dict(judge_model="api:Qwen/Qwen2.5-7B-Instruct",
             judge_model_args={"api_base": "http://localhost:8000/v1", "api_key": "EMPTY"})
s = ak.Sample(input="How often are laptops replaced?", target="Laptops are replaced every 3 years.",
              actual_output="Laptops are replaced every 3 years.",
              actual_trace={"retrieved_contexts": ["Guest wifi passwords rotate every Monday.",
                                                   "Laptops are replaced every 3 years through the IT portal."]})
r = ak.evaluate([s], model="precomputed",
                scorers=[ak.Faithfulness(**judge), ak.ContextPrecision(**judge), ak.ContextRecall(**judge)])
print(r.headline, r.errors)
```

Known limits:

- Claim extraction is non-deterministic: two runs can split one answer into
  different claims.
- All contexts go into one judge prompt. Use `max_context_chars` with a small
  judge.
- The judge must see chunk **texts**. Retrieved ids (as in AgentTune
  trajectories) work with `RetrievalMetrics` only.

### Reference-free RAG metrics

These need **no gold answer and no labelled relevant chunks** -- only some of
the question, retrieved contexts and answer. Use them to grade a RAG pipeline
on data where no reference was ever written. Each declares
`required_fields = frozenset()` (nothing on the sample is mandatory), is
registered by name so `ak.evaluate(metrics=[...])` finds it, and follows the
same numbered-verdict contract as the metrics above: a reply the parser cannot
read raises, so the Runner records an error and the sample gets no score -- a
parse failure is never a 0.

| Class (registry name) | Emits | Needs | Judge calls |
|---|---|---|---|
| `AnswerRelevancy` (`answer_relevancy`) | `answer_relevancy` | question, a non-empty answer | 2 |
| `ResponseGroundedness` (`response_groundedness`) | `response_groundedness` | retrieved contexts, a non-empty answer | 1 |
| `Hallucination` (`hallucination`) | `hallucination` (lower is better) | retrieved contexts, a non-empty answer | 2 |
| `ContextRelevance` (`context_relevance`) | `context_relevance` | question, retrieved contexts | 1 |

- **`answer_relevancy`**: the judge extracts the answer's statements, then
  labels each ADDRESSES or OFF_TOPIC against the question. The score is
  ADDRESSES / statements; `reason` lists the off-topic ones. Needs neither
  context nor a gold answer. (Statement-verdict variant, not the embedding
  reverse-question one.)
- **`response_groundedness`**: the answer is split into sentences locally (no
  extraction call), then the judge grades each SUPPORTED or UNSUPPORTED against
  the contexts. The score is supported / sentences. Unlike `faithfulness` the
  sentence set is deterministic and contradiction is not called out separately.
- **`hallucination`**: reuses `faithfulness`'s claim extraction and verdicts;
  `hallucination = 1 - supported / claims`, so both contradicted and merely
  unsupported claims count. **Lower is better** (`Direction.MINIMIZE`).
  `reason` lists the offending claims, `metadata` holds `n_contradicted`.
- **`context_relevance`**: the mean fraction of retrieved chunks the judge
  labels relevant to the question (judged against the question alone, never a
  reference answer). Runs on its own; `context_precision` already reports this
  same number as a second score, so do not request both together.

## AgentTune bridge

`ak.load_agenttune(path)` reads AgentTune files and detects the shape of each
record:

| AgentTune file | Becomes | Feeds |
|---|---|---|
| `run_eval` / RAG-GRPO dataset rows: `{"prompt", "answer", "message_ids"?}` or `{"prompt", "gold_answer", "question_id", "gold_chunk_ids"}` | Tasks to run. `input` = last user message, `target` = `answer` (the first, if a list) or `gold_answer`, `reference_contexts` = flattened gold ids, `metadata["messages"]` = the full prompt, so `ToolCallAdapter` sends the original system prompt. | Answer metrics (`exact_match`, `quasi_exact_match`, LLM judges, `TaskCompletion`); `RetrievalMetrics` when the endpoint returns retrieved ids as `contexts`. No reference calls, so no tool-call metrics. |
| `TrajectoryDataset` JSONL: `{"task", "steps", "final_response", "metadata": {"conversation", "retrieved_chunk_ids"}}` | Recorded runs for `model="precomputed"`. `actual_trace["tool_calls"]` = one turn per step with tool calls, built from `steps` even when a conversation exists (the rollout can truncate or rewrite the conversation), and these turns are what the tool-call metrics score. `actual_trace["messages"]` = the conversation, kept for the judge and rendering only. `actual_trace["retrieved_contexts"]` = retrieved chunk ids; `metadata["reward"]` = the reward. | `RedundantToolCalls`, `TaskCompletion`; `ToolCallValidity` after you attach `tools`; `ToolCallF1` / `TrajectoryMatch` / `ParallelToolCalls` after you attach `expected_tool_calls`; `RetrievalMetrics` after you attach `reference_contexts`. Not the RAG judges (ids, not text). |
| `trace.jsonl` from train_grpo's `TraceLogger`, or a `TrajectoryStore` export: `{"question", "final_answer", "gold_answer"?, "retrieved_chunk_ids"?, "gold_chunk_ids"?, "tool_calls"?}` | Recorded answers: `actual_output` = `final_answer` without its `<answer>` tag, `reference_contexts` = gold chunk ids, `actual_trace["retrieved_contexts"]` = retrieved ids. Its `tool_calls` stay in `metadata` (no tool names the metrics can read). A store export's `trajectory_id` becomes `id`. | Answer metrics and `RetrievalMetrics`. No tool-call metrics. |
| `run_eval` report JSON (`Report.save`, a `"samples"` list) | Precomputed answers: `input` = question, `target` = gold, `actual_output` = predicted. | Answer metrics only. Its tool calls are flattened names with no turns. |

`load_agenttune` refuses two other AgentTune files with a clear `ValueError`:
the heal audit JSONL that `Project.heal` writes (a lossy per-call projection,
not a trajectory) and a serialized `EventLog`. Read an `EventLog` with
`auditkit.agent_eval.episode_from_eventlog` (see [Agent evaluation](agent_eval.md)).

```python
rows = ak.load_agenttune("mail_eval.jsonl")          # run_eval rows
result = ak.evaluate(rows, model="agent:http://localhost:8000/run",
                     adapter=ak.ToolCallAdapter(),
                     scorers=["exact_match", ak.RetrievalMetrics(k=2)])

runs = ak.load_agenttune("trajectories.jsonl")      # recorded runs
for s in runs:
    s.tools, s.expected_tool_calls = TOOLS, sample.expected_tool_calls
result = ak.evaluate(runs, model="precomputed",
                     scorers=[ak.ToolCallF1(), ak.ParallelToolCalls(), ak.RedundantToolCalls()])
print(result.headline)
```

## End-to-end agent evaluation

The metrics above score one axis of a run at a time. The
[`auditkit.agent_eval`](agent_eval.md) layer wraps them in an episode contract
that evaluates what an agent did and achieved across a whole task. The headline
is a **verified outcome** (final state, artifact, answer, or a custom predicate,
with verdicts `success` / `failure` / `unknown` / `error`); the tool, parallel,
retrieval and judge scores above become separate diagnostic columns.

It reuses this page's parts, it does not replace them. The same scorer classes
run through `agent_eval`'s runner, deployed mode resolves an `agent:` endpoint
via `AutoModel.resolve`, `episode_from_sample` bridges a recorded
`Sample.actual_trace` into an episode, and `episodes_from_agenttune` sits on top
of `load_agenttune`. It adds loss-aware coverage (a flattened report is scored
tool-name only, so argument-sensitive metrics stay ineligible rather than
reading empty arguments as wrong ones), per-case status, both success
denominators, and an offline `rescore` that scores saved episodes with no agent
call. See [Agent evaluation](agent_eval.md).

## How this compares

AuditKit pairs predicted and reference calls with a maximum bipartite matching
(Kuhn's algorithm). BFCL, agentevals and DeepEval pair them greedily
(first fit), which can reject a correct answer when one predicted call fits
several reference calls. RAGAS sorts and zips, and its F1 uses sets, so
duplicate calls vanish. BFCL multi-turn, RAGAS and DeepEval flatten calls
across turns. Of the libraries we reviewed (RAGAS, DeepEval, TruLens, Phoenix,
BFCL, agentevals, inspect_ai, tau-bench and others), none scores parallel tool
calling as a behavior: whether independent calls were batched and dependent
calls were not. On the RAG side, `average_precision` divides by all relevant
items, `ndcg` takes its ideal DCG from the gold grades, and `faithfulness`
fails unsupported claims, where DeepEval fails only contradicted ones.

See [Known Issues](known_issues.md) for the open limits (native tools on
`api:`/`agent:`/`hf:` only, one turn per request, non-deterministic claim
extraction).
