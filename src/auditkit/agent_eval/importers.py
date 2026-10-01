"""Trace adapters: turn recorded runs into :class:`AgentEpisode` (A1, AG-03/04).

Three sources, all dependency-free and loss-aware:

- AgentTune ``load_agenttune`` files (run_eval rows/report, trajectory JSONL,
  RAG ``trace.jsonl``) -- read via the existing loader, with the raw records
  zipped back in so event payloads and order (incl. parallel groups) survive.
- a list of OpenAI chat messages.
- a :class:`~auditkit.sample.Sample` (its ``actual_trace``).

AgentTune is never imported: the shapes are read as plain dicts/JSON (AG-04).
``coverage`` is set honestly per source -- a flattened report is marked
tool-name only, and missing arguments/state are never fabricated (AG-03).
Source reward / verdict / scores are preserved as provenance, never promoted to
the outcome (AG-05).
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from ..provenance import read_provenance
from ..trace import is_call_payload
from .types import (
    INFERRED,
    OBSERVED,
    UNAVAILABLE,
    AgentEpisode,
    AgentEvent,
)


# --------------------------------------------------------------------------
# coverage helpers
# --------------------------------------------------------------------------
def _coverage(**flags: Any) -> dict[str, str]:
    """Build a coverage map from bool/str flags (True->observed, False->unavailable)."""
    out: dict[str, str] = {}
    for key, val in flags.items():
        if isinstance(val, str):
            out[key] = val
        else:
            out[key] = OBSERVED if val else UNAVAILABLE
    return out


def supported_scorers_from_coverage(coverage: dict[str, str]) -> list[str]:
    """Which scorers a trace with this coverage can feed, ignoring the case.

    Used by ``--inspect`` and by the runner's per-case eligibility. A judge is
    always available (diagnostic); argument-sensitive tool metrics need
    ``tool_call_arguments`` observed, name-only traces get the ``_name`` variant.
    """
    supported = ["task_completion"]
    names = coverage.get("tool_call_names") == OBSERVED
    args = coverage.get("tool_call_arguments") == OBSERVED
    if names and args:
        supported += ["tool_call_f1", "trajectory_match", "tool_call_validity", "redundant_tool_calls"]
        if coverage.get("parallel_grouping") == OBSERVED:
            supported.append("parallel_tool_calls")
    elif names:
        supported += ["tool_call_f1_name", "tool_call_validity"]
    # arguments without tool names (an AgentTune TraceLogger record) feed no call metric:
    # every one of them compares tool NAMES first.
    if coverage.get("retrieved_contexts") == OBSERVED:
        supported.append("retrieval")
    return supported


# --------------------------------------------------------------------------
# event builders (shared)
# --------------------------------------------------------------------------
def _action_calls(action: Any) -> list[dict[str, Any]]:
    """Individual call dicts from a step action (rollout ``{"tool_calls": [...]}``
    or flat ``{"name", "arguments"}``); preserves order and parallel groups."""
    if isinstance(action, dict) and isinstance(action.get("tool_calls"), list):
        return [c for c in action["tool_calls"] if isinstance(c, dict)]
    if is_call_payload(action):   # incl. a parameterless Cohere {"tool_name"} call
        return [action]
    return []


# Keys that carry a call's arguments: an action with one of them but no tool name is a call
# whose name was not recorded (a TraceLogger-style record), not a no-call step.
_CALL_ARG_KEYS = ("arguments", "parameters", "args", "input", "query")

# Keys that mark the record as a call even when it carries nothing else. Checked for
# *presence*, not truthiness, so a name that was recorded but left blank ("") still reads
# as a call that lost its name rather than as a step that made none.
#
# Read against AgentTune's EventLog writers (`agenttune/agentic/events.py`), which emit
# three TOOL_CALL shapes: a bare `{"name", "arguments"}` (spine/eval), a DECIDE
# `{"stage": id}` no-call, and the rollout `{"tool_calls": [{"type": "function",
# "function": {"name", "arguments"}}]}` that `rollout_factory.py` builds. The rollout's
# entries are already unwrapped by _action_calls, so its `arguments` is seen; these two
# cover what that leaves -- a zero-argument call, and a blank name.
_CALL_MARKER_KEYS = ("name", "tool_name")

# An OpenAI call marker only counts at the value that means a call. `type` alone is not
# enough: OpenAI content parts and audit rows carry {"type": "text"}, {"type": "refusal"},
# {"type": "image_url"} and are not calls, so a bare presence test would read a step that
# made no call as one that lost its name -- the opposite error, and the same wrong number.
_CALL_MARKER_TYPE = "function"


def _has_unnamed_call(action: Any) -> bool:
    """A recorded action holding a call without a tool name.

    Such a step shows that a call was made but not which tool, so the source's tool names
    are not observed (scoring it as zero calls would be a measured 0.0 for a real call).
    A no-call action (EventLog's terminal ``{}``, DECIDE's ``{"stage": ...}``) is not one.
    """
    def unnamed(c: Any) -> bool:
        # explicit name fields only: normalize_call would read {"arguments": {...}} as BFCL's
        # one-key form, i.e. a call NAMED "arguments"
        fn = c.get("function") if isinstance(c.get("function"), dict) else {}
        named = c.get("name") or c.get("tool_name") or fn.get("name")
        if named:
            return False
        # A name that was recorded but blank still means a call was attempted. So does an
        # OpenAI call marker, which is present even when `function` carries nothing.
        if any(k in c or k in fn for k in _CALL_MARKER_KEYS):
            return True
        if c.get("type") == _CALL_MARKER_TYPE or fn.get("type") == _CALL_MARKER_TYPE:
            return True
        return any(k in c or k in fn for k in _CALL_ARG_KEYS)
    items = _action_calls(action) or ([action] if isinstance(action, dict) else [])
    return any(unnamed(c) for c in items if isinstance(c, dict))


def _events_from_messages(messages: list[dict[str, Any]], source: str) -> list[AgentEvent]:
    """OpenAI transcript -> ordered events. One assistant message's tool_calls
    share a ``turn_id`` (parallel group); a ``tool`` message pairs by id."""
    events: list[AgentEvent] = []
    idx = 0
    for mi, m in enumerate(messages):
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        calls = m.get("tool_calls")
        if role == "assistant" and calls is not None and not isinstance(calls, list):
            raise ValueError(f"messages[{mi}].tool_calls must be a list, got {type(calls).__name__}")
        if role == "assistant" and calls:
            for c in m["tool_calls"]:
                events.append(AgentEvent(index=idx, role="assistant", type="tool_call",
                                         payload=c if isinstance(c, dict) else {"raw": c},
                                         call_id=c.get("id") if isinstance(c, dict) else None,
                                         turn_id=mi, source=source))
                idx += 1
        elif role == "tool":
            events.append(AgentEvent(index=idx, role="tool", type="tool_result",
                                     payload={"output": m.get("content")},
                                     call_id=m.get("tool_call_id"), turn_id=mi, source=source))
            idx += 1
        elif role == "assistant":
            content = m.get("content")
            if content:
                events.append(AgentEvent(index=idx, role="assistant", type="text",
                                         payload={"text": content}, turn_id=mi, source=source))
                idx += 1
        elif role in ("user", "system"):
            events.append(AgentEvent(index=idx, role=role, type="observation",
                                     payload={"text": m.get("content")}, turn_id=mi, source=source))
            idx += 1
    return events


def _last_assistant_text(messages: list[dict[str, Any]]) -> Optional[str]:
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "assistant" and not m.get("tool_calls"):
            return m.get("content")
    return None


def _retrieved_event(index: int, retrieved: Any, source: str) -> AgentEvent:
    return AgentEvent(index=index, role="environment", type="observation",
                      payload={"retrieved_contexts": [str(c) for c in retrieved]}, source=source)


# --------------------------------------------------------------------------
# (b) OpenAI chat messages
# --------------------------------------------------------------------------
def episode_from_openai_messages(messages: list[dict[str, Any]], *, mode: str = "recorded",
                                 case_id: Optional[str] = None, source_id: Optional[str] = None,
                                 final_answer: Optional[str] = None,
                                 retrieved_contexts: Optional[list[str]] = None,
                                 source: str = "openai_messages") -> AgentEpisode:
    """A list of OpenAI chat messages -> one :class:`AgentEpisode`."""
    events = _events_from_messages(messages, source)
    if retrieved_contexts:
        events.append(_retrieved_event(len(events), retrieved_contexts, source))
    if final_answer is None:
        final_answer = _last_assistant_text(messages)
    has_results = any(e.type == "tool_result" for e in events)
    # A transcript with an assistant turn shows every call it made, so zero calls
    # is an observation (a skipped tool, or a correct irrelevance), not a gap.
    seen = (any(e.type == "tool_call" for e in events)
            or any(isinstance(m, dict) and m.get("role") == "assistant" for m in messages))
    coverage = _coverage(
        final_answer=final_answer is not None,
        tool_call_names=seen,
        tool_call_arguments=seen,
        tool_results=has_results,
        call_result_pairing=has_results,
        parallel_grouping=seen,
        timestamps=False,
        retrieved_contexts=bool(retrieved_contexts),
        final_state=False,
    )
    return AgentEpisode(events=events, mode=mode, source_format=source, case_id=case_id,
                        source_id=source_id, final_answer=final_answer, coverage=coverage,
                        counters={"n_tool_calls": sum(1 for e in events if e.type == "tool_call")})


# --------------------------------------------------------------------------
# (c) Sample.actual_trace (and any load_jsonl / user-built Sample)
# --------------------------------------------------------------------------
def episode_from_sample(sample: Any, *, mode: str = "recorded",
                        source: str = "sample_trace") -> AgentEpisode:
    """A :class:`Sample` (its ``actual_trace`` + ``actual_output``) -> episode.

    Trusts ``actual_trace["messages"]`` as the transcript (the OpenAI-messages
    convention). AgentTune trajectory files go through :func:`episodes_from_agenttune`
    instead, which reads the authoritative steps rather than the rewritten
    conversation.
    """
    trace = getattr(sample, "actual_trace", None) or {}
    messages = trace.get("messages")
    turns = trace.get("tool_calls")
    retrieved = trace.get("retrieved_contexts")
    final = getattr(sample, "actual_output", None)
    events: list[AgentEvent] = []
    if messages:
        events = _events_from_messages(messages, source)
        if final is None:
            final = _last_assistant_text(messages)
    elif turns is not None:
        idx = 0
        for ti, turn in enumerate(turns if isinstance(turns, list) else []):
            calls = turn if isinstance(turn, list) else [turn]
            for c in calls:
                events.append(AgentEvent(index=idx, role="assistant", type="tool_call",
                                         payload=c if isinstance(c, dict) else {"raw": c},
                                         call_id=c.get("id") if isinstance(c, dict) else None,
                                         turn_id=ti, source=source))
                idx += 1
    if retrieved:
        events.append(_retrieved_event(len(events), retrieved, source))
    has_results = any(e.type == "tool_result" for e in events)
    # An explicit tool_calls list (even []) or an assistant transcript is a full
    # record of the calls made; only its absence is "no trace".
    seen = (any(e.type == "tool_call" for e in events)
            or (not messages and isinstance(turns, list))
            or any(isinstance(m, dict) and m.get("role") == "assistant"
                   for m in (messages or [])))
    coverage = _coverage(
        final_answer=final is not None,
        tool_call_names=seen,
        tool_call_arguments=seen,
        tool_results=has_results,
        call_result_pairing=has_results,
        parallel_grouping=seen,
        timestamps=False,
        retrieved_contexts=bool(retrieved),
        final_state=False,
    )
    return AgentEpisode(events=events, mode=mode, source_format=source,
                        case_id=getattr(sample, "id", None),
                        source_id=getattr(sample, "id", None), final_answer=final,
                        coverage=coverage, metadata={"task": getattr(sample, "input", None),
                                                     "target": getattr(sample, "target", None)},
                        counters={"n_tool_calls": sum(1 for e in events if e.type == "tool_call")})


# --------------------------------------------------------------------------
# (a) AgentTune load_agenttune files
# --------------------------------------------------------------------------
def _load_agenttune_pairs(path: str) -> list[tuple[str, dict[str, Any], Any]]:
    """``[(format, raw_record, Sample), ...]`` -- raw records zipped 1:1 with the
    Samples ``load_agenttune`` produced (same order, same blank-line skipping)."""
    from ..loaders import load_agenttune

    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    try:
        doc = json.loads(text)
    except ValueError:
        doc = None
    samples = load_agenttune(path)
    if isinstance(doc, dict) and isinstance(doc.get("samples"), list):
        # Report-level provenance (AG-05): the only model identity a run_eval
        # report carries is top-level, so every row keeps it.
        prov = {k: doc[k] for k in _REPORT_PROVENANCE if k in doc}
        raws = [dict(r, _source_report=prov) if isinstance(r, dict) else r for r in doc["samples"]]
        fmts = ["agenttune_report"] * len(raws)
    else:
        # Same blank-line skipping load_agenttune uses, so records align 1:1.
        # split on \n only: splitlines() also breaks on U+2028/U+0085 inside JSON strings
        raws = [json.loads(line) for line in text.split("\n") if line.strip()]
        fmts = [_detect_format(r) for r in raws]
    if len(raws) != len(samples):  # defensive: loader and raw pass disagreed
        raise ValueError(f"{path}: raw record count {len(raws)} != loaded sample count "
                         f"{len(samples)}; cannot pair records")
    return list(zip(fmts, raws, samples))


_REPORT_PROVENANCE = ("model", "use_case", "pass_rate", "n_samples", "n_errors", "timestamp")


def _detect_format(raw: dict[str, Any]) -> str:
    if "question" in raw and "final_answer" in raw:
        return "agenttune_trace"
    if "steps" in raw or "trajectory_id" in raw or "final_response" in raw:
        return "agenttune_trajectory"
    if "prompt" in raw:
        return "agenttune_run_eval_row"
    return "agenttune_report"


def _episode_report(raw: dict[str, Any], sample: Any) -> AgentEpisode:
    names = raw.get("tool_calls") or []
    events = [AgentEvent(index=i, role="assistant", type="tool_call",
                         payload={"name": str(n)}, turn_id=i, source="agenttune_report")
              for i, n in enumerate(names)]
    coverage = _coverage(
        # an errored sample's '' is the absence of an answer, not an observed one
        final_answer=sample.actual_output is not None and not raw.get("error"),
        tool_call_names=bool(names),
        tool_call_arguments=False,   # a flattened report has NAMES only (AG-03)
        tool_results=False, call_result_pairing=False, parallel_grouping=False,
        timestamps=False, retrieved_contexts=False, final_state=False,
    )
    return AgentEpisode(events=events, mode="recorded", source_format="agenttune_report",
                        source_id=None if raw.get("idx") is None else str(raw["idx"]),
                        final_answer=sample.actual_output,
                        # Report.save at 36d4724 drops the per-row `passed`; read it
                        # when a report carries one, never derive it (AG-05).
                        source_verdict=raw.get("passed"),
                        source_scores=raw.get("scores") if isinstance(raw.get("scores"), dict) else None,
                        errors=[str(raw["error"])] if raw.get("error") else [],
                        coverage=coverage,
                        counters={"n_tool_calls": raw.get("n_tools", len(names))},
                        metadata={"task": sample.input, "target": sample.target,
                                  "source_report": raw.get("_source_report") or {}})


def _shared_result(tool_calls: list[Any]) -> bool:
    """Two or more calls carrying the identical result: a copied step blob."""
    results = [str(tc.get("result")) for tc in tool_calls
               if isinstance(tc, dict) and "result" in tc]
    return len(results) > 1 and len(set(results)) < len(results)


def _episode_trace(raw: dict[str, Any], sample: Any) -> AgentEpisode:
    """TraceLogger ``trace.jsonl`` row or ``TrajectoryStore.export_jsonl`` row.

    Checked against real exports at AgentTune 36d4724. Calls carry no name the
    scorers recognize: TraceLogger calls are ``{query, result}``; store-export
    calls add ``tool_name``/``tool_call_id``/``step_number``, which survive in
    the event payload but are not read as a call name, so ``turns()`` stays
    empty. ``result`` is the whole step observation (``str(dict(tool_results))``,
    keyed by tool name), copied onto every call of that step, so a result is
    never a verified per-call pairing. A store export's ``query`` is the Python
    repr of the args dict (a string), not structured arguments.
    """
    tool_calls = list(raw.get("tool_calls") or [])
    # Store exports give each call its step_number: parallel calls of one step
    # share it and become one turn. TraceLogger rows have none -> one turn per call.
    grouped = bool(tool_calls) and all(isinstance(tc, dict) and "step_number" in tc
                                       for tc in tool_calls)
    turn_ids = [tc["step_number"] if grouped else i for i, tc in enumerate(tool_calls)]
    events: list[AgentEvent] = []
    idx = 0
    for tc, turn in zip(tool_calls, turn_ids):
        events.append(AgentEvent(index=idx, role="assistant", type="tool_call",
                                 payload=tc if isinstance(tc, dict) else {"raw": tc},
                                 turn_id=turn, source="agenttune_trace"))
        idx += 1
        if isinstance(tc, dict) and "result" in tc:
            events.append(AgentEvent(index=idx, role="tool", type="tool_result",
                                     payload={"output": tc.get("result")}, turn_id=turn,
                                     source="agenttune_trace"))
            idx += 1
    retrieved = raw.get("retrieved_chunk_ids")
    if retrieved:
        events.append(_retrieved_event(idx, retrieved, "agenttune_trace"))
    has_results = any(isinstance(tc, dict) and "result" in tc for tc in tool_calls)
    queries = [tc.get("query") for tc in tool_calls if isinstance(tc, dict)]
    if not tool_calls:
        arguments = UNAVAILABLE
    elif len(queries) == len(tool_calls) and all(isinstance(q, dict) for q in queries):
        arguments = OBSERVED          # TraceLogger: json-parsed args dict
    else:
        arguments = UNAVAILABLE       # store export: Python-repr string, not args
    if not has_results:
        pairing = UNAVAILABLE
    elif len(set(turn_ids)) < len(turn_ids) or _shared_result(tool_calls):
        pairing = UNAVAILABLE         # one step blob duplicated across parallel calls
    else:
        pairing = INFERRED            # step observation, assumed to be this call's
    coverage = _coverage(
        final_answer=sample.actual_output is not None,
        tool_call_names=False,   # tool_name (store export) is payload-only, not a call name
        tool_call_arguments=arguments,
        tool_results=has_results,
        call_result_pairing=pairing,
        parallel_grouping=grouped, timestamps=False,
        retrieved_contexts=bool(retrieved), final_state=False,
    )
    return AgentEpisode(events=events, mode="recorded", source_format="agenttune_trace",
                        source_id=None if raw.get("trajectory_id") is None else str(raw["trajectory_id"]),
                        final_answer=sample.actual_output,
                        source_reward=raw.get("reward"),
                        source_scores=raw.get("reward_components") if isinstance(raw.get("reward_components"), dict) else None,
                        coverage=coverage,
                        # the record's n_tool_calls is rollout metadata (0 on the flat
                        # path even with a call), so count the calls actually listed
                        counters={"n_tool_calls": len(tool_calls)},
                        metadata={"task": sample.input, "target": sample.target,
                                  "reference_contexts": sample.reference_contexts})


def _episode_trajectory(raw: dict[str, Any], sample: Any) -> AgentEpisode:
    steps = raw.get("steps") or []
    tmeta = raw.get("metadata") or {}
    conversation = tmeta.get("conversation")
    events: list[AgentEvent] = []
    idx = 0
    any_args = False
    any_action = False    # a step recorded its action, call or not
    any_unnamed = False   # some action holds a call without a tool name
    for si, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        if "action" in step:
            any_action = True
            any_unnamed = any_unnamed or _has_unnamed_call(step.get("action"))
        turn = step.get("step_number", si)
        if step.get("thought"):
            events.append(AgentEvent(index=idx, role="assistant", type="reasoning",
                                     payload={"text": step["thought"]}, turn_id=turn,
                                     source="agenttune_trajectory"))
            idx += 1
        for c in _action_calls(step.get("action")):
            fn = c.get("function") if isinstance(c.get("function"), dict) else None
            # arguments observed when the source recorded the key -- even {} is a
            # recorded empty-args call, not a missing one (advisor note).
            if "arguments" in (fn or c) or "parameters" in (fn or c):
                any_args = True
            events.append(AgentEvent(index=idx, role="assistant", type="tool_call",
                                     payload=c, call_id=c.get("id"), turn_id=turn,
                                     source="agenttune_trajectory"))
            idx += 1
        if step.get("observation") is not None:
            # Turn-level result blob (str(dict(tool_results))); not per-call.
            events.append(AgentEvent(index=idx, role="tool", type="tool_result",
                                     payload={"output": step["observation"]}, turn_id=turn,
                                     source="agenttune_trajectory"))
            idx += 1
    retrieved = tmeta.get("retrieved_chunk_ids")
    if retrieved:
        events.append(_retrieved_event(idx, retrieved, "agenttune_trajectory"))
    has_calls = any(e.type == "tool_call" for e in events)
    coverage = _coverage(
        final_answer=sample.actual_output is not None,
        # recorded actions show every call (zero calls included), unless one hides its name
        tool_call_names=(has_calls or any_action) and not any_unnamed,
        tool_call_arguments=any_args,
        tool_results=any(e.type == "tool_result" for e in events),
        # per-call pairing needs the conversation (tool_call_id); steps only give
        # a turn-level result blob (AG-03 / advisor note on Step shape).
        call_result_pairing=bool(conversation),
        parallel_grouping=has_calls,
        timestamps=False,
        retrieved_contexts=bool(retrieved), final_state=False,
    )
    return AgentEpisode(events=events, mode="recorded", source_format="agenttune_trajectory",
                        source_tier="full",
                        source_id=None if raw.get("trajectory_id") is None else str(raw["trajectory_id"]),
                        final_answer=sample.actual_output, source_reward=raw.get("reward"),
                        coverage=coverage,
                        counters={"n_tool_calls": tmeta.get("tool_call_count", sum(
                            1 for e in events if e.type == "tool_call")),
                            "n_turns": len(steps)},
                        metadata={"task": sample.input, "target": sample.target,
                                  "reference_contexts": sample.reference_contexts,
                                  "has_conversation": bool(conversation)})


def _episode_run_eval_row(raw: dict[str, Any], sample: Any) -> AgentEpisode:
    # A task to RUN, not a recorded episode: no events, nothing observed yet.
    coverage = _coverage(**{k: False for k in (
        "final_answer", "tool_call_names", "tool_call_arguments", "tool_results",
        "call_result_pairing", "parallel_grouping", "timestamps",
        "retrieved_contexts", "final_state")})
    return AgentEpisode(events=[], mode="recorded", source_format="agenttune_run_eval_row",
                        source_id=None if raw.get("question_id") is None else str(raw["question_id"]),
                        final_answer=None, coverage=coverage,
                        metadata={"task": sample.input, "target": sample.target,
                                  "reference_contexts": sample.reference_contexts,
                                  "is_task": True,
                                  "messages": raw["prompt"] if isinstance(raw.get("prompt"), list) else None})


_BUILDERS = {
    "agenttune_report": _episode_report,
    "agenttune_trace": _episode_trace,
    "agenttune_trajectory": _episode_trajectory,
    "agenttune_run_eval_row": _episode_run_eval_row,
}


def episodes_from_agenttune(path: str) -> list[AgentEpisode]:
    """Every record of an AgentTune ``load_agenttune`` file -> episodes (AG-04).

    Run_eval rows are tasks (no events); trajectory/trace/report records are
    recorded episodes with honest coverage. A JSONL of serialized EventLogs
    (``{"events": [...], "tier", "id"}`` per line) goes through
    :func:`episode_from_eventlog`.

    *path* may also be an AgentTune run folder: every ``*.jsonl`` in it is
    read (in name order), and the folder's ``lexsi_provenance.json`` lands in
    each episode's ``metadata["provenance"]``. ponytail: ``*.jsonl`` only, so
    configs/checkpoints in the folder are never misread; pass a run_eval
    ``.json`` report as a file.
    """
    if os.path.isdir(path):
        prov = read_provenance(path)
        episodes = [e for name in sorted(os.listdir(path)) if name.endswith(".jsonl")
                    for e in episodes_from_agenttune(os.path.join(path, name))]
        if prov is not None:
            for e in episodes:
                e.metadata["provenance"] = prov
        return episodes
    logs = _eventlog_records(path)
    if logs is not None:
        return [episode_from_eventlog(r) for r in logs]
    return [_BUILDERS[fmt](raw, sample) for fmt, raw, sample in _load_agenttune_pairs(path)]


# --------------------------------------------------------------------------
# (a2) AgentTune EventLog (object or its JSON), duck-typed
# --------------------------------------------------------------------------
def _eventlog_records(path: str) -> Optional[list[dict[str, Any]]]:
    """The records of a JSONL/JSON file of serialized EventLogs, else ``None``."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    try:
        doc = json.loads(text)
        rows = doc if isinstance(doc, list) else [doc]
    except ValueError:
        try:
            rows = [json.loads(line) for line in text.split("\n") if line.strip()]
        except ValueError:
            return None
    if rows and all(isinstance(r, dict) and isinstance(r.get("events"), list) for r in rows):
        return rows
    return None


def _get(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _kind(k: Any) -> str:
    """``EventKind.TOOL_CALL`` (enum), ``"tool_call"`` or ``"EventKind.TOOL_CALL"``."""
    return str(getattr(k, "value", k)).split(".")[-1].lower()


def episode_from_eventlog(log: Any, *, source_id: Optional[str] = None) -> AgentEpisode:
    """An AgentTune ``EventLog`` (or its JSON: ``{"events", "tier", "id"}``) -> episode.

    Duck-typed: AgentTune is never imported. Reads ``events`` (each with
    ``kind``, ``payload``, optional ``token_span``/``logprobs``/``scope``),
    ``tier`` and ``id``. ``TURN_COMPLETE`` closes a turn, so calls inside one
    step form one parallel group; a log without turn boundaries (light tier,
    ``from_eval_dict``) gives each call its own turn and marks grouping and
    pairing unavailable. A ``TOOL_CALL`` whose action holds no call (the
    terminal step's ``{}``, a DECIDE ``{"stage": id}``) is kept as an event but
    never becomes a call. The episode-scope ``REWARD`` is ``source_reward``
    (provenance only). ``id`` is a uuid4 minted by ``EventLog.from_*``, so it
    joins no other AgentTune artifact.
    """
    from ..loaders import _agenttune_answer

    tier = _get(log, "tier")
    raw_events = _get(log, "events")
    if not isinstance(raw_events, list):
        raise ValueError("not an EventLog: expected an 'events' list")
    events: list[AgentEvent] = []
    turn, n_turns = 0, 0
    has_boundaries = any(_kind(_get(e, "kind")) == "turn_complete" for e in raw_events)
    calls_per_turn: dict[int, int] = {}
    final_text: Optional[str] = None
    reward: Optional[float] = None
    any_args = False
    any_action = False   # a TOOL_CALL records the step's action, call or not
    any_unnamed = False  # ... but an action holding a call without a name hides the names

    def add(role: str, typ: str, payload: dict[str, Any], **kw: Any) -> None:
        events.append(AgentEvent(index=len(events), role=role, type=typ, payload=payload,
                                 source="agenttune_eventlog", tier=tier, **kw))

    for ev in raw_events:
        kind = _kind(_get(ev, "kind"))
        payload = dict(_get(ev, "payload") or {})
        for extra in ("token_span", "logprobs", "scope"):
            val = _get(ev, extra)
            if val is not None:
                payload[extra] = list(val) if isinstance(val, tuple) else val
        tid = turn if has_boundaries else None
        if kind == "tool_call":
            any_action = True
            any_unnamed = any_unnamed or _has_unnamed_call(payload.get("action"))
            calls = _action_calls(payload.get("action"))
            if not calls:
                add("assistant", "tool_call", payload, turn_id=tid)  # no call: never scored
            for c in calls:
                fn = c.get("function") if isinstance(c.get("function"), dict) else None
                any_args = any_args or "arguments" in (fn or c) or "parameters" in (fn or c)
                calls_per_turn[turn] = calls_per_turn.get(turn, 0) + 1
                add("assistant", "tool_call", c, call_id=c.get("id", c.get("tool_call_id")), turn_id=tid)
        elif kind == "tool_result":
            add("tool", "tool_result", payload, turn_id=tid,
                call_id=payload.get("tool_call_id", payload.get("call_id")))
        elif kind in ("text", "reasoning"):
            add("assistant", kind, payload, turn_id=tid)
            if kind == "text":
                final_text = payload.get("text")
        elif kind == "reward":
            add("environment", "reward", payload, turn_id=tid)
            if payload.get("scope") == "episode":
                reward = payload.get("value")
        elif kind == "observation":
            add("user", "observation", payload, turn_id=tid)
        else:  # turn_complete, memory_op, anything newer: kept, in order
            add("environment", kind, payload, turn_id=tid)
        if kind == "turn_complete":
            turn += 1
            n_turns += 1
    n_calls = sum(calls_per_turn.values())
    has_results = any(e.type == "tool_result" for e in events)
    answer = None if final_text is None else _agenttune_answer(str(final_text))
    coverage = _coverage(
        final_answer=final_text is not None,
        # a recorded action stream shows every call, so zero calls is observed -- unless
        # some action holds a call whose name was not recorded
        tool_call_names=any_action and not any_unnamed,
        tool_call_arguments=any_args,
        tool_results=has_results,
        # every result naming a recorded call id (Cohere/OpenAI tool_call_id) is an
        # observed pairing; a result with no call id is the step's observation blob:
        # one call per bounded turn is an inferred pairing, anything else is unknown
        call_result_pairing=(OBSERVED if _results_paired(events) else
                             INFERRED if has_results and has_boundaries and n_calls
                             and max(calls_per_turn.values()) == 1 else UNAVAILABLE),
        parallel_grouping=has_boundaries and n_calls > 0,
        timestamps=False, retrieved_contexts=False, final_state=False,
    )
    meta: dict[str, Any] = {}
    if final_text is not None and answer != str(final_text).strip():
        meta["raw_output"] = final_text
    lid = _get(log, "id")
    return AgentEpisode(events=events, mode="recorded", source_format="agenttune_eventlog",
                        source_tier=tier,
                        source_id=source_id or (None if lid is None else str(lid)),
                        final_answer=answer, source_reward=reward, coverage=coverage,
                        counters={"n_tool_calls": n_calls, "n_turns": n_turns},
                        metadata=meta)


def _results_paired(events: list[AgentEvent]) -> bool:
    """True when there are results and each names the id of a recorded call."""
    call_ids = {e.call_id for e in events if e.type == "tool_call" and e.call_id is not None}
    results = [e for e in events if e.type == "tool_result"]
    return bool(results) and all(e.call_id is not None and e.call_id in call_ids for e in results)


def inspect_agenttune(path: str) -> dict[str, Any]:
    """Read-only summary for ``agent import-agenttune --inspect`` (UX-A1).

    No model load, no endpoint call. Reports detected format(s), row count,
    per-episode coverage label, and the scorers each trace can feed.
    """
    from collections import Counter

    episodes = episodes_from_agenttune(path)
    fmt_counts = Counter(e.source_format for e in episodes)
    label_counts = Counter(e.coverage_label() for e in episodes)
    supported = sorted({s for e in episodes
                        for s in supported_scorers_from_coverage(e.coverage)})
    return {
        "path": path,
        "row_count": len(episodes),
        "formats": dict(fmt_counts),
        "coverage_labels": dict(label_counts),
        "eligible_metrics": supported,
        "coverage_sample": episodes[0].coverage if episodes else {},
        "n_tasks": sum(1 for e in episodes if e.metadata.get("is_task")),
    }
