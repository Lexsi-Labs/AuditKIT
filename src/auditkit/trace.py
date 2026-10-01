"""Agent traces: tool calls grouped into turns, and how to match them.

An agent run is scored from its *turns*: each assistant turn is the list of
tool calls it emitted at once. A turn with two or more calls is a parallel
group. Keeping that grouping (instead of flattening every call into one list,
as most eval libraries do) is what makes parallel tool calling measurable:
independent calls spread over several turns, and dependent calls batched into
one turn, are both visible.

Everything here is stdlib-only and accepts the shapes real systems emit:

- OpenAI ``tool_calls`` entries: ``{"id", "type": "function", "function":
  {"name", "arguments": "<json string>"}}``
- flat calls: ``{"name", "arguments" | "args" | "parameters": {...}}``, and
  Cohere's ``{"tool_name", "parameters"}``
- BFCL ground truth: ``{"func_name": {"arg": value}}``
- a transcript of OpenAI chat messages (grouped by assistant message)
- free text with Hermes/Qwen ``<tool_call>{...}</tool_call>`` blocks, a JSON
  list of calls, a ```` ```json ```` fenced block, or Cohere's native action
  list (Command R7B ``<|START_ACTION|>[...]<|END_ACTION|>``, the
  same with the markers stripped, Command-R ``Action: ```json [...]```)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

Turns = list[list["ToolCall"]]


@dataclass
class ToolCall:
    """One tool call: a name and its (decoded) arguments."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    # Set when the arguments string could not be decoded as a JSON object; the
    # call then never matches a reference call with arguments.
    parse_error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "arguments": _strict_json(self.arguments)}
        if self.id:
            d["id"] = self.id
        if self.parse_error:
            d["parse_error"] = self.parse_error
        return d


def _strict_json(v: Any) -> Any:
    """Copy of *v* that is strict JSON: NaN/Infinity (which the parser accepts
    from model text, but JSON forbids) become ``None``, so a saved run stays
    readable by strict consumers (JS, jq)."""
    try:
        return json.loads(json.dumps(v, default=str), parse_constant=lambda _: None)
    except (ValueError, TypeError, RecursionError):
        return v


_DECODER = json.JSONDecoder()
# Deepest JSON nesting read from text. Model text like '[' * 1000 makes json
# raise RecursionError, which escapes every ``except ValueError`` and drops the
# sample from the metric; a value just under that limit parses, then crashes
# the recursive json.dumps/copy steps downstream (scoring, saving the run).
# Real tool arguments are nowhere near this deep.
_MAX_DEPTH = 100


def _check_depth(val: Any) -> Any:
    stack = [(val, 1)]
    while stack:
        v, d = stack.pop()
        if isinstance(v, (dict, list)):
            if d > _MAX_DEPTH:
                raise ValueError(f"JSON nested deeper than {_MAX_DEPTH} levels")
            stack.extend((x, d + 1) for x in (v.values() if isinstance(v, dict) else v))
    return val


def _raw_decode(text: str, idx: int = 0) -> tuple[Any, int]:
    """``raw_decode`` that fails with ValueError only, and on nesting deeper
    than ``_MAX_DEPTH`` -- never RecursionError."""
    try:
        val, end = _DECODER.raw_decode(text, idx)
    except RecursionError:
        raise ValueError("JSON nested too deeply") from None
    return _check_depth(val), end


def _loads(text: str) -> Any:
    """``json.loads`` with the same guarantees as :func:`_raw_decode`."""
    try:
        return _check_depth(json.loads(text))
    except RecursionError:
        raise ValueError("JSON nested too deeply") from None


def _decode_arguments(raw: Any) -> tuple[dict[str, Any], Optional[str]]:
    if raw is None:
        return {}, None
    if isinstance(raw, dict):
        return raw, None
    if isinstance(raw, str):
        if not raw.strip():
            return {}, None
        try:
            val = _loads(raw)
        except ValueError as e:
            return {}, f"arguments are not valid JSON: {e}"
        if isinstance(val, dict):
            return val, None
        return {}, f"arguments decode to {type(val).__name__}, not an object"
    return {}, f"arguments have unsupported type {type(raw).__name__}"


def normalize_call(obj: Any) -> ToolCall:
    """Coerce any supported single-call shape to a :class:`ToolCall`."""
    if isinstance(obj, ToolCall):
        return obj
    if not isinstance(obj, dict):
        raise ValueError(f"cannot read a tool call from {obj!r}")
    fn = obj.get("function")
    if isinstance(fn, dict) and "name" in fn:  # OpenAI tool_calls entry
        args, err = _decode_arguments(fn.get("arguments"))
        return ToolCall(name=str(fn["name"]), arguments=args, id=obj.get("id"), parse_error=err)
    if "name" in obj or "tool_name" in obj:  # tool_name: Cohere
        raw = next((obj[k] for k in ("arguments", "args", "parameters", "input") if k in obj), None)
        args, err = _decode_arguments(raw)
        return ToolCall(name=str(obj.get("name", obj.get("tool_name"))), arguments=args,
                        id=obj.get("id", obj.get("tool_call_id")), parse_error=err)
    if len(obj) == 1 and isinstance(next(iter(obj.values())), dict):
        # BFCL ground truth: {"func": {"arg": value}}. A one-key JSON answer
        # like {"answer": "Paris"} is not a call.
        (name, args), = obj.items()
        return ToolCall(name=str(name), arguments=args)
    raise ValueError(f"cannot read a tool call from {obj!r}")


def _is_message(obj: Any) -> bool:
    return isinstance(obj, dict) and "role" in obj


def turns_from_messages(messages: list[dict[str, Any]]) -> Turns:
    """One turn per assistant message that carries ``tool_calls``."""
    turns: Turns = []
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            turns.append([normalize_call(c) for c in m["tool_calls"]])
    return turns


def to_turns(obj: Any) -> Turns:
    """Coerce a reference or recorded set of calls to turns.

    - ``[[call, call], [call]]`` → turns as given
    - ``[call, call]`` (flat) → ONE turn: the calls expected in one response
    - ``[{"role": ...}, ...]`` → grouped by assistant message
    - ``{"tool_calls": ...}`` / ``{"messages": ...}`` → unwrapped
    - ``[]`` → no calls
    """
    if obj is None:
        return []
    if isinstance(obj, str):
        # A JSON-encoded reference (e.g. a CSV cell). Never read as "no calls".
        try:
            obj = _loads(obj)
        except ValueError as e:
            raise ValueError(f"expected tool calls as JSON, got unparseable string: {obj[:200]!r}") from e
        return to_turns(obj)
    if isinstance(obj, dict):
        if _is_message(obj):
            return turns_from_messages([obj])
        if "tool_calls" in obj:
            return to_turns(obj["tool_calls"])
        if "messages" in obj:
            return turns_from_messages(obj["messages"])
        return [[normalize_call(obj)]]
    if not isinstance(obj, list) or not obj:
        return []
    if all(isinstance(t, list) for t in obj):
        return [[normalize_call(c) for c in t] for t in obj if t]
    if any(_is_message(m) for m in obj):
        return turns_from_messages(obj)
    return [[normalize_call(c) for c in obj]]


def decode_reference(obj: Any) -> Any:
    """A JSON-encoded reference (a CSV cell, a JSONL field) decoded, for hashing (#44).

    So ``'[[{"name": ...}]]'`` and the same list fingerprint alike. Only strings are
    touched, so dict and list references keep their digests byte for byte, and a
    string that doesn't parse is returned as is: a digest never raises, and scoring
    reports the bad reference itself.
    """
    if not isinstance(obj, str):
        return obj
    try:
        return _loads(obj)
    except ValueError:
        return obj


_PATHS_KEY = "any_of"


def to_reference_paths(obj: Any) -> list[Turns]:
    """Coerce a reference to a LIST OF ROUTES, each route a list of turns.

    A reference normally names one route, in any shape :func:`to_turns` reads, and
    comes back as a one-element list. An agent can reach the same correct answer
    through different tool routes, though, and a single-route reference scores a
    valid alternative as wrong on every reference metric. ``{"any_of": [...]}``
    names the alternatives:

        {"any_of": [
            [[{"name": "get_weather", "arguments": {"city": "Paris"}},
              {"name": "get_weather", "arguments": {"city": "Rome"}}]],
            [[{"name": "compare_weather", "arguments": {"cities": ["Paris", "Rome"]}}]],
        ]}

    Each route is itself a reference, so ``[]`` is a route meaning "calling no tool
    is also correct". Routes are never expanded or combined; picking one is the
    caller's job (see :func:`auditkit.metrics.agent.select_reference_path`).

    Backward compatible by construction: ``to_turns`` raises on ``{"any_of": ...}``
    today (its value is a list, not a call), so no valid single-route reference can
    be read as a multi-route one.

    A JSON-encoded reference (a CSV cell, a JSONL field) is decoded first, so
    ``any_of`` reaches a dataset the same way every other reference shape does.
    """
    if isinstance(obj, str):
        try:
            obj = _loads(obj)
        except ValueError as e:
            raise ValueError(f"expected tool calls as JSON, got unparseable string: {obj[:200]!r}") from e
    if not (isinstance(obj, dict) and _PATHS_KEY in obj):
        return [to_turns(obj)]
    if len(obj) != 1:
        # A stray key beside any_of would be silently ignored, and a silently
        # ignored key in a reference is a silently wrong score.
        raise ValueError(
            f"a '{_PATHS_KEY}' reference must carry no other keys, got {sorted(obj)!r}. "
            f"Put per-route notes on the route, not beside '{_PATHS_KEY}'.")
    routes = obj[_PATHS_KEY]
    if not isinstance(routes, list) or not routes:
        raise ValueError(
            f"'{_PATHS_KEY}' must be a non-empty list of routes, got {routes!r}")
    out: list[Turns] = []
    for i, route in enumerate(routes):
        if isinstance(route, dict) and _PATHS_KEY in route:
            raise ValueError(
                f"route {i} nests '{_PATHS_KEY}'; routes are alternatives, not a tree. "
                f"Flatten them into one list.")
        try:
            out.append(to_turns(route))
        except ValueError as e:
            raise ValueError(f"route {i} of '{_PATHS_KEY}' is not a valid reference: {e}") from e
    return out


# ``` fences hold at most one JSON block; no \s* around the lazy group so a
# missing close fence can't backtrack cubically.
_FENCE_RE = re.compile(r"```(?:json)?(.*?)```", re.DOTALL)
# The tags the <tool_call> scanner stops at. A stray </tool_call> is ignored.
_TAG_RE = re.compile(r"</?think>|<tool_call>")
_WS_RE = re.compile(r"\s*")
_OPEN, _CLOSE = "<tool_call>", "</tool_call>"

# Keys a JSON object may carry and still be "a tool call". A structured answer
# like {"name": "Paris", "population": 2_161_000} has "name" but other keys too,
# so it is NOT a call; {"name": "get_time"} or {"name": "f", "args": {...}} is.
_CALL_KEYS = {"name", "arguments", "args", "parameters", "input", "id", "type", "function",
              "tool_name", "tool_call_id"}


def is_call_payload(p: Any) -> bool:
    """A recorded call dict: ``name``, ``function``, or Cohere's ``{"tool_name",
    "parameters"}``. Anything else is kept as an event only -- including an
    AgentTune store-export row, whose ``tool_name`` comes with a Python-repr
    ``query`` string, not arguments."""
    if not isinstance(p, dict):
        return False
    if "name" in p or "function" in p:
        return True
    # Cohere: {"tool_name", "parameters"}, or a parameterless {"tool_name"} call -- but only
    # with call keys, so a store-export row (tool_name + a repr'd "query") stays an event.
    return "tool_name" in p and ("parameters" in p or p.keys() <= _CALL_KEYS)


def _is_call_shape(v: Any) -> bool:
    """True only for a dict shaped like a tool call, not any object with a name."""
    if not isinstance(v, dict):
        return False
    if ("name" in v or "tool_name" in v or isinstance(v.get("function"), dict)) and v.keys() <= _CALL_KEYS:
        return True
    # single-key BFCL ground truth: {"func": {"arg": value}}
    return len(v) == 1 and isinstance(next(iter(v.values())), dict)


def _calls_from(val: Any, strict: bool = False) -> Optional[list[ToolCall]]:
    if strict:
        # Free-text JSON: only read call-shaped objects, so a plain JSON answer
        # ({"name": "Paris", ...}) is not mistaken for a call. Inside <tool_call>
        # tags and the {"tool_calls": ...} wrapper the reading stays lenient.
        if isinstance(val, dict) and "tool_calls" not in val and not _is_call_shape(val):
            return None
        if isinstance(val, list) and not all(_is_call_shape(v) for v in val):
            return None
    try:
        if isinstance(val, list):
            return [normalize_call(v) for v in val]
        if isinstance(val, dict):
            if "tool_calls" in val:
                return [c for t in to_turns(val["tool_calls"]) for c in t]
            return [normalize_call(val)]
    except ValueError:
        return None
    return None


def _loads_calls(text: str, strict: bool = False) -> Optional[list[ToolCall]]:
    try:
        val = _loads(text)
    except ValueError:
        return None
    return _calls_from(val, strict)


def _concatenated_calls(text: str) -> Optional[list[ToolCall]]:
    """Consecutive bare JSON call objects, e.g. ``{..}\\n{..}`` or ``{..}{..}``.

    A model without a tool-call parser (confirmed live: Qwen2.5-Coder) emits
    parallel calls as several bare JSON objects back to back, which a single
    ``json.loads`` rejects as extra data. Walk them with ``raw_decode`` and keep
    them only when every decoded value is call-shaped, so ordinary prose that
    merely starts with ``{`` is not misread. Returns ``None`` when it is not a
    clean run of two or more call objects."""
    idx, n = 0, len(text)
    vals: list[Any] = []
    while idx < n:
        while idx < n and text[idx] in " \t\r\n,":
            idx += 1
        if idx >= n:
            break
        try:
            val, end = _raw_decode(text, idx)
        except ValueError:
            return None
        vals.append(val)
        idx = end
    if len(vals) < 2 or not all(_is_call_shape(v) for v in vals):
        return None
    try:
        return [normalize_call(v) for v in vals]
    except ValueError:
        return None


def strip_thinking(text: str) -> str:
    """Drop ``<think>...</think>`` reasoning from a model reply.

    Removes complete think blocks; a reasoning model whose template opens
    ``<think>`` in the prompt emits only the closing tag, so anything up to and
    including the last ``</think>`` is dropped too (draft calls live there).
    """
    text = re.sub(r"<think>(?:(?!<think>).)*?</think>", "", text, flags=re.DOTALL)
    return text.rsplit("</think>", 1)[-1]


def _tagged_calls(text: str) -> list[ToolCall]:
    """Every Hermes/Qwen ``<tool_call>`` block outside reasoning, in order.

    JSON-aware: the JSON after an open tag is decoded in place, so a tag literal
    inside an argument string (``"see </think>"``, ``"use <tool_call>"``) is
    just text and never ends, opens or strips anything. A closed block that is
    not a call is kept as an error call (it was clearly meant to be one); an
    unclosed block (real, e.g. command-r7b) runs to the next open tag or the
    end and is kept only when it parses, so prose that mentions the tag stays
    ``[]``. Reasoning is dropped as :func:`strip_thinking` does: complete think
    blocks, and everything before a stray ``</think>``. Linear time: each
    search resumes where the last stopped, and the next-close position is
    cached, so thousands of unclosed tags are not quadratic.
    """
    found: list[ToolCall] = []
    pending: list[ToolCall] = []  # calls inside a <think> not closed yet
    thinking = False
    close = -2  # cached index of the next </tool_call> at/after pos; -1 = none left
    pos = 0
    while m := _TAG_RE.search(text, pos):
        pos = m.end()
        if m.group() == "<think>":
            found += pending  # a think block runs from its LAST <think>
            pending, thinking = [], True
            continue
        if m.group() == "</think>":
            if not thinking:  # template opened <think> in the prompt: all before is reasoning
                found = []
            pending, thinking = [], False
            continue
        out = pending if thinking else found
        start = _WS_RE.match(text, pos).end()
        if text[start:start + 1] in ("{", "["):
            try:
                val, end = _raw_decode(text, start)
            except ValueError:
                pass
            else:
                end = _WS_RE.match(text, end).end()
                closed = text.startswith(_CLOSE, end)
                if closed or end == len(text) or text.startswith(_OPEN, end):
                    calls = _calls_from(val)
                    if calls is not None:
                        out.extend(calls)
                    elif closed:
                        out.append(ToolCall(name="", parse_error=f"unparseable <tool_call> block: {text[start:end].strip()[:200]!r}"))
                    pos = end + len(_CLOSE) if closed else end
                    continue
        if close == -2 or -1 < close < pos:
            close = text.find(_CLOSE, pos)
        if close != -1 and text.find(_OPEN, pos, close) == -1:
            # closed before any other open tag, but not a JSON call
            block = text[pos:close].strip()
            out.append(ToolCall(name="", parse_error=f"unparseable <tool_call> block: {block[:200]!r}"))
            pos = close + len(_CLOSE)
        elif close == -1 and text.find(_OPEN, pos) == -1 and _CALL_START_RE.match(text, start):
            # The last block, unclosed, that begins like a call but doesn't parse: a call
            # cut off at max_tokens (what SGLang's parser hands back as raw content when it
            # can't complete the call). Reported as truncated, as a cut Cohere action list
            # is, never read as "no call". Prose that only mentions the tag is untouched.
            name = _CALL_NAME_RE.search(text, start)
            out.append(ToolCall(name=name.group(1) if name else "", parse_error="truncated"))
            pos = len(text)
    return found + pending


_CALL_START_RE = re.compile(r'\{\s*"(?:name|function|tool_name)"\s*:')
_CALL_NAME_RE = re.compile(r'"(?:name|tool_name)"\s*:\s*"([^"\\]*)"')

_COHERE_THINKING_RE = re.compile(r"<\|START_THINKING\|>.*?<\|END_THINKING\|>", re.DOTALL)


def _cohere_calls(text: str) -> Optional[list[ToolCall]]:
    """The first JSON list of ``{"tool_name", "parameters"}`` objects.

    Covers every surface form of Cohere's native tool calls: Command R7B
    ``<|START_ACTION|>[...]<|END_ACTION|>``, the same text decoded
    with ``skip_special_tokens=True`` (markers gone, only ``plan[...]`` left),
    and Command-R / Aya Expanse ``Action: ```json [...]```. The plan inside
    ``<|START_THINKING|>`` is dropped first so a bracket there is not read.
    ponytail: tries raw_decode at every ``[`` (quadratic on bracket-heavy
    text); gated on ``tool_name`` appearing at all, so plain replies never pay.
    """
    if "tool_name" not in text:
        return None
    text = _COHERE_THINKING_RE.sub("", text)
    for m in re.finditer(r"\[", text):
        try:
            val, _ = _raw_decode(text, m.start())
        except ValueError:
            continue
        if val and isinstance(val, list) and all(isinstance(c, dict) and "tool_name" in c for c in val):
            # Command-R answers without a tool through the pseudo-tool "directly-answer":
            # that is an answer, not a call.
            return [normalize_call(c) for c in val if str(c.get("tool_name")) not in _DIRECT_ANSWER]
    return _truncated_cohere_calls(text)


_DIRECT_ANSWER = frozenset({"directly-answer", "directly_answer"})


def _truncated_cohere_calls(text: str) -> Optional[list[ToolCall]]:
    """An action list cut off (e.g. by max_tokens): its complete leading calls, plus one
    call flagged ``parse_error="truncated"`` -- the same as the Hermes path, never a
    silent "no call". Only when an action list clearly started."""
    start = text.find("<|START_ACTION|>")
    if start < 0:
        found = re.search(r"Action:\s*```(?:json)?\s*", text)
        if not found:
            return None
        start = found.end()
    bracket = text.find("[", start)
    if bracket < 0 or "tool_name" not in text[bracket:]:
        return None
    calls: list[ToolCall] = []
    idx = bracket + 1
    while True:
        m = re.compile(r"\s*,?\s*").match(text, idx)
        idx = m.end() if m else idx
        if idx >= len(text) or text[idx] in "]<`":
            break
        try:
            val, idx = _raw_decode(text, idx)
        except ValueError:
            name = re.search(r'"tool_name"\s*:\s*"([^"]+)"', text[idx:])
            calls.append(ToolCall(name=name.group(1) if name else "", arguments={},
                                  parse_error="truncated"))
            break
        if isinstance(val, dict) and "tool_name" in val and str(val["tool_name"]) not in _DIRECT_ANSWER:
            calls.append(normalize_call(val))
    return calls or None


def _answers_directly(call: ToolCall) -> bool:
    """Command-R's ``directly-answer`` pseudo-tool: an answer, never a call."""
    return call.name in _DIRECT_ANSWER


def parse_tool_calls(text: str) -> list[ToolCall]:
    """Tool calls written as text in one response (all one turn).

    Tries, in order: the whole reply as JSON, every Hermes/Qwen
    ``<tool_call>`` block (several blocks in one reply are parallel calls, not
    an error), a Cohere action list (see :func:`_cohere_calls`), the reply as
    JSON once reasoning is stripped, then a fenced JSON block. Returns ``[]``
    when the reply holds no calls -- a plain-text answer. Command-R's
    ``directly-answer`` pseudo-tool is dropped on every path.
    """
    return [c for c in _parse_tool_calls(text) if not _answers_directly(c)]


def _parse_tool_calls(text: str) -> list[ToolCall]:
    if not text:
        return []

    def whole_json(body: str) -> Optional[list[ToolCall]]:
        if body[:1] not in ("{", "["):
            return None
        parsed = _loads_calls(body, strict=True)
        # Several bare JSON call objects back to back (a model with no
        # tool-call parser emitting parallel calls).
        return parsed if parsed is not None else _concatenated_calls(body)

    # A reply that is JSON as a whole holds any tag only inside a string, so it
    # is read before the tag scan and before think-stripping, either of which
    # would mangle an argument like "see </think>" or "use <tool_call>".
    parsed = whole_json(text.strip())
    if parsed is not None:
        return parsed
    calls = _tagged_calls(text)
    if calls:
        return calls
    cohere = _cohere_calls(text)
    if cohere is not None:
        # a Cohere action list was found: it is the answer, even when it held only
        # the "directly-answer" pseudo-tool (then: no calls)
        return cohere
    thought = strip_thinking(text)
    parsed = whole_json(thought.strip())
    if parsed is not None:
        return parsed
    for block in _FENCE_RE.findall(thought):
        parsed = _loads_calls(block.strip(), strict=True)
        if parsed is not None:
            return parsed
    return []


_CALL_MARKUP_RES = (
    re.compile(r"<tool_call>.*?(?:</tool_call>|\Z)", re.DOTALL),                # Hermes / Qwen
    re.compile(r"<\|START_ACTION\|>.*?(?:<\|END_ACTION\|>|\Z)", re.DOTALL),     # Command R7B
    re.compile(r"Action:\s*```.*?(?:```|\Z)", re.DOTALL),                        # Command-R / Aya
)


def strip_tool_calls(text: str) -> str:
    """``text`` without the tool-call markup :func:`parse_tool_calls` reads.

    For the assistant ``content`` stored next to calls parsed out of it: a chat
    template renders both ``content`` and ``tool_calls``, so markup left in the
    content shows every call twice. A fenced block goes only when it holds calls.
    Bare JSON calls have no delimiters, but :func:`parse_tool_calls` reads them
    only when the WHOLE reply (after any ``<think>`` block) is JSON. So when calls
    still parse from what is left, there is no prose to keep and ``""`` is
    returned; prose next to a JSON object is never read as a call and is kept.
    """
    for rx in _CALL_MARKUP_RES:
        text = rx.sub("", text)
    text = _FENCE_RE.sub(lambda m: "" if _loads_calls(m.group(1).strip(), strict=True) is not None
                         else m.group(0), text)
    return "" if parse_tool_calls(text) else text.strip()


def get_trace(context: Any) -> dict[str, Any]:
    """The structured trace the Runner put in the scoring context, or ``{}``."""
    if isinstance(context, dict) and isinstance(context.get("trace"), dict):
        return context["trace"]
    return {}


def predicted_turns(output: str, context: Any) -> Turns:
    """The turns a model/agent actually produced for one sample.

    A structured trace wins (native ``tool_calls`` from an API, a transcript
    from an agent endpoint, or ``Sample.actual_trace``); otherwise the text
    output is parsed as one turn.
    """
    trace = get_trace(context)
    if trace.get("tool_calls") is not None:
        turns = to_turns(trace["tool_calls"])
    elif trace.get("messages"):
        turns = turns_from_messages(trace["messages"])
    else:
        turns = [parse_tool_calls(output)]
    # a structured directly-answer (e.g. a recorded Command-R action) is no call either
    turns = [[c for c in t if not _answers_directly(c)] for t in turns]
    return [t for t in turns if t]


# -- matching ---------------------------------------------------------------

ArgMatcher = Callable[[str, dict, dict], bool]

ARG_MODES = ("exact", "subset", "name")


def args_match(pred: dict, ref: dict, mode: str = "exact") -> bool:
    """Compare argument dicts.

    ``exact``: equal as JSON values (``1 == 1.0``, ``True != 1``). ``subset``: every reference argument is
    present with an equal value; extra predicted arguments (e.g. optional
    defaults) are allowed. ``name``: arguments ignored.
    """
    if mode == "name":
        return True
    if mode == "exact":
        return _json_eq(pred, ref)
    if mode == "subset":
        return all(k in pred and _json_eq(pred[k], v) for k, v in ref.items())
    raise ValueError(f"unknown arg mode {mode!r}; expected one of {ARG_MODES}")


def _json_eq(a: Any, b: Any) -> bool:
    """JSON value equality: ``1 == 1.0``, but ``True != 1`` (Python says equal).
    ``NaN`` equals ``NaN``: the parser accepts it from model text, and a call
    must equal itself (``RedundantToolCalls`` already dedupes it that way).

    Iterative (explicit stack) so nesting depth no longer maps to Python's call
    stack: ``json.loads`` accepts far deeper nesting than a recursive compare
    could survive, so a parseable trace must stay comparable.
    """
    stack = [(a, b)]
    while stack:
        a, b = stack.pop()
        if isinstance(a, bool) or isinstance(b, bool):
            if type(a) is not type(b) or a != b:
                return False
        elif isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                return False
            for k in a:
                stack.append((a[k], b[k]))
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                return False
            stack.extend(zip(a, b))
        elif a != b and not (a != a and b != b):  # x != x only for NaN
            return False
    return True


def call_matches(pred: ToolCall, ref: ToolCall, mode: str = "exact",
                 arg_match: Optional[ArgMatcher] = None) -> bool:
    if pred.name != ref.name or pred.parse_error:
        return False
    if arg_match is not None:
        return bool(arg_match(ref.name, pred.arguments, ref.arguments))
    return args_match(pred.arguments, ref.arguments, mode)


def _augment(adj: list[list[int]], owner: dict[int, int], root: int) -> bool:
    """One Kuhn augmenting-path search from left node ``root`` (``adj[i]``
    lists the right nodes left node ``i`` may take), updating ``owner`` (right
    -> left) in place; returns whether ``root`` got matched.

    Iterative (explicit DFS stack) instead of recursive: the alternating path
    can be as long as the number of reference calls, so a recursive form
    overflows the stack for large references (~1000+ same-named calls).
    """
    seen: set[int] = set()
    stack = [(root, iter(adj[root]))]
    chosen: list[int] = []  # the pred each level recursed through
    while stack:
        i, it = stack[-1]
        for j in it:
            if j in seen:
                continue
            seen.add(j)
            if j not in owner:  # free pred: claim it and reclaim along the path
                owner[j] = i
                for k in range(len(stack) - 1, 0, -1):
                    owner[chosen[k - 1]] = stack[k - 1][0]
                return True
            chosen.append(j)  # recurse into owner[j]
            stack.append((owner[j], iter(adj[owner[j]])))
            break
        else:  # adjacency exhausted: backtrack
            stack.pop()
            if chosen:
                chosen.pop()
    return False


def max_matching(preds: list[ToolCall], refs: list[ToolCall], mode: str = "exact",
                 arg_match: Optional[ArgMatcher] = None) -> dict[int, int]:
    """Maximum one-to-one matching ``{ref_index: pred_index}``.

    Kuhn's augmenting-path algorithm, not greedy first-fit: greedy matching
    (used by BFCL, agentevals and deepeval) rejects correct answers when one
    predicted call is compatible with several reference calls.

    A greedy first pass claims a free compatible pred for each ref, and Kuhn
    then augments only the refs still unmatched (still maximum from any start).
    A ref whose adjacency already failed to augment is skipped: an augment that
    fails once fails forever, and identical refs have identical adjacency. Plain
    Kuhn chained through every earlier owner and was cubic on many identical
    (or, under ``"name"``, same-named) calls.
    ponytail: the adjacency build is O(preds * refs) calls and memory (~1s at
    2000 x 2000); collapse identical calls into counted classes if a real trace
    ever needs 10k+ mutually compatible calls.
    """
    adj = [[j for j, p in enumerate(preds) if call_matches(p, r, mode, arg_match)]
           for r in refs]
    owner: dict[int, int] = {}  # pred_index -> ref_index
    unmatched = []
    for i, js in enumerate(adj):
        j = next((j for j in js if j not in owner), None)
        if j is None:
            unmatched.append(i)
        else:
            owner[j] = i
    failed: set[tuple[int, ...]] = set()
    for i in unmatched:
        key = tuple(adj[i])
        if key not in failed and not _augment(adj, owner, i):
            failed.add(key)
    return {r: p for p, r in owner.items()}


def flatten(turns: Turns) -> list[ToolCall]:
    return [c for t in turns for c in t]


def render_trace(turns: Turns, trace: Optional[dict] = None, limit: int = 6000) -> str:
    """Readable transcript for an LLM judge (tool results included when known)."""
    messages = (trace or {}).get("messages")
    if messages:
        lines = []
        for m in messages:
            role = m.get("role", "?")
            content = m.get("content") or ""
            if isinstance(content, list):
                content = " ".join(str(p.get("text", p)) if isinstance(p, dict) else str(p) for p in content)
            if m.get("tool_calls"):
                calls = ", ".join(json.dumps(normalize_call(c).to_dict(), default=str) for c in m["tool_calls"])
                lines.append(f"[{role}] tool_calls: {calls}" + (f" | {content}" if content else ""))
            else:
                lines.append(f"[{role}] {content}")
        # Same marker the turns branch emits, so the judge sees "no action taken"
        # regardless of whether the trace was messages or turns.
        if not any(m.get("tool_calls") for m in messages):
            lines.append("(no tool calls)")
        text = "\n".join(lines)
    else:
        text = "\n".join(
            f"[turn {i + 1}] " + ", ".join(json.dumps(c.to_dict(), default=str) for c in t)
            for i, t in enumerate(turns)
        ) or "(no tool calls)"
    # Keep the head AND the tail: the task is at the top, the final actions (which
    # decide whether it was completed) are at the bottom -- a head-only cut hides them.
    if len(text) <= limit:
        return text
    return text[: limit // 2] + "\n...(truncated)...\n" + text[-(limit // 2):]
