"""Agent and tool-use metrics.

Reference calls come from ``Sample.expected_tool_calls`` (a list of turns);
the calls the model/agent actually made come from the scoring context's
``trace`` (native ``tool_calls`` from an API, an agent endpoint's transcript,
or ``Sample.actual_trace``), falling back to parsing tool calls out of the text
output. See :mod:`auditkit.trace` for the accepted shapes.

A reference may also name several equally correct routes
(``{"any_of": [...]}``, see :func:`auditkit.trace.to_reference_paths`): the run
is scored against the route it actually took, and the three reference metrics
share that choice, so their scores describe one run. Single-route references
are unaffected, metadata included (:func:`select_reference_path`).

All matching is order-invariant *within* a turn (parallel calls have no order)
and uses a maximum bipartite matching, so a correct answer is never rejected
because of greedy pairing. A metric that has nothing to measure on a sample
(e.g. no parallel groups) returns no score rather than a fake 0.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
from typing import Any, Callable, Optional

from auditkit.metric import Metric
from auditkit.metrics.judge import LLMJudge, _JUDGE_SYSTEM_PROMPT
from auditkit.registry import METRICS
from auditkit.sample import Sample
from auditkit.score import Score
from auditkit.trace import (
    ARG_MODES, ArgMatcher, ToolCall, Turns, _augment, args_match, call_matches, flatten,
    get_trace, max_matching, predicted_turns, render_trace, to_reference_paths,
)
from auditkit.types import Direction, ScoreKind

logger = logging.getLogger(__name__)


def _suffix(arg_mode: str) -> str:
    return "" if arg_mode == "exact" else f"_{arg_mode}"


def addressable(sample: Sample, key: str, default: Any = None) -> Any:
    """A metric-addressable field: the ``Sample`` attribute of that name, else a
    key of ``Sample.metadata`` (which carries ``AgentCase.metadata``).

    This is what makes ``metadata`` a first-class field a reference-free metric or
    an oracle can require and read (reviewer ask (b)): a user drops a custom
    reference field into ``Sample.metadata``/``AgentCase.metadata`` and gates a
    metric on it or feeds it to :class:`~auditkit.agent_eval.outcome.AssertionOracle`.
    """
    val = getattr(sample, key, None)
    if val is not None:
        return val
    meta = getattr(sample, "metadata", None) or {}
    return meta.get(key, default)


def _canonical_args(v: Any) -> Any:
    """Hashable, JSON-value-canonical form of a tool call's arguments, matching
    :func:`auditkit.trace._json_eq`: ``1`` and ``1.0`` collapse, ``True`` stays
    distinct from ``1``, every ``NaN`` collapses to one token, and dict key order
    (and key type, str-coerced) is irrelevant.

    A single shared identity so :class:`RedundantToolCalls` and
    :class:`AgentLoopDetection` agree with :class:`ToolCallF1`'s JSON-value
    matching on number equality (finding 9). A whole-number float keys as its
    ``int`` (``1.0 -> 1``), never via ``float()`` -- ``float()`` would collapse
    two *distinct* large integers (snowflake ids) into one."""
    if isinstance(v, bool):
        return ("bool", v)
    if isinstance(v, int):
        return ("num", v)
    if isinstance(v, float):
        if v != v:  # NaN: _json_eq treats every NaN as equal, so must a repeat
            return ("num", "nan")
        return ("num", int(v) if v.is_integer() else v)
    if isinstance(v, dict):
        return ("obj", tuple(sorted(((str(k), _canonical_args(x)) for k, x in v.items()),
                                    key=lambda kv: kv[0])))
    if isinstance(v, (list, tuple)):
        return ("arr", tuple(_canonical_args(x) for x in v))
    return ("val", v)


def _call_key(call: ToolCall) -> Optional[tuple[str, Any]]:
    """Canonical ``(name, args)`` identity for a call; ``None`` if the call is
    malformed (unparsed arguments), so two different truncated calls never key
    identically. Arguments are canonicalized to JSON values
    (:func:`_canonical_args`), so this agrees with :class:`ToolCallF1` that
    ``f(a=1)`` and ``f(a=1.0)`` are one call."""
    if call.parse_error:
        return None
    return (call.name, _canonical_args(call.arguments))


MAX_REFERENCE_PATHS = 64  # routes one sample may name; each costs a maximum matching


# What each position of select_reference_path's ranking key decides, in order.
ROUTE_DECIDERS = ("f1", "recall", "turn_structure", "call_order", "listed_order")


def _decided_by(keys: list[tuple]) -> str:
    """The first rule that left one route standing: ``"f1"`` is a clear win,
    ``"listed_order"`` means every real signal tied and the listing picked."""
    alive = keys
    for pos, label in enumerate(ROUTE_DECIDERS):
        top = max(k[pos] for k in alive)
        alive = [k for k in alive if k[pos] == top]
        if len(alive) == 1:
            return label
    return ROUTE_DECIDERS[-1]                    # unreachable: -i is unique


def select_reference_path(paths: list[Turns], pred_turns: Turns, mode: str,
                          arg_match: Optional[ArgMatcher] = None) -> tuple[int, list[float]]:
    """Which of several reference routes this run actually took, and each route's F1.

    Pure and deterministic: the choice depends only on ``(paths, pred_turns,
    mode, arg_match)``, so determinism alone makes every reference metric pick
    the same route -- no cache to key, and nothing to invalidate when a context
    is reused across samples.

    Metric-neutral on purpose (call-level F1 over the flattened calls, as
    :class:`ToolCallF1` computes it, turn structure ignored): :class:`TrajectoryMatch`
    and :class:`ParallelToolCalls` rank routes by their own thing, so a route picked
    on F1 by one and on parallelism by another would describe two different runs.

    Best of F1, then recall, then whether the run reproduces this route's turn
    structure exactly, then whether it covers this route in order, then the order
    the routes are listed. A partial run is credited with the route it got
    furthest along rather than the one the author happened to list first; the
    per-route F1s are returned so every score can carry them, so the choice is
    visible and a reader can disagree with it.

    There is deliberately no tiebreak on the fewest unmatched predicted calls,
    though one is easy to imagine. It cannot decide anything: with the predicted
    count P fixed across routes, F1 = 2pr/(p+r) rises strictly with p = m/P, so
    two routes tying on F1 and recall have the same matched count m and the same
    reference count R -- and therefore the same P - m. It would be a key that
    never fires, promising a tiebreak the reader cannot rely on.
    """
    i, f1s, _ = _select_reference_path(paths, pred_turns, mode, arg_match)
    return i, f1s


def _select_reference_path(paths: list[Turns], pred_turns: Turns, mode: str,
                           arg_match: Optional[ArgMatcher] = None) -> tuple[int, list[float], str]:
    """:func:`select_reference_path`, plus which rule decided (:data:`ROUTE_DECIDERS`)."""
    if len(paths) > MAX_REFERENCE_PATHS:
        # Selection costs one maximum matching per route, so this bounds one
        # sample's scoring work. A reference this wide has usually been written
        # out as every combination of some independent choices, which is a case
        # for a per-call alternative slot -- not yet supported -- so the honest
        # advice is to split it, and the error says which half is the problem.
        raise ValueError(
            f"reference names {len(paths)} routes, over the {MAX_REFERENCE_PATHS}-route budget for one "
            f"sample. If the routes differ in only one step, the run is paying for every combination of "
            f"the other steps; split the sample, or wait for per-call alternative slots.")
    preds = flatten(pred_turns)
    best_i, best_key = 0, None
    f1s: list[float] = []
    keys: list[tuple] = []
    for i, route in enumerate(paths):
        refs = flatten(route)
        matched = len(max_matching(preds, refs, mode, arg_match))
        if not refs:  # the irrelevance case, scored as ToolCallF1 scores it
            p = r = f = 1.0 if not preds else 0.0
        else:
            p = matched / len(preds) if preds else 0.0
            r = matched / len(refs)
            f = 2 * p * r / (p + r) if p + r else 0.0
        f1s.append(round(f, 6))
        # Call-level F1 cannot see turn structure, so two routes holding the same
        # calls -- batch them in one response or call them in sequence -- tie on it
        # and on recall, and the listing alone would pick. A run that did exactly
        # one of them would then be scored against the other and take a structural
        # 0.0 for doing the right thing. Rank on the two structural checks, shared
        # and deterministic so all three metrics still agree on one route.
        strict = (len(route) == len(pred_turns)
                  and all(_turn_matches(p, r, mode, arg_match) for p, r in zip(pred_turns, route)))
        key = (f, r, strict, in_order(route, pred_turns, mode, arg_match), -i)
        keys.append(key)
        if best_key is None or key > best_key:
            best_i, best_key = i, key
    return best_i, f1s, _decided_by(keys)


class _ReferenceToolMetric(Metric):
    """Shared plumbing for metrics that compare against expected_tool_calls.

    The reference may name one route or several (``{"any_of": [...]}``). With
    several, all of them score against the route this run actually took
    (:func:`select_reference_path`), so the metrics agree on which run is being
    described, and say which it was in their metadata.
    """

    kind = ScoreKind.AGENT
    direction = Direction.MAXIMIZE
    required_fields = frozenset({"expected_tool_calls"})
    base_name = "tool"

    def __init__(self, arg_mode: str = "exact", arg_match: Optional[ArgMatcher] = None) -> None:
        if arg_mode not in ARG_MODES:
            raise ValueError(f"arg_mode must be one of {ARG_MODES}, got {arg_mode!r}")
        self.arg_mode = arg_mode
        self.arg_match = arg_match
        tag = _suffix(arg_mode)
        if arg_match is not None:
            # A short stable hash of the matcher, so two ToolCallF1(arg_match=...)
            # with *different* matchers emit different score names instead of both
            # ".._custom" (which the Runner would average into one headline number).
            from auditkit.model import _default_callable_name
            tag += "_custom_" + _default_callable_name(arg_match).rsplit(":", 1)[-1]
        self._tag = tag
        self.name = self.base_name + tag

    def identity(self) -> dict:
        # qualname alone is "<lambda>" for every lambda; hash the code too.
        from auditkit.model import _default_callable_name
        fn = self.arg_match
        return {"name": self.name, "arg_mode": self.arg_mode,
                "arg_match": _default_callable_name(fn) if fn else None}

    def _reference_path(self, sample: Sample, output: str,
                        context: Any) -> tuple[Turns, Turns, dict[str, Any]]:
        """The reference route to score this run against, the predicted turns, and
        the route's path metadata (empty for a single-route reference).

        Single-route references get NO path metadata: a ``matched_path: 0`` on
        every existing score would change the metadata of every run that has
        nothing to do with multi-route references, for no gain.
        """
        paths = to_reference_paths(sample.expected_tool_calls)
        pred_turns = predicted_turns(output, context)
        if len(paths) == 1:
            return paths[0], pred_turns, {}
        i, f1s, decided_by = _select_reference_path(paths, pred_turns, self.arg_mode,
                                                    self._effective_arg_match(sample))
        return paths[i], pred_turns, {"matched_path": i, "n_paths": len(paths), "path_f1": f1s,
                                      "path_decided_by": decided_by}

    def _score(self, name: str, value: float, **meta: Any) -> Score:
        return Score(name=name + self._tag, value=float(value), kind=self.kind, metadata=meta)

    @staticmethod
    def _coverage_unavailable(context: Any) -> bool:
        """The trace marks tool-call coverage unobservable -- an ``agent:`` reply
        with no transcript and no explicit ``tool_calls`` (finding 5). The
        reference tool metrics are then ineligible (return no score) rather than
        scoring the reply as observed-zero-calls."""
        return bool(get_trace(context).get("tool_calls_unavailable"))

    def _effective_arg_match(self, sample: Sample) -> Optional[ArgMatcher]:
        """The argument matcher to use for THIS sample. A user-supplied
        ``arg_match`` or a non-``exact`` ``arg_mode`` is used as given. Otherwise,
        when the sample offers tool schemas, exact matching additionally tolerates
        a schema-declared OPTIONAL argument the reference omits (BFCL convention)
        while still rejecting an UNDECLARED extra (finding 10); with no schemas it
        stays strictly exact."""
        if self.arg_match is not None or self.arg_mode != "exact":
            return self.arg_match
        schemas = _tool_schemas(sample.tools or [])
        return _optional_arg_matcher(schemas) if schemas else None


@METRICS.register("tool_call_f1")
class ToolCallF1(_ReferenceToolMetric):
    """Multiset precision / recall / F1 of tool calls, plus exact set match.

    Turn boundaries are ignored here (see :class:`TrajectoryMatch` and
    :class:`ParallelToolCalls` for structure). Duplicate calls count, unlike
    set-based F1. With an empty reference (``expected_tool_calls=[]``, an
    irrelevance case) every score is 1.0 iff the model made no call.

    Emits ``tool_call_precision``, ``tool_call_recall``, ``tool_call_f1`` and
    ``tool_call_exact`` (BFCL parallel semantics: same calls, any order).
    ``arg_mode`` is ``"exact"`` (default), ``"subset"`` (extra predicted
    arguments allowed) or ``"name"`` (tool selection only); ``arg_match`` is a
    ``(tool_name, pred_args, ref_args) -> bool`` override for anything else
    (allowed-value lists, fuzzy strings, numeric tolerance).

    In ``"exact"`` mode, when ``Sample.tools`` carries the tool schemas, a
    schema-declared OPTIONAL argument the reference omits does not break the
    match (BFCL convention); an UNDECLARED extra argument still does.
    """

    base_name = "tool_call_f1"

    def score(self, sample: Sample, output: str, context: Any = None) -> list[Score]:
        if self._coverage_unavailable(context):
            return []
        ref_turns, pred_turns, path_meta = self._reference_path(sample, output, context)
        arg_match = self._effective_arg_match(sample)
        refs, preds = flatten(ref_turns), flatten(pred_turns)
        matched = len(max_matching(preds, refs, self.arg_mode, arg_match))
        if not refs:
            p = r = f = 1.0 if not preds else 0.0
        else:
            p = matched / len(preds) if preds else 0.0
            r = matched / len(refs)
            f = 2 * p * r / (p + r) if p + r else 0.0
        exact = 1.0 if matched == len(refs) == len(preds) else 0.0
        # Which route this run took belongs on every score of the family, not
        # only on the first: a reader comparing tool_call_f1 with
        # trajectory_match has to see that both read the same route.
        meta = {**path_meta, "n_expected": len(refs), "n_predicted": len(preds), "n_matched": matched,
                "predicted": [c.to_dict() for c in preds]}
        return [self._score("tool_call_precision", p, **meta),
                self._score("tool_call_recall", r, **path_meta),
                self._score("tool_call_f1", f, **path_meta),
                self._score("tool_call_exact", exact, **path_meta)]


def _turn_matches(pred: list[ToolCall], ref: list[ToolCall], mode: str, fn: Optional[ArgMatcher]) -> bool:
    return len(pred) == len(ref) and len(max_matching(pred, ref, mode, fn)) == len(ref)


def in_order(ref_turns: Turns, pred_turns: Turns, mode: str = "exact",
             arg_match: Optional[ArgMatcher] = None) -> bool:
    """Every reference turn's calls appear in the predicted calls, turn after turn.

    Extra predicted calls are allowed, so this is the weaker of the two structural
    checks :func:`select_reference_path` ranks routes by. Module-level rather than a
    method because route selection needs it before any metric exists.
    """
    preds = flatten(pred_turns)
    if not flatten(ref_turns):
        # An empty reference (an irrelevance case) is matched in order only
        # when the agent also made no call; any predicted call breaks it
        # (finding 8), consistent with strict/f1 which already score 0 here.
        return not preds
    start = 0
    for ref in ref_turns:
        if not ref:  # empty reference turn covers nothing (to_turns drops these)
            continue
        # One incremental bipartite matching per turn: feed in preds[start:]
        # one call at a time until this turn's calls are all covered. The call
        # that completes the cover is in every full matching, so start advances
        # to just past it -- identical to growing the window and rematching from
        # scratch each step. Each new pred is a new free node, so the matching
        # grows iff an augmenting path starts AT it: a pred matching no call of
        # this turn costs nothing, one matching a still-uncovered call claims it
        # directly, and only one matching covered calls alone runs one search
        # (skipped when a pred with the same edges already failed). Total
        # O(preds * len(turn)) compares plus one search per such pred.
        adj: list[list[int]] = []  # window pred -> ref-call indices it matches
        owner: dict[int, int] = {}  # ref-call index -> window pred
        failed: set[tuple[int, ...]] = set()
        end = start
        while len(owner) < len(ref):
            if end >= len(preds):
                return False
            p = preds[end]
            edges = [r for r, rc in enumerate(ref) if call_matches(p, rc, mode, arg_match)]
            adj.append(edges)
            free = next((r for r in edges if r not in owner), None)
            if free is not None:
                owner[free] = len(adj) - 1
            elif edges and tuple(edges) not in failed and not _augment(adj, owner, len(adj) - 1):
                failed.add(tuple(edges))
            end += 1
        start = end
    return True


@METRICS.register("trajectory_match")
class TrajectoryMatch(_ReferenceToolMetric):
    """Does the sequence of turns match the reference?

    Emits two binary scores:

    - ``trajectory_strict``: same number of turns, and turn *i* holds exactly
      the reference's calls for turn *i* (any order inside a turn).
    - ``trajectory_in_order``: every reference turn's calls appear in the
      predicted calls, turn after turn in reference order; extra calls are
      allowed (e.g. an agent that re-checks something).
    """

    base_name = "trajectory_match"

    def score(self, sample: Sample, output: str, context: Any = None) -> list[Score]:
        if self._coverage_unavailable(context):
            return []
        ref_turns, pred_turns, path_meta = self._reference_path(sample, output, context)
        arg_match = self._effective_arg_match(sample)
        strict = len(ref_turns) == len(pred_turns) and all(
            _turn_matches(p, r, self.arg_mode, arg_match) for p, r in zip(pred_turns, ref_turns))
        return [self._score("trajectory_strict", 1.0 if strict else 0.0,
                            n_expected_turns=len(ref_turns), n_predicted_turns=len(pred_turns), **path_meta),
                self._score("trajectory_in_order",
                            1.0 if self._in_order(ref_turns, pred_turns, arg_match) else 0.0, **path_meta)]

    def _in_order(self, ref_turns: Turns, pred_turns: Turns,
                  arg_match: Optional[ArgMatcher] = None) -> bool:
        return in_order(ref_turns, pred_turns, self.arg_mode, arg_match)


@METRICS.register("parallel_tool_calls")
class ParallelToolCalls(_ReferenceToolMetric):
    """Did the agent batch the calls that belong together, and only those?

    A reference turn with two or more calls is a parallel group: the calls are
    independent and should be emitted in one response. Separate reference
    turns are dependent: a later turn needs an earlier turn's results.

    - ``parallel_recall``: fraction of reference parallel groups whose calls
      were all made, together, in one predicted turn. Low = the model
      serializes independent calls (extra round-trips).
    - ``parallel_precision``: fraction of predicted multi-call turns whose
      (matched) calls all belong to one reference turn. Low = the model
      batches dependent calls, inventing arguments before it has the
      upstream result.
    - ``parallel_detection``: 1.0 when the model made a multi-call turn
      exactly when the reference has one (the should-I-parallelize decision).

    Recall/precision are omitted on samples where they have no denominator.
    Recall is also omitted (never replaced by a guess) in the pathological
    case where the exact group-placement search runs out of budget; the
    ``parallel_detection`` score then carries ``parallel_recall_omitted``.
    For a single-response model (one turn), put only the *next* turn in the
    reference, or dependent later turns will be counted as missed.
    """

    base_name = "parallel_tool_calls"

    def score(self, sample: Sample, output: str, context: Any = None) -> list[Score]:
        if self._coverage_unavailable(context):
            return []
        ref_turns, pred_turns, path_meta = self._reference_path(sample, output, context)
        arg_match = self._effective_arg_match(sample)

        def n_matched(pred: list[ToolCall], ref: list[ToolCall]) -> int:
            return len(max_matching(pred, ref, self.arg_mode, arg_match))

        def fits(t: list[ToolCall], calls: list[ToolCall]) -> bool:
            return n_matched(t, calls) == len(calls)

        # Matched turn against turn, not through one flat call matching: a
        # flat matching may pair a call with an identical call in another
        # turn, splitting a perfectly batched group.
        scores: list[Score] = []
        detection_meta: dict[str, Any] = {}
        groups = sorted((t for t in ref_turns if len(t) >= 2), key=len, reverse=True)
        if groups:
            hits = _pack_groups(groups, pred_turns, fits)
            if hits is None:
                # Never a wrong number: an unproven lower bound would be averaged
                # into the headline as if it were the score.
                detection_meta["parallel_recall_omitted"] = (
                    f"group placement search exceeded its budget ({len(groups)} groups)")
                logger.warning("parallel_recall omitted on sample %s: search budget exceeded "
                               "(%d groups, %d predicted turns)", sample.id, len(groups), len(pred_turns))
            else:
                scores.append(self._score("parallel_recall", hits / len(groups), n_groups=len(groups),
                                          **path_meta))

        ref_by_name = _turns_by_name(ref_turns)
        batched = []
        for pt in pred_turns:
            # Only reference turns sharing a tool name can match any of its calls.
            near = sorted(set().union(*(ref_by_name.get(c.name, ()) for c in pt)))
            n = n_matched(pt, [c for k in near for c in ref_turns[k]])
            if n >= 2:  # pure iff one reference turn explains all its matched calls
                batched.append(max(n_matched(pt, ref_turns[k]) for k in near) == n)
        if batched:
            scores.append(self._score("parallel_precision", sum(batched) / len(batched),
                                      n_batched_turns=len(batched), **path_meta))

        should = any(len(t) >= 2 for t in ref_turns)
        did = any(len(t) >= 2 for t in pred_turns)
        # Same rule for every score of the metric: no route here belongs to
        # parallel_detection alone.
        scores.append(self._score("parallel_detection", 1.0 if should == did else 0.0,
                                  expected_parallel=should, predicted_parallel=did,
                                  **detection_meta, **path_meta))
        return scores


def _turns_by_name(turns: Turns) -> dict[str, set[int]]:
    """Tool name -> indices of the turns that hold a call to it."""
    idx: dict[str, set[int]] = {}
    for k, t in enumerate(turns):
        for c in t:
            idx.setdefault(c.name, set()).add(k)
    return idx


_PACK_BUDGET = 20_000  # joint-fit checks the exact search may spend


def _pack_groups(groups: list[list[ToolCall]], turns: Turns,
                 fits: Callable[[list[ToolCall], list[ToolCall]], bool]) -> Optional[int]:
    """Most reference groups placeable at once, each inside one predicted turn.

    Groups sharing a turn must be jointly matchable into it (each predicted call
    counted once). Greedy (put each group in its first or tightest fit) is a
    lower bound but undercounts when a group fits two turns, or a turn can hold
    two groups, so it only seeds an exact branch-and-bound search. The search
    is skipped when greedy already meets an upper bound (groups that fit any
    turn alone, capped per turn by how many fit its size): a perfect prediction
    is exact in one pass however many groups it has. Iterative, so thousands of
    groups don't overflow the stack.

    ponytail: set packing is NP-hard, so the search stops after _PACK_BUDGET
    joint-fit checks and returns ``None`` (unknown) instead of an unproven
    bound; real traces with a few groups never get near it.
    """
    by_name = _turns_by_name(turns)
    fit = []  # per group: the turns it fits alone
    for g in groups:
        cand = set.intersection(*(by_name.get(c.name, set()) for c in g))
        fit.append([k for k in sorted(cand) if len(turns[k]) >= len(g) and fits(turns[k], g)])
    order = [i for i in range(len(groups)) if fit[i]]
    sizes: dict[int, list[int]] = {}
    for i in order:
        for k in fit[i]:
            sizes.setdefault(k, []).append(len(groups[i]))
    cap = 0
    for k, ss in sizes.items():
        room = len(turns[k])
        for s in sorted(ss):
            if s > room:
                break
            room -= s
            cap += 1
    ub = min(len(order), cap)

    placed: list[list[ToolCall]] = [[] for _ in turns]
    best = 0
    for i in order:  # greedy lower bound
        for k in fit[i]:
            want = placed[k] + groups[i]
            if len(want) <= len(turns[k]) and fits(turns[k], want):
                placed[k] = want
                best += 1
                break
    if best == ub:
        return best

    placed = [[] for _ in turns]
    n, cur, budget = len(order), 0, _PACK_BUDGET
    chosen: list[Optional[int]] = []  # turn chosen for order[d] (None = nowhere)
    stack = [iter(fit[order[0]] + [None])]
    while stack:
        d = len(stack) - 1
        k = next(stack[-1], -1)
        if k == -1 or cur + (n - d) <= best:  # exhausted, or can't beat best: backtrack
            stack.pop()
            if chosen and (c := chosen.pop()) is not None:
                placed[c] = placed[c][:-len(groups[order[d - 1]])]
                cur -= 1
            continue
        g = groups[order[d]]
        if k is not None:
            want = placed[k] + g
            if len(want) > len(turns[k]):
                continue
            if budget == 0:
                return None
            budget -= 1
            if not fits(turns[k], want):
                continue
            placed[k] = want
            cur += 1
        if d + 1 < n:
            chosen.append(k)
            stack.append(iter(fit[order[d + 1]] + [None]))
            continue
        best = max(best, cur)  # leaf: record, then undo this choice
        if best == ub:
            return best
        if k is not None:
            placed[k] = placed[k][:-len(g)]
            cur -= 1
    return best


_JSON_TYPES = {
    "string": (str,), "integer": (int,), "number": (int, float), "boolean": (bool,),
    "array": (list,), "object": (dict,), "null": (type(None),),
}


def _tool_schemas(tools: list[dict]) -> dict[str, dict]:
    out = {}
    for t in tools or []:
        fn = t.get("function", t) if isinstance(t, dict) else {}
        if isinstance(fn, dict) and fn.get("name"):
            # OpenAI "parameters"; Anthropic "input_schema".
            out[fn["name"]] = fn.get("parameters") or fn.get("input_schema") or {}
    return out


def _optional_arg_matcher(schemas: dict[str, dict]) -> ArgMatcher:
    """Exact matching that tolerates a schema-declared OPTIONAL argument the
    reference omits, but still rejects an UNDECLARED extra (BFCL convention,
    finding 10). Every reference argument must be present with an equal value;
    an extra predicted argument is allowed only when the tool's schema declares
    it in ``properties``."""
    def matches(name: str, pred: dict, ref: dict) -> bool:
        if not args_match(pred, ref, "subset"):  # every ref arg present & value-equal
            return False
        declared = (schemas.get(name) or {}).get("properties") or {}
        return all(k in ref or k in declared for k in pred)
    return matches


def _type_ok(value: Any, spec: dict) -> bool:
    types = spec.get("type")
    if types is None:
        return True
    for t in (types if isinstance(types, list) else [types]):
        py = _JSON_TYPES.get(t)
        if py is None:
            return True  # unknown type keyword: don't guess
        # bool is an int subclass in Python; JSON keeps them apart.
        if isinstance(value, bool) and t in ("integer", "number"):
            continue
        if isinstance(value, py):
            return True
    return False


def _join(path: str, key: str) -> str:
    """Human-readable location of a nested argument: ``reminder.minutes_before``
    at depth, the bare key at the top level (so top-level messages read
    ``'city'``, not ``'.city'``)."""
    return f"{path}.{key}" if path else key


def _validate_value(value: Any, spec: Any, path: str) -> list[str]:
    """Problems with *value* against JSON-schema *spec*, recursively (finding 7):
    array ``items``, nested object ``properties``/``required`` and closed objects
    (``properties`` declared, or ``additionalProperties: false``) are all checked.
    *path* names the location for the messages (``attendees[1]``,
    ``reminder.minutes_before``)."""
    if not isinstance(spec, dict):  # JSON Schema draft 6+ boolean subschema (or absent)
        return [f"argument {path!r} not allowed by schema"] if spec is False else []
    if not _type_ok(value, spec):
        return [f"argument {path!r} has type {type(value).__name__}, schema says {spec.get('type')}"]
    problems: list[str] = []
    if "enum" in spec and value not in spec["enum"]:
        problems.append(f"argument {path!r}={value!r} not in enum {spec['enum']}")
    if isinstance(value, dict):
        props = spec.get("properties") or {}
        # A closed object rejects undeclared keys: either it declares its
        # properties, or it explicitly forbids extras with additionalProperties.
        closed = ("properties" in spec) or (spec.get("additionalProperties") is False)
        problems += [f"missing required argument {_join(path, k)!r}"
                     for k in (spec.get("required") or []) if k not in value]
        for k, v in value.items():
            if k in props:
                problems += _validate_value(v, props[k], _join(path, k))
            elif closed:
                problems.append(f"unknown argument {_join(path, k)!r}")
    elif isinstance(value, list):
        items = spec.get("items")
        if isinstance(items, dict):  # a list `items` (draft-4 tuple form) is no constraint
            for i, item in enumerate(value):
                problems += _validate_value(item, items, f"{path}[{i}]")
    return problems


def validate_call(call: ToolCall, schemas: dict[str, dict]) -> list[str]:
    """Problems with *call* against the offered tool schemas (``[]`` = valid).

    Validates recursively (finding 7): a top-level argument, an array's items,
    and a nested object's own properties/required are all checked against the
    schema, and an undeclared key at any depth of a closed object is flagged."""
    if call.parse_error:
        return [call.parse_error]
    if call.name not in schemas:
        return [f"unknown tool {call.name!r}"]
    schema = schemas[call.name]
    if not isinstance(schema, dict):  # `parameters`/`input_schema` was a boolean, not a schema
        schema = {}
    return _validate_value(call.arguments, schema, "")


@METRICS.register("tool_call_validity")
class ToolCallValidity(Metric):
    """Fraction of calls that are valid against the offered tool schemas.

    Needs no reference answer, only ``Sample.tools``. A call is invalid when
    its tool doesn't exist, a required argument is missing, an argument isn't
    in the schema, an argument has the wrong JSON type or is outside its
    ``enum``, or its arguments weren't valid JSON. Validation is recursive:
    array ``items`` and a nested object's own ``properties``/``required`` are
    checked too, not just top-level arguments. Samples with no calls are skipped.
    """

    name = "tool_call_validity"
    kind = ScoreKind.AGENT
    direction = Direction.MAXIMIZE
    required_fields = frozenset({"tools"})

    def score(self, sample: Sample, output: str, context: Any = None) -> list[Score]:
        calls = flatten(predicted_turns(output, context))
        if not calls:
            return []
        schemas = _tool_schemas(sample.tools or [])
        problems = {i: p for i, c in enumerate(calls) if (p := validate_call(c, schemas))}
        return [Score(name=self.name, value=1 - len(problems) / len(calls), kind=self.kind,
                      reason="; ".join(f"{calls[i].name}: {', '.join(p)}" for i, p in problems.items()) or None,
                      metadata={"n_calls": len(calls), "n_invalid": len(problems)})]


@METRICS.register("redundant_tool_calls")
class RedundantToolCalls(Metric):
    """Fraction of calls that exactly repeat an earlier call (lower is better).

    Needs no reference. Catches loops and re-fetching the same data. Samples
    with no calls are skipped.
    """

    name = "redundant_tool_calls"
    kind = ScoreKind.AGENT
    direction = Direction.MINIMIZE

    def score(self, sample: Sample, output: str, context: Any = None) -> list[Score]:
        calls = flatten(predicted_turns(output, context))
        if not calls:
            return []
        # A malformed call keys to None (excluded); ToolCallValidity penalizes
        # those. len(calls) stays the denominator.
        keys = [k for c in calls if (k := _call_key(c)) is not None]
        dupes = len(keys) - len(set(keys))
        return [Score(name=self.name, value=dupes / len(calls), kind=self.kind,
                      metadata={"n_calls": len(calls), "n_duplicates": dupes})]


_TASK_COMPLETION_PROMPT = """\
<task>
{input}
</task>

<expected_outcome>
{expected}
</expected_outcome>

<agent_trajectory>
{trajectory}
</agent_trajectory>

<final_answer>
{output}
</final_answer>

Did the agent accomplish the task? Judge the final answer and the actions in
the trajectory together: an answer that claims success without the actions
that would produce it is not complete. If an expected outcome is given, the
agent's result must agree with it. Ignore style and length."""


@METRICS.register("task_completion")
class TaskCompletion(LLMJudge):
    """LLM judge of whether an agent completed its task, given its trajectory.

    The judge sees the task, the optional expected outcome (``Sample.target``),
    the agent's tool-call transcript, and its final answer, and picks
    ``complete`` (1.0), ``partial`` (0.5) or ``failed`` (0.0).
    """

    def __init__(self, *, judge_model: Any = None, name: str = "task_completion",
                 system_prompt: str = _JUDGE_SYSTEM_PROMPT,
                 judge_model_args: Optional[dict[str, Any]] = None,
                 temperature: Optional[float] = None, max_tokens: Optional[int] = None,
                 top_p: Optional[float] = None,
                 judge_chat_template_kwargs: Optional[dict[str, Any]] = None) -> None:
        super().__init__(
            judge_model=judge_model, name=name, judge_model_args=judge_model_args,
            temperature=temperature, max_tokens=max_tokens, top_p=top_p,
            judge_chat_template_kwargs=judge_chat_template_kwargs,
            prompt=_TASK_COMPLETION_PROMPT, use_cot=True, system_prompt=system_prompt,
            choices={"complete": 1.0, "partial": 0.5, "failed": 0.0},
        )
        self.kind = ScoreKind.AGENT

    def judge(self, sample: Sample, output: str, context: Any = None) -> Score:
        from dataclasses import replace
        from auditkit.model import Request
        transcript = render_trace(predicted_turns(output, context), get_trace(context))
        # {trajectory} is filled from the real transcript below. _render fills every
        # metadata key first, so a dataset column literally named 'trajectory' would
        # otherwise pre-empt the placeholder and the judge would never see the run.
        if sample.metadata and "trajectory" in sample.metadata:
            sample = replace(sample, metadata={k: v for k, v in sample.metadata.items() if k != "trajectory"})
        user = self._render(sample, output).replace("{trajectory}", transcript)
        results = self._model().generate([Request(prompt=self._assemble(user), params=dict(self._gen_params))])
        text = results[0].completions[0].text if results and results[0].completions else ""
        return self._parse(text or "")

    def score(self, sample: Sample, output: str, context: Any = None) -> Score:
        result = self.judge(sample, output, context)
        if result.metadata.get("unknown"):
            # An unreadable verdict is an error, not a silent 0.0 averaged into
            # the headline (finding 6) -- same contract as the RAG verdict judges,
            # which raise so the Runner records the error and skips the sample.
            raise ValueError(f"task_completion judge returned no parseable verdict "
                             f"(complete/partial/failed): {result.reason}")
        result.kind = ScoreKind.AGENT
        return result


# ---------------------------------------------------------------------------
# Reference-free agent metrics (no gold trajectory / answer needed).
# ---------------------------------------------------------------------------

def _tool_names(items: Any) -> set[str]:
    """Tool names out of a list of plain names or OpenAI/Anthropic schemas."""
    names: set[str] = set()
    for t in items or []:
        if isinstance(t, dict):
            fn = t.get("function", t)
            name = fn.get("name") if isinstance(fn, dict) else None
            if name:
                names.add(str(name))
        elif t is not None:
            names.add(str(t))
    return names


def _assistant_texts(context: Any) -> list[str]:
    """Consecutive assistant/reasoning text turns from the trace transcript."""
    out: list[str] = []
    for m in get_trace(context).get("messages") or []:
        if not isinstance(m, dict) or m.get("tool_calls"):
            continue
        if m.get("role") in ("assistant", None):
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                out.append(content.strip())
    return out


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _argument_words(calls: list[ToolCall]) -> set[str]:
    """Every word in the calls' argument values (the entities the agent acted on)."""
    out: set[str] = set()

    def walk(v: Any) -> None:
        if isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)
        elif v is not None:
            out.update(_words(str(v)))

    for c in calls:
        walk(c.arguments)
    return out


def _has_cycle(keys: list[Any]) -> bool:
    """A cycle in the directed graph of consecutive call transitions (DFS).

    Nodes are ``(name, args)`` identities (see :func:`_call_key`), edges are
    consecutive calls. A back edge -- a node reachable from itself -- means the
    agent returned to an earlier state and kept going (a loop).

    ponytail: nodes key on name AND arguments, so ``search(a)->read->search(b)``
    (normal fan-out) is not a cycle; only a genuine revisit of the same call is.
    """
    adj: dict[Any, set[Any]] = {}
    for u, v in zip(keys, keys[1:]):
        # Self-edges (an immediate repeat) belong to the repetition signal, not
        # the cycle signal; a cycle needs distinct nodes (a->b->a).
        if u is not None and v is not None and u != v:
            adj.setdefault(u, set()).add(v)
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[Any, int] = {}
    for start in adj:
        if color.get(start, WHITE) != WHITE:
            continue
        stack = [(start, iter(adj.get(start, ())))]
        color[start] = GRAY
        while stack:
            node, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                color[node] = BLACK
                stack.pop()
                continue
            c = color.get(nxt, WHITE)
            if c == GRAY:  # back edge -> cycle
                return True
            if c == WHITE:
                color[nxt] = GRAY
                stack.append((nxt, iter(adj.get(nxt, ()))))
    return False


@METRICS.register("agent_loop_detection")
class AgentLoopDetection(Metric):
    """Reference-free loop detector: 1.0 = loop-free, 0.0 = a loop was found.

    Deterministic, no judge, needs only the trace (no reference). Flags three
    independent signals (deepeval ``agent_loop_detection``):

    - **repeated calls**: some identical ``(name, arguments)`` call is made
      ``repetition_threshold`` times or more;
    - **reasoning stagnation**: two consecutive assistant/reasoning messages are
      near-identical (``difflib`` ratio >= ``similarity_threshold``);
    - **call-graph cycle**: the sequence of calls revisits an earlier call
      (a back edge in the transition graph, found by DFS).

    ``reason`` names the signals that fired. Samples with no calls and no
    assistant messages are skipped (nothing to measure).
    """

    name = "agent_loop_detection"
    kind = ScoreKind.AGENT
    direction = Direction.MAXIMIZE

    def __init__(self, repetition_threshold: int = 3, similarity_threshold: float = 0.85) -> None:
        if repetition_threshold < 2:
            raise ValueError("repetition_threshold must be >= 2")
        self.repetition_threshold = repetition_threshold
        self.similarity_threshold = similarity_threshold

    def identity(self) -> dict:
        return {"name": self.name, "repetition_threshold": self.repetition_threshold,
                "similarity_threshold": self.similarity_threshold}

    def score(self, sample: Sample, output: str, context: Any = None) -> list[Score]:
        calls = flatten(predicted_turns(output, context))
        texts = _assistant_texts(context)
        if not calls and not texts:
            return []
        signals: list[str] = []

        keys = [_call_key(c) for c in calls]
        counts: dict[Any, int] = {}
        for k in keys:
            if k is not None:
                counts[k] = counts.get(k, 0) + 1
        worst = max(counts.values(), default=0)
        if worst >= self.repetition_threshold:
            top = max(counts, key=counts.get)  # type: ignore[arg-type]
            signals.append(f"repeated call {top[0]!r} x{worst}")

        # Two near-identical consecutive messages are stagnation, UNLESS the words that
        # differ are arguments of the calls the agent made ("... weather in Paris." then
        # "... in Rome." while calling get_weather for each): that is progress through
        # different targets, not a loop.
        arg_words = _argument_words(calls)
        max_ratio = 0.0
        for a, b in zip(texts, texts[1:]):
            diff = set(_words(a)) ^ set(_words(b))
            if diff and diff <= arg_words:
                continue
            max_ratio = max(max_ratio, difflib.SequenceMatcher(None, a, b).ratio())
        if max_ratio >= self.similarity_threshold:
            signals.append(f"reasoning stagnation (similarity {max_ratio:.2f})")

        if _has_cycle(keys):
            signals.append("call-graph cycle")

        looped = bool(signals)
        return [Score(name=self.name, value=0.0 if looped else 1.0, kind=self.kind,
                      reason="; ".join(signals) or None,
                      metadata={"looped": looped, "signals": signals,
                                "max_repeat": worst, "n_calls": len(calls)})]


@METRICS.register("tool_permission")
class ToolPermission(Metric):
    """Reference-free least-privilege check: fraction of calls within policy.

    Deterministic, no judge, no gold trajectory. The policy is an allowlist and
    an optional denylist; a call is unauthorized when it is in the denylist, or
    when an allowlist is set and its tool is not on it (a denial always wins).
    The allowlist comes from ``AgentCase.allowed_tools`` (read here as
    ``Sample.metadata['allowed_tools']`` or, failing that, the offered
    ``Sample.tools``); the denylist from ``denied_tools=`` or
    ``Sample.metadata['denied_tools']``. ``reason`` lists the violations.

    Skipped when no policy is available and no calls were made. deepeval
    ``tool_permission``.
    """

    name = "tool_permission"
    kind = ScoreKind.AGENT
    direction = Direction.MAXIMIZE

    def __init__(self, denied_tools: Optional[list[str]] = None) -> None:
        self.denied_tools = list(denied_tools) if denied_tools is not None else None

    def identity(self) -> dict:
        return {"name": self.name, "denied_tools": self.denied_tools}

    def _policy(self, sample: Sample) -> tuple[Optional[set[str]], set[str]]:
        allowed_spec = addressable(sample, "allowed_tools")
        if allowed_spec is None and sample.tools:
            allowed_spec = sample.tools
        allow = _tool_names(allowed_spec) if allowed_spec is not None else None
        deny = _tool_names(self.denied_tools if self.denied_tools is not None
                           else addressable(sample, "denied_tools"))
        return allow, deny

    def applicable(self, sample: Sample) -> bool:
        allow, deny = self._policy(sample)
        return allow is not None or bool(deny)

    def score(self, sample: Sample, output: str, context: Any = None) -> list[Score]:
        calls = flatten(predicted_turns(output, context))
        if not calls:
            return []
        allow, deny = self._policy(sample)
        violations: list[str] = []
        for c in calls:
            if c.name in deny:
                violations.append(f"{c.name}: denied")
            elif allow is not None and c.name not in allow:
                violations.append(f"{c.name}: not in allowlist")
        value = (len(calls) - len(violations)) / len(calls)
        return [Score(name=self.name, value=value, kind=self.kind,
                      reason="; ".join(dict.fromkeys(violations)) or None,
                      metadata={"n_calls": len(calls), "n_violations": len(violations),
                                "allowlist": sorted(allow) if allow is not None else None,
                                "denylist": sorted(deny)})]


_TOOL_SELECTION_SYSTEM = """\
You are an objective judge deciding whether an AI agent's action is justified
at this specific point in the conversation.

Given the task and everything the agent has done so far, is the agent justified
in making THIS tool call now? Consider whether it reasonably advances the task,
whether the arguments are sensible given what is known, and whether a competent
agent would take this action here. Judge only this one call in context; do not
require it to be the single optimal choice, only a reasonable one."""

_TOOL_SELECTION_PROMPT = """\
<task>
{input}
</task>

<actions_so_far>
{transcript}
</actions_so_far>

<candidate_action>
{call}
</candidate_action>

Is the candidate action justified at this point?"""


@METRICS.register("tool_selection")
class ToolSelectionJudge(LLMJudge):
    """Reference-free LLM judge: was each tool call justified in context?

    For every tool call, the judge sees the task and the transcript of the
    *preceding* calls and decides whether that call was justified at that point;
    the score is the mean over calls (strands ``tool_selection_accuracy`` /
    phoenix ``tool_selection``). No gold trajectory or answer is used. Calls
    whose verdict cannot be parsed are excluded from the mean; if every call is
    unparseable the score is a single ``unknown``-flagged result. Samples with
    no calls are skipped.
    """

    def __init__(self, *, judge_model: Any = None, name: str = "tool_selection",
                 system_prompt: str = _TOOL_SELECTION_SYSTEM,
                 judge_model_args: Optional[dict[str, Any]] = None,
                 temperature: Optional[float] = None, max_tokens: Optional[int] = None,
                 top_p: Optional[float] = None,
                 judge_chat_template_kwargs: Optional[dict[str, Any]] = None) -> None:
        super().__init__(
            judge_model=judge_model, name=name, judge_model_args=judge_model_args,
            prompt=_TOOL_SELECTION_PROMPT, use_cot=True, system_prompt=system_prompt,
            choices={"yes": 1.0, "no": 0.0},
            temperature=temperature, max_tokens=max_tokens, top_p=top_p,
            judge_chat_template_kwargs=judge_chat_template_kwargs,
        )
        self.kind = ScoreKind.AGENT

    def score(self, sample: Sample, output: str, context: Any = None) -> list[Score]:
        turns = predicted_turns(output, context)
        calls = flatten(turns)
        if not calls:
            return []
        from auditkit.model import Request
        task = getattr(sample, "input_text", None) or str(sample.input)
        model = self._model()
        history: list[ToolCall] = []
        values: list[float] = []
        reasons: list[str] = []
        for c in calls:
            transcript = "\n".join(json.dumps(h.to_dict(), default=str) for h in history) or "(none)"
            user = (self.prompt.replace("{input}", task)
                    .replace("{transcript}", transcript)
                    .replace("{call}", json.dumps(c.to_dict(), default=str)))
            results = model.generate([Request(prompt=self._assemble(user), params=dict(self._gen_params))])
            text = results[0].completions[0].text if results and results[0].completions else ""
            sc = self._parse(text or "")
            history.append(c)
            if sc.metadata.get("unknown"):
                reasons.append(f"{c.name}: unparsed")
                continue
            values.append(sc.value)
            reasons.append(f"{c.name}: {sc.metadata.get('choice')}")
        if not values:  # every call unparseable
            return [Score(name=self.name, value=self.unknown_score if self.unknown_score is not None else 0.0,
                          kind=self.kind, reason="UNKNOWN -- no tool-call verdict could be parsed",
                          metadata=self._unknown_meta(n_calls=len(calls)))]
        return [Score(name=self.name, value=sum(values) / len(values), kind=self.kind,
                      reason="; ".join(reasons),
                      metadata={"n_calls": len(calls), "n_judged": len(values)})]
