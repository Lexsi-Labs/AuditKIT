"""The agent-eval runner: modes, scoring, offline rescore (A1/A2).

``recorded`` mode imports/scores saved episodes with NO agent or tool call
(AG-06). ``deployed`` mode calls an ``agent:`` endpoint via the existing
:class:`AgentEndpointModel`, capturing output/trace/errors/latency (AG-09) and
naming the missing field when a metric is ineligible (AG-10). ``harness`` mode
(A4) drives a bounded tool loop that AuditKit owns. Each result carries the
verified outcome (headline), separate diagnostic columns, coverage, resource
use, and an exact status (AG-11).

Repeated trials (``trials>1``, A3) run each case N times and report reliability
(pass@k / all-k / variance / a stable aggregate; see :mod:`.reliability`).
Independence is claimed only when the caller confirms a reset-per-trial contract
(``reset_confirmed=True``); otherwise the counts are shown but flagged (AG-12).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import threading
import time
import warnings
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

from . import importers, outcome as outcome_mod
from ..trace import parse_tool_calls, strip_tool_calls
from .report import AgentCaseResult, AgentEvalResult
from .types import OBSERVED, UNAVAILABLE, AgentCase, AgentEpisode, validate_cases

# Case-level status values (AG-11).
STATUS = ("completed", "budget_exhausted", "method_error", "target_error",
          "judge_error", "ineligible", "not_applicable")

DEFAULT_SCORERS = ["tool_call_f1", "tool_call_validity", "task_completion"]

# Every scorer name the runner accepts (``_make_scorer`` keys + the judge).
KNOWN_SCORERS = ("tool_call_f1", "tool_call_f1_name", "trajectory_match",
                 "parallel_tool_calls", "tool_call_validity", "redundant_tool_calls",
                 "retrieval", "task_completion",
                 # reference-free (no gold trajectory/answer needed)
                 "agent_loop_detection", "tool_permission", "tool_selection")


def validate_scorers(scorers: list[str]) -> None:
    """Fail fast on an unknown scorer name (else every case is a method_error)."""
    unknown = [s for s in scorers if s not in KNOWN_SCORERS]
    if unknown:
        raise ValueError(f"unknown scorer(s) {unknown}; known: {', '.join(KNOWN_SCORERS)}")

# Scorers that compare tool calls against arguments (need args observed).
_ARG_SENSITIVE = frozenset({"tool_call_f1", "trajectory_match", "parallel_tool_calls",
                            "redundant_tool_calls"})


@dataclass
class AgentEvalSpec:
    """One agent evaluation to run."""

    cases: list[AgentCase] = field(default_factory=list)
    mode: str = "recorded"
    agent: Any = None  # deployed: an "agent:<url>" spec or a Model
    agent_opts: dict[str, Any] = field(default_factory=dict)
    episodes: Optional[list[AgentEpisode]] = None  # recorded: score these
    import_path: Optional[str] = None  # recorded: import from an AgentTune file
    trials: int = 1
    reset_confirmed: bool = False  # A3: caller confirms a reset-per-trial contract
    reliability_k: Optional[int] = None  # A3: k for pass@k / all-k (default: decided trials)
    scorers: list[str] = field(default_factory=lambda: list(DEFAULT_SCORERS))
    judge: Any = None  # a TaskCompletion instance or a judge-model spec
    name: str = "agent-eval"
    # harness: generation settings for every policy step (a RunConfig; max_tokens,
    # temperature, top_p, top_k, seed, stop_sequences). None = the backend's defaults.
    config: Any = None

    def identity(self) -> dict[str, Any]:
        return {
            "name": self.name, "mode": self.mode, "trials": self.trials,
            "reset_confirmed": self.reset_confirmed,
            "scorers": list(self.scorers),
            "cases": [c.digest() for c in self.cases],
            "agent": self.agent if isinstance(self.agent, str) else type(self.agent).__name__
            if self.agent is not None else None,
            "judge": _judge_identity(self.judge),
            "generation": _generation_params(self.config),
        }


def _judge_identity(judge: Any) -> Any:
    if judge is None:
        return None
    if isinstance(judge, str):
        return judge
    ident = getattr(judge, "identity", None)
    return ident() if callable(ident) else getattr(judge, "name", type(judge).__name__)


# --------------------------------------------------------------------------
# scorer registry + eligibility
# --------------------------------------------------------------------------
def _make_scorer(name: str, spec: Optional["AgentEvalSpec"] = None) -> Any:
    from ..metrics.agent import (
        AgentLoopDetection,
        ParallelToolCalls,
        RedundantToolCalls,
        ToolCallF1,
        ToolCallValidity,
        ToolPermission,
        ToolSelectionJudge,
        TrajectoryMatch,
    )
    from ..metrics.retrieval import RetrievalMetrics

    judge_model = getattr(spec, "judge", None) if spec is not None else None
    return {
        "tool_call_f1": lambda: ToolCallF1(),
        "tool_call_f1_name": lambda: ToolCallF1(arg_mode="name"),
        "trajectory_match": lambda: TrajectoryMatch(),
        "parallel_tool_calls": lambda: ParallelToolCalls(),
        "tool_call_validity": lambda: ToolCallValidity(),
        "redundant_tool_calls": lambda: RedundantToolCalls(),
        "retrieval": lambda: RetrievalMetrics(),
        "agent_loop_detection": lambda: AgentLoopDetection(),
        "tool_permission": lambda: ToolPermission(),
        "tool_selection": lambda: ToolSelectionJudge(
            judge_model=judge_model if isinstance(judge_model, str) else None),
    }[name]()


def _has_reference(case: AgentCase) -> bool:
    """Whether the case carries a gold tool route at all.

    ``is not None``, never a truthiness test. ``[]`` and ``{"any_of": [[]]}`` are
    references the author wrote on purpose -- "calling no tool is also correct" --
    and they are scoreable, not missing. Downstream the two are literally the
    same value: ``to_reference_paths`` turns ``None``, ``[]`` and
    ``{"any_of": [[]]}`` all into the single empty route ``[[]]``, so once the
    reference leaves the case nothing can tell "no reference" from "no call
    expected". This check is the only place that still can, and getting it wrong
    is silently severe: a truthiness test reports the irrelevance case as an
    ineligible metric, which drops the sample from the denominator instead of
    scoring it 1.0.
    """
    return case.reference_turns is not None


def eligibility(scorer: str, case: AgentCase, episode: AgentEpisode) -> Optional[str]:
    """``None`` when the scorer can run, else the missing reference/trace field.

    Names the exact field so the report can explain why a metric was skipped
    (AG-10). A names-only trace cannot feed argument-sensitive metrics -- that
    would read empty arguments as "wrong args" (fabrication). "No reference" is
    ``reference_turns is None`` and nothing else: an empty reference is a
    reference (see :func:`_has_reference`).
    """
    cov = episode.coverage
    has_calls = bool(episode.turns())          # scorable calls only (a call payload with a name)
    if scorer in ("tool_call_f1", "trajectory_match", "parallel_tool_calls"):
        if not _has_reference(case):
            return "case.reference_turns"
        if episode.calls_unnamed():
            # calls were recorded without tool names (an AgentTune TraceLogger record)
            return "trace.tool_call_names"
        if not has_calls:
            # Tool calls OBSERVED with zero calls is evidence ("the agent called
            # nothing"): scoreable, e.g. irrelevance or a skipped required tool.
            # Unobserved (a sparse answer-only reply) stays ineligible.
            return None if cov.get("tool_call_names") == OBSERVED else "trace.tool_calls"
        if cov.get("tool_call_arguments") != OBSERVED:
            return "trace.tool_call_arguments"
        return None
    if scorer == "tool_call_f1_name":
        if not _has_reference(case):
            return "case.reference_turns"
        if cov.get("tool_call_names") != OBSERVED:
            return "trace.tool_calls"
        return None
    if scorer == "tool_call_validity":
        if not case.allowed_tools:
            return "case.allowed_tools"
        if not has_calls or cov.get("tool_call_names") != OBSERVED:
            return "trace.tool_calls"   # zero calls leaves nothing to validate
        return None
    if scorer == "redundant_tool_calls":
        if not has_calls:
            return "trace.tool_calls"
        if cov.get("tool_call_arguments") != OBSERVED:
            return "trace.tool_call_arguments"
        return None
    if scorer == "retrieval":
        if case.reference_contexts is None:
            return "case.reference_contexts"
        if cov.get("retrieved_contexts") != OBSERVED:
            return "trace.retrieved_contexts"
        return None
    if scorer in ("agent_loop_detection", "tool_permission", "tool_selection"):
        # reference-free: no gold trajectory/answer needed, only the trace
        if not has_calls:
            return "trace.tool_calls"
        if scorer == "tool_permission" and not (
                case.allowed_tools or (case.metadata or {}).get("denied_tools")):
            return "case.allowed_tools"
        return None
    if scorer == "task_completion":
        return None  # diagnostic judge is always eligible
    return None


def _tool_schemas(allowed_tools: Optional[list[Any]]) -> Optional[list[dict[str, Any]]]:
    """Turn ``allowed_tools`` (names or full OpenAI schemas) into schemas for
    ToolCallValidity. With names only, existence is checked but not arguments."""
    if not allowed_tools:
        return None
    schemas = []
    for t in allowed_tools:
        if isinstance(t, dict):
            schemas.append(t)
        else:
            schemas.append({"type": "function", "function": {"name": str(t)}})
    return schemas


def _sample_for(case: AgentCase) -> Any:
    from ..sample import Sample
    from ..types import TaskKind

    meta = dict(case.metadata or {})
    # carry allowed/denied tools so the reference-free ToolPermission metric,
    # which reads them from Sample.metadata, can gate on the case's policy
    if case.allowed_tools and "allowed_tools" not in meta:
        meta["allowed_tools"] = [t if isinstance(t, str)
                                 else (t.get("function", {}).get("name") if isinstance(t, dict) else str(t))
                                 for t in case.allowed_tools]
    return Sample(
        input=case.task,
        target=meta.get("target"),
        kind=TaskKind.AGENT,
        tools=_tool_schemas(case.allowed_tools),
        # The reference crosses unchanged, multi-route form included: coercing it
        # to turns here would collapse {"any_of": [...]} to a single route and
        # score the alternative routes the author named as correct as wrong.
        # Every reference metric selects a route from the same value (see
        # auditkit.metrics.agent), so they cannot disagree about which.
        expected_tool_calls=case.reference_turns,
        reference_contexts=case.reference_contexts,
        metadata=meta,
    )


# --------------------------------------------------------------------------
# case derivation
# --------------------------------------------------------------------------
def case_from_episode(episode: AgentEpisode, *, outcome: Any = None,
                      case_id: Optional[str] = None) -> AgentCase:
    """Derive an :class:`AgentCase` from an imported episode (recorded mode).

    Default outcome is an ``AnswerAssertion`` against the recorded target when
    one exists -- a weak, honest fallback the caller can override.
    """
    meta = episode.metadata or {}
    target = meta.get("target")
    if outcome is None and target is not None:
        outcome = outcome_mod.AnswerAssertion(reference=target)
    cid = case_id or episode.case_id or episode.source_id
    if cid is None:
        raise ValueError("cannot derive a stable case id from this episode; pass case_id= "
                         "(the episode has no case_id or source_id)")
    return AgentCase(
        id=str(cid),
        task=str(meta.get("task") or "(imported task)"),
        category="imported",
        reference_contexts=meta.get("reference_contexts"),
        outcome=outcome,
        metadata={"target": target, "source_id": episode.source_id,
                  "source_format": episode.source_format},
    )


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------
class _LogCapture(logging.Handler):
    """Capture agent_endpoint warnings so endpoint errors stay visible (AG-09)."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []
        self.thread_id: Optional[int] = None

    def emit(self, record: logging.LogRecord) -> None:
        # The logger is process-global: keep only this call's thread, so
        # concurrent runners never mix each other's endpoint errors.
        if record.thread == self.thread_id:
            self.messages.append(record.getMessage())


class AgentEvalRunner:
    """Run an :class:`AgentEvalSpec` to an :class:`AgentEvalResult`."""

    def run(self, spec: AgentEvalSpec) -> AgentEvalResult:
        if not isinstance(spec.trials, int) or spec.trials < 1:
            raise ValueError(f"trials must be a positive int, got {spec.trials!r}")
        validate_cases(spec.cases)
        validate_scorers(spec.scorers)
        identity = spec.identity()
        if spec.mode == "recorded":
            rows, episodes = self._run_recorded(spec)
        elif spec.mode == "deployed":
            rows, episodes, agent_identity = self._run_deployed(spec)
            identity["agent"] = agent_identity
        elif spec.mode == "harness":
            rows, episodes, agent_identity = self._run_harness(spec)
            identity["agent"] = agent_identity
        else:
            raise ValueError(f"unknown mode {spec.mode!r}; use 'recorded', 'deployed' or 'harness'")
        return AgentEvalResult(rows=rows, mode=spec.mode, spec_identity=identity,
                               episodes=episodes)

    # -- trials: score N episodes for one case and aggregate reliability (A3) --
    def _finalize(self, spec: AgentEvalSpec, case: AgentCase,
                  episodes: list[Optional[AgentEpisode]]) -> AgentCaseResult:
        """One row for a case: a single ``_score`` for one trial, else aggregate."""
        if spec.trials == 1 or len(episodes) <= 1:
            return self._score(spec, case, episodes[0] if episodes else None)
        trial_rows = [self._score(spec, case, ep) for ep in episodes]
        return self._aggregate(spec, case, trial_rows, episodes)

    @staticmethod
    def _aggregate(spec: AgentEvalSpec, case: AgentCase, trial_rows: list[AgentCaseResult],
                   episodes: list[Optional[AgentEpisode]]) -> AgentCaseResult:
        from .reliability import aggregate_verdict, reliability

        # A trial that errored or blew its budget never counts as a reliable
        # success (mirrors the summary's release gate): its verdict lands in the
        # reliability n_error bucket, excluded from decided.
        verdicts = [tr.outcome.get("verdict") if tr.status == "completed"
                    else outcome_mod.ERROR for tr in trial_rows]
        rel = reliability(
            verdicts, k=spec.reliability_k, independent=spec.reset_confirmed,
            trial_ids=[ep.trial_id if ep is not None else None for ep in episodes],
            n_requested=spec.trials)
        agg_verdict, agg_reason = aggregate_verdict(verdicts)
        # per-trial compact rows + aggregated disagreement (AG-05 over trials)
        trials_out: list[dict[str, Any]] = []
        per_trial_dis: list[dict[str, Any]] = []
        n_disagree = 0
        kinds: Counter = Counter()
        for tr, ep in zip(trial_rows, episodes):
            trials_out.append({
                "trial_id": ep.trial_id if ep is not None else None,
                "verdict": tr.outcome.get("verdict"),
                "status": tr.status,
                "judge_verdict": (tr.judge or {}).get("verdict"),
                "disagreement": tr.disagreement or {},
            })
            d = tr.disagreement or {}
            per_trial_dis.append(d)
            if d:
                n_disagree += 1
                for kind in d.get("kinds", {}):
                    kinds[kind] += 1
        disagreement: dict[str, Any] = {}
        if n_disagree:
            disagreement = {"n_trials": len(trial_rows), "n_disagree": n_disagree,
                            "rate": n_disagree / len(trial_rows), "kinds": dict(kinds),
                            "trials": per_trial_dis,
                            "resolution": "verified oracle wins each trial; recorded for review"}
        first = trial_rows[0]
        status = "completed" if any(tr.status == "completed" for tr in trial_rows) else \
            Counter(tr.status for tr in trial_rows).most_common(1)[0][0]
        return AgentCaseResult(
            case_id=case.id,
            status=status,
            outcome={"verdict": agg_verdict, "reason": agg_reason,
                     "source": {"type": "trial_aggregate", "n_trials": len(trial_rows)},
                     "diagnostic": False, "aggregate": True},
            coverage=dict(first.coverage),
            coverage_label=first.coverage_label,
            diagnostics=first.diagnostics,
            ineligible=first.ineligible,
            resource_use={**(first.resource_use or {}), "n_trials": len(trial_rows)},
            provenance=first.provenance,
            judge=first.judge,
            stop_reason=first.stop_reason,
            errors=first.errors,
            source_format=first.source_format,
            source_id=first.source_id,
            reliability=rel,
            trials=trials_out,
            disagreement=disagreement,
        )

    # -- recorded (import + score; NO agent call, AG-06) -----------------
    def _run_recorded(self, spec: AgentEvalSpec) -> tuple[list[AgentCaseResult], list[AgentEpisode]]:
        episodes = list(spec.episodes or [])
        if spec.import_path:
            episodes += importers.episodes_from_agenttune(spec.import_path)
        # A task row is something to run, not a recorded episode: never scored here.
        episodes = [ep for ep in episodes if not ep.metadata.get("is_task")]
        rows: list[AgentCaseResult] = []
        scored: list[AgentEpisode] = []
        if spec.cases:
            # Match by explicit id only (case_id, then source_id). A case with no
            # episode is 'not_applicable', never handed another case's episode
            # by position. Indexed once, so matching is linear.
            by_case: dict[str, list[AgentEpisode]] = {}
            by_source: dict[str, list[AgentEpisode]] = {}
            for ep in episodes:
                if ep.case_id is not None:
                    by_case.setdefault(str(ep.case_id), []).append(ep)
                if ep.source_id is not None:
                    by_source.setdefault(str(ep.source_id), []).append(ep)
            used: set[int] = set()
            for case in spec.cases:
                found = by_case.get(case.id) or by_source.get(case.id) or []
                if spec.trials > 1:
                    # Each matched episode is a trial (up to `trials`); aggregate.
                    use = found[:spec.trials]
                    if len(found) > spec.trials:
                        warnings.warn(f"{len(found)} episodes match case id {case.id!r} but "
                                      f"trials={spec.trials}; scoring the first {spec.trials}, "
                                      "the rest are ignored", stacklevel=3)
                    rows.append(self._finalize(spec, case, use))
                    scored.extend(use)
                else:
                    if len(found) > 1:
                        warnings.warn(f"{len(found)} episodes match case id {case.id!r}; scoring "
                                      "the first, the others are ignored", stacklevel=3)
                    ep = found[0] if found else None
                    rows.append(self._score(spec, case, ep))
                    if ep is not None:
                        scored.append(ep)
                used.update(id(e) for e in found)  # extras already warned/used above
            unmatched = [ep for ep in episodes if id(ep) not in used]
            if unmatched:
                ids = [ep.case_id or ep.source_id for ep in unmatched[:5]]
                warnings.warn(f"{len(unmatched)} recorded episode(s) match no case and were "
                              f"not scored (e.g. {ids})", stacklevel=3)
        elif spec.trials > 1:
            # No cases: episodes sharing a derived id are trials of one case.
            groups: "OrderedDict[str, list[AgentEpisode]]" = OrderedDict()
            for i, ep in enumerate(episodes):
                cid = str(ep.case_id or ep.source_id or f"{ep.source_format}:{i}")
                groups.setdefault(cid, []).append(dataclasses.replace(ep, case_id=cid))
            for cid, eps in groups.items():
                use = eps[:spec.trials]
                rows.append(self._finalize(spec, case_from_episode(use[0], case_id=cid), use))
                scored.extend(use)
        else:
            seen: set[str] = set()
            dup: set[str] = set()
            for i, ep in enumerate(episodes):
                cid = str(ep.case_id or ep.source_id or f"{ep.source_format}:{i}")
                if cid in seen:
                    dup.add(cid)
                seen.add(cid)
                # Carry the derived id on the saved episode so a rescore of this
                # run matches it by id, not by a list index that can shift.
                ep = dataclasses.replace(ep, case_id=cid)
                rows.append(self._score(spec, case_from_episode(ep, case_id=cid), ep))
                scored.append(ep)
            if dup:
                warnings.warn(f"duplicate episode id(s) {sorted(dup)[:5]}: several rows share "
                              "one case_id", stacklevel=3)
        return rows, scored

    # -- deployed (call an agent: endpoint, AG-09) -----------------------
    def _run_deployed(self, spec: AgentEvalSpec) -> tuple[list[AgentCaseResult], list[AgentEpisode], Any]:
        from ..model import AutoModel, Request

        if spec.agent is None:
            raise ValueError("deployed mode needs spec.agent (e.g. 'agent:http://host/run')")
        opts = dict(spec.agent_opts)
        state_path = opts.pop("state_path", "final_state")
        artifacts_path = opts.pop("artifacts_path", "artifacts")
        model = AutoModel.resolve(spec.agent, **opts)
        if hasattr(model, "state_path"):          # agent: reads state/artifacts natively
            model.state_path, model.artifacts_path = state_path, artifacts_path
        # Fingerprint the resolved endpoint (URL, response mapping, header
        # digest), never raw agent_opts, which can carry secrets.
        ident = getattr(model, "identity", None)
        agent_identity: Any = ident() if callable(ident) else (
            spec.agent if isinstance(spec.agent, str) else type(model).__name__)
        if isinstance(agent_identity, dict):
            agent_identity = {**agent_identity, "timeout": getattr(model, "timeout", None),
                              "state_path": state_path, "artifacts_path": artifacts_path}
        deadline = opts.get("timeout", getattr(model, "timeout", None))
        endpoint_log = logging.getLogger("auditkit.model.agent_endpoint")
        rows: list[AgentCaseResult] = []
        scored: list[AgentEpisode] = []
        for case in spec.cases:
            eps = [self._deployed_call(model, case, deadline, endpoint_log, trial=t)
                   for t in range(spec.trials)]
            rows.append(self._finalize(spec, case, eps))
            scored.extend(eps)
        return rows, scored, agent_identity

    @staticmethod
    def _deployed_call(model: Any, case: AgentCase, deadline: Optional[float],
                       endpoint_log: logging.Logger, *, trial: int = 0) -> AgentEpisode:
        """One POST to the endpoint -> one episode (an error is captured, not raised)."""
        from ..model import Request

        params: dict[str, Any] = {}
        if case.metadata.get("messages"):
            params["messages"] = case.metadata["messages"]
        schemas = _tool_schemas(case.allowed_tools)
        if schemas:
            params["tools"] = schemas
        req = Request(prompt=case.task, request_type="chat", params=params)
        capture = _LogCapture()
        endpoint_log.addHandler(capture)
        t0 = time.monotonic()
        try:
            result, err = _call_with_deadline(model, req, capture, deadline)
        finally:
            endpoint_log.removeHandler(capture)
        if err is None:
            try:
                ep = AgentEvalRunner._deployed_episode(case, result, list(capture.messages))
                _attach_state(ep, result)
            except Exception as e:  # a malformed-but-JSON reply fails only this case
                err = f"could not read the agent reply: {type(e).__name__}: {e}"
        if err is not None:
            ep = AgentEpisode(mode="deployed", source_format="deployed", case_id=case.id,
                              stop_reason="error", errors=[err] + list(capture.messages),
                              metadata={"task": case.task},
                              counters={"latency_ms": (time.monotonic() - t0) * 1000})
        ep.trial_id = str(trial)
        return ep

    @staticmethod
    def _deployed_episode(case: AgentCase, result: Any, log_messages: list[str]) -> AgentEpisode:
        gen = result.completions[0] if result.completions else None
        finish = getattr(gen, "finish_reason", None)
        trace = (getattr(gen, "trace", None) or {}) if gen else {}
        messages = trace.get("messages")
        if messages:
            ep = importers.episode_from_openai_messages(
                messages, mode="deployed", case_id=case.id,
                final_answer=(gen.text if gen else None),
                retrieved_contexts=trace.get("retrieved_contexts"), source="deployed")
        else:
            ep = importers.episode_from_sample(
                _TraceCarrier(trace, gen.text if gen else None), mode="deployed", source="deployed")
            ep.case_id = case.id
        errors = [m for m in log_messages] if finish == "error" else []
        ep.errors = errors
        ep.stop_reason = finish
        ep.metadata.setdefault("task", case.task)
        ep.counters.update({
            "latency_ms": getattr(result, "latency_ms", None),
            "tokens": getattr(result, "usage", None) or {},
        })
        # Unavailable unless the reply carries it (see _attach_state), so a state
        # oracle stays 'unknown', never fabricated.
        ep.coverage.setdefault("final_state", UNAVAILABLE)
        return ep

    # -- harness-owned bounded loop (A4, AG-15/AG-16/AG-17) --------------
    def _run_harness(self, spec: AgentEvalSpec) -> tuple[list[AgentCaseResult], list[AgentEpisode], Any]:
        """AuditKit owns the turn loop: issue a step, read the policy's reply,
        run any tool calls through a caller-supplied test double, feed the results
        back, and stop on a stop condition or a hard ``max_steps`` bound.

        ``spec.agent`` is the *policy* model (any :class:`Model`, or a spec string
        ``AutoModel.resolve`` accepts). ``agent_opts`` may carry:
          - ``tool_env``: ``(name, arguments) -> result`` test double (AG-17: no
            arbitrary tool execution -- the caller supplies the doubles). If it
            exposes ``snapshot()`` or ``state``, the final environment state is
            captured so a ``FinalStateAssertion`` can verify a real effect.
          - ``max_steps``: the hard step bound (default 8).
        Remaining opts flow to ``AutoModel.resolve``.

        Seam: the deployed ``agent:`` endpoint runs its OWN loop and returns a
        finished reply, so against it the harness completes in one step. A true
        multi-step server loop needs an endpoint that returns a single assistant
        step and accepts intermediate tool results -- beyond the current
        single-shot ``agent:`` contract. That is documented, not faked.
        """
        from ..model import AutoModel

        if spec.agent is None:
            raise ValueError("harness mode needs spec.agent (a policy model, e.g. a Model "
                             "instance or 'openai:gpt-4o-mini')")
        opts = dict(spec.agent_opts)
        tool_env = opts.pop("tool_env", None)      # pop BEFORE resolve (unknown kwargs raise)
        max_steps = int(opts.pop("max_steps", 8))
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")
        model = AutoModel.resolve(spec.agent, **opts)
        ident = getattr(model, "identity", None)
        agent_identity: Any = ident() if callable(ident) else (
            spec.agent if isinstance(spec.agent, str) else type(model).__name__)
        agent_identity = {"policy": agent_identity, "loop": "harness", "max_steps": max_steps,
                          "tool_env": tool_env is not None}
        rows: list[AgentCaseResult] = []
        scored: list[AgentEpisode] = []
        for case in spec.cases:
            eps = [self._harness_episode(model, case, tool_env, max_steps, trial=t,
                                         gen_params=_generation_params(spec.config))
                   for t in range(spec.trials)]
            rows.append(self._finalize(spec, case, eps))
            scored.extend(eps)
        return rows, scored, agent_identity

    @staticmethod
    def _harness_episode(model: Any, case: AgentCase, tool_env: Any, max_steps: int,
                         *, trial: int = 0, gen_params: Optional[dict[str, Any]] = None) -> AgentEpisode:
        from ..model import Request

        messages: list[dict[str, Any]] = [{"role": "user", "content": case.task}]
        schemas = _tool_schemas(case.allowed_tools)
        calls_from_text = False
        final_answer: Optional[str] = None
        stop_reason = "stop"
        errors: list[str] = []
        steps = 0
        t0 = time.monotonic()
        try:
            for steps in range(1, max_steps + 1):
                params: dict[str, Any] = {**(gen_params or {}), "messages": list(messages)}
                if schemas:
                    params["tools"] = schemas
                result = model.generate([Request(prompt=case.task, request_type="chat", params=params)])[0]
                gen = result.completions[0] if result.completions else None
                text = getattr(gen, "text", None)
                trace = (getattr(gen, "trace", None) or {}) if gen else {}
                calls = _first_turn_calls(trace.get("tool_calls"))
                content = text
                bad: list[Any] = []
                if not calls and "tool_calls" not in trace and text:
                    # A text-output policy (hf:, vllm:, Cohere) writes its calls into the
                    # reply (Hermes <tool_call>, Cohere action lists, JSON): read them the
                    # way the metrics do, or the first call is mistaken for the answer.
                    parsed = parse_tool_calls(text)
                    if parsed:
                        calls = [{"name": c.name, "arguments": c.arguments}
                                 for c in parsed if not c.parse_error]
                        bad = [c for c in parsed if c.parse_error]
                        calls_from_text = True
                        content = strip_tool_calls(text)   # the calls live in tool_calls now
                if calls or bad:
                    ids = [f"s{steps}c{i}" for i in range(len(calls))]
                    messages.append({"role": "assistant", "content": content, "tool_calls": [
                        {"id": cid, "type": "function",
                         "function": {"name": c.get("name"),
                                      "arguments": json.dumps(c.get("arguments") or {}, default=str)}}
                        for cid, c in zip(ids, calls)]})
                    if bad:
                        # A truncated/unparseable call (e.g. an action list cut off by
                        # max_tokens) is recorded, never dropped or taken for the answer;
                        # the step it belongs to cannot be executed, so the loop stops.
                        errors.extend(f"step {steps}: unparseable tool call {c.name!r}: {c.parse_error}"
                                      for c in bad)
                        stop_reason = "parse_error"
                        break
                    if tool_env is None:      # no double -> cannot continue the loop safely
                        stop_reason = "no_tool_env"
                        final_answer = text
                        break
                    for cid, c in zip(ids, calls):
                        obs = tool_env(c.get("name"), c.get("arguments") or {})
                        messages.append({"role": "tool", "tool_call_id": cid,
                                         "content": obs if isinstance(obs, str) else json.dumps(obs, default=str)})
                    if steps == max_steps:
                        stop_reason = "max_steps"
                        break
                else:                          # no calls -> the policy is done
                    if text is not None:
                        messages.append({"role": "assistant", "content": text})
                    final_answer = text
                    stop_reason = "stop"
                    break
            else:
                stop_reason = "max_steps"
        except Exception as e:
            errors.append(f"harness step failed: {type(e).__name__}: {e}")
            stop_reason = "error"

        ep = importers.episode_from_openai_messages(
            messages, mode="harness", case_id=case.id, final_answer=final_answer,
            source="harness")
        if stop_reason == "parse_error":      # the unparseable reply is not an answer
            ep.final_answer = None
            ep.coverage["final_answer"] = UNAVAILABLE
        ep.trial_id = str(trial)
        ep.stop_reason = stop_reason
        ep.errors = errors
        ep.metadata.setdefault("task", case.task)
        if calls_from_text:
            ep.metadata["calls_from_text"] = True
        ep.counters.update({"n_turns": steps, "latency_ms": (time.monotonic() - t0) * 1000})
        # Genuine state verification (PRD workflow 3): read the environment double.
        state = _env_state(tool_env)
        if isinstance(state, dict):
            ep.final_state = state
            ep.coverage["final_state"] = OBSERVED
        else:
            ep.coverage.setdefault("final_state", UNAVAILABLE)
        return ep

    # -- scoring (shared by both modes) ----------------------------------
    def _score(self, spec: AgentEvalSpec, case: AgentCase,
               episode: Optional[AgentEpisode]) -> AgentCaseResult:
        if episode is None:
            return AgentCaseResult(
                case_id=case.id, status="not_applicable",
                outcome=outcome_mod.Outcome(outcome_mod.UNKNOWN,
                                            reason="no episode found for this case").to_dict(),
                coverage={}, coverage_label="none")

        # An endpoint error is authoritative (finish_reason=='error'): the run
        # produced no usable answer, so it is 'error', NOT a fabricated failure
        # scored on an empty string (release gate: errors never become zeros).
        if episode.stop_reason == "error":
            reason = "; ".join(episode.errors) or "agent endpoint error (finish_reason=error)"
            return AgentCaseResult(
                case_id=case.id, status="target_error",
                outcome=outcome_mod.Outcome(outcome_mod.ERROR, reason=reason).to_dict(),
                coverage=dict(episode.coverage), coverage_label=episode.coverage_label(),
                ineligible={s: "target_error" for s in spec.scorers},
                resource_use={"latency_ms": episode.counters.get("latency_ms"),
                              "tokens": episode.counters.get("tokens"),
                              "agent_call": spec.mode == "deployed"},
                stop_reason=episode.stop_reason, errors=list(episode.errors),
                source_format=episode.source_format, source_id=episode.source_id)

        sample = _sample_for(case)
        output = episode.final_answer or ""
        context = {"trace": episode.to_trace()}

        diagnostics: dict[str, Any] = {}
        ineligible: dict[str, str] = {}
        method_error: Optional[str] = None
        judge_dict: Optional[dict[str, Any]] = None
        judge_errored = False

        for name in spec.scorers:
            if name == "task_completion":
                if spec.judge is None:
                    ineligible[name] = "spec.judge"  # requested but no judge configured
                    continue
                judge_dict, judge_errored = self._run_judge(spec, case, episode)
                if judge_dict is not None:
                    diagnostics["task_completion"] = {
                        "value": judge_dict.get("evidence", {}).get("value"),
                        "reason": judge_dict.get("reason"),
                        "verdict": judge_dict.get("verdict"),
                        "diagnostic": True,
                    }
                continue
            missing = eligibility(name, case, episode)
            if missing is not None:
                ineligible[name] = missing
                continue
            try:
                scores = _make_scorer(name, spec).score(sample, output, context)
            except Exception as e:  # a scorer bug/edge is a method error, not a 0
                method_error = f"{name}: {e}"
                ineligible[name] = f"error: {e}"
                continue
            for sc in (scores if isinstance(scores, list) else [scores]):
                diagnostics[sc.name] = {"value": sc.value, "reason": sc.reason,
                                        "metadata": sc.metadata}

        # Headline outcome: deterministic oracle is authoritative; a judge only
        # becomes the (labeled diagnostic) headline when no oracle exists (AG-08).
        try:
            verified = outcome_mod.resolve_outcome(episode, case)
        except Exception as e:
            verified = outcome_mod.Outcome(outcome_mod.ERROR, reason=f"oracle raised: {e}")
        if verified.verdict == outcome_mod.ERROR:
            # Any oracle that errored (raised, or a CustomPredicate that caught its
            # own exception) is a method error, whichever oracle it was.
            method_error = method_error or f"oracle: {verified.reason}"
        if case.outcome is None and judge_dict is not None:
            headline = judge_dict  # diagnostic fallback, labeled
        else:
            headline = verified.to_dict()

        # AG-05: when a verified oracle exists, record (but never act on) any
        # judge/source verdict that DISAGREES with it. The oracle already won the
        # headline; the disagreement is surfaced for review, not silently dropped.
        disagreement = _disagreement(case, verified, judge_dict, episode.source_verdict)

        status = self._status(case, episode, headline, ineligible, spec.scorers,
                              method_error, judge_errored)
        return AgentCaseResult(
            case_id=case.id,
            status=status,
            outcome=headline,
            disagreement=disagreement,
            coverage=dict(episode.coverage),
            coverage_label=episode.coverage_label(),
            diagnostics=diagnostics,
            ineligible=ineligible,
            resource_use={
                "latency_ms": episode.counters.get("latency_ms"),
                "tokens": episode.counters.get("tokens"),
                "n_tool_calls": episode.counters.get("n_tool_calls"),
                "agent_call": spec.mode == "deployed",
            },
            provenance={
                "source_reward": episode.source_reward,
                "source_verdict": episode.source_verdict,
                "source_scores": episode.source_scores,
            },
            judge=judge_dict,
            stop_reason=episode.stop_reason,
            errors=list(episode.errors),
            source_format=episode.source_format,
            source_id=episode.source_id,
        )

    def _run_judge(self, spec: AgentEvalSpec, case: AgentCase,
                   episode: AgentEpisode) -> tuple[Optional[dict[str, Any]], bool]:
        judge = self._judge_instance(spec)
        if judge is None:
            return None, False
        oc = outcome_mod.judge_outcome(episode, case, judge)
        return oc.to_dict(), oc.verdict == outcome_mod.ERROR

    @staticmethod
    def _judge_instance(spec: AgentEvalSpec) -> Any:
        if spec.judge is None:
            return None
        if isinstance(spec.judge, str):
            from ..metrics.agent import TaskCompletion
            return TaskCompletion(judge_model=spec.judge)
        return spec.judge

    @staticmethod
    def _status(case: AgentCase, episode: AgentEpisode, headline: dict[str, Any],
                ineligible: dict[str, str], scorers: list[str],
                method_error: Optional[str], judge_errored: bool) -> str:
        # finish_reason=='error' from the endpoint is the authoritative signal,
        # independent of whether the warning text was captured.
        if episode.stop_reason == "error":
            return "target_error"
        if method_error:
            return "method_error"
        # A4: the harness loop stopped because it hit its step bound.
        if episode.stop_reason == "max_steps" or _budget_exhausted(case, episode):
            return "budget_exhausted"
        if case.outcome is None and judge_errored:
            return "judge_error"
        non_judge = [s for s in scorers if s != "task_completion"]
        has_oracle = case.outcome is not None
        all_ineligible = bool(non_judge) and all(s in ineligible for s in non_judge)
        if not has_oracle and all_ineligible and "task_completion" not in scorers:
            return "not_applicable"
        if all_ineligible and not has_oracle:
            return "ineligible"
        return "completed"


# --------------------------------------------------------------------------
# AG-05 disagreement + A4 harness helpers
# --------------------------------------------------------------------------
def _norm_verdict(v: Any) -> Optional[str]:
    """Map an imported source verdict to success/failure/None (unrecognized)."""
    if isinstance(v, bool):
        return outcome_mod.SUCCESS if v else outcome_mod.FAILURE
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("success", "succeeded", "pass", "passed", "true", "1", "ok", "correct"):
        return outcome_mod.SUCCESS
    if s in ("failure", "failed", "fail", "false", "0", "incorrect", "wrong"):
        return outcome_mod.FAILURE
    return None


def _disagreement(case: AgentCase, verified: Any, judge_dict: Optional[dict[str, Any]],
                  source_verdict: Any) -> dict[str, Any]:
    """AG-05: record a judge/source verdict that DISAGREES with a decided oracle.

    Only when the case has an oracle AND the oracle is decided (success/failure).
    An ``unknown``/``error`` oracle is missing evidence, not disagreement. The
    oracle already won the headline; this is recorded, never acted on.
    """
    if getattr(case, "outcome", None) is None:
        return {}
    if verified.verdict not in (outcome_mod.SUCCESS, outcome_mod.FAILURE):
        return {}
    dis: dict[str, Any] = {}
    kinds: dict[str, int] = {}
    jv = (judge_dict or {}).get("verdict")
    if jv in (outcome_mod.SUCCESS, outcome_mod.FAILURE) and jv != verified.verdict:
        dis["judge_vs_oracle"] = {"oracle": verified.verdict, "judge": jv,
                                  "resolution": "verified oracle wins; judge kept as diagnostic"}
        kinds["judge_vs_oracle"] = 1
    sv = _norm_verdict(source_verdict)
    if sv in (outcome_mod.SUCCESS, outcome_mod.FAILURE) and sv != verified.verdict:
        dis["source_vs_oracle"] = {"oracle": verified.verdict, "source": source_verdict,
                                   "source_verdict": sv,
                                   "resolution": "verified oracle wins; source kept as provenance"}
        kinds["source_vs_oracle"] = 1
    if dis:
        dis.update(n_trials=1, n_disagree=1, rate=1.0, kinds=kinds)
    return dis


_GEN_FIELDS = ("max_tokens", "temperature", "top_p", "top_k", "seed", "stop_sequences")


def _generation_params(config: Any) -> dict[str, Any]:
    """The generation fields of a RunConfig (or a plain dict) that are set, plus its
    ``chat_template_kwargs`` (e.g. Qwen3's ``{"enable_thinking": False}``), which reach
    every harness request's chat template the way the Runner sends them."""
    if config is None:
        return {}
    get = config.get if isinstance(config, dict) else (lambda k: getattr(config, k, None))
    out = {k: get(k) for k in _GEN_FIELDS if get(k) is not None}
    if get("chat_template_kwargs"):
        out["chat_template_kwargs"] = dict(get("chat_template_kwargs"))
    return out


def _first_turn_calls(tool_calls: Any) -> list[dict[str, Any]]:
    """The calls of one harness step: the first turn of a turns list, or a flat
    call list, each normalized to ``{"name", "arguments"}``."""
    if not tool_calls or not isinstance(tool_calls, list):
        return []
    turn = tool_calls[0] if isinstance(tool_calls[0], list) else tool_calls
    out: list[dict[str, Any]] = []
    for c in turn:
        if not isinstance(c, dict):
            continue
        if isinstance(c.get("function"), dict):
            fn = c["function"]
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {"_raw": args}
            out.append({"name": fn.get("name"), "arguments": args or {}})
        else:
            out.append({"name": c.get("name"), "arguments": c.get("arguments") or {}})
    return out


def _env_state(tool_env: Any) -> Any:
    """The environment double's final state, via ``snapshot()`` or a ``state`` attr."""
    if tool_env is None:
        return None
    snap = getattr(tool_env, "snapshot", None)
    if callable(snap):
        try:
            return snap()
        except Exception:
            return None
    return getattr(tool_env, "state", None)


def _budget_exhausted(case: AgentCase, episode: AgentEpisode) -> bool:
    b = case.budgets or {}
    counters = episode.counters or {}
    n_calls = counters.get("n_tool_calls")
    if b.get("max_tool_calls") is not None and isinstance(n_calls, int) and n_calls > b["max_tool_calls"]:
        return True
    n_turns = counters.get("n_turns")
    if b.get("max_turns") is not None and isinstance(n_turns, int) and n_turns > b["max_turns"]:
        return True
    lat = counters.get("latency_ms")
    if b.get("max_seconds") is not None and isinstance(lat, (int, float)) and lat > b["max_seconds"] * 1000:
        return True
    return False


def _call_with_deadline(model: Any, req: Any, capture: _LogCapture,
                        deadline: Optional[float]) -> tuple[Any, Optional[str]]:
    """``(result, error)`` for one agent call, bounded by ``deadline``.

    urllib's ``timeout`` is per socket operation, so a byte-trickling reply can
    outlive it; this is the overall wall-clock bound. Any exception from the
    model (or an odd reply it chokes on) fails only this case.
    """
    box: dict[str, Any] = {}

    def work() -> None:
        capture.thread_id = threading.get_ident()
        try:
            box["result"] = model.generate([req])[0]
        except Exception as e:
            box["error"] = f"agent call failed: {type(e).__name__}: {e}"

    # ponytail: an overrun worker is abandoned (daemon) and finishes its socket
    # read in the background; a cancellable transport would need model/ changes.
    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(deadline)
    if t.is_alive():
        return None, f"agent call exceeded the {deadline}s deadline (agent_opts timeout)"
    return box.get("result"), box.get("error")


def _attach_state(ep: AgentEpisode, result: Any) -> None:
    """Copy endpoint-supplied final_state / artifacts (dicts) onto the episode."""
    completions = getattr(result, "completions", None) or []
    trace = (getattr(completions[0], "trace", None) or {}) if completions else {}
    for attr in ("final_state", "artifacts"):
        val = trace.get(attr)
        if isinstance(val, dict):
            setattr(ep, attr, val)
            if attr == "final_state":
                ep.coverage["final_state"] = OBSERVED


class _TraceCarrier:
    """A minimal stand-in so ``episode_from_sample`` can read a deployed trace."""

    def __init__(self, trace: dict[str, Any], output: Optional[str]) -> None:
        self.actual_trace = trace
        self.actual_output = output
        self.id = None
        self.input = None
        self.target = None


# --------------------------------------------------------------------------
# offline rescore (AG-06): score saved episodes with NO agent/tool call
# --------------------------------------------------------------------------
def rescore(episodes: list[AgentEpisode], scorers: list[str], *,
            cases: Optional[list[AgentCase]] = None, judge: Any = None) -> AgentEvalResult:
    """Score already-recorded episodes with chosen scorers, no agent rerun."""
    spec = AgentEvalSpec(
        cases=cases or [], mode="recorded", episodes=episodes,
        scorers=list(scorers), judge=judge)
    return AgentEvalRunner().run(spec)
