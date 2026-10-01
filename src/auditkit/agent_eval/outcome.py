"""Outcome oracles: the verified task result (A2, AG-07/AG-08).

The headline result is a **verified outcome** with verdict ``success``,
``failure``, ``unknown`` or ``error``. Most oracles are deterministic
(final-state, artifact, answer, custom predicate). Two paths use an LLM:

- ``judge_outcome`` (the ``TaskCompletion`` judge) is a diagnostic/fallback
  signal only -- ``diagnostic=True``; a parse failure is ``unknown`` and it
  never overrides a verified state assertion (AG-08).
- :class:`AssertionOracle` is a **first-class** oracle for the no-gold-standard
  case (reviewer ask (c)): a held-out judge decides a human-authored natural
  -language criterion against the transcript/answer, with no gold trajectory or
  answer. Its verdict IS the outcome (``diagnostic=False``). It can carry a
  companion state/artifact oracle that still takes precedence when it verifies a
  real effect -- the "state wins over a judge" rule (AG-08) preserved.

Missing evidence is ``unknown`` -- never fabricated into a pass or a fail (AG-07).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

SUCCESS = "success"
FAILURE = "failure"
UNKNOWN = "unknown"
ERROR = "error"


@dataclass
class Outcome:
    """A verified task result with provenance."""

    verdict: str
    reason: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    source: Optional[dict[str, Any]] = None  # oracle identity
    diagnostic: bool = False  # True when produced by a judge fallback

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "evidence": self.evidence,
            "source": self.source,
            "diagnostic": self.diagnostic,
        }


def _norm(text: Any) -> str:
    """Normalized comparison: lowercase, strip surrounding punctuation/space.
    Thousands separators between digits are dropped first, so "1,000" == "1000"."""
    s = "" if text is None else str(text)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    return re.sub(r"[\s$,%]+", " ", s.lower()).strip().strip(".")


class Oracle:
    """Base class: a deterministic check with a reproducible identity."""

    type = "oracle"
    version = "1"

    def identity(self) -> dict[str, Any]:
        return {"type": self.type, "version": self.version}

    def evaluate(self, episode: Any, case: Any = None) -> Outcome:  # pragma: no cover
        raise NotImplementedError


def _check_value(actual: Any, equals: Any, contains: Any) -> tuple[bool, str]:
    if equals is not None:
        return actual == equals, f"value={actual!r} equals {equals!r}"
    if contains is not None:
        ok = contains in actual if isinstance(actual, (str, list, dict, tuple)) else False
        return ok, f"value={actual!r} contains {contains!r}"
    return actual is not None, f"value present: {actual!r}"


class FinalStateAssertion(Oracle):
    """Assert a key of the episode's verified final state (AG-07).

    A recorded episode with NO state snapshot is scored ``unknown`` -- the
    absence of state is never read as a failure.
    """

    type = "final_state_assertion"

    def __init__(self, key: str, *, equals: Any = None, contains: Any = None,
                 version: str = "1") -> None:
        self.key = key
        self.equals = equals
        self.contains = contains
        self.version = version

    def identity(self) -> dict[str, Any]:
        return {"type": self.type, "version": self.version, "key": self.key,
                "equals": self.equals, "contains": self.contains}

    def evaluate(self, episode: Any, case: Any = None) -> Outcome:
        state = getattr(episode, "final_state", None)
        if not isinstance(state, dict):
            return Outcome(UNKNOWN, reason="no final-state snapshot recorded; "
                           "cannot verify a state assertion (evidence missing)",
                           source=self.identity(), evidence={"final_state": None})
        if self.key not in state:
            return Outcome(UNKNOWN, reason=f"final state has no key {self.key!r}",
                           source=self.identity(), evidence={"keys": sorted(state)})
        ok, detail = _check_value(state[self.key], self.equals, self.contains)
        return Outcome(SUCCESS if ok else FAILURE, reason=f"final_state[{self.key!r}]: {detail}",
                       source=self.identity(), evidence={"final_state": {self.key: state[self.key]}})


class ArtifactAssertion(Oracle):
    """Assert a key of the episode's produced artifacts (key equals/contains)."""

    type = "artifact_assertion"

    def __init__(self, key: str, *, equals: Any = None, contains: Any = None,
                 version: str = "1") -> None:
        self.key = key
        self.equals = equals
        self.contains = contains
        self.version = version

    def identity(self) -> dict[str, Any]:
        return {"type": self.type, "version": self.version, "key": self.key,
                "equals": self.equals, "contains": self.contains}

    def evaluate(self, episode: Any, case: Any = None) -> Outcome:
        artifacts = getattr(episode, "artifacts", None)
        if not isinstance(artifacts, dict):
            return Outcome(UNKNOWN, reason="no artifact evidence recorded (evidence missing)",
                           source=self.identity(), evidence={"artifacts": None})
        if self.key not in artifacts:
            return Outcome(UNKNOWN, reason=f"artifacts have no key {self.key!r}",
                           source=self.identity(), evidence={"keys": sorted(artifacts)})
        ok, detail = _check_value(artifacts[self.key], self.equals, self.contains)
        return Outcome(SUCCESS if ok else FAILURE, reason=f"artifacts[{self.key!r}]: {detail}",
                       source=self.identity(), evidence={"artifacts": {self.key: artifacts[self.key]}})


class AnswerAssertion(Oracle):
    """Assert the final answer against a reference (exact/normalized/contains).

    Weaker evidence than a state check: use it when the task is answered in text.
    When the case requires an external action, a state/artifact oracle takes
    precedence (see :func:`resolve_outcome`).
    """

    type = "answer_assertion"
    MODES = ("exact", "normalized", "contains")

    def __init__(self, reference: Any = None, *, mode: str = "normalized",
                 version: str = "1") -> None:
        if mode not in self.MODES:
            raise ValueError(f"AnswerAssertion mode must be one of {self.MODES}, got {mode!r}")
        self.reference = reference
        self.mode = mode
        self.version = version

    def identity(self) -> dict[str, Any]:
        return {"type": self.type, "version": self.version, "mode": self.mode,
                "reference": self.reference}

    def evaluate(self, episode: Any, case: Any = None) -> Outcome:
        ref = self.reference
        if ref is None and case is not None:
            ref = (case.metadata or {}).get("target")
        answer = getattr(episode, "final_answer", None)
        if ref is None:
            return Outcome(UNKNOWN, reason="no reference answer to check against",
                           source=self.identity())
        if answer is None:
            return Outcome(UNKNOWN, reason="episode has no final answer",
                           source=self.identity())
        if self.mode == "exact":
            ok = str(answer) == str(ref)
        elif self.mode == "contains":
            # token boundaries: "5" is not found inside "15" or "0.5"
            ok = re.search(rf"(?<![\w.]){re.escape(_norm(ref))}(?!\w)", _norm(answer)) is not None
        else:
            ok = _norm(answer) == _norm(ref)
        return Outcome(SUCCESS if ok else FAILURE,
                       reason=f"answer {self.mode} match: {ok}",
                       source=self.identity(),
                       evidence={"answer": str(answer)[:500], "reference": str(ref)[:500]})


class CustomPredicate(Oracle):
    """A caller-supplied ``(episode, case) -> bool | str`` check.

    Requires an explicit ``version`` for reproducibility (AG-07): a run cannot
    be described as reproducible if the predicate has no identity.
    """

    type = "custom_predicate"

    def __init__(self, fn: Callable[[Any, Any], Any], *, version: str,
                 name: Optional[str] = None) -> None:
        if not version or not str(version).strip():
            raise ValueError("CustomPredicate requires an explicit non-empty version= "
                             "so the run can be reproduced")
        self.fn = fn
        self.version = str(version)
        from auditkit.model import _default_callable_name
        self.name = name or _default_callable_name(fn)

    def identity(self) -> dict[str, Any]:
        return {"type": self.type, "version": self.version, "fn": self.name}

    def evaluate(self, episode: Any, case: Any = None) -> Outcome:
        try:
            result = self.fn(episode, case)
        except Exception as e:  # a broken predicate is an error, not a failure
            return Outcome(ERROR, reason=f"custom predicate raised: {e}", source=self.identity())
        if isinstance(result, str):
            # Strict enum: a near-miss like 'fail' or 'FAILURE' is not truthy
            # success, it is an unrecognized verdict.
            if result in (SUCCESS, FAILURE, UNKNOWN, ERROR):
                return Outcome(result, reason=f"custom predicate -> {result}", source=self.identity())
            return Outcome(UNKNOWN, reason=f"custom predicate returned unrecognized verdict "
                           f"string {result[:100]!r}; expected one of "
                           f"{(SUCCESS, FAILURE, UNKNOWN, ERROR)} or a bool",
                           source=self.identity())
        if result is None:
            return Outcome(UNKNOWN, reason="custom predicate returned None", source=self.identity())
        return Outcome(SUCCESS if result else FAILURE,
                       reason=f"custom predicate -> {bool(result)}", source=self.identity())


class AssertionOracle(Oracle):
    """A held-out judge on a human-authored criterion -- the no-gold-standard case.

    First-class (``diagnostic=False``, unlike :func:`judge_outcome`): its verdict
    IS the outcome. The judge decides a natural-language ``criterion`` (strands
    ``expected_assertion``) against the episode's transcript and final answer,
    with NO gold trajectory or reference answer.

    ``criterion`` may be ``None`` and read at evaluation time from
    ``case.metadata[metadata_key]`` (default ``"assertion"``), so a user's custom
    reference field feeds the oracle (reviewer ask (b)). ``judge_model`` is a spec
    string (resolved lazily) or any object with ``generate``.

    ``state_oracle`` is an optional companion deterministic oracle: it runs first,
    and if it returns a *verified* ``success``/``failure`` its Outcome is returned
    unchanged (a verified state assertion still overrides a judge, AG-08). On
    ``unknown`` (no evidence recorded) evaluation falls through to the judge; on
    ``error`` the error is returned (a broken check is not silently ignored).
    """

    type = "assertion"

    def __init__(self, criterion: Optional[str] = None, *, judge_model: Any = None,
                 state_oracle: Optional[Oracle] = None, metadata_key: str = "assertion",
                 version: str = "1", name: str = "assertion") -> None:
        self.criterion = criterion
        self.judge_model = judge_model
        self.state_oracle = state_oracle
        self.metadata_key = metadata_key
        self.version = version
        self.name = name
        self._judge: Any = None

    def identity(self) -> dict[str, Any]:
        from auditkit.metrics.rag_judge import judge_identity
        return {"type": self.type, "version": self.version, "name": self.name,
                "criterion": self.criterion, "metadata_key": self.metadata_key,
                "judge_model": judge_identity(self.judge_model),
                "state_oracle": self.state_oracle.identity() if self.state_oracle else None}

    def _criterion_for(self, case: Any) -> Optional[str]:
        if self.criterion is not None:
            return self.criterion
        meta = getattr(case, "metadata", None) or {}
        return meta.get(self.metadata_key)

    def _build_judge(self, criterion: str) -> Any:
        if self._judge is None:
            from auditkit.metrics.judge import LLMJudge, _JUDGE_SYSTEM_PROMPT
            prompt = ("<task>\n{input}\n</task>\n\n"
                      "<agent_trajectory>\n{trajectory}\n</agent_trajectory>\n\n"
                      "<final_answer>\n{output}\n</final_answer>\n\n"
                      "Judge ONLY this criterion against the trajectory and final "
                      "answer:\n<criterion>\n{criterion}\n</criterion>")
            self._judge = LLMJudge(
                judge_model=self.judge_model, name=self.name, prompt=prompt,
                use_cot=True, system_prompt=_JUDGE_SYSTEM_PROMPT,
                choices={"yes": 1.0, "no": 0.0})
        return self._judge

    def evaluate(self, episode: Any, case: Any = None) -> Outcome:
        if self.state_oracle is not None:
            verified = self.state_oracle.evaluate(episode, case)
            if verified.verdict in (SUCCESS, FAILURE, ERROR):
                return verified  # a verified state/artifact effect wins over the judge

        criterion = self._criterion_for(case)
        if not criterion:
            return Outcome(UNKNOWN, reason="no assertion criterion given (pass criterion= or "
                           f"set case.metadata[{self.metadata_key!r}])", source=self.identity())

        from auditkit.sample import Sample
        from auditkit.trace import predicted_turns, render_trace
        task = case.task if case is not None else (episode.metadata.get("task") or "")
        answer = episode.final_answer or ""
        # A Sample with NO metadata: _render fills placeholders from metadata first,
        # so a column literally named 'trajectory'/'criterion' would pre-empt ours.
        sample = Sample(input=task)
        judge = self._build_judge(criterion)
        trace = episode.to_trace()
        transcript = render_trace(predicted_turns(answer, {"trace": trace}), trace)
        user = (judge._render(sample, answer)
                .replace("{trajectory}", transcript)
                .replace("{criterion}", criterion))
        from auditkit.model import Request
        try:
            results = judge._model().generate([Request(prompt=judge._assemble(user),
                                                       params=dict(judge._gen_params))])
        except Exception as e:  # a broken judge model is an error, not a failure
            return Outcome(ERROR, reason=f"assertion judge raised: {e}", source=self.identity())
        text = results[0].completions[0].text if results and results[0].completions else ""
        score = judge._parse(text or "")
        ident = self.identity()
        if score.metadata.get("unknown"):
            return Outcome(UNKNOWN, reason=f"judge parse failure: {score.reason}",
                           source=ident, evidence={"parse_status": "failed", "raw": score.metadata.get("raw")})
        verdict = SUCCESS if score.value > 0.5 else FAILURE
        return Outcome(verdict, reason=score.reason or f"assertion judge -> {score.value}",
                       source=ident, evidence={"criterion": criterion, "choice": score.metadata.get("choice"),
                                               "value": score.value})


# Deterministic oracles that require an actual external effect (not just text).
_STATE_ORACLES = (FinalStateAssertion, ArtifactAssertion)


def from_spec(spec: Any) -> Optional[Oracle]:
    """Build an oracle from a plain dict spec (used by the CLI config).

    ``{"type": "final_state_assertion", "key": ..., "equals": ...}`` etc.
    A custom predicate cannot be built from JSON (no callable) -- pass an
    ``Oracle`` instance directly in Python for that.
    """
    if spec is None or isinstance(spec, Oracle):
        return spec
    if not isinstance(spec, dict):
        raise ValueError(f"outcome spec must be a dict or Oracle, got {type(spec).__name__}")
    kind = spec.get("type")
    common = {k: spec[k] for k in ("version",) if k in spec}
    if kind == "final_state_assertion":
        return FinalStateAssertion(spec["key"], equals=spec.get("equals"),
                                   contains=spec.get("contains"), **common)
    if kind == "artifact_assertion":
        return ArtifactAssertion(spec["key"], equals=spec.get("equals"),
                                 contains=spec.get("contains"), **common)
    if kind == "answer_assertion":
        return AnswerAssertion(spec.get("reference"), mode=spec.get("mode", "normalized"), **common)
    if kind == "assertion":
        return AssertionOracle(
            spec.get("criterion"), judge_model=spec.get("judge_model"),
            state_oracle=from_spec(spec.get("state_oracle")),
            metadata_key=spec.get("metadata_key", "assertion"),
            name=spec.get("name", "assertion"), **common)
    raise ValueError(f"unknown or non-JSON outcome type {kind!r}; known: "
                     "final_state_assertion, artifact_assertion, answer_assertion, assertion "
                     "(custom_predicate must be passed as an Oracle instance)")


def judge_outcome(episode: Any, case: Any, judge: Any) -> Outcome:
    """Run a ``TaskCompletion``-style judge as a DIAGNOSTIC signal (AG-08).

    Returns an :class:`Outcome` carrying the judge's model identity, reason and
    parse status. A parse failure becomes ``unknown``; the caller never lets a
    judge override a verified state assertion.
    """
    from auditkit.sample import Sample
    ident: dict[str, Any] = {"type": "judge", "name": getattr(judge, "name", "task_completion")}
    model_ident = getattr(getattr(judge, "_model", lambda: None)(), "identity", None)
    if callable(model_ident):
        try:
            ident["judge_model"] = model_ident()
        except Exception:
            pass
    sample = Sample(input=case.task if case else (episode.metadata.get("task") or ""),
                    target=(case.metadata.get("target") if case else None))
    context = {"trace": episode.to_trace()}
    try:
        # Prefer judge.judge() (returns an unknown-flagged Score on an unparseable
        # reply, raises only on a real backend error) over judge.score(), which
        # now raises on an unparseable verdict -- keeps the unknown/error split.
        _judge_fn = getattr(judge, "judge", None) or judge.score
        score = _judge_fn(sample, episode.final_answer or "", context)
    except Exception as e:
        return Outcome(ERROR, reason=f"judge raised: {e}", source=ident, diagnostic=True)
    parse_failed = bool(score.metadata.get("unknown"))
    if parse_failed:
        return Outcome(UNKNOWN, reason=f"judge parse failure: {score.reason}",
                       source=ident, diagnostic=True,
                       evidence={"parse_status": "failed", "raw": score.metadata.get("raw")})
    # A 'partial' judgement (0.5) is not a success -- only a clear 'complete' is.
    verdict = SUCCESS if score.value > 0.5 else FAILURE
    ident["value"] = score.value
    return Outcome(verdict, reason=score.reason or "", source=ident, diagnostic=True,
                   evidence={"parse_status": "ok", "value": score.value,
                             "choice": score.metadata.get("choice")})


def resolve_outcome(episode: Any, case: Any) -> Outcome:
    """The headline verified outcome for one (episode, case).

    Runs the case's oracle and returns its verdict directly. A state/artifact
    oracle is authoritative and takes precedence over answer text (AG-08). An
    :class:`AssertionOracle` is a first-class outcome, not a diagnostic judge:
    its verdict IS the outcome when no state/artifact/answer oracle is present
    (reviewer ask (c)) -- and when such an oracle is set as its ``state_oracle``
    companion, that verified state assertion still overrides the judge. No oracle
    at all -> ``unknown``.
    """
    oracle = getattr(case, "outcome", None) if case is not None else None
    if oracle is None:
        return Outcome(UNKNOWN, reason="no outcome oracle defined for this case",
                       source=None)
    return oracle.evaluate(episode, case)


def requires_external_action(case: Any) -> bool:
    """True when the case's oracle checks a real state/artifact effect.

    Includes an :class:`AssertionOracle` whose companion ``state_oracle`` is a
    state/artifact check (its verified verdict overrides the judge).
    """
    oracle = getattr(case, "outcome", None)
    if isinstance(oracle, _STATE_ORACLES):
        return True
    return isinstance(getattr(oracle, "state_oracle", None), _STATE_ORACLES)
