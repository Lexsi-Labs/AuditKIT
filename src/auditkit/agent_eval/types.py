"""Versioned records for end-to-end agent evaluation (A1 of the AgentTune bridge).

An :class:`AgentCase` is what to evaluate (task, allowed tools, budgets, outcome
oracle). An :class:`AgentEpisode` is one complete recorded/deployed run: an
ordered list of :class:`AgentEvent` plus final answer, evidence, counters and
loss-aware ``coverage``. Everything here is stdlib-only and serializes to strict
JSON so a reviewer can reconstruct the task, evidence and outcome from one file.

Design notes (PRD sections 5-7):
- A missing timestamp is ``None``; monotonic times are never fabricated (AG-02).
- ``coverage`` marks each field ``observed`` / ``inferred`` / ``unavailable`` so
  no missing argument or state is filled with a fabricated default (AG-03).
- ``source_reward`` / ``source_verdict`` / ``source_scores`` are provenance only
  and are never promoted to the AuditKit outcome (AG-05).
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Union

from ..trace import decode_reference, is_call_payload

SCHEMA_VERSION = "agent_eval/1"

# Coverage markers (AG-03).
OBSERVED = "observed"
INFERRED = "inferred"
UNAVAILABLE = "unavailable"

MODES = ("recorded", "deployed", "harness")

# The coverage keys every episode reports on. A summary label (full /
# tool_names_only / answer_only) is derived from these (see ``coverage_label``).
COVERAGE_FIELDS = (
    "final_answer",
    "tool_call_names",
    "tool_call_arguments",
    "tool_results",
    "call_result_pairing",
    "parallel_grouping",
    "timestamps",
    "retrieved_contexts",
    "final_state",
)


def _canonical(obj: Any) -> str:
    """Stable JSON for digests: sorted keys, no whitespace, str fallback."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def strict_json_value(obj: Any) -> Any:
    """``obj`` with non-finite floats (NaN/inf) replaced by ``None``.

    Python's ``json`` writes NaN as a bare ``NaN`` token that strict parsers
    (jq, ``JSON.parse``) reject; an imported NaN reward must not poison the
    whole run file. Anything else passes through unchanged.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: strict_json_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [strict_json_value(v) for v in obj]
    return obj


def dumps_strict(obj: Any, **kw: Any) -> str:
    """``json.dumps`` that always emits strict JSON (see :func:`strict_json_value`)."""
    return json.dumps(strict_json_value(obj), allow_nan=False, default=str, **kw)


REDACTED = "[REDACTED]"

# --------------------------------------------------------------------------
# Automatic secret detection (UX-A9). Conservative: flag + redact, never
# silently pass a leaked secret. Two halves --
#   1. Strong value patterns (below): an unmistakable secret is redacted
#      everywhere, in any string, regardless of the key it sits under.
#   2. A generic high-entropy heuristic, which is the false-positive-prone half,
#      so it is DISABLED for AuditKit's reproducibility fields (digests, ids)
#      and never fires on a hash or a UUID (see ``_SKIP_KEYS`` / ``_is_repro``).
# A dict key whose NAME looks like a credential redacts its whole value, numbers
# included (a PIN or OTP is a number); only bool/None and numbers under a
# count-like key (``password_min_length``, ``retries``, ``ttl``; see
# ``DEFAULT_COUNT_KEYS``) are kept. False positives: pass ``allow=[...]`` to
# whitelist a specific token, or use exact ``keys=``/``secrets=`` and omit
# ``auto`` if the heuristic is too eager for your data.
# --------------------------------------------------------------------------
_SECRET_KEY_RE = re.compile(
    r"(?i)(api[_-]?key|secret|access[_-]?key|client[_-]?secret|auth[_-]?token|"
    r"authorization|password|passwd|credential|private[_-]?key|session[_-]?token|"
    r"bearer|\btoken\b|(?<![A-Za-z])(?:pin|otp)(?![A-Za-z]))")

# Glob patterns (case-insensitive) for keys whose NUMBER under a credential key is
# telemetry, not a credential; ``redact(count_keys=...)`` overrides them.
DEFAULT_COUNT_KEYS = ("*_count", "*_length", "*_len", "*_size", "*_tokens",
                      "min_*", "max_*", "retries", "*_retries", "ttl", "*_ttl",
                      "timeout", "*_timeout", "expires_in", "*_seconds", "*_ms")

_VALUE_PATTERNS = [
    ("openai_key", re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}")),
    ("github_token", re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}")),
    ("slack_token", re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}")),
    ("aws_access_key_id", re.compile(r"(?:AKIA|ASIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA)[0-9A-Z]{16}")),
    ("google_api_key", re.compile(r"AIza[0-9A-Za-z_-]{35}")),
    ("private_key", re.compile(r"-----BEGIN[^\n]*PRIVATE KEY-----[\s\S]*?-----END[^\n]*PRIVATE KEY-----")),
    ("private_key", re.compile(r"-----BEGIN[^\n]*PRIVATE KEY-----")),
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}")),
    # a token shape (>= 16 chars with a digit), so prose like "bearer instruments" is kept
    ("bearer_token", re.compile(r"(?i)bearer\s+(?=[A-Za-z0-9._+/=-]*\d)[A-Za-z0-9._+/=-]{16,}")),
]

# Keys whose values are AuditKit reproducibility identity, never scanned by the
# fuzzy generic heuristic (strong patterns above still apply to them).
# Only true reproducibility-identity names skip the generic-entropy heuristic
# (strong patterns and the _is_repro value check still apply everywhere). Generic
# names like "key"/"version"/"index" are NOT skipped -- a secret can hide there.
_SKIP_KEYS = frozenset({"digest", "headers_digest", "schema_version",
                        "case_id", "source_id", "trial_id", "call_id", "turn_id"})
_TOKEN_RE = re.compile(r"[A-Za-z0-9+/_-]{32,}={0,2}")
_HEX_RE = re.compile(r"(?i)\A[0-9a-f]+\Z")
_UUID_RE = re.compile(r"(?i)\A[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z")


def _is_secret_keyname(key: Any) -> bool:
    return isinstance(key, str) and _SECRET_KEY_RE.search(key) is not None


def _is_repro(tok: str) -> bool:
    """A hash or UUID -- never a leaked credential, and load-bearing for repro."""
    if _UUID_RE.match(tok):
        return True
    return bool(_HEX_RE.match(tok)) and len(tok) in (32, 40, 64)


def _shannon(s: str) -> float:
    from collections import Counter
    n = len(s)
    if n <= 1:
        return 0.0
    counts = Counter(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _high_entropy(tok: str) -> bool:
    if _is_repro(tok):
        return False
    has_alpha = any(c.isalpha() for c in tok)
    has_digit = any(c.isdigit() for c in tok)
    return len(tok) >= 32 and has_alpha and has_digit and _shannon(tok) >= 4.0


def _scan_string(text: str, allow: frozenset, skip_generic: bool) -> tuple[str, list[str]]:
    """``(redacted_text, kinds)`` for one string under the auto detector."""
    kinds: list[str] = []
    for kind, rx in _VALUE_PATTERNS:
        def _repl(m: Any, _kind: str = kind) -> str:
            if m.group(0) in allow:
                return m.group(0)
            kinds.append(_kind)
            return REDACTED
        text = rx.sub(_repl, text)
    if not skip_generic:
        def _grepl(m: Any) -> str:
            tok = m.group(0)
            if tok in allow or not _high_entropy(tok):
                return tok
            kinds.append("high_entropy")
            return REDACTED
        text = _TOKEN_RE.sub(_grepl, text)
    return text, kinds


def redact(obj: Any, *, secrets: Any = (), keys: Any = (),
           auto: bool = False, allow: Any = (),
           count_keys: Any = DEFAULT_COUNT_KEYS) -> Any:
    """A copy of ``obj`` safe to share (UX-A9).

    Every occurrence of a string in ``secrets`` inside any string value (or
    dict key) becomes ``[REDACTED]``, and the whole value of any dict key named
    in ``keys`` (case-insensitive, e.g. ``"api_key"``, ``"authorization"``) is
    replaced. With ``auto=True`` it ALSO auto-detects common secrets (OpenAI/
    GitHub/Slack/Google keys, AWS access-key ids, bearer tokens, JWTs, PEM
    private keys, and high-entropy tokens) and a credential-looking key name --
    conservatively (see the module notes). ``allow=[...]`` whitelists specific
    tokens. Under a credential key a number is redacted too, unless its own key
    matches a ``count_keys`` glob (``*_length``, ``retries``, ...). Apply it to
    ``to_dict()`` output before writing a shareable file.
    """
    secrets = sorted({str(x) for x in secrets if x}, key=len, reverse=True)
    lkeys = {str(k).lower() for k in keys}
    allow = frozenset(str(a) for a in allow)

    def _s(text: str) -> str:
        for sec in secrets:
            text = text.replace(sec, REDACTED)
        return text

    def _redact_subtree(o: Any, key: Any) -> Any:
        if o is None or isinstance(o, bool):
            return o
        if isinstance(o, (int, float)) and isinstance(key, str) and any(
                fnmatch.fnmatchcase(key.lower(), g.lower()) for g in count_keys):
            return o                        # telemetry, e.g. password_min_length / retries
        if isinstance(o, dict):
            return {(_s(k) if isinstance(k, str) else k): _redact_subtree(v, k) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_redact_subtree(v, key) for v in o]
        return REDACTED                     # strings, and numbers such as a PIN

    def _leaf(text: str, key_ctx: Any) -> str:
        text = _s(text)
        if auto:
            text, _ = _scan_string(text, allow, skip_generic=str(key_ctx).lower() in _SKIP_KEYS)
        return text

    def _key(k: Any, taken: dict[Any, Any]) -> Any:
        if not isinstance(k, str):
            return k
        kk = _s(k)
        if auto:
            kk, _ = _scan_string(kk, allow, skip_generic=True)   # a secret used as a key
        n, base = 2, kk
        while kk in taken and kk != k:                            # two keys redacted alike
            kk, n = f"{base}#{n}", n + 1
        return kk

    def _walk(o: Any, key_ctx: Any = None) -> Any:
        if isinstance(o, str):
            return _leaf(o, key_ctx)
        if isinstance(o, dict):
            out: dict[Any, Any] = {}
            for k, v in o.items():
                kk = _key(k, out)
                if str(k).lower() in lkeys:
                    out[kk] = REDACTED
                elif auto and _is_secret_keyname(k):
                    out[kk] = _redact_subtree(v, k)   # a credential key hides its whole value
                else:
                    out[kk] = _walk(v, k)
            return out
        if isinstance(o, (list, tuple)):
            return [_walk(v, key_ctx) for v in o]
        return o

    return _walk(obj)


def detect_secrets(obj: Any, *, allow: Any = ()) -> list[dict[str, Any]]:
    """List the secrets :func:`redact` with ``auto=True`` would redact.

    Returns ``[{"path", "key", "kind"}, ...]`` (no secret value is copied out).
    Use it to warn ("N secrets auto-redacted") rather than to leak: it names
    where and what kind, never the token. ``allow=[...]`` matches :func:`redact`.
    """
    allow = frozenset(str(a) for a in allow)
    findings: list[dict[str, Any]] = []

    def _walk(o: Any, path: str, key_ctx: Any) -> None:
        if isinstance(o, str):
            _, kinds = _scan_string(o, allow, skip_generic=str(key_ctx).lower() in _SKIP_KEYS)
            for kind in kinds:
                findings.append({"path": path, "key": key_ctx, "kind": kind})
        elif isinstance(o, dict):
            for k, v in o.items():
                shown = k
                if isinstance(k, str):
                    shown, kinds = _scan_string(k, allow, skip_generic=True)
                    for kind in kinds:              # a secret used as a key: report it, never echo it
                        findings.append({"path": f"{path}.{shown}" if path else shown,
                                         "key": REDACTED, "kind": kind})
                p = f"{path}.{shown}" if path else str(shown)
                # ``shown`` is the key with any secret in it redacted: it is what a
                # finding may echo, and it equals ``k`` for an ordinary key.
                if _is_secret_keyname(k):
                    findings.append({"path": p, "key": shown, "kind": "secret_key_name"})
                    if not isinstance(v, str):
                        _walk(v, p, shown)   # still scan inside for strong patterns
                else:
                    _walk(v, p, shown)
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                _walk(v, f"{path}[{i}]", key_ctx)

    _walk(obj, "", None)
    return findings


def _oracle_identity(outcome: Any) -> Any:
    """A stable identity for whatever sits in ``AgentCase.outcome``.

    Duck-typed: an oracle object exposes ``.identity()``; a plain dict spec is
    used as-is; ``None`` stays ``None``. This is what makes the case digest
    change when the oracle changes (AG-01) without importing ``outcome`` here.
    """
    if outcome is None:
        return None
    ident = getattr(outcome, "identity", None)
    if callable(ident):
        return ident()
    if isinstance(outcome, dict):
        return outcome
    return repr(outcome)


@dataclass
class AgentEvent:
    """One ordered event in an episode.

    ``index`` is the position in the stream (0-based). ``type`` is the event
    kind (``text``, ``reasoning``, ``tool_call``, ``tool_result``,
    ``observation``, ``reward``, ``turn_complete``). ``turn_id`` groups events
    emitted together -- tool_call events sharing a ``turn_id`` are a parallel
    group. ``call_id`` pairs a ``tool_result`` back to its ``tool_call``.
    ``payload`` keeps the raw source object; ``timestamp`` is ``None`` when the
    source did not record one (never fabricated).
    """

    index: int
    role: str
    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[float] = None
    call_id: Optional[str] = None
    turn_id: Optional[Any] = None
    source: Optional[str] = None
    tier: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AgentEvent":
        return cls(**{k: d.get(k) for k in (
            "index", "role", "type", "payload", "timestamp",
            "call_id", "turn_id", "source", "tier")})


@dataclass
class AgentEpisode:
    """One complete agent run, recorded/deployed/harness."""

    events: list[AgentEvent] = field(default_factory=list)
    mode: str = "recorded"
    source_format: str = "unknown"
    source_tier: Optional[str] = None
    case_id: Optional[str] = None
    trial_id: Optional[str] = None
    source_id: Optional[str] = None
    final_answer: Optional[str] = None
    final_state: Optional[dict[str, Any]] = None
    artifacts: Optional[dict[str, Any]] = None
    counters: dict[str, Any] = field(default_factory=dict)
    stop_reason: Optional[str] = None
    errors: list[str] = field(default_factory=list)
    # Provenance only -- never the AuditKit outcome (AG-05).
    source_reward: Optional[float] = None
    source_verdict: Optional[Any] = None
    source_scores: Optional[dict[str, Any]] = None
    coverage: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"AgentEpisode.mode must be one of {MODES}, got {self.mode!r}")

    # -- derived views ---------------------------------------------------
    def tool_call_events(self) -> list[AgentEvent]:
        return [e for e in self.events if e.type == "tool_call"]

    def turns(self) -> list[list[dict[str, Any]]]:
        """Tool calls grouped into turns, preserving order and parallel groups.

        Events sharing a ``turn_id`` land in one turn; each returns its raw
        payload (a call dict). ``trace.to_turns`` normalizes these for scoring.
        """
        turns: list[list[dict[str, Any]]] = []
        cur_key: Any = object()
        for e in self.tool_call_events():
            p = e.payload
            # Only call-shaped payloads reach the scorers; a name-less record
            # (e.g. a trace.jsonl {"query", "result"}) is kept as an event for
            # provenance but never fabricated into a call.
            if not is_call_payload(p):
                continue
            key = e.turn_id if e.turn_id is not None else ("_ev", e.index)
            if not turns or key != cur_key:
                turns.append([])
                cur_key = key
            turns[-1].append(p)
        return [t for t in turns if t]

    def calls_unnamed(self) -> bool:
        """Calls were recorded, none scorable, and names were never observed.

        Decided by coverage, not by event presence: an importer may keep a
        no-call action (EventLog's terminal ``{}``, a DECIDE ``{"stage": id}``)
        as a ``tool_call`` event while observing that no call was made, and that
        is zero calls, not calls of unknown name.
        """
        return (not self.turns() and bool(self.tool_call_events())
                and self.coverage.get("tool_call_names") != OBSERVED)

    def to_trace(self) -> dict[str, Any]:
        """The structured trace scoring metrics read via ``context["trace"]``.

        Reconstructed from events so events stay the single source of truth for
        order and coverage. Carries ``tool_calls`` (turns), ``retrieved_contexts``
        and a ``messages`` transcript (calls + results) for the judge.
        """
        turns = self.turns()
        if turns or not self.calls_unnamed():
            trace: dict[str, Any] = {"tool_calls": turns}
        else:
            # Calls were recorded but none is a scorable call (e.g. an AgentTune
            # TraceLogger record: arguments, no tool name). Say "unavailable" rather
            # than an empty list, which the metrics would read as "made no calls" (0.0).
            trace = {"tool_calls_unavailable": True}
        contexts: list[str] = []
        for e in self.events:
            if e.type == "observation" and isinstance(e.payload.get("retrieved_contexts"), list):
                contexts = [str(c) for c in e.payload["retrieved_contexts"]]
        if contexts:
            trace["retrieved_contexts"] = contexts
        messages = self._messages()
        if messages:
            trace["messages"] = messages
        return trace

    def _messages(self) -> list[dict[str, Any]]:
        """A best-effort OpenAI-style transcript for the judge/render only.

        Only call-shaped payloads become ``tool_calls`` (a name-less trace.jsonl
        {"query","result"} would otherwise crash ``normalize_call``), and
        consecutive calls sharing a ``turn_id`` are ONE assistant message so the
        judge sees a parallel group as parallel, not as separate turns.
        """
        msgs: list[dict[str, Any]] = []
        pending: list[dict[str, Any]] = []
        pending_turn: Any = object()

        def _flush() -> None:
            if pending:
                msgs.append({"role": "assistant", "content": None, "tool_calls": list(pending)})
                pending.clear()

        for e in self.events:
            if e.type == "tool_call":
                p = e.payload
                if not is_call_payload(p):
                    continue  # not call-shaped; keep out of the transcript
                if pending and e.turn_id != pending_turn:
                    _flush()
                pending_turn = e.turn_id
                pending.append(p)
                continue
            _flush()
            if e.type == "tool_result":
                msgs.append({"role": "tool", "tool_call_id": e.call_id,
                             "content": str(e.payload.get("output", e.payload))})
            elif e.type in ("text", "reasoning"):
                txt = e.payload.get("text")
                if txt:
                    msgs.append({"role": e.role or "assistant", "content": str(txt)})
        _flush()
        return msgs

    def coverage_label(self) -> str:
        """A one-word coverage summary for the report (PRD section 11)."""
        cov = self.coverage
        if cov.get("tool_call_arguments") == OBSERVED:
            return "full"
        if cov.get("tool_call_names") == OBSERVED:
            return "tool_names_only"
        if cov.get("final_answer") == OBSERVED:
            return "answer_only"
        return "none"

    # -- serialization ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "source_format": self.source_format,
            "source_tier": self.source_tier,
            "case_id": self.case_id,
            "trial_id": self.trial_id,
            "source_id": self.source_id,
            "final_answer": self.final_answer,
            "final_state": self.final_state,
            "artifacts": self.artifacts,
            "counters": self.counters,
            "stop_reason": self.stop_reason,
            "errors": list(self.errors),
            "source_reward": self.source_reward,
            "source_verdict": self.source_verdict,
            "source_scores": self.source_scores,
            "coverage": dict(self.coverage),
            "metadata": self.metadata,
            "events": [e.to_dict() for e in self.events],
        }

    def to_json(self, path: Optional[str] = None, *, indent: int = 2) -> str:
        text = dumps_strict(self.to_dict(), indent=indent)
        if path is not None:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        return text

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AgentEpisode":
        return cls(
            events=[AgentEvent.from_dict(e) for e in d.get("events", [])],
            mode=d.get("mode", "recorded"),
            source_format=d.get("source_format", "unknown"),
            source_tier=d.get("source_tier"),
            case_id=d.get("case_id"),
            trial_id=d.get("trial_id"),
            source_id=d.get("source_id"),
            final_answer=d.get("final_answer"),
            final_state=d.get("final_state"),
            artifacts=d.get("artifacts"),
            counters=d.get("counters") or {},
            stop_reason=d.get("stop_reason"),
            errors=list(d.get("errors") or []),
            source_reward=d.get("source_reward"),
            source_verdict=d.get("source_verdict"),
            source_scores=d.get("source_scores"),
            coverage=dict(d.get("coverage") or {}),
            metadata=d.get("metadata") or {},
            schema_version=d.get("schema_version", SCHEMA_VERSION),
        )


@dataclass
class AgentCase:
    """What to evaluate: task, tools, budgets, and the outcome oracle (AG-01).

    ``outcome`` is an oracle object (see :mod:`auditkit.agent_eval.outcome`) or a
    plain dict spec, which is converted to its oracle here (an invalid spec
    raises ``ValueError`` at construction); the case digest folds in its identity so a changed oracle
    changes the digest.

    ``reference_turns`` is the gold tool route, in any form
    :attr:`auditkit.sample.Sample.expected_tool_calls` accepts -- turns, a flat
    call list, messages, a JSON string -- plus ``{"any_of": [...]}`` naming
    ALTERNATIVE routes, for a case an agent can solve more than one way. It is
    stored and passed on as written; choosing a route is the metric layer's job
    (:func:`auditkit.trace.to_reference_paths` reads the shape,
    ``select_reference_path`` picks), so a valid alternative route is not scored
    wrong. It is a diagnostic/policy requirement only when the case says so -- a
    correct outcome on a different valid path still passes. ``[]`` and
    ``{"any_of": [[]]}`` are real references ("calling no tool is correct") and
    are not ``None`` ("no reference given"); see
    :func:`auditkit.agent_eval.runner._has_reference`.
    """

    id: str
    task: str
    category: str = "generic"
    allowed_tools: Optional[list[Any]] = None
    budgets: dict[str, Any] = field(default_factory=dict)
    outcome: Any = None
    # The gold tool route, or ``{"any_of": [...]}`` naming ALTERNATIVE routes:
    # an agent can reach one correct answer several ways, and a reference that
    # names only one of them scores a valid route as wrong. Not coerced here --
    # every reference metric picks its route from the same value, so they cannot
    # disagree about which.
    reference_turns: Optional[Union[list[Any], dict[str, Any], str]] = None
    reference_contexts: Optional[Any] = None
    environment: Optional[dict[str, Any]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not str(self.id).strip():
            raise ValueError("AgentCase.id is required and must be non-empty")
        if not self.task or not str(self.task).strip():
            raise ValueError(f"AgentCase[{self.id!r}].task is required and must be non-empty")
        if self.reference_turns is not None:
            # Read the reference now, not when a metric happens to score it. A
            # malformed multi-route reference is a mistake in the dataset, and a
            # per-sample metric error drops that sample out of the run and lets
            # it finish -- so a dataset of broken references reads as a run that
            # simply scored nothing. Here it names the case instead.
            from auditkit.trace import to_reference_paths
            try:
                to_reference_paths(self.reference_turns)
            except ValueError as e:
                raise ValueError(f"AgentCase[{self.id!r}].reference_turns: {e}") from e
        self.id = str(self.id)
        if isinstance(self.outcome, dict):
            from .outcome import from_spec
            try:
                self.outcome = from_spec(self.outcome)
            except (KeyError, ValueError, TypeError) as e:
                raise ValueError(f"AgentCase[{self.id!r}].outcome: invalid oracle spec "
                                 f"{self.outcome!r}: {e}") from None

    def _oracle_inputs(self) -> dict[str, Any]:
        """Metadata values the oracle reads at evaluation time (duck-typed, no import):
        ``AnswerAssertion()`` with no reference reads ``metadata["target"]``;
        ``AssertionOracle()`` with no criterion reads ``metadata[metadata_key]``;
        a companion ``state_oracle`` is followed. Empty for every other oracle."""
        meta, out = self.metadata or {}, {}
        o = self.outcome
        while o is not None:
            kind = getattr(o, "type", None)
            if kind == "answer_assertion" and getattr(o, "reference", None) is None and "target" in meta:
                out["target"] = meta["target"]
            if kind == "assertion" and getattr(o, "criterion", None) is None:
                key = getattr(o, "metadata_key", "assertion")
                if key in meta:
                    out[key] = meta[key]
            o = getattr(o, "state_oracle", None)
        return out

    def digest(self) -> str:
        """A stable content hash; changes with task or oracle (AG-01), including the
        metadata an oracle reads its reference from (so opposite criteria never share
        a digest). Cases whose oracle reads no metadata keep their previous digest."""
        payload = {
            "id": self.id,
            "task": self.task,
            "category": self.category,
            "allowed_tools": self.allowed_tools,
            "budgets": self.budgets,
            "outcome": _oracle_identity(self.outcome),
            # a JSON-string reference hashes like its decoded form (#44)
            "reference_turns": decode_reference(self.reference_turns),
            "reference_contexts": self.reference_contexts,
            "environment": self.environment,
        }
        inputs = self._oracle_inputs()
        if inputs:
            payload["oracle_inputs"] = inputs
        return "sha256:" + hashlib.sha256(_canonical(payload).encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task": self.task,
            "category": self.category,
            "allowed_tools": self.allowed_tools,
            "budgets": self.budgets,
            "outcome": _oracle_identity(self.outcome),
            "reference_turns": self.reference_turns,
            "reference_contexts": self.reference_contexts,
            "environment": self.environment,
            "metadata": self.metadata,
            "digest": self.digest(),
        }


def validate_cases(cases: list[AgentCase]) -> list[AgentCase]:
    """Fail on duplicate IDs (missing fields already fail in ``__post_init__``)."""
    seen: dict[str, int] = {}
    for i, c in enumerate(cases):
        if not isinstance(c, AgentCase):
            raise TypeError(f"cases[{i}] is not an AgentCase, got {type(c).__name__}")
        if c.id in seen:
            raise ValueError(
                f"duplicate AgentCase id {c.id!r} at positions {seen[c.id]} and {i}")
        seen[c.id] = i
    return cases
