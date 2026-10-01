"""Judge calibration, human review, and operational (load/chaos) checks over the
deterministic RAG-stress metrics (build-order steps (4) and (5) of the RAG-stress
PRD, ``docs/notes/rag-stress-model-risk-research-prd-2026-09.md``).

All three pieces are stdlib and model-free — no LLM call, no real load, no
threads or sleeping. They consume **supplied data** (adjudicated labels, a
:class:`~auditkit.metrics.rag_stress_runner.StressReport`, or scripted
timing/error samples) and follow the same convention as the metrics: an explicit
``unknown`` (value ``None``, not ``0``) whenever the input a check needs is
absent.

Step (4) — judge calibration + human review:

- :func:`judge_calibration` compares a RAG judge's pass/fail verdicts (over the
  faithfulness / citation / abstention dimensions) against a small
  human-adjudicated label set and reports agreement, Cohen's kappa, false-pass
  and false-negative rates with denominators. Deterministic — labels and judge
  outputs are supplied as data (the RAGBench/ARES "test the judge against a
  local adjudicated set" point); this is not itself a judge.
- :class:`HumanReview` + :func:`review_queue` surface, within a budget, the
  ``StressReport`` cases a reviewer should look at (harness/expectation
  disagreement, genuinely ambiguous ``unknown`` scores, cases nothing scored, or
  benign utility loss). The reviewer's detection / correction / override / final
  answer-changed decision is *recorded*, never computed — no auto-resolution.

Step (5) — operational load / chaos (deterministic simulation from samples):

- :func:`operational_stress` replays scripted concurrency / burst / slow-store /
  timeout / partial-search-failure / rate-limit / judge-failure scenarios and
  reports tail latency (p50/p95 from the SUPPLIED timing samples), error budget,
  observed retry/idempotency/recovery, and safe degradation. Nothing is timed or
  slept here; every number comes from the supplied samples.
- :func:`false_answer_under_partial_retrieval` is the safe-degradation check:
  under a partial retrieval failure, did the system answer anyway (bad) or
  abstain (good)?

Out of scope for this step (PRD step (5) also lists them, but the task does not):
jurisdiction-specific evidence packs. Those select which evidence view is
requested per the jurisdiction matrix and are additive on top of these checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from auditkit.metrics.rag_stress_runner import StressCaseResult, StressReport
from auditkit.score import Stat

__all__ = [
    "judge_calibration",
    "HumanReview",
    "review_queue",
    "operational_stress",
    "false_answer_under_partial_retrieval",
    "OPS_KINDS",
]


# ---------------------------------------------------------------------------
# Step 4a: judge calibration against a human-adjudicated label set
# ---------------------------------------------------------------------------


def _as_pass(value: Any) -> bool:
    """Normalize one verdict to pass=True / fail=False.

    Accepts a bool or the strings ``"pass"``/``"fail"`` (case-insensitive) only.
    Anything else raises — a verdict is a trust boundary; do not coerce silently.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low == "pass":
            return True
        if low == "fail":
            return False
    raise ValueError(f"verdict {value!r} not in {{True, False, 'pass', 'fail'}}")


def _kappa_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The 2x2 confusion + agreement / kappa / false-pass / false-negative.

    ``rows`` each carry ``human`` and ``judge`` verdicts (already filtered to
    rows where both are present). ``pass`` is the positive class:

    - ``false_pass_rate`` = P(judge=pass | human=fail): the judge lets a
      human-failed answer through (the model-risk-dangerous miss; the PRD's
      false-pass).
    - ``false_negative_rate`` = P(judge=fail | human=pass): the judge rejects a
      human-passed answer (a false alarm; the PRD's false-fail).

    Rates whose denominator is empty, and kappa when chance agreement is 1.0
    (a degenerate all-one-class set), are ``None`` — never a silent 0.
    """
    pp = pf = fp = ff = 0  # (human, judge): pass/pass, pass/fail, fail/pass, fail/fail
    for r in rows:
        h, j = _as_pass(r["human"]), _as_pass(r["judge"])
        if h and j:
            pp += 1
        elif h and not j:
            pf += 1
        elif not h and j:
            fp += 1
        else:
            ff += 1
    n = pp + pf + fp + ff
    n_human_pass, n_human_fail = pp + pf, fp + ff
    po = (pp + ff) / n
    # Chance agreement from the marginals (Cohen's kappa).
    pe = ((pp + pf) * (pp + fp) + (fp + ff) * (pf + ff)) / (n * n)
    kappa = None if pe >= 1.0 else (po - pe) / (1.0 - pe)
    return {
        "n": n,
        "confusion": {"pp": pp, "pf": pf, "fp": fp, "ff": ff},
        "n_human_pass": n_human_pass,
        "n_human_fail": n_human_fail,
        "agreement": po,
        "cohens_kappa": kappa,
        "false_pass_rate": (fp / n_human_fail) if n_human_fail else None,
        "false_negative_rate": (pf / n_human_pass) if n_human_pass else None,
    }


def judge_calibration(labeled_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Calibrate a RAG judge against a human-adjudicated label set (RST-08).

    Each row is ``{"human": <verdict>, "judge": <verdict>, "dimension"?: str}``
    where a verdict is a bool or ``"pass"``/``"fail"``. Rows missing a ``human``
    label are not counted (you cannot calibrate against nothing); rows with a
    human label but no ``judge`` verdict are counted separately
    (``n_missing_judge``) and excluded from the confusion.

    Returns a strict-JSON dict with overall agreement, Cohen's kappa, false-pass
    and false-negative rates (with the 2x2 counts and denominators the PRD asks
    to record), plus a ``by_dimension`` breakdown when rows carry ``dimension``
    (RST-08's per-slice agreement for routing low-agreement slices to review).
    When no row carries a human label the whole result is ``status: "unknown"``
    with every rate ``None`` — never a silent 0.
    """
    labeled = [r for r in labeled_rows if r.get("human") is not None]
    n_missing_judge = sum(1 for r in labeled if r.get("judge") is None)
    scored = [r for r in labeled if r.get("judge") is not None]
    n_unlabeled = len(labeled_rows) - len(labeled)
    if not scored:
        return {
            "status": "unknown",
            "reason": "no rows carry both a human label and a judge verdict",
            "n": 0,
            "n_unlabeled": n_unlabeled,
            "n_missing_judge": n_missing_judge,
            "agreement": None,
            "cohens_kappa": None,
            "false_pass_rate": None,
            "false_negative_rate": None,
            "by_dimension": {},
        }
    out: dict[str, Any] = {"status": "scored"}
    out.update(_kappa_block(scored))
    out["n_unlabeled"] = n_unlabeled
    out["n_missing_judge"] = n_missing_judge
    by_dim: dict[str, Any] = {}
    if any("dimension" in r for r in scored):
        dims = sorted({str(r.get("dimension")) for r in scored})
        for d in dims:
            by_dim[d] = _kappa_block([r for r in scored if str(r.get("dimension")) == d])
    out["by_dimension"] = by_dim
    return out


# ---------------------------------------------------------------------------
# Step 4b: human review queue over a StressReport (no auto-resolution)
# ---------------------------------------------------------------------------


@dataclass
class HumanReview:
    """One case surfaced for human review, plus the reviewer's recorded decision.

    The top fields are populated by :func:`review_queue` from the report. The
    reviewer fields (``detected`` / ``correction`` / ``override`` /
    ``final_answer_changed``) default to ``None`` and are filled by a human via
    :meth:`record` — the queue never computes them (no auto-resolution).
    """

    case_id: str
    stress_kind: str
    priority: int
    queued_reason: str
    first_failure_stage: Optional[str] = None
    expected_failure: Optional[str] = None
    stress_flagged: list[str] = field(default_factory=list)
    unknown_metrics: list[str] = field(default_factory=list)
    # Reviewer decision — recorded, not computed.
    detected: Optional[bool] = None            # did the reviewer confirm an issue?
    correction: Optional[str] = None           # the correction the reviewer applied
    override: Optional[str] = None             # reviewer verdict overriding the harness
    final_answer_changed: Optional[bool] = None  # did the customer-facing answer change?

    def record(self, *, detected: Optional[bool] = None, correction: Optional[str] = None,
               override: Optional[str] = None,
               final_answer_changed: Optional[bool] = None) -> "HumanReview":
        """Record a reviewer's decision (only the fields supplied); returns self."""
        if detected is not None:
            self.detected = detected
        if correction is not None:
            self.correction = correction
        if override is not None:
            self.override = override
        if final_answer_changed is not None:
            self.final_answer_changed = final_answer_changed
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "stress_kind": self.stress_kind,
            "priority": self.priority,
            "queued_reason": self.queued_reason,
            "first_failure_stage": self.first_failure_stage,
            "expected_failure": self.expected_failure,
            "stress_flagged": list(self.stress_flagged),
            "unknown_metrics": list(self.unknown_metrics),
            "detected": self.detected,
            "correction": self.correction,
            "override": self.override,
            "final_answer_changed": self.final_answer_changed,
        }


def _review_signals(r: StressCaseResult) -> dict[str, Any]:
    """Which review signals fire for one case, and its priority.

    A ``StressReport`` carries no per-case confidence number, so the proxy for
    "lowest-confidence" is a genuinely unresolvable score. ``not_tested`` (absent
    gold for a metric this case does not target) is a fixture gap, not an
    ambiguity a reviewer resolves, so it is deliberately NOT a signal — otherwise
    every fixture case would queue and the budget would be meaningless.

    Signals:

    - ``disagreement`` — the harness disagrees with the injected expectation:
      the ``expected_failure`` metric did not flag, or something unexpected did.
    - ``unknown`` metrics — status ``unknown`` (gold present, unresolvable).
    - ``nothing_scored`` — no metric scored at all (the operational /
      partial-retrieval shape: retrieval never fired).
    - ``utility_loss`` — the benign control arm failed a metric.
    """
    unknown_metrics = [m for m in r.stress if r.stress[m]["status"] == "unknown"]
    nothing_scored = not any(v["status"] == "scored" for v in r.stress.values())
    disagreement = r.expected_failure is not None and (
        r.expected_failure not in r.stress_flagged
        or bool(set(r.stress_flagged) - {r.expected_failure}))
    utility_loss = not r.benign_control_passed
    reasons = []
    if disagreement:
        reasons.append("harness/expectation disagreement")
    if unknown_metrics:
        reasons.append(f"unknown: {sorted(unknown_metrics)}")
    if nothing_scored:
        reasons.append("nothing scored (retrieval never fired)")
    if utility_loss:
        reasons.append("benign utility loss")
    priority = (100 * disagreement + 10 * nothing_scored
                + 5 * len(unknown_metrics) + 1 * utility_loss)
    return {"priority": priority, "queued_reason": "; ".join(reasons),
            "unknown_metrics": sorted(unknown_metrics),
            "is_candidate": bool(reasons)}


def review_queue(stress_report: StressReport, *, budget: int) -> list[HumanReview]:
    """The top-``budget`` cases from ``stress_report`` that a human should review.

    Candidates are cases where a review signal fires (see :func:`_review_signals`);
    clean cases are never queued. Ordered by priority (disagreement > nothing
    scored > unknown scores > utility loss), ties broken by ``case_id`` for
    determinism. Returns :class:`HumanReview` records with the reviewer decision
    left blank (``None``) — the queue surfaces work, it does not resolve it.
    """
    if budget < 0:
        raise ValueError(f"budget must be non-negative, got {budget}")
    scored = []
    for r in stress_report.cases:
        sig = _review_signals(r)
        if sig["is_candidate"]:
            scored.append((r, sig))
    scored.sort(key=lambda rs: (-rs[1]["priority"], rs[0].case_id))
    return [
        HumanReview(
            case_id=r.case_id, stress_kind=r.stress_kind, priority=sig["priority"],
            queued_reason=sig["queued_reason"], first_failure_stage=r.first_failure_stage,
            expected_failure=r.expected_failure, stress_flagged=list(r.stress_flagged),
            unknown_metrics=sig["unknown_metrics"])
        for r, sig in scored[:budget]
    ]


# ---------------------------------------------------------------------------
# Step 5: operational load / chaos (deterministic, from supplied samples)
# ---------------------------------------------------------------------------

OPS_KINDS: tuple[str, ...] = (
    "concurrency", "burst", "slow_store", "timeout",
    "partial_search_failure", "rate_limit", "judge_failure",
)

_OK_OUTCOMES = frozenset({"ok", "success"})


def false_answer_under_partial_retrieval(scenario: dict[str, Any]) -> Optional[bool]:
    """Under a partial retrieval failure, did the system answer anyway (bad)?

    Returns ``True`` (a false answer — unsafe degradation) if the scenario had a
    partial retrieval failure and the system answered; ``False`` (safe: it
    abstained) if it abstained; ``None`` when the check does not apply
    (``partial_retrieval`` absent or falsey) or when it applies but no answer
    signal is present (``answered``/``abstained`` both absent) — never a silent
    pass. The answer signal is read from ``answered`` or, failing that, the
    negation of ``abstained``.
    """
    if not scenario.get("partial_retrieval"):
        return None
    answered = scenario.get("answered")
    if answered is None and scenario.get("abstained") is not None:
        answered = not scenario["abstained"]
    if answered is None:
        return None
    return bool(answered)


def _scenario_result(scenario: dict[str, Any]) -> dict[str, Any]:
    kind = scenario.get("kind")
    if kind not in OPS_KINDS:
        raise ValueError(f"unknown ops kind {kind!r}; choose from {OPS_KINDS}")
    lat = scenario.get("latencies_ms")
    if lat:
        stat = Stat("lat")
        for v in lat:
            stat.add(v)
        p50: Optional[float] = stat.percentile(50)
        p95: Optional[float] = stat.percentile(95)
    else:
        p50 = p95 = None  # unknown: no timing samples supplied

    outcomes = scenario.get("outcomes")
    n_requests: Optional[int] = None
    error_rate: Optional[float] = None
    timeout_rate: Optional[float] = None
    if outcomes:
        n_requests = len(outcomes)
        error_rate = sum(1 for o in outcomes if o not in _OK_OUTCOMES) / n_requests
        timeout_rate = sum(1 for o in outcomes if o == "timeout") / n_requests
    elif lat:
        n_requests = len(lat)

    budget = scenario.get("error_budget")
    within_budget = None if (budget is None or error_rate is None) else error_rate <= budget
    slo = scenario.get("slo_ms")
    p95_within_slo = None if (slo is None or p95 is None) else p95 <= slo

    false_answer = false_answer_under_partial_retrieval(scenario)
    return {
        "name": scenario.get("name"),
        "kind": kind,
        "n_requests": n_requests,
        "p50_ms": p50,
        "p95_ms": p95,
        "error_rate": error_rate,
        "error_budget": budget,
        "within_error_budget": within_budget,
        "timeout_rate": timeout_rate,
        "p95_within_slo": p95_within_slo,
        "partial_retrieval": bool(scenario.get("partial_retrieval")) if "partial_retrieval" in scenario else None,
        "false_answer_under_partial_retrieval": false_answer,
        "safe_degradation": None if false_answer is None else (not false_answer),
        # Observed operational behaviour — recorded, not computed.
        "retried": scenario.get("retried"),
        "idempotent": scenario.get("idempotent"),
        "recovered": scenario.get("recovered"),
    }


def operational_stress(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    """Score scripted operational load/chaos scenarios (RST-07, deterministic).

    Each scenario is a dict: ``name``, ``kind`` (one of :data:`OPS_KINDS`), and
    any of ``latencies_ms`` (supplied timing samples → p50/p95), ``outcomes``
    (per-request ``"ok"``/``"error"``/``"timeout"``/``"rate_limited"`` →
    error/timeout rate), ``error_budget``, ``slo_ms``, ``partial_retrieval``,
    ``answered``/``abstained``, and observed ``retried``/``idempotent``/
    ``recovered`` flags. Nothing is timed or slept — every number is computed
    from the supplied samples; a field absent means the corresponding measure is
    ``None`` (unknown), never 0.

    Returns a strict-JSON dict with the per-scenario results plus a safe-
    degradation rollup: ``any_false_answer`` (``True`` if any determinable
    scenario answered under partial retrieval, ``None`` if none were
    determinable) reported beside ``n_partial_retrieval_checked`` /
    ``n_partial_retrieval_unknown`` so a pass cannot hide an untested slice.
    """
    results = [_scenario_result(s) for s in scenarios]
    partials = [r for r in results if r["partial_retrieval"]]
    determinable = [r["false_answer_under_partial_retrieval"] for r in partials
                    if r["false_answer_under_partial_retrieval"] is not None]
    any_false = None if not determinable else any(determinable)
    return {
        "n_scenarios": len(results),
        "scenarios": results,
        "any_false_answer": any_false,
        "n_partial_retrieval": len(partials),
        "n_partial_retrieval_checked": len(determinable),
        "n_partial_retrieval_unknown": len(partials) - len(determinable),
    }
