"""Agent-eval results: per-case rows and a coverage-aware summary (A2, AG-11).

The headline is the **verified outcome**; answer/tool/retrieval scores and the
judge are separate diagnostic columns. The summary shows BOTH a decided-only
denominator (excluding ``unknown``) and an over-all-cases denominator, plus
trace coverage counts and cost -- matching PRD section 11. Missing/unknown is
never turned into a zero score.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from .outcome import FAILURE, SUCCESS, UNKNOWN
from .types import AgentEpisode, dumps_strict

SCHEMA_VERSION = "agent_eval_result/1"


def _jvo_records(dis: dict[str, Any]) -> list[dict[str, Any]]:
    """The judge-vs-oracle disagreement records in a row's ``disagreement`` dict
    (top-level for a single trial, plus any per-trial examples for a multi-trial row)."""
    recs: list[dict[str, Any]] = []
    if isinstance(dis.get("judge_vs_oracle"), dict):
        recs.append(dis["judge_vs_oracle"])
    for t in dis.get("trials", []) or []:
        if isinstance(t, dict) and isinstance(t.get("judge_vs_oracle"), dict):
            recs.append(t["judge_vs_oracle"])
    return recs


@dataclass
class AgentCaseResult:
    """One case's result: outcome headline + diagnostics + status."""

    case_id: str
    status: str
    outcome: dict[str, Any]
    coverage: dict[str, str] = field(default_factory=dict)
    coverage_label: str = "answer_only"
    diagnostics: dict[str, Any] = field(default_factory=dict)
    ineligible: dict[str, str] = field(default_factory=dict)
    resource_use: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    judge: Optional[dict[str, Any]] = None
    stop_reason: Optional[str] = None
    errors: list[str] = field(default_factory=list)
    source_format: Optional[str] = None
    source_id: Optional[str] = None
    # A3 reliability (empty for a single trial) and its per-trial rows.
    reliability: dict[str, Any] = field(default_factory=dict)
    trials: list[dict[str, Any]] = field(default_factory=list)
    # AG-05: judge/source vs verified-oracle disagreement (oracle wins; recorded).
    disagreement: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AgentCaseResult":
        return cls(**{k: d[k] for k in (
            "case_id", "status", "outcome", "coverage", "coverage_label",
            "diagnostics", "ineligible", "resource_use", "provenance", "judge",
            "stop_reason", "errors", "source_format", "source_id",
            "reliability", "trials", "disagreement") if k in d})


@dataclass
class AgentEvalResult:
    """The full run: per-case rows plus run identity."""

    rows: list[AgentCaseResult] = field(default_factory=list)
    mode: str = "recorded"
    spec_identity: dict[str, Any] = field(default_factory=dict)
    # The scored episodes, kept so `agent rescore` can replay a run offline --
    # including a deployed run, WITHOUT re-spending endpoint calls.
    episodes: list[AgentEpisode] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "spec_identity": self.spec_identity,
            "rows": [r.to_dict() for r in self.rows],
            "episodes": [e.to_dict() for e in self.episodes],
        }

    def to_json(self, path: Optional[str] = None, *, indent: int = 2) -> str:
        text = dumps_strict(self.to_dict(), indent=indent)
        if path is not None:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        return text

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AgentEvalResult":
        return cls(
            rows=[AgentCaseResult.from_dict(r) for r in d.get("rows", [])],
            mode=d.get("mode", "recorded"),
            spec_identity=d.get("spec_identity") or {},
            episodes=[AgentEpisode.from_dict(e) for e in d.get("episodes", [])],
            schema_version=d.get("schema_version", SCHEMA_VERSION),
        )

    @classmethod
    def from_json(cls, path: str) -> "AgentEvalResult":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    # -- summary (PRD section 11) ----------------------------------------
    def summary(self) -> str:
        n = len(self.rows)
        # Each row lands in exactly one bucket so the counts add up to n. A case
        # that overran its budget is reported as such, never as a verified success.
        over_budget = sum(1 for r in self.rows if r.status == "budget_exhausted")
        verdicts = [r.outcome.get("verdict") for r in self.rows
                    if r.status != "budget_exhausted"]
        success = verdicts.count(SUCCESS)
        failure = verdicts.count(FAILURE)
        unknown = verdicts.count(UNKNOWN)
        errored = len(verdicts) - success - failure - unknown  # error (or unrecognized)
        decided = success + failure  # excludes unknown/error (never counted as 0)
        endpoint_errors = sum(1 for r in self.rows if r.status == "target_error")
        statuses = Counter(r.status for r in self.rows)

        labels = [r.coverage_label for r in self.rows]
        full = labels.count("full")
        names_only = labels.count("tool_names_only")
        answer_only = labels.count("answer_only")

        validity_eligible = sum(1 for r in self.rows if "tool_call_validity" in r.diagnostics)
        validity_pass = sum(1 for r in self.rows
                            if r.diagnostics.get("tool_call_validity", {}).get("value") == 1.0)

        lats = [(r.resource_use or {}).get("latency_ms") for r in self.rows
                if (r.resource_use or {}).get("latency_ms") is not None]
        median_lat = f"{statistics.median(lats) / 1000:.2f} s" if lats else "n/a"
        agent_calls = sum(1 for r in self.rows if (r.resource_use or {}).get("agent_call"))

        decided_str = f"{success}/{decided} decided" if decided else "0 decided"
        lines = [
            f"Agent eval   Mode: {self.mode}   Cases: {n}",
            f"Verified success: {success}/{n} ({decided_str})   Failure: {failure}   "
            f"Unknown outcome: {unknown}   Error: {errored}   Over budget: {over_budget}",
            f"Endpoint errors: {endpoint_errors}",
            "Status: " + ", ".join(f"{k} {v}" for k, v in sorted(statuses.items())),
            f"Trace coverage: full {full}, tool names only {names_only}, answer only {answer_only}",
            f"Tool validity: {validity_pass}/{validity_eligible} eligible",
            f"Median latency: {median_lat}          Agent calls: {agent_calls}",
        ]
        # Reliability line (A3): only when some case ran repeated trials.
        multi = [r for r in self.rows if (r.reliability or {}).get("n_trials", 1) > 1]
        if multi:
            rel = Counter((r.reliability or {}).get("status") for r in multi)
            unconfirmed = sum(1 for r in multi if not (r.reliability or {}).get("independent"))
            line = (f"Reliability: {len(multi)} multi-trial case(s) -- "
                    f"reliable {rel.get('reliable', 0)}, mixed {rel.get('mixed', 0)}, "
                    f"unreliable {rel.get('unreliable', 0)}, unknown {rel.get('unknown', 0)}")
            if unconfirmed:
                line += f"  ({unconfirmed} with independence unconfirmed)"
            lines.append(line)
        # Review lines: judge/oracle disagreement (AG-05: oracle wins, recorded);
        # answers that claim success while the verified outcome failed; and cases
        # with no state evidence that are NOT counted as failures.
        jvo = [rec for r in self.rows for rec in _jvo_records(r.disagreement or {})]
        claim_gap = sum(1 for rec in jvo
                        if rec.get("judge") == SUCCESS and rec.get("oracle") == FAILURE)
        disagree_cases = len(self.disagreements())
        no_state = sum(1 for r in self.rows
                       if r.outcome.get("verdict") == UNKNOWN
                       and "state" in (r.outcome.get("reason") or "").lower())
        review = []
        if disagree_cases:
            review.append(f"{disagree_cases} cases: judge/source disagrees with the verified "
                          "oracle (oracle wins; disagreement recorded for review)")
        if claim_gap:
            review.append(f"{claim_gap} answers claim success but verified outcome fails")
        if no_state:
            review.append(f"{no_state} cases lack state evidence and are not counted as failures")
        if review:
            lines.append("Review: " + "\n        ".join(review))
        return "\n".join(lines)

    def disagreements(self, min_rate: float = 0.0) -> list[AgentCaseResult]:
        """Rows where the judge or source verdict disagreed with the verified oracle.

        AG-05: the verified oracle already won each headline; this filter surfaces
        the recorded disagreements for review. ``min_rate`` keeps only cases whose
        disagreement rate (over trials, or 1.0 for a single trial) is at least the
        threshold -- e.g. ``min_rate=0.5`` for cases where a majority of trials
        disagreed.
        """
        out = []
        for r in self.rows:
            d = r.disagreement or {}
            if not d:
                continue
            rate = d.get("rate", 1.0)
            if rate > 0 and rate >= min_rate:
                out.append(r)
        return out

    def cases_report(self) -> str:
        """One block per case: status, verdict and reason, coverage, skipped
        metrics, diagnostics, stop reason and errors (the drilldown)."""
        out = []
        for r in self.rows:
            oc = r.outcome or {}
            out.append(f"[{r.case_id}] status={r.status} outcome={oc.get('verdict')} "
                       f"coverage={r.coverage_label} stop={r.stop_reason}")
            if oc.get("reason"):
                out.append(f"    reason: {oc['reason']}")
            for name, d in (r.diagnostics or {}).items():
                out.append(f"    {name}: {d.get('value') if isinstance(d, dict) else d}")
            for name, why in (r.ineligible or {}).items():
                out.append(f"    skipped {name}: {why}")
            for e in r.errors or []:
                out.append(f"    error: {str(e)[:300]}")
        return "\n".join(out)
