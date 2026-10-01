"""Reliability across repeated trials (A3, AG-12/AG-13, UX-A6).

Given the per-trial verdicts of one case, summarize how *reliably* the agent
succeeds -- not whether it passed once by luck. Everything is deterministic
given the trial verdicts and reuses the existing HumanEval estimators
:func:`auditkit.score.pass_at_k` (capability: >=1 of k) and
:func:`auditkit.score.pass_hat_k` (all-k reliability: all k correct).

Honesty rules (PRD section 8):
- Estimators are computed over **decided** trials only (``success`` + ``failure``);
  ``unknown``/``error`` are reported separately, never counted as a 0.
- ``trials==1`` (or no decided trial) -> ``status="unknown"`` with an explicit
  reason; no reliability is claimed.
- Independence is NOT inferred from a ``trials=N`` config or from distinct
  recorded episodes (AuditKit cannot see whether they shared a session/state).
  It is ``True`` only when the caller confirms a reset-per-trial contract
  (``reset_confirmed=True``). Otherwise the counts are shown but flagged as not
  an independence-backed reliability claim (UX-A6).
"""

from __future__ import annotations

from typing import Any, Optional

from .outcome import FAILURE, SUCCESS, UNKNOWN


def aggregate_verdict(verdicts: list[str]) -> tuple[str, str]:
    """A single stable outcome for a multi-trial case: majority over decided trials.

    Returns ``(verdict, reason)``. A tie between success and failure is
    ``unknown`` (the trials disagree, no majority) rather than a coin-flip pass.
    """
    n_success = verdicts.count(SUCCESS)
    n_failure = verdicts.count(FAILURE)
    decided = n_success + n_failure
    if decided == 0:
        return UNKNOWN, "no decided trials (all unknown/error)"
    if n_success > n_failure:
        agg = SUCCESS
    elif n_failure > n_success:
        agg = FAILURE
    else:
        return UNKNOWN, f"trials disagree with no majority ({n_success}/{decided} succeeded)"
    return agg, f"{n_success}/{decided} decided trials succeeded (aggregate={agg})"


def reliability(verdicts: list[str], *, k: Optional[int] = None,
                independent: bool = False, trial_ids: Optional[list[Any]] = None,
                n_requested: Optional[int] = None) -> dict[str, Any]:
    """Summarize reliability from a case's per-trial verdicts.

    ``k`` defaults to the number of decided trials. ``independent`` gates the
    reliability *claim*: when ``False`` the estimators are still reported but
    flagged (UX-A6). ``trial_ids`` lets us flag a duplicate id (a fake that
    reused one trial). ``n_requested`` records the configured trial count so a
    short match (recorded mode) is visible.
    """
    n_total = len(verdicts)
    n_success = verdicts.count(SUCCESS)
    n_failure = verdicts.count(FAILURE)
    n_unknown = verdicts.count(UNKNOWN)
    n_error = n_total - n_success - n_failure - n_unknown
    decided = n_success + n_failure

    out: dict[str, Any] = {
        "n_trials": n_total,
        "n_trials_requested": n_total if n_requested is None else n_requested,
        "n_success": n_success,
        "n_failure": n_failure,
        "n_unknown": n_unknown,
        "n_error": n_error,
        "n_decided": decided,
        "independent": bool(independent),
    }
    flags: list[str] = []
    if trial_ids:
        present = [t for t in trial_ids if t is not None]
        if len(present) != len(set(present)):
            flags.append("duplicate trial_id across trials: trials may not be independent")
    if not independent:
        flags.append("independence unconfirmed (no reset-per-trial contract); counts shown, "
                     "the estimators are not an independence-backed reliability claim")

    if n_total <= 1:
        out.update(status="unknown", reason="single trial; nothing to compare", flags=flags)
        return out
    if decided <= 1:
        # one decided trial (the rest error/unknown) is "passed once by luck",
        # not a reliability estimate -- do not report a full pass@k over it.
        out.update(status="unknown",
                   reason=f"only {decided} decided trial(s); cannot estimate reliability",
                   flags=flags)
        return out

    from ..score import pass_at_k, pass_hat_k
    kk = decided if k is None else max(1, min(int(k), decided))
    p = n_success / decided
    out.update(
        k=kk,
        pass_at_k=pass_at_k(decided, n_success, kk),   # capability: >=1 of k pass
        all_k=pass_hat_k(decided, n_success, kk),       # reliability: all k pass
        success_rate=p,
        consistency=max(n_success, n_failure) / decided,  # fraction matching the modal verdict
        variance=p * (1 - p),                             # of the 0/1 success indicator
        status="reliable" if n_success == decided else ("unreliable" if n_success == 0 else "mixed"),
        reason=f"{n_success}/{decided} decided trials succeeded",
        flags=flags,
    )
    return out
