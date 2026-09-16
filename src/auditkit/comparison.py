"""Baseline-anchored comparison of two runs — the "did pruning/quantizing hurt?" view.

`RunComparison(baseline, candidate)` is a read-side helper over two immutable
`RunResult`s produced with the same eval config (e.g. a base model vs a pruned or
quantized one). It gives:

- **per-task deltas** (engine-agnostic: benchmark runs key per-task metrics in
  `stats` as ``"task:metric"``; native runs carry a `task` on each `Prediction`
  and a per-score breakdown in `metadata["scores"]` — both are folded in),
- **percentage differences** per metric (the summary reports the absolute delta
  and the relative % -- no pass/warn/fail verdict and no better/worse framing;
  a fixed threshold isn't meaningful across every metric, so it's the reader's
  job to interpret the numbers. An opt-in threshold gate is still available via
  .grade() with caller-chosen thresholds),
- **retention %** (candidate/baseline for MAXIMIZE, inverted for MINIMIZE),
- **significance** via the shared by-sample-id paired bootstrap,
- a **size/latency tradeoff** view over caller-supplied numbers,
- and the **regressed / improved** sample browser (delegated to `RunDiff`).

Lineage-based *auto* baseline selection is deliberately a platform concern; here
you hand it the two runs.
"""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

from ._bootstrap import paired_bootstrap, score_pairs
from .diff import (
    DeltaGrade, RunDiff, direction_for, grade_delta, metric_directions,
    relative_pct,
)
from .report import Prediction, RunResult
from .types import Direction

logger = logging.getLogger(__name__)

# NOT_COMPARABLE deliberately ranks below every real grade: a single metric
# that was only scored on one side shouldn't be able to make grade()'s
# worst-case reduction report a FAIL-equivalent when every metric that
# actually WAS compared passed cleanly. If every metric is NOT_COMPARABLE
# (nothing at all could be compared), that's still what grade() reports --
# there's nothing better to fall back to.
_GRADE_ORDER = {
    DeltaGrade.NOT_COMPARABLE: -1,
    DeltaGrade.PASS: 0,
    DeltaGrade.WARN: 1,
    DeltaGrade.FAIL: 2,
}


def _delta_line(label: str, d) -> str:
    """One human line for a MetricDelta/TaskDelta: baseline -> candidate, the
    absolute delta AND the relative %, with the sample count/std. No verdict and
    no better/worse framing -- the reader interprets the numbers for their own
    metric. ASCII only (printed to consoles that can't encode arrows)."""
    if d.delta is None:
        return (f"{label}: N/A -- only scored on one side "
                f"(n={d.baseline_count}->{d.candidate_count})")
    pct_s = f"{d.pct_change:+.1f}% rel" if d.pct_change is not None else "rel n/a"
    return (
        f"{label}: {d.baseline:.4f} -> {d.candidate:.4f}  "
        f"delta={d.delta:+.4f} ({pct_s})  "
        f"(n={d.baseline_count}->{d.candidate_count}, "
        f"std={d.baseline_std:.4f}->{d.candidate_std:.4f})"
    )


@dataclass
class MetricDelta:
    metric: str
    # None when the metric was scored on only one side (grade is then
    # NOT_COMPARABLE) -- never a fabricated 0.0 for whichever side didn't
    # score it.
    baseline: Optional[float]
    candidate: Optional[float]
    delta: Optional[float]
    direction: str
    # Threshold-based pass/warn/fail. Kept for callers who explicitly want a
    # gate, but NOT what the default summary reports -- a fixed % threshold
    # isn't meaningful across every metric (see summary()). Read the percentage
    # difference (`delta` / `pct_change`) and decide acceptability yourself.
    grade: DeltaGrade
    # How many real samples fed each mean, and how spread out they were --
    # .mean alone can't tell you a regression rode on far fewer samples than
    # the baseline, or that a flat mean hid much wider variance. See
    # RunComparison.summary()'s coverage warning, which reads these.
    baseline_count: int = 0
    candidate_count: int = 0
    baseline_std: float = 0.0
    candidate_std: float = 0.0
    # The change as a percentage of the baseline (candidate vs baseline). None
    # when incomparable or the baseline is <= 0 (percent-of-zero is undefined).
    pct_change: Optional[float] = None


@dataclass
class TaskDelta:
    task: str
    metric: str
    baseline: Optional[float]
    candidate: Optional[float]
    delta: Optional[float]
    direction: str
    grade: DeltaGrade
    baseline_count: int = 0
    candidate_count: int = 0
    baseline_std: float = 0.0
    candidate_std: float = 0.0
    pct_change: Optional[float] = None


class RunComparison:
    """Compare a ``candidate`` run against a ``baseline`` run on the same config."""

    def __init__(
        self,
        baseline: RunResult,
        candidate: RunResult,
        *,
        pass_threshold: float = 0.02,
        warn_threshold: float = 0.05,
    ) -> None:
        self.baseline = baseline
        self.candidate = candidate
        self.pass_threshold = pass_threshold
        self.warn_threshold = warn_threshold
        self._diff = RunDiff(baseline, candidate)

    # -- metric directions (read from stored per-score dicts) --------------
    def _directions(self) -> dict[str, Direction]:
        return metric_directions((self.baseline, self.candidate))

    def _direction(self, metric: str) -> Direction:
        return direction_for(metric, self._directions())

    def _grade(self, delta: Optional[float], direction: Direction) -> DeltaGrade:
        # None means "this metric wasn't scored on both sides" -- nothing to
        # grade, not a fabricated PASS/FAIL against a phantom zero delta.
        if delta is None:
            return DeltaGrade.NOT_COMPARABLE
        return grade_delta(delta, direction, self.pass_threshold, self.warn_threshold)

    # -- overall per-metric deltas ----------------------------------------
    def metric_deltas(self) -> list[MetricDelta]:
        out: list[MetricDelta] = []
        for name, info in self._diff.metric_deltas().items():
            d = self._direction(name)
            b_stat = self.baseline.stats.get(name)
            c_stat = self.candidate.stats.get(name)
            out.append(MetricDelta(
                metric=name, baseline=info["baseline"], candidate=info["contrast"],
                delta=info["delta"], direction=d.value, grade=self._grade(info["delta"], d),
                baseline_count=b_stat.count if b_stat else 0,
                candidate_count=c_stat.count if c_stat else 0,
                baseline_std=b_stat.std if b_stat else 0.0,
                candidate_std=c_stat.std if c_stat else 0.0,
                pct_change=relative_pct(info["baseline"], info["delta"]),
            ))
        return out

    # -- per-task deltas (engine-agnostic) --------------------------------
    def _per_task_stats(self, run: RunResult) -> dict[tuple[str, str], dict[str, float]]:
        stats: dict[tuple[str, str], dict[str, float]] = {}
        # Benchmark engine: stats keyed "task:metric" -- Stat already tracks count/std.
        for key, stat in run.stats.items():
            if ":" in key:
                task, metric = key.split(":", 1)
                stats[(task, metric)] = {"mean": stat.mean, "count": stat.count, "std": stat.std}
        # Native engine: group predictions by task via their per-score dicts.
        acc: dict[tuple[str, str], list[float]] = defaultdict(list)
        for p in run.predictions:
            for s in p.metadata.get("scores", []):
                name, val = s.get("name"), s.get("value")
                if name is not None and isinstance(val, (int, float)):
                    acc[(p.task or "", name)].append(float(val))
        for k, vals in acc.items():
            if vals and k not in stats:  # don't override benchmark stats
                stats[k] = {
                    "mean": sum(vals) / len(vals),
                    "count": len(vals),
                    "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                }
        return stats

    def per_task_deltas(self) -> list[TaskDelta]:
        b = self._per_task_stats(self.baseline)
        c = self._per_task_stats(self.candidate)
        out: list[TaskDelta] = []
        for (task, metric) in sorted(set(b) | set(c)):
            bs, cs = b.get((task, metric)), c.get((task, metric))
            d = self._direction(metric)
            # None (not a fabricated 0.0) when either side never scored this
            # task/metric pair at all -- same fix as the blended metric_deltas()
            # above, applied at the per-task granularity.
            delta = (cs["mean"] - bs["mean"]) if (bs is not None and cs is not None) else None
            out.append(TaskDelta(
                task=task, metric=metric,
                baseline=bs["mean"] if bs else None, candidate=cs["mean"] if cs else None,
                delta=delta, direction=d.value, grade=self._grade(delta, d),
                baseline_count=bs["count"] if bs else 0, candidate_count=cs["count"] if cs else 0,
                baseline_std=bs["std"] if bs else 0.0, candidate_std=cs["std"] if cs else 0.0,
                pct_change=relative_pct(bs["mean"] if bs else None, delta),
            ))
        return out

    # -- logging escalation: fires wherever these are called, not just via
    # .summary()/.coverage_warnings() -- separate from metric_deltas()/
    # per_task_deltas() themselves (coverage_warnings() calls those, so
    # warning *inside* them would recurse).
    def _warn_if_issues(self) -> None:
        if self.has_failures:
            fs = self.failure_summary()
            logger.warning(
                "RunComparison: baseline had %d failed sample(s), candidate had %d "
                "-- excluded from means, not counted as wrong. See .failure_summary().",
                fs["baseline_failed"], fs["candidate_failed"],
            )
        cw = self.coverage_warnings()
        if cw:
            logger.warning("RunComparison: sample coverage differs -- %s", cw)

    # -- grading rollups ---------------------------------------------------
    def grades(self) -> dict[str, DeltaGrade]:
        """Grade per overall metric."""
        self._warn_if_issues()
        return {md.metric: md.grade for md in self.metric_deltas()}

    def grade(self) -> DeltaGrade:
        """Worst grade across all tasks/metrics — the ship / no-ship headline."""
        self._warn_if_issues()
        grades = [td.grade for td in self.per_task_deltas()] or [md.grade for md in self.metric_deltas()]
        return max(grades, key=lambda g: _GRADE_ORDER[g]) if grades else DeltaGrade.PASS

    # -- retention ---------------------------------------------------------
    def retention(self, metric: Optional[str] = None):
        """Fraction of the baseline's quality the candidate keeps (1.0 = parity).

        candidate/baseline for MAXIMIZE metrics, baseline/candidate for MINIMIZE
        (so 'lower is better' metrics also read as 'higher retention = better').
        Returns a dict over all metrics, or a single float for a named metric.

        Retention is a ratio, and a ratio is only a meaningful "how much
        quality did we keep" answer for a non-negative-range metric -- for a
        metric that can go negative (some judge/reward scores), a ratio of
        two negative numbers can look like an improvement or a loss with the
        wrong sign entirely (e.g. baseline=-2.0, candidate=-1.0 is a real
        improvement, closer to zero, but -1.0/-2.0 = 0.5 reads as "lost half
        the quality"). Rather than inventing a formula for an ill-defined
        case, this logs a warning and reports ``nan`` -- an honest "not
        computable," not a silently misleading number.
        """
        self._warn_if_issues()
        mds = {md.metric: md for md in self.metric_deltas()}
        out: dict[str, float] = {}
        for name in ([metric] if metric else list(mds)):
            md = mds.get(name)
            if md is None or md.baseline is None or md.candidate is None:
                continue
            if md.baseline < 0 or md.candidate < 0:
                logger.warning(
                    "retention(%r): baseline=%.6g candidate=%.6g -- not meaningful as a "
                    "ratio for a metric with negative values, reporting nan instead of "
                    "a misleading number.", name, md.baseline, md.candidate,
                )
                out[name] = float("nan")
                continue
            if self._direction(name) == Direction.MAXIMIZE:
                out[name] = md.candidate / md.baseline if md.baseline else float("nan")
            else:
                out[name] = md.baseline / md.candidate if md.candidate else float("nan")
        return out.get(metric) if metric else out

    # -- significance (shared, by sample_id) ------------------------------
    def significance(self, metric: Optional[str] = None) -> dict[str, Any]:
        return paired_bootstrap(score_pairs(self.baseline, self.candidate, metric))

    # -- size / latency tradeoff (everything measured, nothing caller-supplied) --
    def tradeoff(self, *, metric: Optional[str] = None) -> dict[str, Any]:
        """Quality-vs-cost view. Every number here is measured, never passed
        in by a caller: latency from ``RunResult.perf`` (timed by ``Runner``
        while it ran), size from ``RunResult.model_size`` (introspected for
        local backends, identity-only for hosted-API ones -- see
        ``Model.model_info()``). When either side is API-based there is no
        on-disk size to ratio, so ``size_ratio``/``quality_per_mb`` are
        omitted and ``api_based`` names the two models instead."""
        self._warn_if_issues()
        mds = {md.metric: md for md in self.metric_deltas()}
        if metric is None:
            metric = next(iter(mds), None)
        out: dict[str, Any] = {"metric": metric}
        if metric in mds:
            out["baseline_quality"] = mds[metric].baseline
            out["candidate_quality"] = mds[metric].candidate
            out["retention"] = self.retention(metric)

        # or {}: model_size is None (not {}) when RunConfig.track_performance
        # was never enabled on that run -- treated the same as "measured but
        # nothing recorded" here, since tradeoff() already omits size_ratio
        # etc. whenever a side has no size_mb, whatever the reason.
        base_size, cand_size = self.baseline.model_size or {}, self.candidate.model_size or {}
        baseline_size_mb, candidate_size_mb = base_size.get("size_mb"), cand_size.get("size_mb")
        if baseline_size_mb and candidate_size_mb:
            out["baseline_size_mb"] = baseline_size_mb
            out["candidate_size_mb"] = candidate_size_mb
            out["size_ratio"] = candidate_size_mb / baseline_size_mb  # <1 = smaller
            if metric in mds:
                # retention(), not the raw candidate value: for a MINIMIZE
                # metric (latency, cost) a bigger raw number is WORSE, so
                # dividing it by size and calling the result "quality" would
                # be backwards. retention() already normalizes for direction
                # (1.0 = parity, regardless of whether the metric counts up
                # or down), so "quality per mb" means the same thing for
                # every metric.
                out["quality_per_mb"] = self.retention(metric) / candidate_size_mb
        elif base_size or cand_size:
            # No real size_mb on at least one side -- either it's a hosted
            # -API/callable model (never measurable at all) or a local
            # backend that's marked is_local but doesn't actually introspect
            # size yet (e.g. VLLMModel today -- see model/vllm_gen.py). Either
            # way there's no size ratio to report; name the two models
            # instead of silently reporting nothing at all (the previous
            # behavior when both sides were "local" but neither had size_mb).
            out["api_based"] = {
                "baseline": base_size.get("model_name", self.baseline.model_spec),
                "candidate": cand_size.get("model_name", self.candidate.model_spec),
            }

        baseline_latency_ms = ((self.baseline.perf or {}).get("latency_ms") or {}).get("mean") or None
        candidate_latency_ms = ((self.candidate.perf or {}).get("latency_ms") or {}).get("mean") or None
        if baseline_latency_ms and candidate_latency_ms:
            out["baseline_latency_ms"] = baseline_latency_ms
            out["candidate_latency_ms"] = candidate_latency_ms
            out["latency_ratio"] = candidate_latency_ms / baseline_latency_ms
            out["speedup"] = baseline_latency_ms / candidate_latency_ms  # >1 = faster
        return out

    # -- performance (latency/throughput, measured automatically) ---------
    def performance(self) -> dict[str, Any]:
        """Baseline-vs-candidate latency/throughput, straight from each run's
        measured ``RunResult.perf`` -- no caller-supplied numbers needed
        (unlike ``tradeoff()``, which folds in quality + optional size).

        A side's fields here are ``None`` (not ``{}``/``0.0``) when that run
        never enabled ``RunConfig.track_performance`` at all -- distinct from
        an empty/zeroed value, which means it *was* tracked but nothing was
        timed (e.g. a ``reads_actual_output`` run). Mixing a tracked and an
        untracked run is fine: the tracked side still reports its real
        numbers, the untracked side reports ``None`` for its own fields, and
        ``speedup`` (which needs both) is simply omitted.
        """
        base_tracked = self.baseline.perf is not None
        cand_tracked = self.candidate.perf is not None
        base_lat = (self.baseline.perf.get("latency_ms") or {}) if base_tracked else None
        cand_lat = (self.candidate.perf.get("latency_ms") or {}) if cand_tracked else None
        base_thr = (self.baseline.perf.get("throughput") or {}) if base_tracked else None
        cand_thr = (self.candidate.perf.get("throughput") or {}) if cand_tracked else None
        out: dict[str, Any] = {
            "baseline_latency_ms": base_lat,
            "candidate_latency_ms": cand_lat,
            "baseline_throughput_rps": base_thr.get("rps", 0.0) if base_thr is not None else None,
            "candidate_throughput_rps": cand_thr.get("rps", 0.0) if cand_thr is not None else None,
            "baseline_output_tokens_per_sec": base_thr.get("output_tokens_per_sec", 0.0) if base_thr is not None else None,
            "candidate_output_tokens_per_sec": cand_thr.get("output_tokens_per_sec", 0.0) if cand_thr is not None else None,
        }
        if base_lat is not None and cand_lat is not None and base_lat.get("count") and cand_lat.get("count"):
            out["speedup"] = base_lat["mean"] / cand_lat["mean"] if cand_lat["mean"] else float("nan")
        return out

    # -- cost (real measured tokens; pricing is the one thing you must supply) --
    def cost(self, baseline_pricing: dict[str, float], candidate_pricing: dict[str, float]) -> dict[str, Any]:
        """Dollar cost of baseline vs. candidate. Token counts are real,
        provider-reported numbers already captured on each ``RunResult``
        (see ``RunResult.token_usage``/``.cost()``) -- never estimated here.
        ``$``/token pricing has no library default (it isn't measurable and
        goes stale) so it's the one thing you pass in, per side, since a
        baseline and candidate are often different paid models with
        different prices. Returns ``{}`` for either side that recorded no
        token usage at all (nothing to price)."""
        out: dict[str, Any] = {}
        base_cost = self.baseline.cost(baseline_pricing)
        cand_cost = self.candidate.cost(candidate_pricing)
        if base_cost is not None:
            out["baseline_cost"] = base_cost
        if cand_cost is not None:
            out["candidate_cost"] = cand_cost
        if base_cost and cand_cost:
            out["cost_ratio"] = cand_cost / base_cost
        return out

    # -- sample browser (delegated to RunDiff, aligned by sample_id) ------
    def regressed(self, metric_name: str = "") -> list[Prediction]:
        """Samples correct in ``baseline`` but wrong in ``candidate`` -- what broke.

        Named for what it reports (a two-way flip between exactly these two
        runs), not "recently" -- there's no history or tracking involved, just
        this one baseline-vs-candidate comparison. Was ``newly_wrong()``.
        """
        return self._diff.regressed(metric_name)

    def improved(self, metric_name: str = "") -> list[Prediction]:
        """Samples wrong in ``baseline`` but fixed in ``candidate``. Was ``newly_correct()``."""
        return self._diff.improved(metric_name)

    def still_wrong(self, metric_name: str = "") -> list[Prediction]:
        return self._diff.still_wrong(metric_name)

    def sample_diff(self, metric_name: str = "") -> list[dict]:
        return self._diff.sample_diff(metric_name)

    def sample_summary(self, metric_name: str = "") -> dict[str, int]:
        return self._diff.sample_summary(metric_name)

    # -- failed-sample visibility -------------------------------------------
    def failure_summary(self) -> dict[str, Any]:
        """Per-run failed-sample counts and errors.

        ``metric_deltas()``/``per_task_deltas()``/``retention()``/``tradeoff()``
        all read per-sample scores from ``Prediction.metadata["scores"]``, which
        is empty on a sample that errored during annotation/scoring (see
        ``Runner.run()``'s per-sample fallback) -- such samples are silently
        **excluded** from those means, not counted as a miss. A run with several
        failures can therefore report a better mean than a clean run would,
        purely because its denominator shrank. Call this (or check
        ``has_failures``) before trusting a grade; ``regressed()`` is the one
        view that *does* catch a failure correctly (via ``Prediction.correct``,
        which is ``False`` on the fallback), everything else needs this check.
        """
        return {
            "baseline_failed": self.baseline.failed_count,
            "candidate_failed": self.candidate.failed_count,
            "baseline_errors": self.baseline.errors,
            "candidate_errors": self.candidate.errors,
        }

    @property
    def has_failures(self) -> bool:
        return bool(self.baseline.failed_count or self.candidate.failed_count)

    def coverage_warnings(self) -> list[str]:
        """Metrics (or task/metric pairs) whose baseline/candidate sample
        counts diverge.

        ``metric_deltas()``/``per_task_deltas()`` derive their metric set purely
        from whichever names actually appear in each run's stats/scores -- there
        is no check against an "expected" scorer list, and a metric that failed
        on every sample (or was scored under a different config) simply never
        shows up, silently. A count mismatch between baseline and candidate for
        the *same* metric name is the visible symptom of that: it means the two
        means were computed over different sample sets, not a like-for-like
        comparison -- treat a large mismatch as a reason to distrust that
        metric's grade, not just a curiosity.

        Checks both the blended (overall) counts *and* the per-task counts --
        two tasks can individually have mismatched N (5 vs 3, 5 vs 7) while the
        *totals* coincidentally agree (10 vs 10), invisible to a blended-only
        check.
        """
        warnings = []
        for md in self.metric_deltas():
            if md.baseline_count != md.candidate_count:
                warnings.append(
                    f"{md.metric}: baseline n={md.baseline_count}, candidate n={md.candidate_count}"
                )
        for td in self.per_task_deltas():
            if td.baseline_count != td.candidate_count:
                warnings.append(
                    f"{td.task}/{td.metric}: baseline n={td.baseline_count}, "
                    f"candidate n={td.candidate_count}"
                )
        return warnings

    def summary(self) -> str:
        lines = []
        if self.has_failures:
            fs = self.failure_summary()
            lines.append(
                f"  WARNING: baseline had {fs['baseline_failed']} failed sample(s), "
                f"candidate had {fs['candidate_failed']} -- EXCLUDED from the means "
                f"below, not counted as wrong. See .failure_summary()."
            )
        cw = self.coverage_warnings()
        if cw:
            lines.append(
                "  WARNING: sample coverage differs per metric (means computed over "
                "different Ns, not a like-for-like comparison):"
            )
            lines += [f"    {w}" for w in cw]
        mds = self.metric_deltas()
        # Just the percentage differences per metric -- no pass/warn/fail and no
        # better/worse framing. A fixed threshold isn't meaningful across every
        # metric, so it's the reader's job to interpret the numbers for their own
        # metrics. (An opt-in gate is still available via .grade()/.grades() with
        # your own thresholds.)
        lines += [
            f"Comparison: {self.baseline.run_id} (baseline) -> {self.candidate.run_id} (candidate)",
            "  per-metric (baseline -> candidate | absolute delta | relative %):",
        ]
        for md in mds:
            lines.append("    " + _delta_line(md.metric, md))
        tasks = self.per_task_deltas()
        # Show per-task detail whenever there's more than one distinct task --
        # comparing this to len(metric_deltas()) instead (the old check) could
        # wrongly hide a genuinely different per-task breakdown whenever the
        # row counts happened to coincide numerically (e.g. 2 metrics each
        # scored on a different, non-overlapping single task).
        if len({td.task for td in tasks}) > 1:
            lines.append("  per-task:")
            for td in tasks:
                lines.append("    " + _delta_line(f"{td.task}/{td.metric}", td))
        sm = self.sample_summary()
        lines.append(
            f"  samples: {sm['newly_wrong']} went correct->wrong, "
            f"{sm['newly_correct']} went wrong->correct"
        )
        perf = self.performance()
        if (perf["baseline_latency_ms"] is not None and perf["candidate_latency_ms"] is not None
                and perf["baseline_latency_ms"].get("count") and perf["candidate_latency_ms"].get("count")):
            bl, cl = perf["baseline_latency_ms"], perf["candidate_latency_ms"]
            line = (
                f"  performance: latency {bl['mean']:.0f}ms -> {cl['mean']:.0f}ms, "
                f"throughput {perf['baseline_throughput_rps']:.2f} -> "
                f"{perf['candidate_throughput_rps']:.2f} req/s "
                f"(speedup={perf['speedup']:.2f}x)"
            )
            # Only when the backend actually reported token usage (hosted
            # APIs) -- 0 tok/s for a local/callable backend would
            # misleadingly read as "measured zero" rather than "unavailable".
            if perf["baseline_output_tokens_per_sec"] or perf["candidate_output_tokens_per_sec"]:
                line += (
                    f", {perf['baseline_output_tokens_per_sec']:.1f} -> "
                    f"{perf['candidate_output_tokens_per_sec']:.1f} out-tok/s"
                )
            lines.append(line)
        size_line = self._model_size_line()
        if size_line:
            lines.append(f"  {size_line}")
        return "\n".join(lines)

    def _model_size_line(self) -> str:
        """Model size for both sides, or '' if neither run recorded any
        (RunResult.model_size is None when RunConfig.track_performance was
        never enabled on that run, {} for a tracked run with nothing to
        report -- both treated the same way here)."""
        base_size, cand_size = self.baseline.model_size or {}, self.candidate.model_size or {}
        if not base_size and not cand_size:
            return ""
        baseline_size_mb, candidate_size_mb = base_size.get("size_mb"), cand_size.get("size_mb")
        if baseline_size_mb and candidate_size_mb:
            return (
                f"model size: {baseline_size_mb:.1f} MB -> {candidate_size_mb:.1f} MB "
                f"({base_size.get('model_name', '?')} -> {cand_size.get('model_name', '?')})"
            )
        # Either a hosted-API/callable model (nothing to measure) or a local
        # backend marked is_local that doesn't actually introspect size yet
        # (e.g. VLLMModel) -- either way, no fake "0.0 MB" placeholder.
        base_name = base_size.get("model_name", self.baseline.model_spec)
        cand_name = cand_size.get("model_name", self.candidate.model_spec)
        return f"model: {base_name} -> {cand_name} (no measurable size)"
