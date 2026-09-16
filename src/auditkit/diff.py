from __future__ import annotations

from enum import Enum
from typing import Iterable, Optional

from .report import RunResult, Prediction
from .types import Direction


class DeltaGrade(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    # A metric that was scored on only one side (added after the baseline
    # ran, or inapplicable to every sample there) has no real delta to
    # grade at all. Previously the missing side silently defaulted to 0.0,
    # producing a fabricated (and often FAIL-grade) delta indistinguishable
    # from a genuine regression to zero. This is the explicit "nothing to
    # compare" signal instead.
    NOT_COMPARABLE = "not_comparable"


def grade_delta(
    delta: float,
    direction: Direction = Direction.MAXIMIZE,
    pass_threshold: float = 0.02,
    warn_threshold: float = 0.05,
) -> DeltaGrade:
    """Grade a candidate−baseline delta, accounting for the metric's direction.

    The *regression* is how much worse the candidate is: for a MAXIMIZE metric a
    drop (negative delta) is a regression; for a MINIMIZE metric (latency, WER,
    toxicity) a rise (positive delta) is. An improvement, or a regression within
    ``pass_threshold``, is PASS; within ``warn_threshold``, WARN; beyond, FAIL.
    """
    regression = -delta if direction == Direction.MAXIMIZE else delta
    if regression <= pass_threshold:
        return DeltaGrade.PASS
    if regression <= warn_threshold:
        return DeltaGrade.WARN
    return DeltaGrade.FAIL


def metric_directions(runs: Iterable[RunResult]) -> dict[str, Direction]:
    """Every metric name -> its declared :class:`Direction`, read straight
    from each run's stored per-score dicts (``Prediction.metadata["scores"]``,
    populated by ``Runner.score_one()`` from ``Metric.direction`` -- see
    ``metric.py``'s compulsory-direction enforcement). Shared by
    :class:`RunDiff`, :class:`~auditkit.comparison.RunComparison` (2 runs),
    and :class:`~auditkit.model_compare.CompareResult` (N runs) so all three
    read direction the exact same way, not independently-maintained lookups
    that could drift apart.
    """
    dirs: dict[str, Direction] = {}
    for run in runs:
        for p in run.predictions:
            for s in p.metadata.get("scores", []):
                name, raw = s.get("name"), s.get("direction")
                if name and name not in dirs and raw is not None:
                    try:
                        dirs[name] = Direction(raw)
                    except ValueError:
                        dirs[name] = Direction.MAXIMIZE
    return dirs


def direction_for(metric: str, dirs: dict[str, Direction]) -> Direction:
    """Look up *metric*'s direction in *dirs* (from :func:`metric_directions`),
    falling back from a benchmark-style ``"task:metric"`` key to the base
    metric name, and finally to MAXIMIZE if the metric was never scored at
    all (nothing to look up)."""
    base = metric.split(":", 1)[1] if ":" in metric else metric
    return dirs.get(metric) or dirs.get(base) or Direction.MAXIMIZE


def best_model(scores: dict[str, float], direction: Direction) -> Optional[str]:
    """The winning model name for *scores* (``{model_name: value}``),
    respecting *direction* -- the lowest value wins for MINIMIZE metrics
    (latency, error rate, ...), not just whichever is numerically largest."""
    if not scores:
        return None
    if direction == Direction.MINIMIZE:
        return min(scores, key=scores.get)
    return max(scores, key=scores.get)


def relative_pct(baseline: Optional[float], delta: Optional[float]) -> Optional[float]:
    """The change as a percentage of the baseline (``delta / baseline * 100``).

    This is the "percentage difference in the metric" the comparison surfaces so
    the reader can judge whether a change matters, rather than the library
    imposing a fixed pass/warn/fail threshold that isn't meaningful across every
    metric. Returns ``None`` when the baseline is missing or ``<= 0`` (a percent
    of zero/negative is undefined -- fall back to the absolute delta there).
    """
    if baseline is None or delta is None or baseline <= 0:
        return None
    return delta / baseline * 100.0




class RunDiff:
    def __init__(self, baseline: RunResult, contrast: RunResult) -> None:
        self.baseline = baseline
        self.contrast = contrast

    def _directions(self) -> dict[str, Direction]:
        return metric_directions((self.baseline, self.contrast))

    def _direction(self, metric: str) -> Direction:
        return direction_for(metric, self._directions())

    def metric_deltas(self) -> dict[str, dict[str, float]]:
        all_metrics = set(self.baseline.stats.keys()) | set(self.contrast.stats.keys())
        result: dict[str, dict[str, float]] = {}
        for name in sorted(all_metrics):
            in_baseline = name in self.baseline.stats
            in_contrast = name in self.contrast.stats
            b_mean = self.baseline.stats[name].mean if in_baseline else None
            c_mean = self.contrast.stats[name].mean if in_contrast else None
            result[name] = {
                "baseline": b_mean,
                "contrast": c_mean,
                # None (not 0.0 - c_mean's old fallback) when either side
                # never scored this metric at all -- a fabricated delta
                # against a phantom zero is worse than an honest "no data".
                "delta": (c_mean - b_mean) if (in_baseline and in_contrast) else None,
                "comparable": in_baseline and in_contrast,
            }
        return result

    def grade(self, metric_name: str,
              pass_threshold: float = 0.02,
              warn_threshold: float = 0.05) -> DeltaGrade:
        deltas = self.metric_deltas()
        if metric_name not in deltas:
            raise KeyError(f"metric {metric_name!r} not found in deltas")
        info = deltas[metric_name]
        if not info["comparable"]:
            return DeltaGrade.NOT_COMPARABLE
        return grade_delta(info["delta"], self._direction(metric_name), pass_threshold, warn_threshold)

    def grades(self, pass_threshold: float = 0.02,
               warn_threshold: float = 0.05) -> dict[str, DeltaGrade]:
        return {name: self.grade(name, pass_threshold, warn_threshold)
                for name in self.metric_deltas()}

    @staticmethod
    def _predictions_by_id(result: RunResult) -> dict[str, Prediction]:
        return {p.sample_id: p for p in result.predictions}

    @staticmethod
    def _is_correct(pred: Prediction | None, metric_name: str) -> bool | None:
        if pred is None:
            return None
        if not metric_name:
            return pred.correct
        for score_doc in pred.metadata.get("scores", []):
            if score_doc.get("name") == metric_name:
                # Use the metric's own gate (passed) when it has a threshold;
                # bool(value) was wrong for any thresholded/continuous score
                # (e.g. value=0.6, threshold=0.7, passed=False -- bool(0.6)
                # is truthy, silently reporting a failing sample as correct).
                # Falls back to the same value==1.0 convention Runner.score_one()
                # itself uses for Prediction.correct when there's no threshold.
                passed = score_doc.get("passed")
                if passed is not None:
                    return bool(passed)
                return score_doc.get("value") == 1.0
        return None

    def sample_diff(self, metric_name: str = "") -> list[dict]:
        base_by_id = self._predictions_by_id(self.baseline)
        cont_by_id = self._predictions_by_id(self.contrast)
        all_ids = set(base_by_id.keys()) | set(cont_by_id.keys())

        entries = []
        for sid in sorted(all_ids):
            bp = base_by_id.get(sid)
            cp = cont_by_id.get(sid)
            if bp is None or cp is None:
                change = "unknown"
            else:
                b_correct = self._is_correct(bp, metric_name)
                c_correct = self._is_correct(cp, metric_name)
                if b_correct is True and c_correct is False:
                    change = "newly_wrong"
                elif b_correct is False and c_correct is True:
                    change = "newly_correct"
                elif b_correct is False and c_correct is False:
                    change = "still_wrong"
                elif b_correct is True and c_correct is True:
                    change = "still_correct"
                else:
                    change = "unknown"

            entries.append({
                "sample_id": sid,
                "baseline_correct": self._is_correct(bp, metric_name) if bp else None,
                "contrast_correct": self._is_correct(cp, metric_name) if cp else None,
                "change": change,
            })
        return entries

    def _filter_by_change(self, change: str, metric_name: str = "") -> list[Prediction]:
        cont_by_id = self._predictions_by_id(self.contrast)
        result = []
        for entry in self.sample_diff(metric_name):
            if entry["change"] == change:
                sid = entry["sample_id"]
                cp = cont_by_id.get(sid)
                if cp is not None:
                    result.append(cp)
        return result

    def regressed(self, metric_name: str = "") -> list[Prediction]:
        """Samples correct in ``baseline`` but wrong in ``contrast`` -- what broke.

        Was ``newly_wrong()`` -- renamed because "newly" implied history/
        tracking over time that doesn't exist here; this is always a one-shot,
        two-run comparison.
        """
        return self._filter_by_change("newly_wrong", metric_name)

    def improved(self, metric_name: str = "") -> list[Prediction]:
        """Samples wrong in ``baseline`` but fixed in ``contrast``. Was ``newly_correct()``."""
        return self._filter_by_change("newly_correct", metric_name)

    def still_wrong(self, metric_name: str = "") -> list[Prediction]:
        return self._filter_by_change("still_wrong", metric_name)

    def still_correct(self, metric_name: str = "") -> list[Prediction]:
        return self._filter_by_change("still_correct", metric_name)

    def sample_summary(self, metric_name: str = "") -> dict[str, int]:
        counts: dict[str, int] = {
            "newly_wrong": 0,
            "newly_correct": 0,
            "still_wrong": 0,
            "still_correct": 0,
            "unknown": 0,
        }
        for entry in self.sample_diff(metric_name):
            counts[entry["change"]] += 1
        return counts

    def summary(self) -> str:
        # Reports the change per metric -- absolute delta AND relative % -- and
        # nothing else. No pass/warn/fail verdict, and no better/worse framing:
        # a fixed threshold isn't meaningful across every metric, and it's the
        # reader's job to interpret the numbers for their own metrics. (An opt-in
        # threshold gate is still available via .grade()/.grades().) ASCII only
        # -- this string is often printed to a console that can't encode
        # arrows/greek.
        lines = [f"baseline {self.baseline.run_id} -> contrast {self.contrast.run_id}"]
        for name, info in sorted(self.metric_deltas().items()):
            if not info["comparable"]:
                lines.append(f"  {name}: N/A -- only scored on one side")
                continue
            pct = relative_pct(info["baseline"], info["delta"])
            pct_s = f"{pct:+.1f}% rel" if pct is not None else "rel n/a"
            lines.append(f"  {name}: {info['baseline']:.4f} -> {info['contrast']:.4f}  "
                         f"delta={info['delta']:+.4f} ({pct_s})")
        return "\n".join(lines)
