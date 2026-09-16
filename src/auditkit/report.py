"""The result of a run: :class:`Prediction` rows and the :class:`RunResult`.

Every run returns the same shape regardless of task or technique: aggregate
:class:`~auditkit.score.Stat` per metric, a ``headline`` of means, and one
:class:`Prediction` per sample — the answer browser. Reporting helpers
(:meth:`RunResult.summary`, :meth:`RunResult.metric_table`,
:meth:`RunResult.wrong_only`, :meth:`RunResult.save`) read off that.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from .score import Stat


@dataclass
class Prediction:
    """One sample's record: what was asked, what came back, and how it scored."""

    run_id: str
    task: str
    sample_id: str
    prompt: str
    raw_output: Optional[str]
    parsed_answer: Optional[str]
    expected: Optional[str]
    correct: Optional[bool]
    score: Optional[float]
    choice_likelihoods: Optional[list[float]] = None
    context: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_doc(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task": self.task,
            "sample_id": self.sample_id,
            "prompt": self.prompt,
            "raw_output": self.raw_output,
            "parsed_answer": self.parsed_answer,
            "expected": self.expected,
            "correct": self.correct,
            "score": self.score,
            "choice_likelihoods": self.choice_likelihoods,
            "context": self.context,
            "metadata": self.metadata,
        }


@dataclass
class RunResult:
    """What ``evaluate()`` returns: aggregates, per-sample rows, and a headline."""

    run_id: str
    fingerprint: str
    stats: dict[str, Stat]
    predictions: list[Prediction]
    headline: dict[str, float]
    config: Any = None
    model_spec: Any = None
    errors: list[dict] = field(default_factory=list)
    failed_count: int = 0
    experiment_name: str | None = None
    tags: list[str] = field(default_factory=list)
    # Efficiency/systems metrics — latency (ms) and throughput (req/s),
    # measured by timing the run's own model calls (see Runner.run()).
    # None unless RunConfig.track_performance=True was set on this run (the
    # default is False -- most callers scoring a quick eval don't want this
    # measured). When tracking IS on: {} on the reads_actual_output path (no
    # model call happens there at all) or when a run made zero requests --
    # that's a genuinely different state from "never asked to measure" (None).
    perf: Optional[dict[str, Any]] = None
    # Model size/identity — never caller-supplied. Introspected (real
    # parameter count/sparsity/size) for local backends; identity-only
    # ({"is_local": False, "model_name": ...}) for hosted-API backends,
    # since there is no checkpoint to measure. See Model.model_info().
    # None unless RunConfig.track_performance=True (see `perf` above).
    model_size: Optional[dict[str, Any]] = None
    # Real, provider-reported token counts (prompt_tokens/completion_tokens/
    # total_tokens) — populated automatically by Runner from each backend's
    # actual API response (see model/openai.py etc.), never estimated. All
    # zero for local/callable backends that don't report usage. This is the
    # only input .cost() needs from the run itself; $/token pricing is
    # deliberately NOT bundled in the library (see .cost()'s docstring).
    # None unless RunConfig.track_performance=True (see `perf` above).
    token_usage: Optional[dict[str, int]] = None

    def cost(self, pricing: dict[str, float]) -> Optional[float]:
        """Dollar cost of this run, given *pricing* — never a library default.

        There is no live pricing API and provider prices change over time, so
        a hardcoded table would silently go stale; you must supply
        ``{"input_per_1m": <$ per 1M prompt tokens>, "output_per_1m": <$ per
        1M completion tokens>}`` yourself (check your provider's current
        pricing page). Returns ``None`` if this run recorded no token usage
        at all (a local/callable backend, nothing was ever generated, or
        ``RunConfig.track_performance`` was never enabled on this run) —
        there is nothing to price.
        """
        if not self.token_usage or not self.token_usage.get("total_tokens"):
            return None
        return (
            self.token_usage["prompt_tokens"] / 1_000_000 * pricing["input_per_1m"]
            + self.token_usage["completion_tokens"] / 1_000_000 * pricing["output_per_1m"]
        )

    @property
    def samples(self) -> list[Prediction]:
        return self.predictions

    def metric_table(self) -> list[dict[str, Any]]:
        """One row per metric: name, mean, stderr, and n."""
        rows = []
        for name, stat in self.stats.items():
            rows.append({
                "name": name,
                "mean": stat.mean,
                "stderr": stat.stderr,
                "n": stat.count,
            })
        return rows

    def summary(self) -> str:
        """A compact human-readable metric table (mean ± stderr over n)."""
        lines = [f"run {self.run_id} (fingerprint {self.fingerprint})"]
        for row in self.metric_table():
            lines.append(
                f"  {row['name']}: {row['mean']:.4f} ± {row['stderr']:.4f}  (n={row['n']})"
            )
        perf_line = self._perf_line()
        if perf_line:
            lines.append(f"  {perf_line}")
        size_line = self._model_size_line()
        if size_line:
            lines.append(f"  {size_line}")
        if self.token_usage and self.token_usage.get("total_tokens"):
            lines.append(
                f"  tokens: {self.token_usage['prompt_tokens']:,} in + "
                f"{self.token_usage['completion_tokens']:,} out = "
                f"{self.token_usage['total_tokens']:,} total "
                f"(pass pricing to .cost() for a $ figure)"
            )
        return "\n".join(lines)

    def _perf_line(self) -> str:
        """One-line latency/throughput summary, or '' if nothing was measured
        (the reads_actual_output path makes no model call to time, or
        RunConfig.track_performance was never enabled on this run)."""
        if not self.perf:
            return ""
        lat = self.perf.get("latency_ms") or {}
        thr = self.perf.get("throughput") or {}
        if not lat.get("count"):
            return ""
        line = (
            f"perf: latency={lat['mean']:.0f}ms "
            f"(n={lat['count']} calls) | throughput={thr.get('rps', 0.0):.2f} req/s"
        )
        # Only shown when the backend actually reported token usage (hosted
        # APIs) -- 0 tok/s for a local/callable backend would misleadingly
        # read as "measured zero" rather than "not available here".
        if thr.get("total_tokens_per_sec"):
            line += f" | {thr['output_tokens_per_sec']:.1f} out-tok/s, {thr['total_tokens_per_sec']:.1f} total-tok/s"
        return line

    def _model_size_line(self) -> str:
        """One-line model size summary, or '' if nothing was recorded at all."""
        ms = self.model_size
        if not ms:
            return ""
        if not ms.get("is_local"):
            return f"model: {ms.get('model_name', '?')} (not a local model -- no measurable size)"
        return (
            f"model: {ms.get('model_name', '?')} -- "
            f"{ms.get('total_params', 0):,} params "
            f"({ms.get('sparsity', 0.0):.1%} sparse), {ms.get('size_mb', 0.0):.1f} MB"
        )

    def wrong_only(self) -> list[Prediction]:
        """The predictions that were scored incorrect (the regression browser)."""
        return [p for p in self.predictions if p.correct is False]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "fingerprint": self.fingerprint,
            "headline": self.headline,
            "stats": {name: stat.to_dict() for name, stat in self.stats.items()},
            "predictions": [p.to_doc() for p in self.predictions],
            "config": self.config.to_dict() if self.config else None,
            "model_spec": self.model_spec,
            "errors": self.errors,
            "failed_count": self.failed_count,
            "experiment_name": self.experiment_name,
            "tags": self.tags,
            "perf": self.perf,
            "model_size": self.model_size,
            "token_usage": self.token_usage,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunResult:
        from .score import Stat
        from .runspec import RunConfig
        stats = {}
        for name, sdata in data.get("stats", {}).items():
            stats[name] = Stat.from_dict({"name": name, **sdata})
        predictions = [Prediction(**p) for p in data.get("predictions", [])]
        return cls(
            run_id=data["run_id"],
            fingerprint=data["fingerprint"],
            stats=stats,
            predictions=predictions,
            headline=data.get("headline", {}),
            config=RunConfig.from_dict(data["config"]) if data.get("config") else None,
            model_spec=data.get("model_spec"),
            errors=data.get("errors", []),
            failed_count=data.get("failed_count", 0),
            experiment_name=data.get("experiment_name"),
            tags=data.get("tags", []),
            perf=data.get("perf", {}),
            model_size=data.get("model_size", {}),
            token_usage=data.get("token_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
        )

    @classmethod
    def load(cls, path: str) -> RunResult:
        import json
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def save(self, path: str, fmt: str = "json") -> None:
        """Persist the run as ``json`` (full) or ``csv`` (per-prediction rows)."""
        if fmt == "json":
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(self.to_dict(), fh, indent=2, default=str)
        elif fmt == "csv":
            import csv as csv_mod
            predictions = self.predictions
            if not predictions:
                with open(path, "w") as fh:
                    pass
                return
            fields = list(predictions[0].to_doc().keys())
            with open(path, "w", newline="") as fh:
                writer = csv_mod.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                for p in predictions:
                    writer.writerow(p.to_doc())
        else:
            raise ValueError(f"unknown format {fmt!r}; use 'json' or 'csv'")

    def push_to_hub(
        self,
        repo_id: str,
        private: bool = False,
        token: str | None = None,
    ) -> str:
        """Push predictions to a Hub dataset and stamp the AuditKIT card."""
        from .hf_publish import push_rows_to_hub

        model = ""
        if isinstance(self.model_spec, str):
            model = self.model_spec
        elif isinstance(self.model_spec, dict):
            model = str(self.model_spec.get("name") or self.model_spec.get("model") or "")
        return push_rows_to_hub(
            [p.to_doc() for p in self.predictions],
            repo_id,
            private=private,
            token=token,
            kind="run",
            method=self.experiment_name or "evaluate",
            model=model,
        )
