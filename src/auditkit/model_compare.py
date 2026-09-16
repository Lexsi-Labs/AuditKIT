"""Multi-model comparison: run the same evaluation across several models."""
from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence, Union

from .adapter import Adapter
from .api import evaluate
from .diff import best_model, direction_for, metric_directions
from .model import AutoModel, Model
from .report import RunResult
from .runspec import RunConfig
from .sample import Sample
from .scenario import Scenario

logger = logging.getLogger(__name__)


@dataclass
class PerMetricComparison:
    """Comparison of one metric across models."""

    metric: str
    scores: dict[str, float]
    winner: str | None
    deltas: dict[str, dict[str, float]]


@dataclass
class CompareResult:
    """Result of comparing multiple models on the same dataset."""

    runs: dict[str, RunResult]
    dataset_size: int
    scorers: list[str]
    # Models that failed to run AT ALL (raised before producing a RunResult --
    # e.g. an invalid/decommissioned model spec), keyed by name, value = the
    # error message. Previously an exception from any one model crashed
    # compare_models() entirely, losing every other model's already-computed
    # results too; these are now caught and the comparison continues without
    # that model, with the failure recorded here instead of silently lost.
    errors: dict[str, str] = field(default_factory=dict)

    def _warn_if_issues(self) -> None:
        if self.errors:
            logger.warning("CompareResult: %d model(s) failed to run entirely: %s",
                            len(self.errors), self.errors)
        cw = self.coverage_warnings()
        if cw:
            logger.warning("CompareResult: coverage issues across models -- %s", cw)

    def per_metric(self) -> list[PerMetricComparison]:
        """For each metric, show each model's average score and pairwise deltas."""
        self._warn_if_issues()
        by_metric: dict[str, dict[str, float]] = {}
        for model_name, run in self.runs.items():
            for metric_name, value in run.headline.items():
                by_metric.setdefault(metric_name, {})[model_name] = value

        dirs = metric_directions(self.runs.values())
        results = []
        for metric, model_scores in by_metric.items():
            winner = best_model(model_scores, direction_for(metric, dirs))
            deltas = {}
            for m1 in model_scores:
                deltas[m1] = {}
                for m2 in model_scores:
                    if m1 != m2:
                        deltas[m1][m2] = round(model_scores[m1] - model_scores[m2], 4)
            results.append(PerMetricComparison(
                metric=metric,
                scores=model_scores,
                winner=winner,
                deltas=deltas,
            ))
        return results

    def winner(self, metric: str | None = None) -> str | None:
        """Return the model with the highest average on *metric* (or the first metric)."""
        metrics = self.per_metric()
        if not metrics:
            return None
        target = metric or metrics[0].metric
        for m in metrics:
            if m.metric == target:
                return m.winner
        return None

    def significance(self, model_a: str, model_b: str, metric: str | None = None) -> dict[str, Any]:
        """Paired-bootstrap significance test between two models on a given metric.

        Per-sample scores are aligned **by sample_id** (via the shared
        :func:`~auditkit._bootstrap.score_pairs`), so the test is robust even if
        the two runs' prediction order diverges.
        """
        from ._bootstrap import paired_bootstrap, score_pairs

        run_a = self.runs.get(model_a)
        run_b = self.runs.get(model_b)
        if not run_a or not run_b:
            return {"error": f"Missing run for {model_a} or {model_b}"}
        target = metric or next(iter(run_a.headline.keys()), None)
        if target is None:
            return {"error": "No metrics available"}

        bs = paired_bootstrap(score_pairs(run_a, run_b, target))
        if "error" in bs:
            return {"error": bs["error"], "model_a": model_a, "model_b": model_b, "metric": target}
        return {
            "model_a": model_a,
            "model_b": model_b,
            "metric": target,
            "mean_a": bs["mean_baseline"],
            "mean_b": bs["mean_candidate"],
            "delta": round(bs["mean_baseline"] - bs["mean_candidate"], 6),  # a − b
            "p_value": bs["p_value"],
            "significant": bs["significant"],
        }

    def pairwise(self, baseline_name: str, candidate_name: str, **kw: Any):
        """A baseline-anchored :class:`~auditkit.comparison.RunComparison` between
        two of the already-run models (e.g. base vs pruned)."""
        from .comparison import RunComparison

        base, cand = self.runs.get(baseline_name), self.runs.get(candidate_name)
        if base is None or cand is None:
            raise KeyError(
                f"unknown model name(s) {baseline_name!r}/{candidate_name!r}; "
                f"have {list(self.runs)}"
            )
        return RunComparison(base, cand, **kw)

    # -- N-model coverage visibility -----------------------------------------
    def coverage_warnings(self) -> list[str]:
        """Flags failed samples and per-metric sample-count mismatches across
        ALL compared models at once -- the N-model analog of
        :meth:`~auditkit.comparison.RunComparison.coverage_warnings`/
        ``has_failures``. ``compare_models()`` itself performs none of this
        checking on its own; ``per_metric()``/``summary()`` read each run's
        already-aggregated ``.headline`` and would silently show a clean
        table even if one model failed on several samples or a metric never
        scored successfully for one model at all.
        """
        warnings: list[str] = []
        for name, err in self.errors.items():
            warnings.append(f"{name}: FAILED TO RUN ENTIRELY -- {err}")
        for name, run in self.runs.items():
            if run.failed_count:
                warnings.append(f"{name}: {run.failed_count} failed sample(s) -- see run.errors")

        all_metrics: set[str] = set()
        for run in self.runs.values():
            all_metrics |= set(run.stats.keys())
        for metric in sorted(all_metrics):
            counts = {name: run.stats[metric].count for name, run in self.runs.items() if metric in run.stats}
            missing = [name for name in self.runs if metric not in self.runs[name].stats]
            if missing:
                warnings.append(f"{metric}: missing entirely from {missing} (never scored successfully there)")
            elif len(set(counts.values())) > 1:
                warnings.append(f"{metric}: sample counts differ across models -> {counts}")

        # Per-task check: two tasks can individually mismatch in N while the
        # blended totals above coincidentally agree.
        for row in self.per_task_table():
            counts = {name: row.get(f"{name}_n", 0) for name in self.runs}
            if len(set(counts.values())) > 1:
                warnings.append(f"{row['task']}/{row['metric']}: sample counts differ across models -> {counts}")
        return warnings

    @property
    def has_failures(self) -> bool:
        return bool(self.errors) or any(run.failed_count for run in self.runs.values())

    def comparison_table(self) -> list[dict[str, Any]]:
        """One row per metric, with every model's score/n/std side by side and
        the winner marked -- the structured form behind ``summary()``'s printed
        table, for programmatic use (dashboards, CI gates, etc.)."""
        all_metrics: set[str] = set()
        for run in self.runs.values():
            all_metrics |= set(run.stats.keys())
        dirs = metric_directions(self.runs.values())
        rows = []
        for metric in sorted(all_metrics):
            scores: dict[str, float] = {}
            row: dict[str, Any] = {"metric": metric}
            for name, run in self.runs.items():
                stat = run.stats.get(metric)
                row[f"{name}_score"] = stat.mean if stat else None
                row[f"{name}_n"] = stat.count if stat else 0
                row[f"{name}_std"] = stat.std if stat else None
                if stat:
                    scores[name] = stat.mean
            row["winner"] = best_model(scores, direction_for(metric, dirs))
            rows.append(row)
        return rows

    def per_task_table(self) -> list[dict[str, Any]]:
        """One row per (task, metric), every model's score/n/std side by side --
        the N-model analog of ``RunComparison.per_task_deltas()``.

        ``comparison_table()``/``summary()`` blend every task into one number
        per metric -- two models that fail in completely different,
        non-overlapping ways (one bombs task A, the other bombs task B) can
        show the *identical* blended score and be indistinguishable there.
        This breaks it out per task so that's visible.
        """
        # (task, metric) -> {model_name: {"mean", "count", "std"}}
        table: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
        for name, run in self.runs.items():
            # Benchmark engine: stats keyed "task:metric" -- Stat already has count/std.
            for key, stat in run.stats.items():
                if ":" in key:
                    task, metric = key.split(":", 1)
                    table.setdefault((task, metric), {})[name] = {
                        "mean": stat.mean, "count": stat.count, "std": stat.std,
                    }
            # Native engine: group predictions by task via their per-score dicts.
            acc: dict[tuple[str, str], list[float]] = {}
            for p in run.predictions:
                for s in p.metadata.get("scores", []):
                    mname, val = s.get("name"), s.get("value")
                    if mname is not None and isinstance(val, (int, float)):
                        acc.setdefault((p.task or "", mname), []).append(float(val))
            for (task, metric), vals in acc.items():
                if vals and name not in table.get((task, metric), {}):
                    table.setdefault((task, metric), {})[name] = {
                        "mean": sum(vals) / len(vals),
                        "count": len(vals),
                        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                    }

        dirs = metric_directions(self.runs.values())
        rows = []
        for task, metric in sorted(table):
            per_model = table[(task, metric)]
            row: dict[str, Any] = {"task": task, "metric": metric}
            scores: dict[str, float] = {}
            for name in self.runs:
                entry = per_model.get(name)
                row[f"{name}_score"] = entry["mean"] if entry else None
                row[f"{name}_n"] = entry["count"] if entry else 0
                row[f"{name}_std"] = entry["std"] if entry else None
                if entry:
                    scores[name] = entry["mean"]
            row["winner"] = best_model(scores, direction_for(metric, dirs))
            rows.append(row)
        return rows

    def task_macro_average_table(self) -> list[dict[str, Any]]:
        """One row per metric: each model's macro-average across tasks --
        the mean of that model's own per-task means, each task weighted
        equally regardless of how many samples it has.

        Different from ``comparison_table()`` (which blends every sample
        together, so a task with more samples counts for more) whenever
        tasks have unequal sample counts -- e.g. a model at 100% on a
        1-sample task and 0% on a 3-sample task blends (micro-average) to
        0.25, but macro-averages to 0.5 (each task counted once, equally).
        Neither is "more correct" -- micro answers "how good overall,
        weighted by real sample volume," macro answers "how good on the
        average task, regardless of how many examples it had."
        """
        task_rows = self.per_task_table()
        acc: dict[str, dict[str, list[float]]] = {}
        for row in task_rows:
            metric = row["metric"]
            for name in self.runs:
                val = row.get(f"{name}_score")
                if val is not None:
                    acc.setdefault(metric, {}).setdefault(name, []).append(val)

        dirs = metric_directions(self.runs.values())
        rows = []
        for metric in sorted(acc):
            per_model = acc[metric]
            row: dict[str, Any] = {"metric": metric}
            scores: dict[str, float] = {}
            for name in self.runs:
                vals = per_model.get(name, [])
                macro_mean = sum(vals) / len(vals) if vals else None
                row[f"{name}_macro_mean"] = macro_mean
                row[f"{name}_num_tasks"] = len(vals)
                if macro_mean is not None:
                    scores[name] = macro_mean
            row["winner"] = best_model(scores, direction_for(metric, dirs))
            rows.append(row)
        return rows

    def performance_table(self) -> list[dict[str, Any]]:
        """Each model's measured latency/throughput side by side -- the
        N-model analog of ``RunComparison.performance()``. Straight from
        each run's ``RunResult.perf`` (populated by ``Runner`` while it ran);
        a model with no timed calls (e.g. a ``reads_actual_output`` run), or
        one whose ``RunConfig.track_performance`` was never enabled at all,
        reports ``None``s rather than being omitted or crashing, so the row
        shape stays uniform across models."""
        rows = []
        for name, run in self.runs.items():
            perf = run.perf
            lat = (perf.get("latency_ms") or {}) if perf is not None else None
            thr = (perf.get("throughput") or {}) if perf is not None else None
            rows.append({
                "model": name,
                "latency_mean_ms": lat.get("mean") if lat is not None else None,
                "throughput_rps": thr.get("rps") if thr is not None else None,
                "output_tokens_per_sec": thr.get("output_tokens_per_sec") if thr is not None else None,
                "total_tokens_per_sec": thr.get("total_tokens_per_sec") if thr is not None else None,
                "calls": lat.get("count", 0) if lat is not None else None,
            })
        return rows

    def cost_table(self, pricing: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
        """Dollar cost per model, given *pricing* -- ``{model_name:
        {"input_per_1m": ..., "output_per_1m": ...}}``. Token counts are the
        real, provider-reported numbers already on each ``RunResult``
        (``.token_usage``); there is no library default price table (see
        ``RunResult.cost()``), so a model with no entry in *pricing*, no
        recorded token usage at all, or ``RunConfig.track_performance``
        never enabled, reports ``cost=None`` and ``None`` token counts
        rather than a guess or a crash."""
        rows = []
        for name, run in self.runs.items():
            model_pricing = pricing.get(name)
            cost = run.cost(model_pricing) if model_pricing else None
            usage = run.token_usage
            rows.append({
                "model": name,
                "prompt_tokens": usage.get("prompt_tokens", 0) if usage is not None else None,
                "completion_tokens": usage.get("completion_tokens", 0) if usage is not None else None,
                "cost": cost,
            })
        return rows

    def size_table(self) -> list[dict[str, Any]]:
        """Each model's size, side by side -- never caller-supplied. Local
        backends report introspected params/sparsity/MB; hosted-API backends
        (or anything with no ``model_size`` recorded) report just an
        identity so the row still names the model instead of a blank."""
        rows = []
        for name, run in self.runs.items():
            ms = run.model_size or {}
            rows.append({
                "model": name,
                "is_local": ms.get("is_local", False),
                "model_name": ms.get("model_name"),
                "total_params": ms.get("total_params"),
                "sparsity": ms.get("sparsity"),
                "size_mb": ms.get("size_mb"),
            })
        return rows

    def summary(self) -> str:
        lines = [f"Model Comparison — {self.dataset_size} samples, {len(self.scorers)} scorers", ""]

        lines.append("Runs:")
        for name, run in self.runs.items():
            flag = f"  [!! {run.failed_count} FAILED]" if run.failed_count else ""
            lines.append(f"  {name}: run_id={run.run_id}{flag}")
        for name, err in self.errors.items():
            lines.append(f"  {name}: DID NOT RUN -- {err}")
        lines.append("")

        cw = self.coverage_warnings()
        if cw:
            lines.append("WARNING -- coverage issues across models:")
            lines += [f"  {w}" for w in cw]
            lines.append("")

        metrics = self.per_metric()
        if not metrics:
            lines.append("(no metrics)")
            return "\n".join(lines)

        model_names = list(self.runs.keys())
        table = {row["metric"]: row for row in self.comparison_table()}
        header = f"{'Metric':<25}" + "".join(f"{m:<26}" for m in model_names)
        lines.append(header)
        lines.append("-" * len(header))
        for m in metrics:
            row = f"{m.metric:<25}"
            for name in model_names:
                val = m.scores.get(name, 0)
                n = table[m.metric].get(f"{name}_n", 0)
                marker = " <-" if m.winner == name else ""
                cell = f"{val:.4f} (n={n}){marker}"
                row += f"{cell:<26}"
            lines.append(row)

        task_rows = self.per_task_table()
        # Only show the per-task breakout if it says something the blended
        # table above doesn't -- i.e. there's more than one distinct task.
        if len({r["task"] for r in task_rows}) > 1:
            lines.append("")
            lines.append("Per-task:")
            task_header = f"  {'Task/Metric':<25}" + "".join(f"{m:<26}" for m in model_names)
            lines.append(task_header)
            lines.append("  " + "-" * (len(task_header) - 2))
            for row in task_rows:
                label = f"{row['task']}/{row['metric']}"
                line = f"  {label:<25}"
                for name in model_names:
                    val = row.get(f"{name}_score")
                    n = row.get(f"{name}_n", 0)
                    marker = " <-" if row["winner"] == name else ""
                    cell = "n/a" if val is None else f"{val:.4f} (n={n}){marker}"
                    line += f"{cell:<26}"
                lines.append(line)

            lines.append("")
            lines.append("Macro-average across tasks (each task weighted equally, not by sample count):")
            macro_header = f"  {'Metric':<25}" + "".join(f"{m:<26}" for m in model_names)
            lines.append(macro_header)
            lines.append("  " + "-" * (len(macro_header) - 2))
            for row in self.task_macro_average_table():
                line = f"  {row['metric']:<25}"
                for name in model_names:
                    val = row.get(f"{name}_macro_mean")
                    marker = " <-" if row["winner"] == name else ""
                    cell = "n/a" if val is None else f"{val:.4f}{marker}"
                    line += f"{cell:<26}"
                lines.append(line)

        size_rows = [r for r in self.size_table() if r["model_name"] is not None]
        if size_rows:
            lines.append("")
            lines.append("Model size:")
            for row in size_rows:
                if row["is_local"] and row["size_mb"] is not None:
                    lines.append(
                        f"  {row['model']} ({row['model_name']}): "
                        f"{row['total_params']:,} params "
                        f"({row['sparsity']:.1%} sparse), {row['size_mb']:.1f} MB"
                    )
                elif row["is_local"]:
                    # is_local=True but no size_mb -- a local backend that
                    # doesn't actually introspect size yet (e.g. VLLMModel),
                    # not a hosted-API model. Different reason, same "nothing
                    # to report" outcome -- said explicitly either way.
                    lines.append(f"  {row['model']} ({row['model_name']}): local model, size not introspected")
                else:
                    lines.append(f"  {row['model']} ({row['model_name']}): not a local model -- no measurable size")

        perf_rows = [r for r in self.performance_table() if r["calls"]]
        if perf_rows:
            lines.append("")
            lines.append("Performance (measured):")
            perf_header = f"  {'Model':<20}{'latency (ms)':<14}{'req/s':<10}{'out-tok/s':<12}"
            lines.append(perf_header)
            lines.append("  " + "-" * (len(perf_header) - 2))
            for row in self.performance_table():
                if not row["calls"]:
                    lines.append(f"  {row['model']:<20}{'n/a':<14}{'n/a':<10}{'n/a':<12}")
                    continue
                # output_tokens_per_sec is 0 for local/callable backends that
                # never report token usage -- shown as "n/a", not a
                # misleading measured zero.
                tok_s = row.get("output_tokens_per_sec") or 0.0
                tok_cell = f"{tok_s:.1f}" if tok_s else "n/a"
                lines.append(
                    f"  {row['model']:<20}{row['latency_mean_ms']:<14.0f}"
                    f"{row['throughput_rps']:<10.2f}{tok_cell:<12}"
                )

        lines.append("")
        lines.append("Significance (pairwise bootstrap p < 0.05):")
        for i, a in enumerate(model_names):
            for b in model_names[i + 1:]:
                sig = self.significance(a, b)
                p = sig.get("p_value", 1)
                label = "significant" if sig.get("significant") else "not significant"
                lines.append(f"  {a} vs {b}: p={p:.4f} ({label})")

        return "\n".join(lines)

    def push_to_hub(
        self,
        repo_id: str,
        private: bool = False,
        token: str | None = None,
    ) -> str:
        """Push per-model predictions to a Hub dataset and stamp the AuditKIT card."""
        from .hf_publish import push_rows_to_hub

        rows: list[dict[str, Any]] = []
        for name, run in self.runs.items():
            for p in run.predictions:
                doc = p.to_doc()
                doc["model"] = name
                rows.append(doc)
        return push_rows_to_hub(
            rows,
            repo_id,
            private=private,
            token=token,
            kind="compare",
            method="compare_models",
        )


def compare_models(
    models: Sequence[Union[str, Model, Callable[[list[str]], list[str]]]],
    dataset: Union[Scenario, Sequence[Sample], Callable[[], Iterable[Sample]], str],
    scorers: Any = None,
    *,
    model_names: list[str] | None = None,
    adapter: Union[Adapter, str, None] = None,
    annotators: Any = None,
    extract_with: str | None = None,
    config: RunConfig | None = None,
    configs: dict[str, RunConfig] | None = None,
    model_opts: dict[str, dict[str, Any]] | None = None,
    experiment_name: str | None = None,
    tags: list[str] | None = None,
    **opts: Any,
) -> CompareResult:
    """Evaluate the same dataset against multiple models and compare results.

    Args:
        models: List of model specs (strings, Model instances, or callables).
        dataset: Dataset to evaluate (same for all models).
        scorers: Metrics to use (same for all models -- comparison across
            models assumes a shared metric set, so this is not overridable
            per-model via ``model_opts``).
        model_names: Optional human-readable names for each model.
        adapter: Adapter shared across all models unless overridden per-model
            via ``model_opts``. Same default as ``evaluate()`` (``None`` ->
            ``GenerationAdapter``) when not given here.
        annotators: Annotator(s) shared across all models unless overridden
            per-model via ``model_opts``.
        extract_with: Annotator name whose extraction feeds scoring, shared
            across all models unless overridden per-model via ``model_opts``.
        config: Run configuration shared across all evaluations.
        configs: Optional per-model ``RunConfig`` overrides, keyed by model name
            (falls back to ``config`` when a name is absent). Use this when a
            quantized/pruned model legitimately needs different knobs than the base.
        model_opts: Optional per-model overrides, keyed by model name. Each
            entry is itself a dict that may contain any of ``adapter``/
            ``annotators``/``extract_with`` (overriding the shared value
            above, for that model only) plus arbitrary backend construction
            kwargs (e.g. ``dtype``/``device``, layered over the shared
            ``**opts``) -- anything not recognized as a pipeline override is
            forwarded to model resolution unchanged.
        experiment_name: If set, each model's run is persisted under
            ``f"{experiment_name}:{name}"`` (mirrors ``evaluate_many()``).
            Explicit real params, not swept into ``**opts`` -- previously
            passing ``experiment_name=``/``tags=`` here silently leaked them
            into ``AutoModel.resolve(**opts)``, i.e. into each model's own
            constructor kwargs, and for a real hosted API backend that meant
            landing in the live HTTP request body, which the provider then
            rejected outright (a real, confirmed Groq 400).
        tags: Passed to each model's underlying ``evaluate()`` call.
        **opts: Extra kwargs forwarded to model resolution for every model.

    Returns:
        A CompareResult with per-model runs and comparison methods.
    """
    names = model_names or [str(m) if isinstance(m, str) else getattr(m, "name", f"model_{i}") for i, m in enumerate(models)]
    if len(names) != len(models):
        raise ValueError("model_names must match number of models")

    runs: dict[str, RunResult] = {}
    errors: dict[str, str] = {}
    for i, (name, model_spec) in enumerate(zip(names, models)):
        run_config = (configs or {}).get(name, config)
        this_opts = dict((model_opts or {}).get(name, {}))
        run_adapter = this_opts.pop("adapter", adapter)
        run_annotators = this_opts.pop("annotators", annotators)
        run_extract_with = this_opts.pop("extract_with", extract_with)
        run_opts = {**opts, **this_opts}
        # Resolve the model HERE (rather than letting evaluate() do it) so we
        # hold the instance and can free its memory before the next model
        # loads -- keeping peak GPU use at one model instead of accumulating
        # across the comparison. Only tear down a model we resolved ourselves;
        # a Model instance the caller passed in is theirs to keep/reuse.
        owns = not isinstance(model_spec, Model)
        # vLLM sleep mode is one engine per process. Enable it only when a
        # later vllm: model still has to load, so unload() can sleep(level=2)
        # and free the T4. The last vllm: engine stays a normal LLM().
        # Confirmed live: enabling it on every model raised
        # "Sleep mode can only be used for one instance per process."
        if (
            owns
            and isinstance(model_spec, str)
            and model_spec.startswith("vllm:")
            and any(
                isinstance(m, str) and m.startswith("vllm:")
                for m in models[i + 1:]
            )
        ):
            run_opts.setdefault("enable_sleep_mode", True)
        resolved: Model | None = None
        try:
            resolved = AutoModel.resolve(model_spec, **run_opts)
            # opts already applied at resolve(); don't re-forward them to
            # evaluate (a resolved instance ignores them anyway).
            result = evaluate(
                dataset, model=resolved, scorers=scorers, config=run_config,
                adapter=run_adapter, annotators=run_annotators, extract_with=run_extract_with,
                experiment_name=f"{experiment_name}:{name}" if experiment_name else None,
                tags=tags,
            )
        except Exception as e:
            # One bad model (an invalid/decommissioned spec, a real API
            # error that exhausted retries, ...) used to crash this whole
            # function, discarding every other model's already-computed
            # results too. Record it and keep going instead -- callers can
            # see exactly which model(s) failed via CompareResult.errors/
            # .coverage_warnings(), rather than losing everything.
            detail = str(e)
            if e.__cause__ is not None:
                # Runner._retry_generate() re-raises as a generic ModelError
                # ("generate failed after N retries"), chaining the real
                # underlying exception via `raise ... from` -- surface that
                # cause too, or the recorded message is nearly useless.
                detail = f"{detail} (caused by {type(e.__cause__).__name__}: {e.__cause__})"
            errors[name] = f"{type(e).__name__}: {detail}"
            logger.error("compare_models(): model %r failed entirely: %s", name, errors[name])
            continue
        finally:
            # Free this model's local weights/GPU memory before the next one
            # loads (no-op for API/callable backends). Runs on both the success
            # and error paths; only for a model we resolved ourselves. The
            # RunResult stores a stringified model_spec, not the live model, so
            # unloading after evaluate() returns never affects the result.
            if owns and resolved is not None:
                try:
                    resolved.unload()
                except Exception:  # noqa: BLE001 -- teardown is best-effort
                    logger.debug("compare_models(): unload() failed for %r", name, exc_info=True)
        runs[name] = result

    first = next(iter(models), None)
    if first is None:
        raise ValueError("At least one model is required")
    if isinstance(dataset, str):
        from .registry import SCENARIOS
        scenario = SCENARIOS.get(dataset)()
        samples = list(scenario.samples())
    elif isinstance(dataset, Scenario):
        samples = list(dataset.samples())
    elif callable(dataset):
        samples = list(dataset())
    else:
        samples = list(dataset)

    scorer_names = list(next(iter(runs.values())).headline.keys()) if runs else []

    return CompareResult(
        runs=runs,
        dataset_size=len(samples),
        scorers=scorer_names,
        errors=errors,
    )
