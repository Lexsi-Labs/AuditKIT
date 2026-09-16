"""The public evaluation API: :func:`evaluate`.

This is what users import::

    import auditkit as ak
    r = ak.evaluate([ak.Sample(input="hi", target="HI")], model=lambda prompts: ["HI" for _ in prompts])
    r.headline  # {"exact_match": 1.0}
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Optional, Sequence, Union

from .adapter import Adapter, GenerationAdapter
from .annotator import Annotator
from .experiment import Experiment, ExperimentDB
from .loaders import load_csv  # noqa: F401 — re-exported via __init__
from .metric import ExactMatch, Metric, QuasiExactMatch
from .model import (
    Model,
    AutoModel,
)
from .report import RunResult
from .runner import Runner
from .runspec import RunConfig, RunSpec
from .sample import Sample
from .scenario import CallableScenario, ListScenario, Scenario
from .scorers import FunctionScorer, ScorerMetric, ScorerType, scorer  # noqa: F401
from .errors import AuditKitError, RegistryError
from .registry import ADAPTERS, ANNOTATORS, METRICS, SCENARIOS

logger = logging.getLogger("auditkit")

#: Known metric names resolvable from a string.
_BUILTIN_METRICS: dict[str, type[Metric]] = {
    "exact_match": ExactMatch,
    "quasi_exact_match": QuasiExactMatch,
}


def _to_scenario(
    dataset: Union[Scenario, Sequence[Sample], Callable[[], Iterable[Sample]], str],
) -> Scenario:
    """Coerce any supported dataset form to a :class:`Scenario`."""
    if isinstance(dataset, Scenario):
        return dataset
    if isinstance(dataset, str):
        try:
            return SCENARIOS.get(dataset)()
        except RegistryError:
            available = ", ".join(SCENARIOS.names())
            raise AuditKitError(f"Unknown dataset '{dataset}'. Available: [{available}]")
    if callable(dataset):
        return CallableScenario(dataset)
    # treat it as a list/iterable of Samples
    return ListScenario(list(dataset))


def _to_metrics(
    scorers: Any,
    samples: Sequence[Sample],
) -> list[Metric]:
    """Coerce scorers/user metrics to a list of internal :class:`Metric`.

    The resolution rules (from HANDOFF §Cycle-8 *step 4*):

    - ``None``: auto-select. If any sample is golden (has a ``target``),
      return ``[ExactMatch()]``; otherwise return ``[]``.
    - ``str``: look up in the built-in map (``"exact_match"``,
      ``"quasi_exact_match"``), falling back to the :data:`METRICS` registry
      (populated by every metric class in ``metrics/*.py``) for any other
      registered name (e.g. ``"acc_norm"``, ``"bleu"``, ``"lexical_groundedness"``).
    - :class:`Metric`: returned as-is.
    - :class:`ScorerType` (``FunctionScorer`` / callable): wrap in
      :class:`ScorerMetric`.
    - ``list``: recurse on each element.
    """
    if scorers is None:
        if any(s.is_golden for s in samples):
            return [ExactMatch()]
        return []

    if isinstance(scorers, str):
        cls = _BUILTIN_METRICS.get(scorers)
        if cls is None:
            try:
                cls = METRICS.get(scorers)
            except RegistryError:
                cls = None
        if cls is None:
            known = sorted(set(_BUILTIN_METRICS) | set(METRICS.names()))
            raise ValueError(f"unknown metric {scorers!r}; known: {known}")
        return [cls()]

    if isinstance(scorers, Metric):
        return [scorers]

    if isinstance(scorers, FunctionScorer) or callable(scorers):
        # Wrap a bare callable as a FunctionScorer if needed, then as Metric
        if not isinstance(scorers, FunctionScorer):
            scorers = FunctionScorer(scorers)
        return [ScorerMetric(scorers)]

    if isinstance(scorers, list):
        result: list[Metric] = []
        for item in scorers:
            result.extend(_to_metrics(item, samples))
        return result

    raise TypeError(f"cannot interpret scorers={scorers!r}")


def _to_adapter(adapter: Union[Adapter, str, None], samples: Any = None) -> Adapter:
    """Coerce ``None``/``"auto"``/a name string/an :class:`Adapter` to an :class:`Adapter`.

    ``"auto"`` routes on the dataset's shape (``choices`` -> MCQ,
    ``retrieval_context`` -> RAG, else generation); it needs ``samples``.
    """
    if adapter is None:
        return GenerationAdapter()
    if adapter == "auto":
        from .router import route_adapter
        return route_adapter(samples if samples is not None else [])
    if isinstance(adapter, str):
        try:
            return ADAPTERS.get(adapter)()
        except RegistryError:
            raise AuditKitError(f"Unknown adapter {adapter!r}. Available: {ADAPTERS.names()}")
    return adapter


def _to_annotators(annotators: Any) -> list[Annotator]:
    """Coerce ``None``/an :class:`Annotator`/a name string/a list of any of
    those to ``list[Annotator]``."""
    if annotators is None:
        return []
    if isinstance(annotators, Annotator):
        return [annotators]
    if isinstance(annotators, str):
        try:
            return [ANNOTATORS.get(annotators)()]
        except RegistryError:
            raise AuditKitError(f"Unknown annotator {annotators!r}. Available: {ANNOTATORS.names()}")
    if isinstance(annotators, list):
        result: list[Annotator] = []
        for item in annotators:
            result.extend(_to_annotators(item))
        return result
    raise TypeError(f"cannot interpret annotators={annotators!r}")


def _record_experiment(
    result: RunResult, experiment_name: str | None, tags: list[str] | None
) -> RunResult:
    """Stamp experiment/tags on a result and persist it if named."""
    result.experiment_name = experiment_name
    result.tags = tags or []
    if experiment_name:
        exp = Experiment(name=experiment_name)
        exp.add(result)
        ExperimentDB().save(exp)
    return result


def evaluate(
    dataset: Union[Scenario, Sequence[Sample], Callable[[], Iterable[Sample]], str],
    model: Union[Model, Callable[[list[str]], list[str]], str],
    scorers: Any = None,
    *,
    engine: str = "native",
    adapter: Union[Adapter, str, None] = None,
    annotators: Any = None,
    extract_with: str | None = None,
    config: Optional[RunConfig] = None,
    verbose: bool = False,
    experiment_name: str | None = None,
    tags: list[str] | None = None,
    **opts: Any,
) -> RunResult:
    """Evaluate a model on a dataset and return a :class:`RunResult`.

    The single front door for both engines. ``engine="native"`` (default) runs
    the owned spine (Adapter → Model → Metric); ``engine="lmeval"`` runs the
    lm-evaluation-harness with its full task machinery (see :func:`run_lmeval`),
    treating ``dataset`` as the task name(s).

    Parameters
    ----------
    dataset
        A :class:`Scenario`, a list of :class:`Sample`, a callable that yields
        samples, or a scenario name (native). With ``engine="lmeval"``, an
        lm-eval task name, comma-separated string, or list of names.
    model
        A :class:`Model` instance, a ``list[str] -> list[str]`` callable, or a
        string model spec (e.g. ``"hf:gpt2"``, ``"groq:llama-3.3-70b-versatile"``,
        ``"precomputed"`` for samples with ``actual_output`` already set). With
        ``engine="lmeval"`` a string spec is required (``"hf:gpt2"``, …).
    scorers
        ``None`` (auto-select ``ExactMatch`` if golden samples exist),
        a metric name string, a :class:`Metric`, a :class:`FunctionScorer`,
        a ``(sample, output) -> float|Score`` callable, or a list of any of
        the above. Ignored by ``engine="lmeval"`` (lm-eval owns its scoring).
    adapter
        An :class:`Adapter` instance, a registered adapter name string (e.g.
        ``"mcq"``, ``"chat"``, ``"rag"`` — see ``ADAPTERS.names()``), or
        ``None``. Defaults to :class:`GenerationAdapter`.
    annotators
        An :class:`Annotator` instance, a registered name string (e.g.
        ``"regex"`` for :class:`RegexAnnotator`), a list of either, or
        ``None``. Annotator output lands in each :class:`Score`/metric's
        ``context`` under the annotator's name; pass ``extract_with=`` to
        actually score against one annotator's extracted value.
    extract_with
        Names an annotator (by its ``.name``) whose ``context["extracted"]``
        value should be scored instead of the raw model output — e.g. a
        ``RegexAnnotator(r"FINAL ANSWER:\\s*(\\d+)", group=1)`` pulling
        ``"42"`` out of a longer chain-of-thought reply. ``None`` (default):
        score the raw output, unchanged from before this existed.
    config
        A :class:`RunConfig` with evaluation knobs. Defaults to ``RunConfig()``.
    **opts
        Extra keyword args passed to the model constructor (native) or to
        lm-eval's ``model_args`` (``engine="lmeval"``: ``base_url``, ``dtype``, …).
    """
    if engine == "lmeval":
        from .lmeval_engine import run_benchmark

        result = run_benchmark(
            dataset, model, config=config, run_name=experiment_name or "", **opts
        )
        return _record_experiment(result, experiment_name, tags)
    if engine != "native":
        raise ValueError(f"unknown engine {engine!r}; use 'native' or 'lmeval'")

    scenario = _to_scenario(dataset)
    resolved_model = AutoModel.resolve(model, **opts)
    samples = list(scenario.samples())
    metrics = _to_metrics(scorers, samples)
    cfg = config or RunConfig()

    spec = RunSpec(
        scenario=scenario,
        model=resolved_model,
        adapter=_to_adapter(adapter, samples),
        metrics=metrics,
        annotators=_to_annotators(annotators),
        extracted_by=extract_with,
        config=cfg,
    )

    result = Runner().run(spec, verbose=verbose)
    return _record_experiment(result, experiment_name, tags)


def run_lmeval(
    tasks: Union[str, list[str]],
    model: str,
    *,
    num_fewshot: int | None = None,
    limit: int | None = None,
    config: Optional[RunConfig] = None,
    experiment_name: str | None = None,
    tags: list[str] | None = None,
    **opts: Any,
) -> RunResult:
    """Run academic benchmarks via lm-evaluation-harness (its full task machinery).

    The dedicated front door for the lm-eval engine -- unlike :func:`evaluate`
    (which defaults to the native spine and only reaches lm-eval via
    ``engine="lmeval"``), this function is *always* lm-eval, unconditionally;
    there is no native/lmeval switch here. lm-eval owns the task, prompt
    template, filters, and scoring; this returns the same uniform
    :class:`RunResult` (headline + per-sample answer browser) so
    :func:`compare`, diffs, and experiment tracking work identically.

    Requires ``pip install auditkit[lmeval]``. For gated models (Llama,
    Gemma, ...) pass ``hf_token=`` (or ``token=``); it is forwarded to lm-eval's
    ``model_args`` and mirrored into ``HF_TOKEN`` for gated dataset access.

    Model-backend options (``base_url``, ``dtype``, ``device``, ``hf_token``, …)
    and lm-eval run knobs all pass through. Common run knobs: ``apply_chat_template``
    (set ``True`` for instruct/chat/fine-tuned models), ``gen_kwargs`` (generation
    params for generative tasks), ``system_instruction``, ``fewshot_as_multiturn``.
    Anything else ``simple_evaluate`` accepts goes via ``lmeval_kwargs={...}``.

    Examples
    --------
    ::

        ak.run_lmeval(["arc_challenge", "gsm8k"], model="hf:gpt2", num_fewshot=5)
        ak.run_lmeval("mmlu", model="vllm:meta-llama/Llama-3.2-1B", limit=100)
        ak.run_lmeval("gsm8k", model="api:my-model", base_url="https://…/v1/completions")
        ak.run_lmeval("mmlu", model="hf:meta-llama/Llama-3.2-1B", hf_token="hf_…")  # gated
        ak.run_lmeval("ifeval", model="hf:my-finetune", apply_chat_template=True)   # instruct
        ak.run_lmeval("gsm8k", model="hf:gpt2", gen_kwargs="temperature=0,max_gen_toks=256")
    """
    from .lmeval_engine import run_benchmark

    cfg = config
    if cfg is None:
        cfg = RunConfig(num_fewshot=num_fewshot, limit=limit)
    result = run_benchmark(
        tasks, model, config=cfg, run_name=experiment_name or "", **opts
    )
    return _record_experiment(result, experiment_name, tags)


def generate(
    dataset: Union[Scenario, Sequence[Sample], Callable[[], Iterable[Sample]], str],
    model: Union[Model, Callable[[list[str]], list[str]], str],
    *,
    adapter: Union[Adapter, str, None] = None,
    config: Optional[RunConfig] = None,
    verbose: bool = False,
    **opts: Any,
) -> list[Sample]:
    """Generate answers for a dataset and return samples with ``actual_output`` set.

    Stage 1 of the generate→score flow. Runs ``model`` over the dataset's inputs
    and returns copies of the samples with their answers filled in. Score them
    afterwards — repeatedly, with different scorers, without re-generating::

        answers = ak.generate(data, model="hf:my-pruned-model")
        ak.evaluate(answers, model="precomputed", scorers=[judge])
        ak.evaluate(answers, model="precomputed", scorers=[ak.Factuality(judge_model="openai:gpt-4o-mini")])
    """
    from dataclasses import replace

    scenario = _to_scenario(dataset)
    resolved_model = AutoModel.resolve(model, **opts)
    samples = list(scenario.samples())
    spec = RunSpec(
        scenario=ListScenario(samples),
        model=resolved_model,
        adapter=_to_adapter(adapter, samples),
        metrics=[],
        config=config or RunConfig(),
    )
    result = Runner().run(spec, verbose=verbose)
    return [replace(s, actual_output=p.raw_output)
            for s, p in zip(samples, result.predictions)]


def evaluate_many(
    datasets: Union[dict[str, Any], Sequence[Any]],
    model: Union[Model, Callable[[list[str]], list[str]], str],
    *,
    scorers: Any = None,
    adapter: Union[Adapter, str, None] = None,
    engine: str = "native",
    config: Optional[RunConfig] = None,
    verbose: bool = False,
    experiment_name: str | None = None,
    tags: list[str] | None = None,
    on_error: str = "raise",
    **opts: Any,
) -> dict[str, RunResult]:
    """Evaluate one model across **several datasets** in a single call.

    :func:`evaluate` takes exactly one dataset, one adapter, and one scorer set —
    because a run is a single reproducible unit (one ``RunSpec``, one fingerprint).
    Different benchmarks legitimately need different scorers (SORRY-Bench rewards
    refusal; OR-Bench penalises *over*-refusal; GSM8K wants ``exact_match``) and
    sometimes a different adapter, and each deserves its **own** headline number
    and its own cache entry. So this runs each dataset as its own ``evaluate()``
    call, **sequentially** (not concurrently), and returns one
    :class:`RunResult` per dataset — never a blended mean.

    Parameters
    ----------
    datasets
        Either a ``dict[name -> spec]`` (recommended — you name each benchmark),
        or a ``list[spec]`` (names are derived: a dataset string, else the
        scenario's ``.name``, else ``"dataset_{i}"``). Each ``spec`` is either:

        * a bare dataset — anything :func:`evaluate` accepts as ``dataset``
          (a ``list[Sample]``, a :class:`Scenario`, a callable, or a scenario
          name) — scored with the shared ``scorers``/``adapter`` below; or
        * a **tuple** ``(dataset,)``, ``(dataset, scorers)``, or
          ``(dataset, scorers, adapter)`` to override the scorers/adapter for
          just that benchmark. (A tuple is unambiguous here — no dataset form is
          itself a tuple.)
    model
        Resolved **once** up front (native engine) and reused for every dataset,
        so a local ``hf:`` checkpoint is loaded a single time rather than
        re-loaded per benchmark. ``**opts`` are the model constructor kwargs.
    scorers, adapter
        Shared defaults used for any dataset whose ``spec`` doesn't override them.
    on_error
        ``"raise"`` (default) — the first failing dataset aborts the batch, same
        as calling :func:`evaluate` directly. ``"skip"`` — log the failure and
        continue, omitting that dataset from the returned dict (so one bad
        dataset load doesn't lose the benchmarks that did run).
    experiment_name
        When set, each dataset's run is persisted under ``f"{experiment_name}:{name}"``
        so the per-benchmark runs stay individually addressable in the ExperimentDB.

    Returns
    -------
    dict[str, RunResult]
        One entry per dataset, keyed by name, in input order.

    Examples
    --------
    Heterogeneous safety benches, each with its own judge, one model, one call::

        results = ak.evaluate_many(
            {
                "sorry_bench": (sorry_samples, [refusal_judge]),      # refusal = good
                "or_bench":    (or_samples,    [over_refusal_judge]), # over-refusal = bad
                "gsm8k_local": (gsm_samples,   ["exact_match"]),
            },
            model="groq:llama-3.3-70b-versatile",
        )
        for name, r in results.items():
            print(name, r.headline)

    A list of same-shaped datasets sharing one scorer::

        results = ak.evaluate_many([ds_a, ds_b, ds_c], model="hf:gpt2",
                                   scorers=["exact_match"], device="cpu")
    """
    if on_error not in ("raise", "skip"):
        raise ValueError(f"on_error must be 'raise' or 'skip', got {on_error!r}")

    # Resolve the model ONCE (native) so a heavy local checkpoint isn't reloaded
    # per dataset. lm-eval builds its own backend from a string spec, so leave
    # the spec (and opts) untouched on that path.
    if engine == "native":
        model = AutoModel.resolve(model, **opts)
        opts = {}

    items = datasets.items() if isinstance(datasets, dict) else enumerate(datasets)
    results: dict[str, RunResult] = {}
    for key, spec in items:
        if isinstance(spec, tuple):
            ds = spec[0]
            ds_scorers = spec[1] if len(spec) > 1 else scorers
            ds_adapter = spec[2] if len(spec) > 2 else adapter
        else:
            ds, ds_scorers, ds_adapter = spec, scorers, adapter

        if isinstance(key, str):
            name = key
        elif isinstance(ds, str):
            name = ds
        else:
            name = getattr(ds, "name", None) or f"dataset_{key}"

        try:
            results[name] = evaluate(
                ds, model=model, scorers=ds_scorers, adapter=ds_adapter,
                engine=engine, config=config, verbose=verbose,
                experiment_name=f"{experiment_name}:{name}" if experiment_name else None,
                tags=tags, **opts,
            )
        except Exception as e:  # noqa: BLE001 — isolation is opt-in via on_error
            if on_error == "raise":
                raise
            logger.error("evaluate_many: dataset %r failed, skipping: %s", name, e)
    return results


def compare(baseline_or_results, candidate=None, *, metric: str | None = None, **kw):
    """Two shapes:

    - ``compare(baseline_run, candidate_run)`` — two :class:`RunResult`s → a
      :class:`~auditkit.comparison.RunComparison` (per-task deltas, pass/warn/fail,
      newly-wrong browser, retention, significance). This is the base-vs-pruned view.
    - ``compare([run1, run2, ...], metric=...)`` — a list of runs → a leaderboard
      (one dict per run, sorted by ``metric`` or the first headline metric).

    ``kw`` (``pass_threshold``/``warn_threshold``) is forwarded to ``RunComparison``.
    """
    if candidate is not None:
        from .comparison import RunComparison
        return RunComparison(baseline_or_results, candidate, **kw)

    results = baseline_or_results
    if not results:
        return []
    first = metric or next(iter(results[0].headline.keys()), None)
    rows = []
    for r in results:
        row = {"run_id": r.run_id, "fingerprint": r.fingerprint}
        row.update(r.headline)
        rows.append(row)
    if first:
        rows.sort(key=lambda x: x.get(first, 0), reverse=True)
    return rows


__all__ = [
    "run_lmeval",
    "compare",
    "evaluate",
    "evaluate_many",
    "generate",
    "load_csv",
    "scorer",
]
