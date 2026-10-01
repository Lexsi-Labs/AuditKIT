"""The runner: five separately-testable stages that drive one evaluation.

``build_requests`` turns samples into requests, ``execute`` sends them to the
model in one batch and realigns the results, ``annotate`` runs any annotators,
``score_one`` applies the metrics to a sample's output and records a
:class:`Prediction`, and ``aggregate`` rolls per-sample scores into per-metric
:class:`~auditkit.score.Stat`. :meth:`Runner.run` chains them into a
:class:`RunResult`. Keeping the stages independent is what makes each testable in
isolation and lets techniques swap one stage without touching the others.
"""

from __future__ import annotations

import logging
import math
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import asdict
from typing import Any, Optional

from .adapter import Adapter
from .cache import DiskCache
from .errors import CapabilityError, ExtraNotInstalled, ModelError, ModelTimeout
from .metric import Metric
from .metrics.perf import LatencyStats, Throughput
from .model import Generated, Model, Request, Result_, _prompt_text
from .provenance import input_ref
from .report import Prediction, RunResult
from .runspec import RunConfig, RunSpec
from .sample import Sample
from .scenario import Scenario
from .score import Score, Stat
from .types import Capability

logger = logging.getLogger(__name__)

_generate_lock = threading.Lock()


def _score_doc(score: Score) -> dict[str, Any]:
    """Every field on *score* that actually carries information.

    ``Score`` supports rich, judge-grade output (``reason``, ``threshold``,
    ``label``, per-score ``metadata``, ...), but most deterministic metrics
    never set most of it -- serializing every field unconditionally would
    bury the useful cases (a judge's verdict reasoning) under a wall of
    ``None``s and defaults for the common case (``ExactMatch`` only ever
    sets ``name``/``value``). Keep only fields that aren't ``None`` or an
    empty container, plus the derived ``passed`` (True/False/None against
    ``threshold``), which isn't itself a dataclass field so ``asdict``
    wouldn't pick it up.
    """
    doc = {k: v for k, v in asdict(score).items() if v not in (None, {}, [])}
    if score.passed is not None:
        doc["passed"] = score.passed
    return doc


# ParallelToolCalls' scores; a name carries an arg-mode suffix (parallel_recall_subset)
_PARALLEL_SCORES = ("parallel_recall", "parallel_precision", "parallel_detection")


def _run_notes(model: Any) -> dict[str, Any]:
    """The backend's own facts about the run (``Model.run_notes``); a failure
    there never takes down a finished run."""
    try:
        return dict(model.run_notes() or {}) if callable(getattr(model, "run_notes", None)) else {}
    except Exception as e:  # noqa: BLE001
        logger.warning("run_notes() failed for %s: %s", getattr(model, "name", model), e)
        return {}


def _token_count(value: Any) -> int:
    """A backend-reported token count as an int; a string ("5") is parsed and
    anything non-numeric or non-finite (NaN/Infinity) counts as 0, so neither
    fails the sample nor puts NaN into saved (strict) JSON."""
    try:
        value = float(value or 0)
    except (TypeError, ValueError):
        return 0
    return int(value) if math.isfinite(value) else 0


# Finish reasons whose EMPTY reply is not an answer (see Runner.execute): recorded as errors.
_NO_ANSWER_FINISH = ("length", "error", "abort", "content_filter")


def _raised_during_import(exc: BaseException) -> bool:
    """True when *exc* was raised while a module was being imported (e.g. torchaudio
    refusing a CUDA-mismatched torch inside ``import vllm``). Python strips its import
    machinery from tracebacks, but the imported module's top-level code still shows as
    a ``<module>`` frame -- which a generate() call only runs by importing something."""
    return any(f.name == "<module>" for f in traceback.extract_tb(exc.__traceback__))


def _is_setup_error(exc: BaseException) -> bool:
    """A failure every retry would repeat: a missing or broken import, a missing
    model/file, or a bad argument/configuration. Transient failures (connection
    resets, timeouts, 5xx/429) are not setup errors and keep their retries."""
    e: Optional[BaseException] = exc
    for _ in range(8):                      # follow explicit ``raise ... from`` causes
        if e is None:
            break
        if isinstance(e, (ImportError, NotImplementedError, TypeError, ValueError,
                          FileNotFoundError, IsADirectoryError, NotADirectoryError)):
            return True
        if isinstance(e, OSError) and not isinstance(e, (ConnectionError, TimeoutError)) and (
                "not a valid model identifier" in str(e) or "is not a local folder" in str(e)):
            return True
        if _raised_during_import(e):
            return True
        e = e.__cause__
    return False


def unscored_status(score: Score) -> Optional[str]:
    """``"unknown"`` / ``"not_tested"`` when *score* is not a measurement, else ``None``.

    The codebase convention for "no measurement" is ``metadata["unknown"] = True``
    with a placeholder value; ``metadata["status"]`` or ``label`` may say
    ``not_tested`` (nothing to measure). ``metadata["count_in_aggregate"]`` means the
    producer explicitly asked for the value to be used (e.g. a judge's opt-in
    ``unknown_score``), so it is treated as a measurement.
    """
    meta = score.metadata or {}
    if not meta.get("unknown") or meta.get("count_in_aggregate"):
        return None
    status = meta.get("status") or getattr(score, "label", None) or "unknown"
    return "not_tested" if status == "not_tested" else "unknown"

class Runner:
    """Drives a :class:`RunSpec` to a :class:`RunResult` in five stages."""

    def build_requests(
        self, scenario: Scenario, adapter: Adapter, config: RunConfig
    ) -> list[tuple[Sample, list[Request]]]:
        samples = list(scenario.samples())
        if config.limit is not None:
            samples = samples[: config.limit]
        split_cfg = config.split
        if split_cfg is not None:
            train, val, test = self._split_samples(samples, split_cfg)
            samples = test or val or train
            if hasattr(adapter, 'pool') and train:
                adapter.pool = train
        built = [(s, adapter.adapt(s, config)) for s in samples]
        # Images ride on every request of their sample, whatever the adapter.
        for s, reqs in built:
            if s.images:
                for r in reqs:
                    r.params["images"] = s.images
        # Chat-template kwargs from the RunConfig reach every request; a request's own win.
        if config.chat_template_kwargs:
            for _, reqs in built:
                for r in reqs:
                    r.params["chat_template_kwargs"] = {**config.chat_template_kwargs,
                                                       **(r.params.get("chat_template_kwargs") or {})}
        return built

    def _split_samples(self, samples, split_cfg):
        n = len(samples)
        t = split_cfg
        if t.strategy == "sequential":
            train_end = int(n * t.train_ratio)
            val_end = train_end + int(n * t.val_ratio)
            return samples[:train_end], samples[train_end:val_end], samples[val_end:]
        elif t.strategy == "random":
            import random
            rng = random.Random(t.seed)
            idx = list(range(n))
            rng.shuffle(idx)
            shuffled = [samples[i] for i in idx]
            train_end = int(n * t.train_ratio)
            val_end = train_end + int(n * t.val_ratio)
            return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]
        elif t.strategy == "stratified":
            from collections import defaultdict
            groups = defaultdict(list)
            for i, s in enumerate(samples):
                key = str(getattr(s, 'kind', None) or str(getattr(s, 'id', None) or i))
                groups[key].append(s)
            train, val, test = [], [], []
            for g in groups.values():
                gn = len(g)
                te = int(gn * t.train_ratio)
                ve = te + int(gn * t.val_ratio)
                train.extend(g[:te])
                val.extend(g[te:ve])
                test.extend(g[ve:])
            return train, val, test
        else:
            return samples, [], []

    def execute(
        self, model: Model, batch: list[tuple[Sample, list[Request]]],
        concurrency: int = 1, max_retries: int = 3, retry_delay: float = 1.0,
        timeout: float | None = None,
        latency: LatencyStats | None = None, throughput: Throughput | None = None,
    ) -> list[tuple[Sample, list[Result_]]]:
        flat, counts = self._flatten(batch)
        if not flat:
            return [(s, []) for s, _ in batch]
        if getattr(model, "per_request", False):
            # agent:/api: fail, retry (429/5xx/reset) and time out each request
            # on its own and never raise, so a batch retry here would only
            # re-send already-answered (side-effecting) requests, and a
            # chunk-wide timeout would drop them and orphan the rest of the
            # chunk. Hand the settings to the backend instead.
            # ponytail: sets attributes on the model object; per-call plumbing
            # if one model instance must serve runs with different settings.
            model.max_retries, model.retry_delay = max_retries, retry_delay
            if timeout is not None:
                model.timeout = timeout
            max_retries, timeout = 0, None

        # Fan out across threads ONLY when asked for it AND the model declares it
        # is safe to call concurrently. A single local model (HFGenModel/vLLM) is
        # not thread-safe, so calling generate() from many threads at once races
        # the same model — corrupt output or a crash. When we don't parallelize we
        # make a single batched generate() call, which also lets batched backends
        # process every request at once instead of fragmenting the batch.
        parallel = concurrency > 1 and getattr(model, "threadsafe", False)
        if not parallel:
            results = self._retry_generate(model, flat, max_retries, retry_delay, timeout, latency, throughput)
        else:
            # Never make more chunks than there are requests, so we never call
            # generate([]) on empty chunks (wasteful, and some backends choke).
            n_chunks = min(concurrency, len(flat))
            chunks = self._chunk(flat, n_chunks)
            # Parallel chunks OVERLAP in wall-clock, so letting each worker call
            # throughput.record() with its own elapsed sums overlapping durations
            # -- inflating total_time_ms ~n_chunks-fold and collapsing rps to the
            # serial rate. Pass throughput=None into the workers (latency still
            # records per call, which is honest -- those are call-level samples)
            # and record ONE span for the whole parallel region. len(flat) is
            # safe: a chunk that exhausts retries raises ModelError out of
            # pool.map, so we only reach the record line when every request was
            # served. Ceiling: the span also counts any retry-backoff sleeps that
            # fall inside the region (the per-call design excludes them) -- a
            # small effect that only fires when a chunk actually retries.
            parallel_start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=n_chunks) as pool:
                chunk_results = list(pool.map(
                    lambda c: self._retry_generate(model, c, max_retries, retry_delay, timeout, latency, None),
                    chunks,
                ))
            if throughput is not None:
                throughput.record((time.perf_counter() - parallel_start) * 1000.0, len(flat))
            results = [r for cr in chunk_results for r in cr]

        out: list[tuple[Sample, list[Result_]]] = []
        cursor = 0
        for (sample, _requests), n in zip(batch, counts):
            out.append((sample, results[cursor:cursor + n]))
            cursor += n
        return out

    def _retry_generate(self, model, requests, max_retries, retry_delay, timeout=None,
                         latency: LatencyStats | None = None, throughput: Throughput | None = None):
        """Calls ``model.generate(requests)`` with retry/backoff.

        When *latency*/*throughput* are given, records one sample per actual
        model-call attempt — success or failure alike, since a failing call
        still occupied real model/network time — but never the artificial
        ``time.sleep()`` backoff between retries, which is our own throttling,
        not the model's latency. This is a *call*-level measurement, not a
        true per-request one: batched backends (the default — see
        ``execute()``'s comment) answer many requests in a single call, so
        there is no per-request timestamp to read. One sample per call,
        tagged with how many requests it served, is the honest granularity
        actually available without breaking batching; it still yields a real
        distribution across a run's several chunks/attempts and a real
        requests/sec figure.
        """
        def _record(elapsed_ms: float, served: bool) -> None:
            # `served` distinguishes a successful call (which actually
            # returned results for every one of `requests`) from a failed
            # attempt: a failing call still occupied real model/network
            # time, so it's a legitimate latency sample -- but it served
            # zero requests, not len(requests), so it must NOT add to
            # Throughput's request count. Previously every attempt credited
            # the full batch size regardless of success, so a run that hit
            # even one retry over-counted total_requests (and therefore
            # inflated requests/sec) by the number of failed attempts.
            if latency is None and throughput is None:
                return
            with _generate_lock:
                if latency is not None:
                    latency.record(elapsed_ms)
                if throughput is not None:
                    throughput.record(elapsed_ms, len(requests) if served else 0)

        last_exc = None
        for attempt in range(max_retries + 1):
            start = time.perf_counter()
            try:
                if timeout is not None:
                    # future.result(timeout) must bound the WAIT. A `with` block
                    # would shutdown(wait=True) on exit and block until the slow
                    # generate() actually returned, so the timeout never cut
                    # anything (wall was bounded by generate(), not by `timeout`).
                    # shutdown(wait=False) returns at the budget. Ceiling: Python
                    # can't kill the worker, so the orphaned generate() runs on in
                    # the background (result dropped; joined by concurrent.futures'
                    # atexit hook at process exit) -- on a non-threadsafe local
                    # backend that makes `timeout` best-effort, since the orphan
                    # can overlap the next call the `parallel` guard serializes.
                    pool = ThreadPoolExecutor(max_workers=1)
                    future = pool.submit(model.generate, requests)
                    try:
                        result = future.result(timeout=timeout)
                    finally:
                        pool.shutdown(wait=False)
                else:
                    result = model.generate(requests)
            except FuturesTimeout:
                _record((time.perf_counter() - start) * 1000.0, served=False)
                raise ModelTimeout(f"generate timed out after {timeout}s")
            except (ExtraNotInstalled, CapabilityError):
                # Setup errors: every retry fails the same way, and wrapping
                # them would hide the fix behind "generate failed after N retries".
                raise
            except Exception as e:
                _record((time.perf_counter() - start) * 1000.0, served=False)
                if _is_setup_error(e):
                    raise           # same reason as above: surface the real cause at once
                last_exc = e
                if attempt < max_retries:
                    time.sleep(retry_delay * (2 ** attempt))
                    logger.warning("Retry %d/%d: %s", attempt + 1, max_retries, e)
                continue
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            per_request = [getattr(r, "latency_ms", None) for r in (result or [])]
            if (latency is not None and per_request
                    and all(isinstance(v, (int, float)) for v in per_request)):
                # A backend that times each request itself (api:, agent: through
                # generate_each) gives a real per-request latency: record those.
                # Throughput stays call-level: it is a wall-clock rate.
                with _generate_lock:
                    for v in per_request:
                        latency.record(float(v))
                    if throughput is not None:
                        throughput.record(elapsed_ms, len(requests))
            else:
                _record(elapsed_ms, served=True)
            return result
        raise ModelError(f"generate failed after {max_retries} retries: "
                         f"{type(last_exc).__name__}: {last_exc}") from last_exc

    def _flatten(self, batch):
        flat = []
        counts = []
        for _sample, reqs in batch:
            flat.extend(reqs)
            counts.append(len(reqs))
        return flat, counts

    def _chunk(self, items, n_chunks):
        k, m = divmod(len(items), n_chunks)
        return [items[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_chunks)]

    def annotate(
        self, annotators: list[Any], sample: Sample, results: list[Result_]
    ) -> dict[str, Any]:
        context: dict[str, Any] = {}
        for annotator in annotators:
            context[getattr(annotator, "name", type(annotator).__name__)] = annotator.annotate(
                sample, results
            )
        return context

    def score_one(
        self,
        metrics: list[Metric],
        sample: Sample,
        output: str,
        context: Any,
        run_id: str,
        errors: list | None = None,
        prompt: Optional[str] = None,
        extracted_by: str | None = None,
    ) -> tuple[list[Score], Prediction]:
        # extracted_by names an annotator whose context["extracted"] value
        # should be scored instead of the raw output (e.g. RegexAnnotator
        # pulling "42" out of "...FINAL ANSWER: 42"). Missing/misspelled
        # name degrades to raw output with a warning, not a crash -- a typo
        # here shouldn't take down a whole run.
        scoring_output = output
        if extracted_by:
            entry = context.get(extracted_by) if isinstance(context, dict) else None
            if isinstance(entry, dict) and "extracted" in entry:
                scoring_output = entry["extracted"]
            else:
                logger.warning("extracted_by=%r has no 'extracted' entry in context", extracted_by)
        scores: list[Score] = []
        failed_metrics: list[str] = []
        primary_failed = False
        for metric in metrics:
            if not metric.applicable(sample):
                continue
            try:
                produced = metric.score(sample, scoring_output, context)
            except ExtraNotInstalled:
                # A missing extra is a run-wide configuration problem, not a
                # per-sample scoring failure -- every remaining sample would
                # hit the exact same error. Swallowing it into a fake 0.0
                # score (the old behavior) is indistinguishable in
                # `headline` from "the model's output genuinely scored
                # zero," silently corrupting the aggregate. Fail loudly and
                # immediately instead, so the missing dependency is obvious.
                raise
            except Exception as e:
                # Don't fake a 0.0 Score here -- that's indistinguishable from
                # "the model's output genuinely scored zero" in every mean/
                # aggregate downstream (the exact corruption ExtraNotInstalled
                # above is deliberately not swallowed into either). A metric
                # crashing on this sample is a computation failure, not a
                # data point -- skip it so this sample's absence from
                # `metric.name`'s count is the honest signal, not a fabricated
                # score. `errors` still records exactly what happened.
                logger.error("Metric %s failed on sample %s: %s", metric.name, sample.id, e)
                if errors is not None:
                    errors.append({"sample_id": sample.id, "metric": metric.name, "error": str(e)})
                primary_failed = primary_failed or not scores
                failed_metrics.append(metric.name)
                continue
            produced_list = produced if isinstance(produced, list) else [produced]
            for s in produced_list:
                # A Score flagged metadata["unknown"] is not a measurement (its value
                # is a placeholder). "unknown" -- the metric could not decide, e.g. an
                # unreadable judge verdict -- is recorded exactly like a metric that
                # raised (same rule as task_completion). "not_tested" -- nothing to
                # measure on this sample -- is not an error. Neither is averaged
                # (see aggregate()); both stay on the prediction with their flag.
                if unscored_status(s) == "unknown":
                    reason = f"unknown result, not scored: {s.reason}" if s.reason else "unknown result, not scored"
                    logger.warning("Metric %s gave no usable result on sample %s: %s", s.name, sample.id, reason)
                    if errors is not None:
                        errors.append({"sample_id": sample.id, "metric": s.name, "error": reason})
                    if s.name not in failed_metrics:
                        failed_metrics.append(s.name)
                # Authoritative, not a fallback: metric.direction is required
                # (Metric.__init_subclass__ enforces it), so every Score a
                # metric produces carries ITS metric's declared direction,
                # regardless of what value the metric's own score() happened
                # to set (or forgot to). This is what makes direction
                # actually reliable for RunComparison/compare_models grading.
                s.direction = metric.direction
            scores.extend(produced_list)

        # The first applicable metric is the primary one. If it crashed, the
        # next metric must not stand in for it, and a sample any metric failed
        # on (e.g. malformed tool_calls/contexts) is never "correct".
        primary = scores[0] if scores and not primary_failed else None
        correct: Optional[bool]
        if failed_metrics:
            correct = False
        elif primary is None:
            correct = None
        elif primary.passed is not None:
            correct = primary.passed
        else:
            correct = primary.value == 1.0

        prediction = Prediction(
            run_id=run_id,
            task=sample.task or sample.kind.value,
            sample_id=sample.id or "",
            # The adapter-built prompt actually sent to the model when known
            # (a RAGAdapter's injected context, a ChatAdapter's system-prompt
            # wrapping, ...) -- falls back to the raw sample input only when
            # no request was built (e.g. an errored-out sample).
            prompt=prompt if prompt is not None else sample.input_text,
            raw_output=output,
            parsed_answer=scoring_output,
            expected=sample.target,
            correct=correct,
            score=primary.value if primary else None,
            context=context,
            metadata={"scores": [_score_doc(s) for s in scores],
                      **({"metric_errors": failed_metrics} if failed_metrics else {})},
        )
        return scores, prediction

    def aggregate(self, scores: list[Score]) -> dict[str, Stat]:
        """Per-metric Stats over real measurements only.

        A Score flagged ``metadata["unknown"]`` (``unknown`` / ``not_tested``) carries
        a placeholder value, not a measurement, so it is left out -- averaging it
        in would move the headline (a placeholder 0.0 made a lower-is-better
        Brier look better). They are counted by :meth:`unscored_counts`. A score
        whose producer explicitly opted in (``metadata["count_in_aggregate"]``,
        e.g. ``LLMJudge(unknown_score=0.5)``) is averaged as asked.
        """
        stats: dict[str, Stat] = {}
        for score in scores:
            if unscored_status(score) is not None:
                continue
            stats.setdefault(score.name, Stat(score.name)).add(score.value)
        return stats

    @staticmethod
    def unscored_counts(scores: list[Score]) -> dict[str, dict[str, int]]:
        """``{metric: {"unknown": n, "not_tested": m}}`` for results left out of the aggregate."""
        out: dict[str, dict[str, int]] = {}
        for score in scores:
            status = unscored_status(score)
            if status is not None:
                bucket = out.setdefault(score.name, {"unknown": 0, "not_tested": 0})
                bucket[status] += 1
        return out

    def run(self, spec: RunSpec, *, verbose: bool = False) -> RunResult:
        fingerprint = spec.fingerprint()
        config = spec.config

        # What this run consumed, with each input's lexsi_provenance.json (a
        # dataset folder via ak.load_dataset, an hf:/vllm: model folder).
        inputs = [i for i in ((getattr(spec.scenario, "metadata", None) or {}).get("lexsi_input"),) if i]
        model_ref = getattr(spec.model, "_model_name", None)
        if isinstance(model_ref, str):
            inputs.append(input_ref("model", model_ref))

        cache = DiskCache()
        cached = cache.get(fingerprint)
        if cached is not None:
            cached.metadata = {**cached.metadata, "inputs": inputs}  # lineage of THIS call
            return cached

        run_id = spec.run_name or fingerprint
        _run_notes(spec.model)   # drop what an earlier, aborted run left behind

        batch = self.build_requests(spec.scenario, spec.adapter, config)
        logger.info("Starting run with %d samples", len(batch))
        if (any(s.tools for s, _ in batch) and getattr(spec.adapter, "method", "") != "tools"
                and not getattr(spec.model, "reads_actual_output", False)):
            # The model would never see the tools and every tool metric would
            # quietly score it as "made no call" -- say so instead.
            logger.warning(
                "Samples carry tools but the adapter is %r, so the model is not "
                "shown them. Pass adapter=ToolCallAdapter() (or adapter='auto').",
                getattr(spec.adapter, "method", type(spec.adapter).__name__))

        has_loglikelihood = any(
            r.request_type == "loglikelihood"
            for _, reqs in batch
            for r in reqs
        )

        all_scores: list[Score] = []
        predictions: list[Prediction] = []
        errors: list[dict] = []
        failed_count: int = 0
        latency = LatencyStats()
        throughput = Throughput()
        # Real, provider-reported token counts -- only ever non-zero when a
        # backend actually populates Result_.usage (the hosted-API backends;
        # see openai.py/anthropic.py/groq_gen.py/api_gen.py/litellm_gen.py).
        # Local backends and callables leave this at zero -- never estimated
        # by counting characters or guessing a tokenizer.
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        def _accumulate_usage(results: list[Result_]) -> None:
            for r in results:
                for key in token_usage:
                    token_usage[key] += _token_count(r.usage.get(key))

        if getattr(spec.model, "reads_actual_output", False):
            # Score pre-generated answers (Sample.actual_output) — no model call.
            for index, (sample, _reqs) in enumerate(batch):
                if sample.id is None:
                    sample.id = str(index)
                output = sample.actual_output or ""
                context = self.annotate(
                    spec.annotators, sample, [Result_(completions=[Generated(text=output)])]
                )
                if sample.actual_trace:
                    # A recorded agent/RAG run (tool-call turns, transcript,
                    # retrieved contexts) scored offline, same key as live runs.
                    context["trace"] = sample.actual_trace
                scores, prediction = self.score_one(
                    spec.metrics, sample, output, context, run_id, errors,
                    extracted_by=spec.extracted_by,
                )
                all_scores.extend(scores)
                predictions.append(prediction)
        elif has_loglikelihood:
            if not spec.model.supports(Capability.LOGLIKELIHOOD):
                raise CapabilityError(
                    f"{spec.model.name} does not declare LOGLIKELIHOOD"
                )
            flat = [r for _, reqs in batch for r in reqs]
            logger.info("Executing %d requests", len(flat))
            last_exc = None
            for attempt in range(config.max_retries + 1):
                start = time.perf_counter()
                try:
                    lls = spec.model.loglikelihood(flat)
                except Exception as e:
                    # A failing attempt occupied real time (legitimate
                    # latency sample) but served zero requests -- crediting
                    # it with len(flat) would double-count against the
                    # eventual successful retry (see the identical fix in
                    # _retry_generate() above).
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    latency.record(elapsed_ms)
                    throughput.record(elapsed_ms, 0)
                    last_exc = e
                    if attempt < config.max_retries:
                        time.sleep(config.retry_delay * (2 ** attempt))
                        logger.warning("Retry %d/%d for loglikelihood: %s", attempt + 1, config.max_retries, e)
                    continue
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                latency.record(elapsed_ms)
                throughput.record(elapsed_ms, len(flat))
                break
            else:
                raise ModelError(f"loglikelihood failed after {config.max_retries} retries") from last_exc
            logger.info("Got %d responses", len(lls))
            cursor = 0
            for index, (sample, reqs) in enumerate(batch):
                try:
                    if sample.id is None:
                        sample.id = str(index)
                    n = len(reqs)
                    sample_lls = lls[cursor: cursor + n]
                    cursor += n
                    choice_probs = [ll.logprob for ll in sample_lls]
                    argmax_idx = max(
                        range(len(choice_probs)), key=lambda i: choice_probs[i]
                    )
                    output = sample.choices[argmax_idx] if sample.choices else ""
                    context = self.annotate(
                        spec.annotators, sample, [Result_(completions=[Generated(text=output)])]
                    )
                    context["choice_likelihoods"] = choice_probs
                    prompt = _prompt_text(reqs[0].prompt) if reqs else None
                    # extracted_by deliberately NOT threaded through here:
                    # `output` on this path is already sample.choices[argmax_idx]
                    # -- the exact, correct, final choice text picked by
                    # comparing logprobs, not free-form generated text. There
                    # is nothing to clean up, and applying extraction here is
                    # actively dangerous for choice-index metrics (Acc/AccNorm
                    # interpret a bare digit-string as a CHOICE INDEX, not "a
                    # number that happened to appear in the choice text") --
                    # a correct pick can silently get scored as a different,
                    # wrong choice, or as no match at all. Annotators still
                    # run on this branch (context is populated normally, for
                    # logging/inspection), only the extraction hookup into
                    # scoring is structurally disabled here.
                    scores, prediction = self.score_one(
                        spec.metrics, sample, output, context, run_id, errors, prompt=prompt,
                    )
                    all_scores.extend(scores)
                    predictions.append(prediction)
                except Exception as e:
                    logger.error("Sample %s failed: %s", sample.id, e)
                    failed_count += 1
                    predictions.append(Prediction(
                        run_id=run_id,
                        task=getattr(sample, 'task', '') or '',
                        sample_id=sample.id or str(index),
                        prompt=str(sample.input),
                        raw_output="",
                        parsed_answer=None,
                        expected=None,
                        correct=False,
                        score=0.0,
                    ))
                    errors.append({"sample_id": sample.id or str(index), "error": str(e)})
                    continue
                logger.debug("Scored %d/%d samples", index + 1, len(batch))
                if verbose and (index + 1) % 10 == 0:
                    print(".", end="", flush=True)
        else:
            requests = [r for _, reqs in batch for r in reqs]
            if any(r.params.get("tools") for r in requests) and not spec.model.supports(Capability.TOOLS):
                # Tool schemas travel as a request param; a backend without
                # native tool support would silently drop them and the model
                # would answer without ever seeing the tools.
                raise CapabilityError(
                    f"{spec.model.name} does not accept native tool schemas. Use an "
                    f"'api:'/'agent:'/'hf:' backend, or ToolCallAdapter(mode='prompt') to "
                    f"describe the tools in the prompt instead."
                )
            logger.info("Executing %d requests", len(requests))
            executed = self.execute(
                spec.model, batch, concurrency=config.concurrency, max_retries=config.max_retries,
                retry_delay=config.retry_delay, timeout=config.timeout,
                latency=latency, throughput=throughput,
            )
            logger.info("Got %d responses", len(executed))
            for index, (sample, results) in enumerate(executed):
                try:
                    if sample.id is None:
                        sample.id = str(index)
                    output = results[0].text if results else ""
                    _accumulate_usage(results)
                    context = self.annotate(spec.annotators, sample, results)
                    trace = results[0].completions[0].trace if results and results[0].completions else None
                    if trace:
                        # Structured output (native tool calls, an agent's
                        # transcript, retrieved contexts) for agent/RAG metrics.
                        context["trace"] = trace
                    # An empty reply truncated at max_tokens (finish_reason
                    # 'length', typical of reasoning models), a per-request
                    # backend failure ('error'), a request the server aborted
                    # ('abort', SGLang) or a filtered one ('content_filter')
                    # carries no real answer. Scoring "" silently can even mark
                    # an irrelevance sample "correct", so record it as a failed
                    # sample (the except below) instead. Non-empty text or any
                    # other finish_reason scores normally.
                    gen = results[0].completions[0] if results and results[0].completions else None
                    if (gen is not None and gen.finish_reason in _NO_ANSWER_FINISH
                            and not trace and not output.strip()):
                        hint = ("reasoning model truncated at max_tokens? raise RunConfig.max_tokens"
                                if gen.finish_reason == "length" else "the server gave no answer")
                        raise ModelError(gen.error or (
                            f"empty output with finish_reason={gen.finish_reason!r} and no trace ({hint})"
                        ))
                    # execute() preserves batch's order/length, so batch[index]
                    # is the (sample, requests) pair this result came from.
                    reqs = batch[index][1]
                    prompt = _prompt_text(reqs[0].prompt) if reqs else None
                    scores, prediction = self.score_one(
                        spec.metrics, sample, output, context, run_id, errors, prompt=prompt,
                        extracted_by=spec.extracted_by,
                    )
                    all_scores.extend(scores)
                    predictions.append(prediction)
                except Exception as e:
                    logger.error("Sample %s failed: %s", sample.id, e)
                    failed_count += 1
                    predictions.append(Prediction(
                        run_id=run_id,
                        task=getattr(sample, 'task', '') or '',
                        sample_id=sample.id or str(index),
                        prompt=str(sample.input),
                        raw_output="",
                        parsed_answer=None,
                        expected=None,
                        correct=False,
                        score=0.0,
                    ))
                    errors.append({"sample_id": sample.id or str(index), "error": str(e)})
                    continue
                logger.debug("Scored %d/%d samples", index + 1, len(executed))
                if verbose and (index + 1) % 10 == 0:
                    print(".", end="", flush=True)

        notes = _run_notes(spec.model)
        if "parallel_tool_calls" in notes.get("unverified_request_fields", {}):
            # The one-call cap may never have reached the model: say so on every
            # parallel score, so a reader doesn't take it as the capped behaviour.
            for s in all_scores:
                if s.name.startswith(_PARALLEL_SCORES):
                    s.metadata = {**(s.metadata or {}), "cap_unverified": True}
            for p in predictions:
                for doc in (p.metadata or {}).get("scores", []):
                    if str(doc.get("name", "")).startswith(_PARALLEL_SCORES):
                        doc["metadata"] = {**doc.get("metadata", {}), "cap_unverified": True}
        stats = self.aggregate(all_scores)
        unscored = self.unscored_counts(all_scores)
        for name, n in Counter(e["metric"] for e in errors if "metric" in e).items():
            logger.warning("Metric %s failed on %d sample(s); they are left out of its headline "
                           "and marked incorrect (see RunResult.errors)", name, n)
        logger.info("Run complete – %d metrics computed", len(stats))
        if verbose:
            print(f" done – {len(stats)} metrics")
        headline = {name: stat.mean for name, stat in stats.items()}
        # All three of perf/model_size/token_usage are populated unless the
        # caller opts out via RunConfig.track_performance=False (default
        # True -- model_info() below can be a real, if best-effort,
        # introspection cost for local backends, so callers who don't want
        # that can disable it). This is a *reporting* gate, not a timing
        # one -- latency/throughput/usage were already measured for free
        # above as part of running the requests; we just don't surface them
        # when disabled.
        if config.track_performance:
            # Empty on the `reads_actual_output` path (no model call was made
            # at all -- nothing to time) and on the loglikelihood/generative
            # paths when there were zero requests; LatencyStats/Throughput
            # .stats() both degrade to zeroed dicts rather than raising.
            perf = {"latency_ms": latency.stats(), "throughput": throughput.stats()}
            # Token throughput: derived from two numbers that already exist
            # separately (real elapsed call time, real token counts from the
            # backend's own API response) but were never combined into a
            # tokens/sec figure. total_time_ms is the same denominator
            # throughput["rps"] already uses -- real wall-clock time spent in
            # model calls, not the whole run (annotation/scoring time excluded).
            total_time_s = perf["throughput"].get("total_time_ms", 0.0) / 1000.0
            if total_time_s > 0:
                perf["throughput"]["output_tokens_per_sec"] = token_usage["completion_tokens"] / total_time_s
                perf["throughput"]["total_tokens_per_sec"] = token_usage["total_tokens"] / total_time_s
            else:
                perf["throughput"]["output_tokens_per_sec"] = 0.0
                perf["throughput"]["total_tokens_per_sec"] = 0.0
            # Never caller-supplied: introspected for local backends (real
            # parameter count/sparsity/size), identity-only for hosted-API
            # backends (nothing to measure -- see Model.model_info()).
            # Guarded because a backend's own introspection (e.g. loading a
            # model just to inspect it) is best-effort and shouldn't take
            # down an otherwise successful run. Skipped entirely (not just
            # hidden) when tracking is off, since this is the one field here
            # that can cost real time to compute, not just to report.
            try:
                model_size = spec.model.model_info() if spec.model else {}
            except Exception as e:
                logger.warning("model_info() failed for %s: %s", getattr(spec.model, "name", spec.model), e)
                model_size = {}
        else:
            perf = None
            model_size = None
            token_usage = None
        result = RunResult(
            run_id=run_id,
            fingerprint=fingerprint,
            stats=stats,
            predictions=predictions,
            headline=headline,
            config=spec.config,
            model_spec=str(spec.model) if spec.model else None,
            errors=errors,
            unscored=unscored,
            # a sample whose metric crashed is a failed sample, not a silent skip
            failed_count=failed_count + sum(
                1 for p in predictions if (getattr(p, "metadata", None) or {}).get("metric_errors")),
            model_size=model_size,
            token_usage=token_usage,
            perf=perf,
            metadata={"inputs": inputs, **notes},
        )
        cache.set(fingerprint, result)
        return result
