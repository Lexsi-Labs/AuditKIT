"""The reproducible description of a run: :class:`RunConfig` and :class:`RunSpec`.

A ``RunSpec`` binds together everything a run needs — the scenario, the model,
the adapter, the metrics, the config — and can distil itself to a stable
:meth:`RunSpec.fingerprint`. Two runs with the same fingerprint are the same
experiment; changing the model, the seed, the few-shot count, or the judge
changes the fingerprint, which is what makes results comparable and cacheable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING, Union

if TYPE_CHECKING:  # avoid import cycles; these are only type hints here
    from .adapter import Adapter
    from .metric import Metric
    from .model import Model
    from .scenario import Scenario


@dataclass
class RunConfig:
    """The knobs that shape a run (all optional, all with sane defaults)."""

    num_fewshot: Union[int, None] = None
    limit: Union[int, None] = None
    # None (not 0): every other generation-affecting field here defaults to
    # None so _gen_params() correctly skips it when unset. A concrete 0
    # defeats that -- seed=0 was being force-included in every generation
    # request whether the caller asked for it or not, and some providers
    # (confirmed: at least one Groq model) reject the seed param outright,
    # turning every request into a 400. Explicitly pass seed=0 if you want
    # a real, reproducible seed sent to the backend.
    seed: Union[int, None] = None
    trials: int = 1
    batch_size: Union[str, int] = "auto"
    # Default 1 (not higher) because Runner._chunk() always splits requests
    # into exactly `concurrency` chunks regardless of request count -- with
    # fewer requests than concurrency, that produces empty chunks and wastes
    # real model/API calls on them.
    # concurrency<=1 takes Runner.execute()'s no-chunking path entirely, so
    # this is the only value that's safe unconditionally.
    concurrency: int = 1
    split: SplitConfig | None = None
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    stop_sequences: list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    num_completions: int | None = None
    best_of: int | None = None
    timeout: float | None = None
    max_retries: int = 3
    retry_delay: float = 1.0
    judge_model: Union[str, None] = None
    judge_prompt_version: Union[str, None] = None
    # On by default: RunResult.perf/model_size/token_usage are populated
    # unless explicitly disabled. Set False to skip surfacing them (they are
    # still measured for free during the run either way -- this only gates
    # whether they're attached to the result; see Runner.run()) -- useful if
    # local-backend model_size introspection's real, if best-effort, cost is
    # unwanted. Part of the fingerprint (RunConfig.to_dict() is hashed
    # whole), so toggling this forces a fresh run rather than replaying a
    # cached perf-less result.
    track_performance: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunConfig:
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        valid = {k: v for k, v in data.items() if k in field_names}
        return cls(**valid)


@dataclass
class SplitConfig:
    strategy: str = "sequential"
    train_ratio: float = 0.0
    val_ratio: float = 0.0
    test_ratio: float = 1.0
    fold: int = 0
    # Fixed, not None: Runner._split_samples()'s "random"/"stratified" paths
    # seed `random.Random(seed)` fresh on every call. `None` draws OS entropy
    # each time, so reusing the identical SplitConfig (or RunConfig) across a
    # baseline and a candidate evaluate() call -- the normal "same eval
    # config" comparison pattern -- would silently reshuffle differently each
    # time, breaking RunComparison's sample_id-based alignment (newly_wrong,
    # significance) even though nothing about the config looked different.
    # A caller who genuinely wants a different shuffle per run still can, by
    # passing a different seed= explicitly.
    seed: int | None = 0


@dataclass
class RunSpec:
    """Everything a run needs, plus a stable fingerprint over what matters."""

    scenario: "Scenario"
    model: "Model"
    adapter: "Adapter"
    metrics: list["Metric"]
    annotators: list[Any] = field(default_factory=list)
    # Names an annotator (by its .name) whose context["extracted"] value
    # should be scored instead of the raw output -- e.g. a RegexAnnotator
    # pulling "42" out of "...FINAL ANSWER: 42". None (default): score the
    # raw output, today's behavior, unchanged.
    extracted_by: Union[str, None] = None
    config: RunConfig = field(default_factory=RunConfig)
    run_name: str = ""

    def fingerprint(self) -> str:
        """A stable sha256[:16] over the identity-defining parts of the run.

        Hashes the *entire* config (via ``RunConfig.to_dict()``) rather than a
        hand-picked subset of fields, so a run-affecting setting (``limit``,
        ``batch_size``, a new field added later, ...) can never be silently
        left out of the fingerprint the way ``limit`` previously was.
        """
        def _identity(obj: Any, default_attr: str) -> Any:
            fn = getattr(obj, "identity", None)
            if callable(fn):
                return fn()
            return getattr(obj, default_attr, type(obj).__name__)

        key = {
            "engine": "native",
            "model": _identity(self.model, "name"),
            "scenario": getattr(self.scenario, "name", type(self.scenario).__name__),
            # identity() captures the *prompt* (adapter template/system prompt,
            # judge prompt/choices), not just the type — so changing a prompt
            # changes the fingerprint and can't return a stale cached result.
            "adapter": _identity(self.adapter, "method"),
            "metrics": sorted(
                json.dumps(_identity(m, "name"), sort_keys=True, default=str)
                for m in self.metrics
            ),
            # Annotators have no scoring effect unless extracted_by names
            # one -- but hashing both unconditionally means changing a
            # regex pattern (or which annotator feeds scoring) always
            # invalidates the cache, never silently reuses a stale result
            # scored under a different pattern.
            "annotators": sorted(
                json.dumps(_identity(a, "name"), sort_keys=True, default=str)
                for a in self.annotators
            ),
            "extracted_by": self.extracted_by,
            "run_name": self.run_name,
            "config": self.config.to_dict(),
        }
        blob = json.dumps(key, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]
