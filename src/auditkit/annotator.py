from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from typing import TYPE_CHECKING

from .registry import ANNOTATORS
from ._identity_guard import warn_if_identity_incomplete

if TYPE_CHECKING:
    from .sample import Sample
    from .model import Result_


class Annotator(ABC):
    name: str = "annotator"

    @abstractmethod
    def annotate(self, sample: Sample, results: list[Result_]) -> dict[str, Any]:
        ...

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        warn_if_identity_incomplete(cls, Annotator, "annotate")

    def identity(self) -> dict:
        """The config that defines this annotator's behavior, for the run
        fingerprint. Override when an annotator has real config (a pattern,
        a prompt, ...) — otherwise two differently-configured instances of
        the same annotator class would silently collide in the fingerprint,
        same class of bug already fixed for adapters/judges elsewhere.
        ``__init_subclass__`` above warns at class-definition time when a
        subclass takes constructor arguments but skips this override (unless
        it instead makes ``self.name`` itself parameter-derived, which
        already protects the fingerprint — see ``_identity_guard.py``)."""
        return {"name": self.name}


# The default extraction pattern: everything, spanning newlines. It exists so
# `pattern` can be optional, which is what makes a bare registry name --
# `ANNOTATORS.get("regex")()`, the form `ak.evaluate(annotators="regex")` and the
# platform's `annotators: ["regex"]` both resolve to -- actually constructible.
# Before this, every registered annotator required a constructor argument, so a
# bare name raised TypeError partway through a run that had already been
# submitted and allocated compute.
#
# For ThinkingStripAnnotator this default is the useful case rather than a
# degenerate one: strip the <think> block, keep everything else.
_MATCH_ALL = r"(?s).+"


@ANNOTATORS.register("regex")
class RegexAnnotator(Annotator):
    """Extracts a substring from the model output via a user-supplied regex.

    The extracted value lands in the run's ``context`` under this
    annotator's ``name`` as ``{"extracted": ..., "matched": bool, "raw":
    ...}``. Pass this annotator's ``name`` as ``extract_with=`` on
    ``ak.evaluate()`` to actually score against the extracted value instead
    of the raw output — without that, the extraction still happens but sits
    unused in ``context``, available to any custom :class:`Metric` that
    reads it directly.

    Most useful on the generative and precomputed (``actual_output``)
    paths, where output is free text. On the loglikelihood/MCQ path,
    ``extract_with`` is intentionally never honored by the ``Runner`` --
    output there is already the exact, correct choice text picked by
    comparing logprobs, and extracting a number from inside that text is a
    different operation than the choice-index-based metrics (``Acc``/
    ``AccNorm``) expect from a bare digit-string, which can silently
    misattribute a correct pick as wrong. This annotator still runs and
    populates ``context`` there for inspection; only the extraction hookup
    into scoring is disabled on that specific path.

    ``cast`` converts the extracted string (e.g. ``int``, ``float``, or any
    ``str -> Any`` callable) -- ``None`` (default) keeps it a plain string,
    unchanged from before this existed. A failing cast degrades to the
    original string plus ``"cast_failed": True`` in the returned dict rather
    than raising, so one malformed sample can't crash a whole run; pass
    ``strict=True`` to raise instead, if you'd rather fail fast.
    """

    def __init__(
        self,
        pattern: str = _MATCH_ALL,
        group: int | str = 0,
        flags: int = 0,
        on_no_match: str = "",
        name: str = "regex",
        cast: Optional[Callable[[str], Any]] = None,
        strict: bool = False,
    ) -> None:
        self.pattern = pattern
        self.flags = flags
        self._regex = re.compile(pattern, flags)
        self._group = group
        self._on_no_match = on_no_match
        self.name = name
        self._cast = cast
        self._strict = strict

    def identity(self) -> dict:
        return {
            "name": self.name, "pattern": self.pattern, "group": self._group,
            "flags": self.flags, "on_no_match": self._on_no_match,
            "cast": getattr(self._cast, "__name__", repr(self._cast)) if self._cast else None,
            "strict": self._strict,
        }

    def annotate(self, sample: Sample, results: list[Result_]) -> dict[str, Any]:
        text = results[0].text if results else ""
        m = self._regex.search(text)
        if m is None:
            return {"extracted": self._on_no_match, "matched": False, "raw": text}
        value = m.group(self._group)
        if value is None:  # matched, but this particular group didn't participate
            value = self._on_no_match

        if self._cast is None:
            return {"extracted": value, "matched": True, "raw": text}
        try:
            return {"extracted": self._cast(value), "matched": True, "raw": text}
        except (ValueError, TypeError) as e:
            if self._strict:
                raise ValueError(
                    f"{self.name}: cast {self._cast!r} failed on extracted value {value!r}: {e}"
                ) from e
            return {"extracted": value, "matched": True, "raw": text, "cast_failed": True}


@ANNOTATORS.register("thinking_strip")
class ThinkingStripAnnotator(Annotator):
    """Extracts a value via regex, after first stripping ``<think>...</think>``
    reasoning blocks some models embed directly inside their own output.

    A few real hosted reasoning models (seen live against Groq's
    ``qwen/qwen3.6-27b``) put their entire draft reasoning process inline in
    the same text as the final answer, wrapped in ``<think>`` tags — and
    that draft often mentions the target pattern (e.g. ``"ANSWER: 42"``)
    multiple times before reaching the real final line. A plain
    :class:`RegexAnnotator`'s ``re.search()`` matches the *first*
    occurrence, which is frequently still inside the reasoning, not the
    real answer. This strips the ``<think>`` block(s) first, then takes the
    *last* match in what remains, then (by default) trims trailing
    backtick/parenthetical commentary some models append after the value
    (seen live, e.g. ``"18` (or `$18`, but usually just the number is
    fine.)"``).
    """

    def __init__(
        self,
        pattern: str = _MATCH_ALL,
        group: int | str = 0,
        flags: int = 0,
        on_no_match: str = "",
        name: str = "thinking_strip",
        strip_pattern: str = r"<think>.*?</think>",
        trim_trailing_noise: bool = True,
    ) -> None:
        self.pattern = pattern
        self.flags = flags
        self._regex = re.compile(pattern, flags)
        self._group = group
        self._on_no_match = on_no_match
        self.name = name
        self.strip_pattern = strip_pattern
        self._strip_regex = re.compile(strip_pattern, re.DOTALL)
        self._trim_trailing_noise = trim_trailing_noise

    def identity(self) -> dict:
        return {
            "name": self.name, "pattern": self.pattern, "group": self._group,
            "flags": self.flags, "on_no_match": self._on_no_match,
            "strip_pattern": self.strip_pattern,
            "trim_trailing_noise": self._trim_trailing_noise,
        }

    def annotate(self, sample: Sample, results: list[Result_]) -> dict[str, Any]:
        text = results[0].text if results else ""
        stripped = self._strip_regex.sub("", text)
        matches = list(self._regex.finditer(stripped))
        if not matches:
            return {"extracted": self._on_no_match, "matched": False, "raw": text}
        value = matches[-1].group(self._group)
        if value is None:  # matched, but this particular group didn't participate
            return {"extracted": self._on_no_match, "matched": False, "raw": text}
        if self._trim_trailing_noise:
            value = re.split(r"[`(]", value)[0].strip()
        return {"extracted": value, "matched": True, "raw": text}


@ANNOTATORS.register("llm")
class LLMAnnotator(Annotator):
    """Extracts (or transforms) a value from the model output by asking a
    second model to do it — the same model-resolution machinery as
    ``LLMJudge``/generation (a spec string like ``"openai:gpt-4o-mini"``,
    resolved lazily, or any object with ``.generate``).

    Useful where a regex can't do the job: pulling a value out of free-form
    prose, normalizing an answer's phrasing, translating, summarizing, or
    any other extraction that needs judgment rather than a fixed pattern.

    Parameters
    ----------
    model
        A model spec string or any object with ``generate(list[Request]) ->
        list[Result_]`` — same contract as ``LLMJudge.judge_model``.
    prompt
        User-prompt template. ``{input}``, ``{output}`` (the sample's raw
        model output), ``{expected}`` (alias ``{target}``), ``{context}``,
        and any ``sample.metadata`` key are filled in.
    system_prompt
        Optional system-level instructions, prepended to every call.
    model_args
        Connection-level kwargs forwarded to ``AutoModel.resolve`` for a
        string spec (``api_key``/``api_base``/``device``/``hf_token``) —
        not generation settings, same split as ``LLMJudge``.
    temperature, max_tokens, top_p
        Generation settings for the annotator model's own call — like
        ``LLMJudge``, this call doesn't go through an ``Adapter``/
        ``RunConfig``, so this is the only way to control them.
    cast, strict
        Same contract as ``RegexAnnotator``: ``cast`` converts the model's
        (stripped) reply via a ``str -> Any`` callable; a failing cast
        degrades to the original string plus ``"cast_failed": True`` unless
        ``strict=True``, in which case it raises.
    on_empty
        Fallback value (default ``""``) when the model returns an empty
        reply.
    """

    def __init__(
        self,
        *,
        model: Any = None,
        prompt: str = "Extract the requested value from the OUTPUT below. "
                      "Respond with only that value, nothing else.\n"
                      "Input: {input}\nOutput: {output}",
        system_prompt: Optional[str] = None,
        name: str = "llm_annotator",
        model_args: Optional[dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        cast: Optional[Callable[[str], Any]] = None,
        strict: bool = False,
        on_empty: str = "",
    ) -> None:
        if model is None:
            raise ValueError(f"{name}: requires a model")
        self.name = name
        self._model_spec = model
        self._model_args = model_args or {}
        self._resolved_model: Any = None
        self.prompt = prompt
        self.system_prompt = system_prompt
        self._gen_params = {
            k: v for k, v in
            {"temperature": temperature, "max_tokens": max_tokens, "top_p": top_p}.items()
            if v is not None
        }
        self._cast = cast
        self._strict = strict
        self._on_empty = on_empty

    def _model(self) -> Any:
        if self._resolved_model is None:
            spec = self._model_spec
            if isinstance(spec, str):
                from .model import AutoModel
                self._resolved_model = AutoModel.resolve(spec, **self._model_args)
            else:
                self._resolved_model = spec
        return self._resolved_model

    def _model_identity(self) -> Any:
        spec = self._model_spec
        if isinstance(spec, str):
            return spec
        fn = getattr(spec, "identity", None)
        if callable(fn):
            return fn()
        return getattr(spec, "name", type(spec).__name__)

    def _render(self, sample: Sample, output: str) -> str:
        fields = {
            "input": getattr(sample, "input_text", None) or str(sample.input),
            "output": output,
            "expected": sample.target or "",
            "target": sample.target or "",
            "context": "\n".join(sample.retrieval_context or []),
        }
        for k, v in (sample.metadata or {}).items():
            fields.setdefault(k, v)
        text = self.prompt
        for key, val in fields.items():
            text = text.replace("{" + key + "}", str(val))
        return text

    def _assemble(self, user_prompt: str) -> str:
        if self.system_prompt:
            return self.system_prompt + "\n\n" + user_prompt
        return user_prompt

    def identity(self) -> dict:
        return {
            "name": self.name, "kind": "llm_annotator", "model": self._model_identity(),
            "prompt": self.prompt, "system_prompt": self.system_prompt,
            "gen_params": self._gen_params,
            "cast": getattr(self._cast, "__name__", repr(self._cast)) if self._cast else None,
            "strict": self._strict, "on_empty": self._on_empty,
        }

    def annotate(self, sample: Sample, results: list[Result_]) -> dict[str, Any]:
        from .model import Request

        text = results[0].text if results else ""
        full = self._assemble(self._render(sample, text))
        model = self._model()
        gen_results = model.generate([Request(prompt=full, params=dict(self._gen_params))])
        reply = ""
        if gen_results and getattr(gen_results[0], "completions", None):
            reply = gen_results[0].completions[0].text or ""
        reply = reply.strip()

        if not reply:
            return {"extracted": self._on_empty, "matched": False, "raw": reply}
        if self._cast is None:
            return {"extracted": reply, "matched": True, "raw": reply}
        try:
            return {"extracted": self._cast(reply), "matched": True, "raw": reply}
        except (ValueError, TypeError) as e:
            if self._strict:
                raise ValueError(
                    f"{self.name}: cast {self._cast!r} failed on reply {reply!r}: {e}"
                ) from e
            return {"extracted": reply, "matched": True, "raw": reply, "cast_failed": True}
