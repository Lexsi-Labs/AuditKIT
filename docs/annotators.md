# Annotators — extracting a clean value from raw output before scoring

Annotators are a separate, opt-in step between generation and scoring:
they pull a value out of a model's raw text (`"Let me think...\nFINAL
ANSWER: 42"` → `"42"`) without changing what actually got generated.
Extraction and scoring are independent — an annotator can run and populate
`Prediction.context` purely for inspection, or its output can drive the
metrics via `extract_with=`. This page traces the real code path
(`src/auditkit/annotator.py`, `src/auditkit/runner.py`) with runnable
examples, the same standard as [LLM-as-Judge Rendering](llm_judge_rendering.md).

## Why this exists

A model's raw generation is rarely the bare value you want to score against
a gold answer. Chain-of-thought reasoning, a full sentence where you wanted a
short span, or unpredictable surrounding text shouldn't force every metric
to re-implement its own text cleanup. An annotator does the cleanup once,
and any metric can then score against the cleaned value instead of the raw
one.

```python
import auditkit as ak
from auditkit.annotator import RegexAnnotator
from auditkit.model import CallableModel
from auditkit.sample import Sample

model = CallableModel(lambda prompts: ["Reasoning...\nFINAL ANSWER: 42" for _ in prompts])
sample = Sample(input="What is 6*7?", target="42")

# Without extraction: raw_output != target, ExactMatch fails.
result = ak.evaluate([sample], model=model, scorers="exact_match")
print(result.headline)  # {'exact_match': 0.0}

# With extraction: the annotator's clean value drives scoring instead.
answer = RegexAnnotator(r"FINAL ANSWER:\s*(.+)", group=1, name="answer")
result = ak.evaluate(
    [sample], model=model, scorers="exact_match",
    annotators=answer, extract_with="answer",
)
print(result.headline)                          # {'exact_match': 1.0}
print(result.predictions[0].raw_output)          # "Reasoning...\nFINAL ANSWER: 42" -- unchanged
print(result.predictions[0].parsed_answer)       # "42" -- what was actually scored
print(result.predictions[0].context["answer"])   # {'extracted': '42', 'matched': True, 'raw': '...'}
```

`raw_output` always keeps the true, unmodified generation. `parsed_answer`
and the metric scores only change when `extract_with=` names an annotator
that actually ran.

## The base contract

Every annotator implements one method:

```python
class Annotator(ABC):
    name: str = "annotator"

    @abstractmethod
    def annotate(self, sample: Sample, results: list[Result_]) -> dict[str, Any]:
        ...
```

`Runner.annotate()` calls `annotate(sample, results)` for every configured
annotator and stores the returned dict at `context[annotator.name]`. The
one contract every built-in annotator (and `extract_with=`) relies on:
**the returned dict should have an `"extracted"` key** if you want
`extract_with=` to be able to use it — `Runner.score_one()` looks up
`context[extracted_by]["extracted"]` specifically.

`results` is the real model output for that sample, wrapped the same way
across every adapter/execution path (generative, precomputed/
`actual_output`, and — with one deliberate exception, see below —
loglikelihood/MCQ) — `results[0].text` is always the string to work with.

## `RegexAnnotator` — deterministic, zero extra model calls

```python
RegexAnnotator(
    pattern: str,
    group: int | str = 0,
    flags: int = 0,
    on_no_match: str = "",
    name: str = "regex",
    cast: Callable[[str], Any] | None = None,
    strict: bool = False,
)
```

| Param | Meaning |
|---|---|
| `pattern` | The regex to search the raw output for (`re.search`, not `re.match` — doesn't need to anchor at the start). |
| `group` | Which capture group to extract; `0` (default) is the whole match. |
| `flags` | Plain `re` flags, e.g. `re.IGNORECASE`. |
| `on_no_match` | Fallback value (default `""`) when the pattern doesn't match at all. |
| `name` | The key this annotator's output lands under in `Prediction.context`, and what `extract_with=` refers to. |
| `cast` | Convert the extracted string via any `str -> Any` callable (`int`, `float`, a custom function). `None` (default) keeps it a string. |
| `strict` | If `cast` fails: `False` (default) degrades to the original string plus `"cast_failed": True` in the context dict; `True` raises instead. |

```python
answer = RegexAnnotator(r"FINAL ANSWER:\s*(-?\d+)", group=1, name="answer", cast=int)
```

## `ThinkingStripAnnotator` — for models that embed reasoning inline

Some hosted reasoning models (seen live against Groq's `qwen/qwen3.6-27b`)
put their entire draft reasoning process directly in the same text as the
final answer, wrapped in `<think>...</think>` tags — and that draft often
mentions the target pattern (e.g. `"ANSWER: 42"`) several times while
thinking out loud, before reaching the real final line. A plain
`RegexAnnotator`'s `re.search()` matches the *first* occurrence, which is
frequently still inside the reasoning, not the real answer — and if the
model changes its mind mid-thought, that's the *wrong* value.

```python
ThinkingStripAnnotator(
    pattern: str,                              # required, same contract as RegexAnnotator
    group: int | str = 0,
    flags: int = 0,
    on_no_match: str = "",
    name: str = "thinking_strip",
    strip_pattern: str = r"<think>.*?</think>",  # removed before matching
    trim_trailing_noise: bool = True,             # trims trailing `...`/`(...)` commentary after the value
)
```

It strips the `<think>` block(s) first, then takes the *last* match in what
remains (not the first), then — by default — trims trailing
backtick/parenthetical commentary some models append after the value
(seen live, e.g. `"18\` (or \`$18\`, but usually just the number is
fine.)"`).

```python
from auditkit.annotator import ThinkingStripAnnotator

answer = ThinkingStripAnnotator(r"ANSWER:\s*([^\n]+)", group=1, name="answer")
```

## `LLMAnnotator` — a second model call, for text with no fixed marker

Same model-resolution contract as `LLMJudge` — a spec string, resolved
lazily, or any object with `.generate()`:

```python
LLMAnnotator(
    model: Any = None,                # required
    prompt: str = "Extract the requested value ...",
    system_prompt: str | None = None,
    name: str = "llm_annotator",
    model_args: dict | None = None,   # connection-level only: api_key/device/api_base
    temperature: float | None = None, # this call's own generation settings --
    max_tokens: int | None = None,    # doesn't go through Adapter/RunConfig,
    top_p: float | None = None,       # so these are the only way to set them
    cast: Callable[[str], Any] | None = None,
    strict: bool = False,
    on_empty: str = "",               # fallback when the model returns nothing
)
```

`prompt` fills the same placeholder vocabulary `LLMJudge` uses
(`{input}`/`{output}`/`{expected}`/`{context}`, plus any `sample.metadata`
key) — `{output}` is the raw text being extracted from.

```python
from auditkit.annotator import LLMAnnotator

extractor = LLMAnnotator(
    model="groq:llama-3.1-8b-instant",
    model_args={"api_key": groq_api_key},
    prompt="Extract ONLY the final numeric answer. Respond with just the number.\n\nSolution: {output}",
    temperature=0.0,
    max_tokens=10,
    name="answer",
)
```

Use this over `RegexAnnotator` when there's no fixed marker to anchor a
pattern to — e.g. a RAG answer phrased as a full sentence
("The Denver Broncos represented the AFC at Super Bowl 50.") with no
`FINAL ANSWER:`-style marker a regex could match against. The tradeoff:
extraction quality now depends on the extraction prompt and the model
doing the extracting, same as any other LLM call — it can pick the wrong
entity on ambiguous multi-entity text, the same way any LLM call can be
wrong.

## Multiple annotators in one run — only one drives scoring

```python
result = ak.evaluate(
    samples, model=model, scorers=["exact_match"],
    annotators=[regex_answer, llm_answer],   # both run
    extract_with="llm_answer",               # only this one feeds the metrics
)
```

Every configured annotator runs and is inspectable via
`Prediction.context[annotator.name]`, regardless of which one (if any) is
named in `extract_with=`. This is useful for comparing extraction
strategies side by side without paying for two separate evaluation runs.

## The one place `extract_with=` is structurally disabled: loglikelihood/MCQ

On the `mcq_loglikelihood` path, `output` is already
`sample.choices[argmax_idx]` — the exact, correct choice text picked by
comparing real log-probabilities, not free text needing cleanup. `Acc`/
`AccNorm` interpret a bare digit string as a **choice index**, not "a
number that happened to appear in the text" — extracting a number embedded
in the correct choice's own wording and scoring *that* as an index is a
different, unrelated operation, and can silently turn a correct pick into
a wrong (or out-of-range) score. `Runner.run()` still runs every configured
annotator on this path (so `Prediction.context` is populated, for
inspection) — only the `extract_with=` scoring hookup is disabled, and it
isn't a configuration flag you can turn back on; it's removed from that one
call site in the source.

## `cast=`/`strict=` — type conversion, not just string cleanup

Both annotator types support the same contract:

```python
answer = RegexAnnotator(r"\d+", cast=int)
ctx = answer.annotate(sample, results)
# {"extracted": 42, "matched": True, "raw": "..."}   -- an int, not "42"

strict_answer = RegexAnnotator(r"\d+", cast=int, strict=True)
# a non-numeric match raises ValueError instead of degrading silently
```

A failing cast never crashes the whole run by default (`strict=False`) —
one malformed sample degrades to the original string plus
`"cast_failed": True` in its context dict, visible per-sample without
stopping everything else.

## Fingerprinting — annotator config is part of the run's identity

Changing a `RegexAnnotator`'s pattern, an `LLMAnnotator`'s prompt, or
either one's `cast=`, changes `RunSpec.fingerprint()` — a cached result
scored under a different extraction config is never silently reused.
Every built-in annotator's `identity()` reflects its real config, and
`LLMAnnotator.identity()` specifically delegates to the annotator model's
own `identity()` (not just its `.name`), so pointing it at a different
model checkpoint also correctly invalidates the cache.

## Writing your own annotator

```python
from auditkit.annotator import Annotator

class LengthFlagAnnotator(Annotator):
    def __init__(self, max_words: int = 50):
        self.max_words = max_words

    def identity(self) -> dict:
        # Every constructor argument that affects annotate()'s output must
        # be listed here -- otherwise two differently-configured instances
        # silently share one fingerprint. See "A warning to expect" below.
        return {"name": self.name, "max_words": self.max_words}

    def annotate(self, sample, results):
        text = results[0].text if results else ""
        n = len(text.split())
        return {"extracted": text, "matched": n <= self.max_words, "too_long": n > self.max_words}
```

Register it (only needed if you want to select it by string name):

```python
from auditkit.registry import ANNOTATORS

ANNOTATORS.register("length_flag")(LengthFlagAnnotator)
```

Otherwise just pass an instance directly to `annotators=` — no registration
required.

**A warning to expect**: if your subclass takes real `__init__` parameters
but you forget the `identity()` override above, `Annotator.__init_subclass__`
emits a `UserWarning` naming the unguarded parameters, the moment the class
is *defined* (at import time — before you ever instantiate it or run
anything):

```
UserWarning: LengthFlagAnnotator defines __init__ parameters (max_words) but
does not override Annotator.identity() (and doesn't derive self.name from
them either). Differently-configured instances will silently produce the
same RunSpec.fingerprint() and can reuse a stale cached RunResult scored
under a different config. Add an identity() override that includes every
constructor argument affecting annotate()'s output.
```

This is a warning, not an error — nothing stops you from ignoring it — but
it exists because this exact silent-collision bug was found live, in
shipped code, multiple times (see the [Changelog](community/changelog.md)).
The same guard exists for custom `Adapter` and `Metric` subclasses.

## Related

- [Scorer Reference](scorers_reference.md) — every metric's `context` keys,
  including which ones a `extract_with=`-selected annotator can and can't
  drive.
- [LLM-as-Judge Rendering](llm_judge_rendering.md) — `LLMAnnotator` reuses
  the exact same prompt-placeholder mechanism `LLMJudge` uses.
- [Known Issues](known_issues.md) — the 3 pairwise/preference metrics
  (`win_rate`, `elo_score`, `preference_accuracy`) read fixed `context`
  keys an annotator can't populate — `extract_with=` has no effect on
  them regardless of configuration.
