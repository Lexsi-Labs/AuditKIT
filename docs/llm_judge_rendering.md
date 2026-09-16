# How LLM-as-Judge prompt rendering actually works

This is a mechanics doc: given a scorer — especially `LLMJudge` — how does it
know what to run and what its inputs are? Short answer: there's no
introspection or "figuring out" involved anywhere in this path. Every scorer
implements one fixed method signature, and `LLMJudge`'s prompt placeholders
are resolved by a plain, fixed-vocabulary string substitution. This page
traces the real code path end to end
(`src/auditkit/metric.py`, `src/auditkit/metrics/judge.py`) so the behavior
is verifiable, not just asserted.

## 1. Every scorer has the same fixed call signature — nothing is inspected

`Metric` (the base class every scorer, including `LLMJudge`, implements) has
one abstract method:

```python
class Metric(ABC):
    @abstractmethod
    def score(self, sample: Sample, output: str, context: Any = None) -> Union[Score, list[Score]]: ...
```

`Runner.score_one()` always calls `metric.score(sample, output, context)`
with exactly those three positional arguments: the `Sample`, the model's raw
output string, and an optional context object (retrieved docs, few-shot
info, ...). This is identical for a deterministic metric like `ExactMatch`
and for `LLMJudge` — the runner never inspects a scorer's signature to figure
out what it wants; every scorer is written to accept the same three things,
whether or not it uses all of them.

`JudgeMetric.score()` (the shared base for LLM-judge scorers) just forwards
to `self.judge(sample, output, context)`.

## 2. `{input}` / `{output}` / `{expected}` are a fixed, small vocabulary — resolved by plain string replacement

When you write a custom `LLMJudge` prompt like:

```python
judge = LLMJudge(
    judge_model="groq:llama-3.1-8b-instant",
    prompt="Question: {input}\nExpected: {expected}\nGiven answer: {output}",
    choices={"good": 1.0, "bad": 0.0},
)
```

the placeholders are filled in by `LLMJudge._render()`:

```python
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
```

The mapping is fixed and small:

| Placeholder | Source |
|---|---|
| `{input}` | `sample.input_text` |
| `{output}` | the `output` argument — the model's real generation |
| `{expected}` / `{target}` | `sample.target` (both names work, same value) |
| `{context}` | `sample.retrieval_context`, joined with newlines |
| anything else, e.g. `{persona}` | looked up in `sample.metadata`, if present |

That last row is what makes custom prompts genuinely adjustable per-sample,
not just at judge-construction time: **any key in `sample.metadata` becomes
a placeholder automatically**, with no extra wiring. For example:

```python
Sample(
    input="Why is the sky blue?",
    target="Rayleigh scattering of sunlight by the atmosphere",
    metadata={"persona": "a curious 5-year-old", "criteria": "simple, no jargon, friendly tone"},
)

judge = LLMJudge(
    judge_model=MODEL,
    prompt=(
        "Question: {input}\n"
        "Key facts expected: {expected}\n"
        "Judging criteria: {criteria}\n"
        "Answer given: {output}\n\n"
        "Does the answer satisfy the judging criteria AND cover the key facts?"
    ),
    choices={"meets_criteria": 1.0, "partially_meets": 0.5, "does_not_meet": 0.0},
    use_cot=True,
)
```

`{criteria}` here isn't a built-in field — it resolves purely because
`sample.metadata["criteria"]` exists. Change or add metadata keys per sample
and the same judge prompt adapts per sample, without touching the judge's
code.

**Why `str.replace()` and not `str.format()`:** `.format()` would raise
`KeyError`/`IndexError` on any stray literal `{`/`}` in your prompt text that
isn't one of the known fields (a JSON example in the prompt, a code snippet,
etc.). `.replace()` only ever substitutes the exact known keys and leaves
everything else untouched — so a prompt containing unrelated braces doesn't
break the pipeline.

## 3. What's auto-appended vs. what you write

`_assemble()` wraps your rendered prompt with two things you don't write
yourself:

```python
def _assemble(self, user_prompt: str) -> str:
    parts = []
    if self.system_prompt:
        parts.append(self.system_prompt)
    parts.append(user_prompt)
    parts.append(self._instructions())
    return "\n\n".join(parts)
```

- `self.system_prompt` — only if you passed one, prepended as-is.
- `self._instructions()` — auto-generated from your `choices=` or `scale=`,
  e.g. for `choices={"good": 1.0, "bad": 0.0}` with `use_cot=True`:

  > First, briefly explain your reasoning. Then, on the final line, output
  > exactly `CHOICE: <one of these labels exactly: good, bad>`.

  You control the *labels* (via `choices`) and whether reasoning is asked for
  (via `use_cot`), but not the literal wording of this instruction line — it
  comes from `LLMJudge`, not from your `prompt=`.

## 4. It "just runs" — one real generation call, then a regex parse

```python
def judge(self, sample: Sample, output: str, context: Any = None) -> Score:
    model = self._model()
    full = self._assemble(self._render(sample, output))
    results = model.generate([Request(prompt=full, params=dict(self._gen_params))])
    text = ""
    if results and getattr(results[0], "completions", None):
        text = results[0].completions[0].text or ""
    return self._parse(text)
```

`params=dict(self._gen_params)` is how `temperature=`/`max_tokens=`/`top_p=`
(passed to `LLMJudge(...)`'s constructor) actually reach the judge model's
own call — the judge call never goes through an `Adapter`/`RunConfig`, so
this is the only path those settings have.

No structured output / function-calling / JSON mode is used — the fully
assembled prompt (system prompt + your rendered template + the auto-appended
`CHOICE:`/`SCORE:` instruction) is sent as one plain text generation request,
and the reply is parsed with a regex looking for the `CHOICE:`/`SCORE:`
marker line, falling back to scanning only the **last line** of the reply if
no marker is found (the fallback deliberately doesn't scan the whole reply —
see [`_last_line` in `judge.py`](https://github.com/Lexsi-Labs/AuditKIT/blob/main/src/auditkit/metrics/judge.py), fixed to
avoid matching an incidental word like "no" appearing earlier in the judge's
own chain-of-thought reasoning as if it were the real verdict). If neither
matches, the score falls through to an explicit `Unknown` (`metadata["unknown"]
= True`), never a silent guess.

## 5. `GEval`, `Factuality`, `ClosedQA`, `Relevance` use the exact same mechanism

These are all `LLMJudge` subclasses — they don't have their own rendering
logic. They just pre-fill `prompt=`/`choices=`/`scale=`/`use_cot=` in their
`__init__` (e.g. `GEval` builds its `prompt` from a rubric list, `Factuality`
ships a fixed A–E prompt). Everything in this doc — the fixed field
vocabulary, the `sample.metadata` extension, `str.replace()` substitution,
one real generation call, marker-based parsing — applies identically to all
of them.

## Related

- [`metrics.md`](metrics.md) — full metric catalog.
- **Prompt-aware fingerprinting**: `LLMJudge.identity()` (in
  `src/auditkit/metrics/judge.py`) folds `prompt`/`choices`/`scale`/
  `system_prompt` into the run's fingerprint (`RunSpec.fingerprint()`), so
  changing your judge prompt correctly invalidates any cached result scored
  under the old prompt rather than silently reusing it.
