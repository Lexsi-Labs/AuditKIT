# EncoderJudge — LLM-as-judge via a real encoder classifier

Every judge in [`metrics/judge.py`](api.md) (`LLMJudge`, `GEval`, `Factuality`,
...) needs a **generative** model: it asks a second LLM to produce free
text, then parses a verdict out of that text with a regex/marker.
`EncoderJudge` (`src/auditkit/metrics/encoder_judge.py`) works differently:
it uses an **encoder classifier** (BERT, RoBERTa,
DeBERTa, ELECTRA, ALBERT, or any other architecture Hugging Face's
`AutoModelForSequenceClassification` covers) directly as the judge. It
never generates anything: it classifies a (candidate, reference) pair in
one forward pass and reads the verdict straight off the model's own output
distribution — no free-text parsing, no CHOICE:/SCORE: marker to get
wrong.

This page has been verified against **five real checkpoints across four
distinct encoder architectures** (BERT, RoBERTa, DeBERTa, ELECTRA), plus a
real independent multi-label classifier — every code block below actually
ran, with real output pasted in, not invented.

## Why use this instead of `LLMJudge`?

- **No API key, no generation cost.** A small encoder classifier
  (100M–400M params) runs on CPU in a fraction of a second per sample —
  no hosted API call, no local generative model to load.
- **Deterministic.** `is_deterministic = True` — the same input always
  produces the same output distribution (no sampling temperature to
  control, no CHOICE-parsing ambiguity).
- **Different failure modes than a prompted judge.** A
  generative judge can be swayed by verbosity, tone, or a submission that
  contains prompt-injection-style text (`LLMJudge`'s shared system prompt
  explicitly guards against this). An encoder classifier has no
  instruction-following behavior to manipulate at all — it was trained to
  do exactly one thing (classify a text pair) and can't be talked out of
  it.
- **Trade-off:** an encoder classifier is not as flexible as a prompted
  judge — it can't follow a custom rubric, can't reason step by step, and
  is exactly as good (or as biased) as whatever fixed task it was
  fine-tuned for (typically NLI/entailment). Use `EncoderJudge` when NLI-
  style "does the output entail the reference" is actually the judgment
  you want; use `LLMJudge`/`GEval` for anything requiring nuanced,
  rubric-driven reasoning.

## Quickstart

```python
import auditkit as ak
from auditkit.metrics.encoder_judge import EncoderJudge

judge = EncoderJudge(model_name="microsoft/deberta-base-mnli")

sample = ak.Sample(input="What did the cat do?", target="The cat sat on the mat.")
score = judge.score(sample, "The cat sat on the mat.")
print(score.value, score.reason)
# 0.9973967224359512 predicted='ENTAILMENT' probability=0.995 aggregation=probability

score = judge.score(sample, "A completely unrelated sentence about rockets.")
print(score.value, score.reason)
# 0.0005356273031793535 predicted='CONTRADICTION' probability=0.999 aggregation=probability
```

Works with `evaluate()`/`compare_models()` exactly like every other metric:

```python
result = ak.evaluate(
    [sample], model=lambda p: ["The cat sat on the mat."],
    scorers=[EncoderJudge(model_name="microsoft/deberta-base-mnli")],
)
print(result.headline)  # {'encoder_judge': 0.9973967224359512}
```

## "Templating" for an encoder — what it actually means

A chat model's prompt template decides the literal text sent to the model.
An encoder classifier built for pair-input tasks (NLI, similarity,
cross-encoder ranking) doesn't take a rendered prompt string at all — it
takes exactly **two text spans**, which its tokenizer encodes together via
real pair-sequence support (`tokenizer(text, text_pair)` → `[CLS] text
[SEP] text_pair [SEP]` for BERT-style models, with an architecture-specific
equivalent for RoBERTa/DeBERTa/ELECTRA/etc.). There is no template string
to hand-write.

So for `EncoderJudge`, "template" means **which rendered fields fill those
two slots** — `text_template=`/`text_pair_template=` are plain
`{input}`/`{output}`/`{target}` (alias `{expected}`)/`{context}` format
strings (the same placeholder set `LLMJudge` uses), and the actual
pair-encoding is handled by `transformers`' own pipeline:

```python
judge = EncoderJudge(
    model_name="microsoft/deberta-base-mnli",
    text_template="{output}",         # first span: the candidate output (default)
    text_pair_template="{target}",    # second span: the reference (default)
)
```

**Do not manually concatenate a literal `"[SEP]"` into one string instead**
— that was tried in an earlier version of this idea
(`FactualConsistency` in `metrics/hallucination.py`) and doesn't reliably
match what a given tokenizer's real special-token handling would produce.
Always pass two real spans and let the tokenizer's pair-encoding do the
work.

**Single-sequence classifiers** (no reference to compare against — e.g. a
standalone sentiment/toxicity classifier) set `text_pair_template=None`:

```python
judge = EncoderJudge(
    model_name="distilbert-base-uncased-finetuned-sst-2-english",
    text_template="{output}", text_pair_template=None,
    label_map={"POSITIVE": 1.0, "NEGATIVE": 0.0},
)
sample = ak.Sample(input="q", target="")
print(judge.score(sample, "I love this product, it is amazing!").value)
# 0.9998867511749268
print(judge.score(sample, "This is the worst thing I have ever bought.").value)
# 0.00021629415277857333
```

## Label mapping — auto-detected, or explicit

An encoder classifier's output is a fixed label set the checkpoint was
fine-tuned with (`model.config.id2label`) — `EncoderJudge` needs to know
which labels count as a "good" verdict.

**Auto-detection** (no `label_map=` passed) works when the checkpoint has
real, human-readable label names — recognizes common NLI/sentiment-style
names: `entailment`/`positive`/`yes`/`true` → `1.0`, `contradiction`/
`negative`/`no`/`false` → `0.0`, `neutral` → `0.5`.

```python
EncoderJudge(model_name="microsoft/deberta-base-mnli")  # id2label: {0: 'CONTRADICTION', 1: 'NEUTRAL', 2: 'ENTAILMENT'}
EncoderJudge(model_name="cross-encoder/nli-distilroberta-base")  # id2label: {0: 'contradiction', 1: 'entailment', 2: 'neutral'}
```
Both auto-detect correctly and were verified live — real, discriminating
scores (>0.9 for identical text, <0.01 for unrelated text) on both a
DeBERTa and a RoBERTa checkpoint.

**Many real checkpoints only expose generic labels** (`LABEL_0`,
`LABEL_1`, `LABEL_2`) with no way to know what they mean from the config
alone. `EncoderJudge` **raises rather than guessing**:

```python
EncoderJudge(model_name="textattack/bert-base-uncased-MNLI").score(sample, "...")
# ValueError: EncoderJudge: couldn't auto-detect a label_map for this
# checkpoint's labels ['LABEL_0', 'LABEL_1', 'LABEL_2'] -- pass label_map=
# explicitly, ...
```

Pass `label_map=` explicitly — by label name or by numeric index (both
work identically):

```python
EncoderJudge(model_name="textattack/bert-base-uncased-MNLI",
             label_map={"LABEL_0": 0.0, "LABEL_1": 1.0, "LABEL_2": 0.5})
# identical to:
EncoderJudge(model_name="textattack/bert-base-uncased-MNLI",
             label_map={0: 0.0, 1: 1.0, 2: 0.5})
```

### Real gotcha, confirmed live: generic label *order* is not standardized

Don't assume one checkpoint's `LABEL_0`/`LABEL_1`/`LABEL_2` convention
carries over to another. Verified live: `howey/electra-base-mnli` (a real,
different checkpoint) uses `LABEL_0=entailment, LABEL_1=neutral,
LABEL_2=contradiction` — the **opposite** order from what
`textattack/bert-base-uncased-MNLI` and several other checkpoints use.
Applying the wrong assumed order silently scores everything backwards
(confirmed: assuming the common order for `howey/electra-base-mnli` scored
*identical* text near `0.0` and *unrelated* text near `0.5`, exactly
inverted from correct).

**Determine the real order empirically** before trusting a `label_map` on
an unfamiliar checkpoint — a couple of obviously-true/obviously-false test
pairs, checking which label index actually lights up:

```python
from transformers import AutoConfig
print(AutoConfig.from_pretrained("howey/electra-base-mnli").id2label)
# {0: 'LABEL_0', 1: 'LABEL_1', 2: 'LABEL_2'} -- no help, still generic

# Probe with a known example instead:
probe = EncoderJudge(model_name="howey/electra-base-mnli", label_map={0: 1.0, 1: 0.5, 2: 0.0})
s = ak.Sample(input="q", target="The weather today is sunny and warm.")
print(probe.score(s, "The weather today is sunny and warm.").reason)
# predicted='LABEL_0' probability=0.995 -- LABEL_0 lit up for identical text -> LABEL_0 is really entailment
```
Once confirmed, `label_map={0: 1.0, 1: 0.5, 2: 0.0}` is the correct,
verified mapping for that specific checkpoint.

## Comparing texts of different lengths

Length mismatch between the candidate output and the reference (a one-line
output vs. a paragraph-long reference, or vice versa) isn't a problem by
itself — self-attention lets every token attend to every other token
regardless of position, and the tokenizer's pair-encoding handles two
spans of any relative size. Verified live: a short output scored correctly
against a long reference and a long output against a short reference,
both landing at real, sensible confidence values.

**Real bug found and fixed while testing this:** checkpoints using
*absolute* position embeddings (BERT, RoBERTa, ELECTRA) have a hard
512-token limit on the **combined** pair — `EncoderJudge` used to crash
outright (`RuntimeError: The size of tensor a (1233) must match the size
of tensor b (512) ...`) if both texts together exceeded it, since
truncation wasn't enabled. Fixed: `score()` now always calls the pipeline
with `truncation=True`. DeBERTa (*relative* position embeddings, no hard
limit — its tokenizer even reports `model_max_length` as the classic
"unbounded" sentinel value seen elsewhere in this codebase's `bert_score`
fix) was never affected by the crash, which is exactly why it wasn't
caught by the smaller examples used everywhere else on this page — only
surfaced by deliberately testing very long inputs against a BERT
checkpoint specifically. `truncation=True` is confirmed harmless for
DeBERTa too (no `OverflowError`, unlike that earlier `bert_score` case).

Practical effect: past ~512 combined tokens on an absolute-position-
embedding checkpoint, the tail end of whichever span runs longest gets
silently truncated rather than crashing — same trade-off any transformer
classifier makes, not something `EncoderJudge` can avoid while still using
the checkpoint's own tokenizer correctly.

## Aggregation: `"probability"` vs `"argmax"`

- **`"probability"`** (default): the score is the softmax-weighted sum of
  every label's probability times its mapped value — a continuous score
  reflecting the classifier's actual confidence (an 85%-confident
  entailment call scores `0.85`, not a rounded `1.0`).
- **`"argmax"`**: the score is just the mapped value of the single
  highest-probability label — a cruder, binary-flavored pass/fail.

```python
judge = EncoderJudge(model_name="microsoft/deberta-base-mnli", aggregation="argmax")
print(judge.score(sample, "The cat sat on the mat.").value)  # 1.0
print(judge.score(sample, "A completely unrelated sentence about rockets.").value)  # 0.0
```

### Real gotcha, confirmed live: `"probability"` assumes mutually-exclusive labels

`"probability"`'s weighted-sum formula is correct for single-label
classification (NLI/sentiment/similarity — probabilities across all labels
sum to ~1). For an **independent multi-label** classifier (e.g.
`unitary/toxic-bert` — sigmoid outputs, several labels can be true at
once), mapping a label to `0.0` means "this label contributes nothing," —
**not** "the complement of this label." There is no way to express
"1 − P(toxic)" through the weighted-sum formula alone:

```python
judge = EncoderJudge(model_name="unitary/toxic-bert").score(sample, "...")
# ValueError: couldn't auto-detect a label_map for ['toxic', 'severe_toxic',
# 'obscene', 'threat', 'insult', 'identity_hate'] -- none of these are
# recognized NLI/sentiment-style names either, so it correctly refuses to guess.

judge = EncoderJudge(
    model_name="unitary/toxic-bert",
    text_template="{output}", text_pair_template=None,
    label_map={"toxic": 1.0},
)
s = ak.Sample(input="q", target="")
print(judge.score(s, "Thank you so much for your help today.").value)   # ~0.0005 -- clean
print(judge.score(s, "You are all idiots and I hate everyone here.").value)  # ~0.99 -- toxic
```
This measures "how toxic," not "how safe." If you specifically want a
safety/non-toxicity score, use this repo's dedicated `ToxicityScore`
metric instead — `EncoderJudge` is built for the NLI-as-judge case, and
genericity over encoder architectures doesn't extend to inventing
semantics a raw weighted sum can't actually express.

## Verified checkpoints (real, live-tested)

| Checkpoint | Architecture | Labels | Label map |
|---|---|---|---|
| `microsoft/deberta-base-mnli` | DeBERTa (v1) | real names | auto-detected |
| `cross-encoder/nli-distilroberta-base` | RoBERTa | real names | auto-detected |
| `cross-encoder/nli-MiniLM2-L6-H768` | RoBERTa (MiniLM) | real names | auto-detected |
| `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | DeBERTa **v3** (`DebertaV2ForSequenceClassification`, a different implementation from v1 above) | real names | auto-detected — weaker discrimination on some pairs than DeBERTa-v1/RoBERTa, a real model-quality difference, not a code issue |
| `textattack/bert-base-uncased-MNLI` | BERT | `LABEL_0/1/2` | explicit, required |
| `howey/electra-base-mnli` | ELECTRA | `LABEL_0/1/2` (reversed order vs. the above) | explicit, required |
| `distilbert-base-uncased-finetuned-sst-2-english` | DistilBERT | real names, single-sequence | explicit (2-way, not NLI) |
| `unitary/toxic-bert` | BERT | real names, **multi-label** | explicit, required |

Genericity comes from using `transformers`' own `AutoModelForSequenceClassification`/
`AutoTokenizer`/`pipeline()` throughout — no architecture-specific code in
`EncoderJudge` itself; any future encoder architecture HuggingFace's
`Auto*` classes support should work the same way, though only the
checkpoints above have actually been run.

**Not covered:** encoder-*decoder* architectures with a classification head
bolted on (e.g. `facebook/bart-large-mnli`, `BartForSequenceClassification`)
weren't tested — `EncoderJudge` targets pure encoder classifiers
specifically; a BART-style model may or may not work through the same
`AutoModelForSequenceClassification` path, unverified either way.

## Works with `Annotator`s exactly like any other metric

The `Annotator` pipeline (`RegexAnnotator`, `ThinkingStripAnnotator`, ...)
extracts a clean value from the raw output of the model *being evaluated*,
entirely before any `Metric` — `EncoderJudge` included — is ever invoked.
This makes it fully metric-agnostic: the same `annotators=`/`extract_with=`
wiring that works for `ExactMatch`/`LLMJudge`/anything else works
identically here, with no special-casing needed. Verified live across two
different encoder architectures and two different annotator types:

```python
from auditkit.annotator import RegexAnnotator

def rambling_model(prompts):
    return ["Let me think step by step... the answer must be: The cat sat on the mat."
            for _ in prompts]

sample = ak.Sample(input="What did the cat do?", target="The cat sat on the mat.")
answer = RegexAnnotator(r"answer must be:\s*(.+)", group=1, name="answer")

result = ak.evaluate(
    [sample], model=rambling_model,
    scorers=[EncoderJudge(model_name="microsoft/deberta-base-mnli")],
    annotators=answer, extract_with="answer",
)
print(result.headline)                          # {'encoder_judge': 0.9973967224359512}
print(result.predictions[0].raw_output)         # the full rambling text, unchanged
print(result.predictions[0].parsed_answer)      # 'The cat sat on the mat.' -- what was actually classified
```
Without the annotator, the same call scores `0.9848...` — the rambling
preamble genuinely dilutes the classifier's confidence, confirming the
extraction has a real effect, not just correct wiring with no impact.

## Prebuilt judges — zero-config, ready to use

Two `EncoderJudge` subclasses, each with the correct, checkpoint-specific
config (`model_name`/`label_map`/template shape) baked in as defaults —
the same pattern `Factuality`/`ClosedQA`/`Relevance` already use for
`LLMJudge` (a pre-filled configuration, not a different mechanism). Every
default is still a normal constructor argument, fully overridable. See
[Encoder Judge Prebuilts](encoder_judge_prebuilts.md) for a dedicated,
example-driven walkthrough of both.

> **Candor — this used to be six.** The reduction happened in two rounds.
> First, `DebertaNLIJudge`/`DebertaV3NLIJudge`/`RobertaNLIJudge` were
> collapsed into one, `FactualityEncoderJudge` (keeping DeBERTa v1 — the
> strongest of the three: no hard input-length limit, and empirically more
> confident/discriminative than DeBERTa-v3 on ambiguous pairs) — all three
> used checkpoints with real, auto-detectable labels, so having three
> separate classes added a name each but no real behavior difference over
> passing `model_name=` to one class directly. Then `BertNLIJudge`/
> `ElectraNLIJudge` were removed entirely: both did the exact same *task*
> as `FactualityEncoderJudge` (entailment/factual-consistency) — their only
> distinguishing feature was an unrecoverable, empirically-verified generic
> label order on their specific checkpoints, which is real knowledge but
> not a distinct enough task to justify two more classes. That knowledge
> is preserved below as a worked example instead. `SentimentEncoderJudge`
> (renamed from `DistilBertSentimentJudge`) is the only prebuilt left with
> a genuinely different task from `FactualityEncoderJudge`.

```python
import auditkit as ak

sample = ak.Sample(input="q", target="The cat sat on the mat.")

ak.FactualityEncoderJudge().score(sample, "The cat sat on the mat.")  # 0.9974

ak.SentimentEncoderJudge().score(ak.Sample(input="q", target=""), "I love this!")  # 0.9999
```

| Class | Registry name | Checkpoint | Architecture | Label config |
|---|---|---|---|---|
| `FactualityEncoderJudge` | `factuality_encoder_judge` | `microsoft/deberta-base-mnli` | DeBERTa v1 | auto-detected (real names) |
| `SentimentEncoderJudge` | `sentiment_encoder_judge` | `distilbert-base-uncased-finetuned-sst-2-english` | DistilBERT | baked-in explicit map (`POSITIVE=1.0, NEGATIVE=0.0`); single-sequence (`text_pair_template=None`, no `target` required) — the one prebuilt judge that isn't NLI |

Both accept the exact same constructor arguments as `EncoderJudge` itself
(`text_template=`, `text_pair_template=`, `label_map=`, `aggregation=`,
`device=`, `name=`) — the class just changes which values are the
*defaults*:

```python
# Override aggregation
ak.FactualityEncoderJudge(aggregation="argmax")

# Swap the underlying checkpoint but keep the class's other defaults --
# only sensible if the replacement checkpoint's labels actually match
# what the class assumes; otherwise pass label_map= too.
ak.FactualityEncoderJudge(model_name="some-other-deberta-nli-checkpoint")
```

### Worked example: a checkpoint with only generic labels

Not every entailment checkpoint has real label names — some (e.g.
`textattack/bert-base-uncased-MNLI`) only expose generic `LABEL_0/1/2`,
which bare `EncoderJudge` correctly refuses to auto-detect (see the
gotcha above) rather than guess. This is exactly the situation
`BertNLIJudge`/`ElectraNLIJudge` used to paper over with a whole class
each; passing `label_map=` explicitly does the same thing with no extra
class needed — you just have to determine the real order yourself first
(by testing a few unambiguous pairs and checking `metadata["predicted_label"]`).

Two real, already-verified examples, confirmed to have the **opposite**
label order from each other despite both exposing the same generic
`LABEL_0/1/2` names — proof a generic checkpoint's order can never be
assumed from another checkpoint's convention:

```python
# textattack/bert-base-uncased-MNLI: LABEL_0=contradiction, LABEL_1=entailment, LABEL_2=neutral
bert_judge = ak.EncoderJudge(
    model_name="textattack/bert-base-uncased-MNLI",
    label_map={"LABEL_0": 0.0, "LABEL_1": 1.0, "LABEL_2": 0.5},
)

# howey/electra-base-mnli: LABEL_0=entailment, LABEL_1=neutral, LABEL_2=contradiction -- REVERSED
electra_judge = ak.EncoderJudge(
    model_name="howey/electra-base-mnli",
    label_map={0: 1.0, 1: 0.5, 2: 0.0},
)
```

## Full parameter reference

```python
EncoderJudge(
    model_name: str = "microsoft/deberta-base-mnli",
    *,
    text_template: str = "{output}",
    text_pair_template: str | None = "{target}",
    label_map: dict[str | int, float] | None = None,
    aggregation: Literal["probability", "argmax"] = "probability",
    device: str | None = None,   # None auto-detects CUDA/MPS/CPU
    name: str = "encoder_judge",
)
```

Every score carries `reason` (predicted label + probability) and
`metadata["probabilities"]` (the full label→probability distribution) —
inspect `result.predictions[i].scores` (or the `Score` returned directly
from `.score()`) to see exactly what the classifier saw, the same
transparency `LLMJudge` gives via its raw judge-model reply.

Real tests: `tests/test_encoder_judge.py` (20 tests, all against real
checkpoints, no mocks).
