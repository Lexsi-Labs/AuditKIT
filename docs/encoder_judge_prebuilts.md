# The two `EncoderJudge` prebuilts

`EncoderJudge` (`src/auditkit/metrics/encoder_judge.py`) ships with two
zero-config subclasses — `FactualityEncoderJudge` and
`SentimentEncoderJudge`. This doc explains what each one actually does,
when to reach for which, and why the library used to ship six of these
and now ships two. For the full mechanism (templating, label auto-
detection, aggregation modes, edge cases), see
[Encoder Judge](encoder_judge.md); this doc is specifically about the two
prebuilts.

---

## `FactualityEncoderJudge`

```python
FactualityEncoderJudge(
    model_name: str = "microsoft/deberta-base-mnli",
    text_template: str = "{output}",
    text_pair_template: Optional[str] = "{target}",
    label_map: Optional[dict] = None,   # auto-detected from real label names
    aggregation: Literal["probability", "argmax"] = "probability",
    device: Optional[str] = None,
    name: str = "factuality_encoder_judge",
)
```

**Task**: entailment / factual-consistency checking. Given a candidate
answer (`output`) and a reference (`target`), classifies whether the
candidate **entails** (is consistent with), **contradicts**, or is
**neutral** to the reference — "does this claim hold up against that
reference?"

**Mechanism**: `microsoft/deberta-base-mnli` (DeBERTa v1, fine-tuned on
MNLI). Real label names (`ENTAILMENT`/`NEUTRAL`/`CONTRADICTION`), so
`label_map` auto-detects correctly with zero configuration. Relative
position embeddings mean no hard input-length limit, unlike BERT/RoBERTa/
ELECTRA's 512-token absolute limit — this checkpoint tolerates long
combined input without crashing.

**Real numbers, verified live**:
```python
sample = ak.Sample(input="q", target="The cat sat on the mat.")
FactualityEncoderJudge().score(sample, "The cat sat on the mat.")                       # 0.9974 (entailed)
FactualityEncoderJudge().score(sample, "A completely unrelated sentence about rockets.") # 0.0367 (contradicted)
```

**The one thing to get right: give it real sentences, not bare tokens.**
`FactualityEncoderJudge` is an NLI classifier — it needs actual language
to reason entailment over. A bare short answer like `"72"` vs `"72"`
carries no linguistic content, so it discriminates weakly on numeric or
single-word answers. Two ways to use it well:
- **Best**: use it on datasets where answers are naturally full sentences
  (factual QA, claim verification, paraphrase checking) — no workaround
  needed, just the default templates.
- **Workaround, if your answers are short**: render both sides into full
  sentences including context, e.g.
  `text_template="Question: {input}\nAnswer: {output}"` /
  `text_pair_template="Question: {input}\nAnswer: {target}"` — verified
  live against real GSM8K-shaped numeric answers: ~0.99 on real matches,
  ~0.001 on real mismatches (see `01_full_evaluation_pipeline.ipynb`'s
  Pass 3 and `15_groq_llmannotator_encoderjudge.ipynb`).

**Where it's demonstrated with a real model end to end**:
`15_groq_llmannotator_encoderjudge.ipynb` (real `groq:llama-3.3-70b-versatile`
generation + `LLMAnnotator` cleanup + `FactualityEncoderJudge` on the
cleaned sentence) and `01_full_evaluation_pipeline.ipynb`'s Pass 3 (same
idea, GSM8K, the templating workaround for numeric answers).

---

## `SentimentEncoderJudge`

```python
SentimentEncoderJudge(
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    text_template: str = "{output}",
    text_pair_template: Optional[str] = None,   # single-sequence -- no reference needed
    label_map: Optional[dict] = None,           # {"POSITIVE": 1.0, "NEGATIVE": 0.0}
    aggregation: Literal["probability", "argmax"] = "probability",
    device: Optional[str] = None,
    name: str = "sentiment_encoder_judge",
)
```

**Task**: 2-way sentiment classification of **one** piece of text alone —
no reference/`target` needed at all. This is the one task that differs
from `FactualityEncoderJudge`: it doesn't compare two texts, it reads
the tone of one.

**Mechanism**: `distilbert-base-uncased-finetuned-sst-2-english`
(DistilBERT, fine-tuned on SST-2 movie-review sentiment). Real label
names (`POSITIVE`/`NEGATIVE`), auto-detected. `text_pair_template=None`
is what makes it single-sequence — `required_fields = frozenset()`, so
`target` is never required on the `Sample`.

**Real numbers, verified live**:
```python
sample = ak.Sample(input="q", target="")  # target unused
SentimentEncoderJudge().score(sample, "I love this!")  # 0.9999 (positive)
SentimentEncoderJudge().score(sample, "I hate this!")  # 0.0004 (negative)
```

**A real caveat, confirmed live, not asserted**: this checkpoint is
**strictly binary** — there's no neutral class. A genuinely neutral
sentence (`"The package arrived on Tuesday."`) doesn't land in the
middle; it gets pushed toward whichever of the two labels edges out
(scored 0.983 — reads as confidently "positive" even though the content
isn't positive at all). This is a real limitation of the checkpoint's own
training task, not a bug — don't use this class where you actually need a
neutral option.

**Where it's demonstrated live**: `18_encoder_judge_prebuilts.ipynb`
(every cell executed for real) and `15_groq_llmannotator_encoderjudge.ipynb`.

---

## Why only two (not six)

The library originally shipped 6 prebuilts. The reduction happened in two
rounds, each removing classes that didn't earn a separate name:

1. **`DebertaNLIJudge`/`DebertaV3NLIJudge`/`RobertaNLIJudge` → collapsed
   into `FactualityEncoderJudge`.** All three used checkpoints with real,
   auto-detectable labels — having three class names added nothing over
   passing `model_name=` to one class directly. DeBERTa v1 was kept as
   the default: no hard input-length limit, and empirically more
   confident/discriminative than DeBERTa-v3 on ambiguous pairs.
2. **`BertNLIJudge`/`ElectraNLIJudge` → removed entirely.** Both did the
   *exact same task* as `FactualityEncoderJudge` (entailment/factual-
   consistency) — their only distinguishing feature was an unrecoverable,
   empirically-verified generic label order on their specific checkpoints
   (`textattack/bert-base-uncased-MNLI`: `LABEL_0=contradiction,
   LABEL_1=entailment, LABEL_2=neutral`; `howey/electra-base-mnli`: the
   **opposite** order). Real, useful knowledge — but not a distinct task,
   so it didn't justify two more classes with no behavior difference
   beyond which checkpoint. That knowledge is preserved as a worked
   example in [Encoder Judge](encoder_judge.md) instead:
   ```python
   bert_judge = ak.EncoderJudge(
       model_name="textattack/bert-base-uncased-MNLI",
       label_map={"LABEL_0": 0.0, "LABEL_1": 1.0, "LABEL_2": 0.5},
   )
   electra_judge = ak.EncoderJudge(
       model_name="howey/electra-base-mnli",
       label_map={0: 1.0, 1: 0.5, 2: 0.0},
   )
   ```

`SentimentEncoderJudge` (renamed from `DistilBertSentimentJudge`) is the
only one that survived both rounds untouched — it's the one prebuilt with
a task distinct from `FactualityEncoderJudge`.

## Quick decision guide

| You need to... | Use |
|---|---|
| Check if a generated answer is consistent with / contradicts a reference (QA correctness, claim verification, paraphrase detection) | `FactualityEncoderJudge` |
| Check the tone (positive/negative) of a single piece of text, no reference | `SentimentEncoderJudge` |
| Judge a checkpoint with only generic `LABEL_0/1/2` labels | Bare `EncoderJudge(model_name=..., label_map=...)` — see the worked example above |
| Judge something neither entailment nor 2-way sentiment can express (toxicity, multi-label classification, a task-specific fine-tune) | Bare `EncoderJudge(model_name=..., label_map=...)` with your own checkpoint |
