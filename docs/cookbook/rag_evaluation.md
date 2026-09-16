# Cookbook: Evaluating a RAG Pipeline

End-to-end RAG evaluation with the real API — `RAGAdapter`, real `Sample`
fields (`target`, `retrieval_context`), and the real RAG metrics. Every
snippet on this page has been run and its output verified.

## Step 1: Samples carry `retrieval_context` directly

```python
import auditkit as ak

dataset = [
    ak.Sample(
        input="How do I reset my password?",
        target="Settings > Security > Reset Password",
        retrieval_context=[
            "Users can reset their password from the Security section in Settings.",
            "A confirmation email is sent to the registered email address.",
        ],
    ),
    ak.Sample(
        input="What is your return policy?",
        target="30-day returns for unused items",
        retrieval_context=[
            "Return policy: 30 days from delivery, unused items only.",
            "Items must be in original packaging with all accessories.",
        ],
    ),
]
```

`retrieval_context` is a real `Sample` field (not something you invent via
`metadata`) — it's what `RAGAdapter` reads, and what the RAG metrics below
check the model's output against.

## Step 2: `RAGAdapter` injects the context into the prompt for you

```python
from auditkit.model import CallableModel

def fake_rag_model(prompts):
    # Stand-in for your real RAG generation call. In production this is
    # any model spec ("openai:gpt-4o-mini", "groq:llama-3.1-8b-instant", a
    # local "hf:..." model, or your own function/API wrapped exactly like
    # this) -- RAGAdapter doesn't care what generates the answer, only that
    # it receives a prompt with the retrieved context already appended.
    out = []
    for p in prompts:
        if "password" in p.lower():
            out.append("You can reset your password by going to Settings > "
                        "Security > Reset Password. A confirmation email will be sent to you.")
        else:
            out.append("Our policy allows 30-day returns for unused items in original packaging.")
    return out

model = CallableModel(fake_rag_model)
adapter = ak.RAGAdapter(context_separator="\n\nContext:\n")

print(adapter.adapt(dataset[0], ak.RunConfig())[0].prompt)
```
```
How do I reset my password?

Context:
Users can reset their password from the Security section in Settings.
A confirmation email is sent to the registered email address.
```

`RAGAdapter` raises `ValueError` immediately if a sample has no
`retrieval_context` at all — it's meant specifically for RAG-shaped data,
not a general-purpose adapter with an optional context feature.

## Step 3: Score with the real RAG metrics

```python
result = ak.evaluate(
    dataset, model=model, adapter=adapter,
    scorers=[ak.LexicalGroundedness(), ak.ContextOverlap(), ak.ContextCoverage(), ak.AnswerOverlap()],
)
print(result.headline)
```
```
{'lexical_groundedness': 0.477, 'context_overlap': 1.0, 'context_coverage': 0.3, 'answer_overlap': 1.0}
```

What each one actually checks (see the RAG section of the
[Scorer Reference](../scorers_reference.md) for the exact matching rule):

| Metric | Checks |
|---|---|
| `LexicalGroundedness` | Is the model's output grounded in the retrieved context — word-boundary matching, not naive substring containment. Named for the lexical-overlap heuristic it computes, distinct from RAGAS's LLM-based `Faithfulness`. |
| `ContextOverlap` | Is the retrieved context actually relevant to what the model said. Distinct from RAGAS's LLM-based `ContextPrecision`. |
| `ContextCoverage` | Does the retrieved context contain what the *target* answer needs — **deliberately independent of the model's output**, since it measures retrieval quality, not generation quality (same scoping intent as RAGAS's `context_recall`, but a raw vocabulary-overlap heuristic, not RAGAS's method). |
| `AnswerOverlap` | Does the output's vocabulary overlap with the target's — doesn't look at `retrieval_context` at all, despite living in the same module. Distinct from RAGAS's `ResponseRelevancy`, which compares the answer to the *question* via embedding similarity. |

## Step 4: Extraction, if you're scoring against a short gold answer

RAG answers are usually phrased as full sentences ("You can reset your
password by going to Settings > Security..."), while a gold answer is
often a short span ("Settings > Security > Reset Password"). An exact-match
metric fails on that mismatch even when the answer is correct — this isn't
a metric bug, it's a real phrasing gap. Use an annotator to close it (full
mechanics: [Annotators](../annotators.md)):

```python
from auditkit.annotator import LLMAnnotator
from auditkit.metric import ExactMatch
from auditkit.cache import DiskCache

# Reuses the real generations from step 3 (Sample.actual_output), no new
# generation call -- same pattern LLMJudge re-scoring uses.
annotated = [
    ak.Sample(input=s.input, target=s.target, retrieval_context=s.retrieval_context,
               actual_output=p.raw_output)
    for s, p in zip(dataset, result.predictions)
]

def fake_extractor_model(prompts):
    # Stand-in for a real extraction call (e.g. a cheap "hf:..."/"groq:..."
    # model). In production this is any real model -- reusing `model`
    # from step 2 wouldn't work here, since that function is hardcoded to
    # produce full-sentence *answers*, not follow this prompt's extraction
    # instruction.
    out = []
    for p in prompts:
        if "password" in p.lower():
            out.append("Settings > Security > Reset Password")
        else:
            out.append("30-day returns for unused items")
    return out

extractor = LLMAnnotator(
    model=CallableModel(fake_extractor_model),
    prompt="Extract ONLY the short answer span, no full sentence.\n\nResponse: {output}",
    name="answer",
)

DiskCache().clear()
extracted_result = ak.evaluate(
    annotated, model="precomputed", adapter=adapter,
    scorers=[ExactMatch()], annotators=extractor, extract_with="answer",
)
print(extracted_result.headline)
for s, p in zip(annotated, extracted_result.predictions):
    print(" extracted:", p.parsed_answer, "| expected:", s.target)
```
```
{'exact_match': 1.0}
 extracted: Settings > Security > Reset Password | expected: Settings > Security > Reset Password
 extracted: 30-day returns for unused items | expected: 30-day returns for unused items
```

`RegexAnnotator` doesn't fit here — there's no fixed marker in a free-form
RAG answer to anchor a pattern to, which is exactly the case
`LLMAnnotator` is for. In practice, extraction quality then depends on the
extraction prompt and the model doing the extracting — it can pick the
wrong entity on genuinely ambiguous multi-entity sentences, so treat it as
a real (if usually reliable) model call, not a guaranteed-correct parser.

## Key takeaways

- Use **`retrieval_context`** (a real `Sample` field), not something
  improvised via `metadata`.
- **`LexicalGroundedness`/`ContextOverlap`** check the model's output against
  retrieved context; **`ContextCoverage`** checks retrieval quality
  independent of the output; **`AnswerOverlap`** checks the output
  against the target, not the context at all — pick metrics based on what
  you actually want to know, not just "the RAG ones."
- If your gold answers are short spans but your model answers in full
  sentences, **add an `LLMAnnotator`** before scoring with exact-match-style
  metrics rather than accepting a systematically low score that doesn't
  reflect actual correctness.
- Every metric's real matching rule (word-boundary vs. substring, whether
  it reads `output` at all) is in the [Scorer Reference](../scorers_reference.md)
  — several of these are crude, honestly-documented proxies, not
  state-of-the-art RAG evaluation; check [Known Issues](../known_issues.md)
  before trusting a number you haven't spot-checked.
