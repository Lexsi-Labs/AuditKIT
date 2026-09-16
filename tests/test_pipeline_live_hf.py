"""Full pipeline, end to end, with real data and a real model -- no mocking.

Every other adapter test in this suite uses ``EchoModel``/``CallableModel`` so
it can run offline in milliseconds. This file is the opposite: it pulls real
slices of real HuggingFace datasets -- ``allenai/ai2_arc`` (MCQ), ``rajpurkar/
squad`` (RAG), ``nyu-mll/glue``/``sst2`` (fixed-label classification),
``openai/openai_humaneval`` (code completion) -- and runs them through a real
(tiny) HuggingFace causal LM, ``sshleifer/tiny-gpt2``, exercising every
built-in :class:`~auditkit.adapter.Adapter` plus two examples of extending the
adapter system for a dataset shape none of the built-ins fit: a genuinely new
:class:`ClassificationAdapter` for SST-2's fixed shared label set (unlike
``MCQAdapter``, which needs per-sample ``choices``), and reusing the existing
``TemplateAdapter`` for HumanEval (no new adapter needed there -- it's just a
different template string).

The model has random/tiny weights, so outputs are gibberish by design -- this
is a plumbing test (does each adapter's real prompt reach a real model and
come back as a real `Prediction`?), not a quality/accuracy test.

Needs network + ``pip install -e ".[transformers,interop]"``; skipped
automatically when either extra or network access is unavailable, same as
the rest of the suite runs fully offline by default.
"""
from __future__ import annotations

import pytest

pytest.importorskip("transformers")
pytest.importorskip("datasets")

from auditkit.adapter import (
    Adapter, ChatAdapter, FewShotAdapter, GenerationAdapter, InstructionAdapter,
    MCQAdapter, RAGAdapter, TemplateAdapter,
)
from auditkit.api import evaluate
from auditkit.cache import DiskCache
from auditkit.model import Request
from auditkit.runspec import RunConfig
from auditkit.sample import Sample


class ClassificationAdapter(Adapter):
    """A custom adapter for fixed-label classification (e.g. SST-2 sentiment).

    Unlike :class:`~auditkit.adapter.MCQAdapter`, the label set here is the
    *same for every sample* rather than per-sample ``choices`` -- there's no
    clean way to express "every sample shares this fixed label set" with the
    shipped adapters without mutating every ``Sample`` to carry an identical
    ``choices`` list, so this is a case where writing a new ``Adapter``
    subclass genuinely earns its keep.
    """

    method = "classification"

    def __init__(self, labels: list[str]) -> None:
        self.labels = labels

    def identity(self) -> dict:
        return {"labels": self.labels}

    def adapt(self, sample: Sample, config: RunConfig) -> list[Request]:
        options = ", ".join(self.labels)
        prompt = (
            f"Classify the sentiment as one of: {options}.\n\n"
            f"Sentence: {sample.input_text}\nLabel:"
        )
        return [Request(prompt=prompt, request_type="generate")]

MODEL_NAME = "sshleifer/tiny-gpt2"


def _skip_if_offline(fn):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 -- any network/hub failure should skip, not fail CI
        pytest.skip(f"HF Hub unavailable: {type(e).__name__}: {e}")


@pytest.fixture(scope="module")
def arc_samples() -> list[Sample]:
    """8 real ARC-Challenge MCQ items: question + real 4-way choices + answer key."""
    def _load():
        import datasets
        rows = datasets.load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test[:8]")
        return [
            Sample(input=r["question"], choices=r["choices"]["text"], target=r["answerKey"])
            for r in rows
        ]
    return _skip_if_offline(_load)


@pytest.fixture(scope="module")
def squad_samples() -> list[Sample]:
    """3 real SQuAD items with a genuine context paragraph as retrieval_context."""
    def _load():
        import datasets
        rows = datasets.load_dataset("rajpurkar/squad", split="validation[:3]")
        return [
            Sample(input=r["question"], target=r["answers"]["text"][0],
                   retrieval_context=[r["context"]])
            for r in rows
        ]
    return _skip_if_offline(_load)


@pytest.fixture(scope="module")
def sst2_samples() -> list[Sample]:
    """5 real SST-2 sentences with a fixed shared 2-way label set."""
    def _load():
        import datasets
        rows = datasets.load_dataset("nyu-mll/glue", "sst2", split="validation[:5]")
        return [
            Sample(input=r["sentence"].strip(), target="positive" if r["label"] == 1 else "negative")
            for r in rows
        ]
    return _skip_if_offline(_load)


@pytest.fixture(scope="module")
def humaneval_samples() -> list[Sample]:
    """3 real HumanEval code-completion prompts (function signature + docstring)."""
    def _load():
        import datasets
        rows = datasets.load_dataset("openai/openai_humaneval", split="test[:3]")
        return [Sample(input=r["prompt"], target=r["canonical_solution"]) for r in rows]
    return _skip_if_offline(_load)


@pytest.fixture(scope="module")
def hf_model():
    """The real (tiny) HF backend, loaded once and reused across every adapter test."""
    def _load():
        from auditkit.model.hf_gen import HFGenModel
        m = HFGenModel(model=MODEL_NAME, device="cpu")
        m._ensure_pipeline()  # force the real weight download/load here, not on first test
        return m
    return _skip_if_offline(_load)


@pytest.fixture(scope="module")
def real_gpt2_model():
    """A real, trained (not random-weight) small model -- unlike `hf_model`
    (`sshleifer/tiny-gpt2`, deliberately tiny/random weights, fine for
    plumbing tests but useless for anything checking actual output
    plausibility), plain `gpt2` (124M, real trained weights) is needed for
    loglikelihood tests that assert a sensible continuation scores higher
    than a nonsense one -- that assertion would be unreliably flaky against
    near-random weights.
    """
    def _load():
        from auditkit.model.hf_gen import HFGenModel
        m = HFGenModel(model="gpt2", device="cpu")
        m._ensure_pipeline()
        return m
    return _skip_if_offline(_load)


def _run(data, model, adapter, config=None):
    DiskCache().clear()  # each adapter's config differs, but clearing avoids any stale cross-test cache hit
    # concurrency=1: Runner._chunk() always makes exactly `concurrency` chunks,
    # so the default (8) leaves mostly-empty chunks with our handful of real
    # samples, and a real transformers pipeline errors on an empty batch
    # (unlike EchoModel, which tolerates it). Left as a test-side workaround,
    # not a runner.py change.
    # max_tokens=8: generation settings are RunConfig's job, not the model
    # constructor's -- this keeps each real generation call short/fast, the
    # same intent the old HFGenModel(max_new_tokens=8) constructor kwarg served.
    return evaluate(data, model=model, adapter=adapter, config=config or RunConfig(concurrency=1, max_tokens=8))


class TestAdaptersAgainstRealData:
    """One test per built-in adapter, each printing the real prompt/output it produced."""

    def test_generation(self, arc_samples, hf_model, capsys):
        r = _run(arc_samples[:3], hf_model, GenerationAdapter())
        assert len(r.predictions) == 3
        with capsys.disabled():
            print("\n[generation]")
            for p in r.predictions:
                print(f"  prompt={p.prompt!r}\n  output={p.raw_output!r}")

    def test_mcq(self, arc_samples, hf_model, capsys):
        r = _run(arc_samples[:3], hf_model, MCQAdapter("mcq_joint"))
        assert len(r.predictions) == 3
        with capsys.disabled():
            print("\n[mcq (mcq_joint)]")
            for p in r.predictions:
                print(f"  prompt={p.prompt!r}\n  output={p.raw_output!r} expected={p.expected!r}")

    def test_chat(self, arc_samples, hf_model, capsys):
        r = _run(arc_samples[:3], hf_model, ChatAdapter("You are a science tutor."))
        assert len(r.predictions) == 3
        assert all(p.prompt.startswith("System: You are a science tutor.") for p in r.predictions)
        with capsys.disabled():
            print("\n[chat]")
            for p in r.predictions:
                print(f"  prompt={p.prompt!r}\n  output={p.raw_output!r}")

    def test_instruction(self, arc_samples, hf_model, capsys):
        r = _run(arc_samples[:3], hf_model, InstructionAdapter("Answer the science question:"))
        assert len(r.predictions) == 3
        assert all(p.prompt.startswith("Answer the science question:") for p in r.predictions)
        with capsys.disabled():
            print("\n[instruction]")
            for p in r.predictions:
                print(f"  prompt={p.prompt!r}\n  output={p.raw_output!r}")

    def test_template(self, arc_samples, hf_model, capsys):
        r = _run(arc_samples[:3], hf_model, TemplateAdapter("Q: {input}\nA:"))
        assert len(r.predictions) == 3
        assert all(p.prompt.startswith("Q: ") and p.prompt.endswith("\nA:") for p in r.predictions)
        with capsys.disabled():
            print("\n[template]")
            for p in r.predictions:
                print(f"  prompt={p.prompt!r}\n  output={p.raw_output!r}")

    def test_fewshot(self, arc_samples, hf_model, capsys):
        pool, eval_samples = arc_samples[:3], arc_samples[3:6]
        r = _run(eval_samples, hf_model, FewShotAdapter(num_shots=2, pool=pool))
        assert len(r.predictions) == 3
        # the first pool question should actually appear in the built prompt
        assert pool[0].input in r.predictions[0].prompt
        with capsys.disabled():
            print("\n[fewshot, 2-shot from real ARC pool]")
            for p in r.predictions:
                print(f"  prompt={p.prompt!r}\n  output={p.raw_output!r}")

    def test_rag(self, squad_samples, hf_model, capsys):
        r = _run(squad_samples, hf_model, RAGAdapter())
        assert len(r.predictions) == 3
        # the real SQuAD context paragraph must actually be in the built prompt
        for p, s in zip(r.predictions, squad_samples):
            assert s.retrieval_context[0] in p.prompt
        with capsys.disabled():
            print("\n[rag, real SQuAD context]")
            for p in r.predictions:
                print(f"  prompt={p.prompt[:200]!r}...\n  output={p.raw_output!r}")

    def test_classification_custom_adapter(self, sst2_samples, hf_model, capsys):
        """New dataset shape (fixed shared labels) -> a genuinely new custom adapter."""
        r = _run(sst2_samples, hf_model, ClassificationAdapter(labels=["negative", "positive"]))
        assert len(r.predictions) == len(sst2_samples)
        assert all(p.prompt.startswith("Classify the sentiment") for p in r.predictions)
        with capsys.disabled():
            print("\n[custom ClassificationAdapter, real SST-2]")
            for p in r.predictions:
                print(f"  prompt={p.prompt!r}\n  output={p.raw_output!r} expected={p.expected!r}")

    def test_code_completion_reuses_template_adapter(self, humaneval_samples, hf_model, capsys):
        """New dataset shape (code) but NO new adapter needed -- TemplateAdapter already fits."""
        r = _run(
            humaneval_samples, hf_model,
            TemplateAdapter("# Complete the following Python function.\n{input}"),
        )
        assert len(r.predictions) == len(humaneval_samples)
        assert all(p.prompt.startswith("# Complete the following Python function.\n") for p in r.predictions)
        with capsys.disabled():
            print("\n[TemplateAdapter reused for real HumanEval, no new adapter needed]")
            for p in r.predictions:
                print(f"  prompt={p.prompt[:150]!r}...\n  output={p.raw_output[-120:]!r}")

    def test_rag_without_context_raises(self, hf_model):
        """The RAGAdapter fix: real data without retrieval_context must fail loudly."""
        no_ctx = [Sample(input="What color is the sky?", target="blue")]
        with pytest.raises(ValueError, match="retrieval_context"):
            _run(no_ctx, hf_model, RAGAdapter())

    def test_mcq_without_choices_raises(self, hf_model):
        """load_csv/plain samples have no choices -- MCQAdapter must fail loudly, not silently."""
        no_choices = [Sample(input="Pick one.", target="A")]
        with pytest.raises(ValueError, match="choices"):
            _run(no_choices, hf_model, MCQAdapter("mcq_joint"))


class TestHFGenModelRealLogLikelihood:
    """HFGenModel.loglikelihood() -- real teacher-forced scoring, real model.

    Was previously unimplemented for every real backend (only test doubles
    faked it). Verified two ways: a plausibility sanity check (a sensible
    continuation must score higher than a nonsense one), and a full
    MCQAdapter(mcq_loglikelihood) -> Acc/AccNorm pipeline run, which also
    caught a real, previously-undiscovered bug in _resolve_choice_index()
    (it stripped the candidate but not `choices` before comparing, so a
    correct pick against whitespace-prefixed choices -- the standard,
    correct convention for GPT-2/BPE-style loglikelihood scoring -- could
    never match). Both are fixed; this is the regression guard.
    """

    def test_plausible_continuation_scores_higher_than_nonsense(self, real_gpt2_model):
        from auditkit.model import Request

        reqs = [
            Request(prompt="The capital of France is", request_type="loglikelihood",
                    params={"target": " Paris"}),
            Request(prompt="The capital of France is", request_type="loglikelihood",
                    params={"target": " a banana"}),
        ]
        lls = real_gpt2_model.loglikelihood(reqs)
        assert len(lls) == 2
        assert lls[0].logprob > lls[1].logprob

    def test_real_mcq_loglikelihood_pipeline_scores_correctly(self, real_gpt2_model):
        from auditkit.metric import Acc, AccNorm

        DiskCache().clear()
        samples = [
            Sample(input="The capital of France is",
                   choices=[" a banana", " Paris", " a shoe", " Tuesday"], target=1),
        ]
        result = evaluate(
            samples, model=real_gpt2_model, adapter=MCQAdapter(method="mcq_loglikelihood"),
            scorers=[Acc(), AccNorm()], config=RunConfig(concurrency=1),
        )
        pred = result.predictions[0]
        # The real check: whatever choice the real model's real logprobs
        # picked, Acc/AccNorm must correctly recognize a match against
        # whitespace-prefixed choice text (the _resolve_choice_index fix).
        picked_idx = samples[0].choices.index(pred.raw_output)
        assert pred.correct == (picked_idx == samples[0].target)
