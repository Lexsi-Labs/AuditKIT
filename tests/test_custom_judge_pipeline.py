"""Full custom pipeline, end to end, with a real Groq model -- no mocking.

Exercises the complete chain with genuinely custom pieces at every stage:

- A custom :class:`~auditkit.adapter.Adapter` (``PersonaAdapter``, not one of
  the shipped built-ins) that reads a per-sample ``metadata["persona"]`` field
  to build its generation prompt.
- A custom :class:`~auditkit.metrics.judge.LLMJudge` whose own prompt template
  references a *different* metadata field (``metadata["criteria"]``) --
  proving metadata flows independently into both the generation and the
  scoring stage.
- Both generation and judging go through the real Groq API
  (``llama-3.1-8b-instant``), with ``concurrency=1`` (which sidesteps the empty-chunk bug).

Needs a real ``GROQ_API_KEY``; skipped automatically when it isn't set, same
as the rest of the suite runs fully offline by default.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("requests")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
pytestmark = pytest.mark.skipif(not GROQ_API_KEY, reason="GROQ_API_KEY not set")

from auditkit.adapter import Adapter
from auditkit.api import evaluate
from auditkit.cache import DiskCache
from auditkit.metrics.judge import LLMJudge
from auditkit.model import Request
from auditkit.runspec import RunConfig
from auditkit.sample import Sample

MODEL = "groq:llama-3.1-8b-instant"

# The exact input data used throughout this test.
SAMPLES = [
    Sample(
        input="Why is the sky blue?",
        target="Rayleigh scattering of sunlight by the atmosphere",
        metadata={"persona": "a curious 5-year-old", "criteria": "simple, no jargon, friendly tone"},
    ),
    Sample(
        input="What causes the seasons on Earth?",
        target="Earth's axial tilt as it orbits the sun",
        metadata={"persona": "a college physics student", "criteria": "technically precise, uses correct terminology"},
    ),
]


class PersonaAdapter(Adapter):
    """Custom adapter: builds its prompt from sample.metadata["persona"]."""

    method = "persona"

    def identity(self) -> dict:
        return {"method": self.method}

    def adapt(self, sample: Sample, config: RunConfig) -> list[Request]:
        persona = sample.metadata.get("persona", "a general audience")
        prompt = (f"Explain the following as if you were talking to {persona}. "
                  f"Keep it brief.\n\nQuestion: {sample.input_text}")
        return [Request(prompt=prompt, request_type="generate")]


def _judge() -> LLMJudge:
    return LLMJudge(
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


@pytest.fixture(scope="module")
def pipeline_result():
    """Runs the real pipeline exactly ONCE for the whole module -- Groq's
    free-tier rate limit is easily hit if each test re-runs its own real
    generation + judge calls independently."""
    DiskCache().clear()
    return evaluate(
        SAMPLES,
        model=MODEL,
        adapter=PersonaAdapter(),
        scorers=[_judge()],
        config=RunConfig(concurrency=1, temperature=0.3, max_tokens=150),
    )


@pytest.fixture(scope="module")
def judge_scores(pipeline_result):
    """Score.reason/metadata are discarded by Runner.score_one() before
    reaching Prediction (only (name, value) survives) -- call the judge
    directly, once per sample, with the real generation output already
    produced above, to recover the full verdict."""
    judge = _judge()
    return [judge.judge(s, p.raw_output) for s, p in zip(SAMPLES, pipeline_result.predictions)]


class TestCustomJudgePipeline:
    def test_adapter_builds_persona_specific_prompts(self, pipeline_result):
        assert len(pipeline_result.predictions) == len(SAMPLES)
        assert pipeline_result.predictions[0].prompt.startswith(
            "Explain the following as if you were talking to a curious 5-year-old."
        )
        assert pipeline_result.predictions[1].prompt.startswith(
            "Explain the following as if you were talking to a college physics student."
        )

    def test_real_generation_is_nonempty_and_on_topic(self, pipeline_result, capsys):
        for p, s in zip(pipeline_result.predictions, SAMPLES):
            assert p.raw_output and len(p.raw_output) > 20
            with capsys.disabled():
                print(f"\n[{s.metadata['persona']}] generation: {p.raw_output!r}")

    def test_judge_prompt_includes_custom_metadata_field(self, pipeline_result):
        """The judge's own template references metadata["criteria"] -- a field
        the adapter never used -- proving metadata reaches both stages
        independently."""
        judge = _judge()
        rendered = judge._assemble(judge._render(SAMPLES[0], pipeline_result.predictions[0].raw_output))
        assert "simple, no jargon, friendly tone" in rendered
        assert "Rayleigh scattering" in rendered

    def test_judge_scores_are_valid_and_reasoned(self, judge_scores, capsys):
        for score, s in zip(judge_scores, SAMPLES):
            assert score.value in (0.0, 0.5, 1.0)
            assert score.metadata.get("unknown") is not True
            assert "CHOICE:" in score.metadata["raw"]
            with capsys.disabled():
                print(f"\n[{s.metadata['persona']}] verdict={score.metadata['choice']} reason={score.reason[:150]!r}")

    def test_fingerprint_reflects_sample_content(self):
        """Two different samples (different input) get different fingerprints --
        the scenario-level check from earlier in this session, re-verified here.
        Pure fingerprint math, no real API call needed."""
        from auditkit.model import AutoModel
        from auditkit.runspec import RunSpec
        from auditkit.scenario import ListScenario

        model = AutoModel.resolve(MODEL, api_key=GROQ_API_KEY)
        judge = _judge()
        fp0 = RunSpec(scenario=ListScenario([SAMPLES[0]]), model=model,
                      adapter=PersonaAdapter(), metrics=[judge]).fingerprint()
        fp1 = RunSpec(scenario=ListScenario([SAMPLES[1]]), model=model,
                      adapter=PersonaAdapter(), metrics=[judge]).fingerprint()
        assert fp0 != fp1
