"""Tests for the LLMJudge classifier/numeric judge and the fingerprint identity.

The judge model is a fake (canned text), so these run offline.
"""

from __future__ import annotations

import auditkit as ak
from auditkit.metrics.judge import LLMJudge, GEval, RubricItem
from auditkit.model import Generated, Result_
from auditkit.runspec import RunConfig, RunSpec
from auditkit.scenario import ListScenario
from auditkit.model import EchoModel
from auditkit.metric import ExactMatch
from auditkit.adapter import TemplateAdapter, GenerationAdapter
from auditkit.sample import Sample
from auditkit.types import ScoreKind


class FakeJudge:
    """A judge model that returns whatever canned text it was given."""

    name = "fake-judge"

    def __init__(self, reply: str):
        self.reply = reply
        self.seen: list[str] = []

    def generate(self, requests):
        self.seen = [str(r.prompt) for r in requests]
        return [Result_(completions=[Generated(text=self.reply)]) for _ in requests]


# --- classifier mode ------------------------------------------------------

def test_classifier_maps_choice_to_score():
    j = LLMJudge(judge_model=FakeJudge("CHOICE: correct"),
                 prompt="Q: {input}\nA: {output}",
                 choices={"correct": 1.0, "partial": 0.5, "incorrect": 0.0})
    s = j.score(Sample(input="2+2?", target="4"), "4")
    assert s.value == 1.0
    assert s.kind == ScoreKind.JUDGE
    assert s.metadata["choice"] == "correct"
    assert s.reason  # reason is populated


def test_classifier_cot_uses_last_marker():
    reply = "The answer matches the expected value.\nCHOICE: correct"
    j = LLMJudge(judge_model=FakeJudge(reply), choices={"correct": 1.0, "incorrect": 0.0},
                 use_cot=True, prompt="{output}")
    s = j.score(Sample(input="q", target="a"), "a")
    assert s.value == 1.0
    assert "matches the expected" in s.reason


def test_classifier_case_and_punctuation_insensitive():
    j = LLMJudge(judge_model=FakeJudge("CHOICE: Incorrect."),
                 choices={"correct": 1.0, "incorrect": 0.0}, prompt="{output}")
    assert j.score(Sample(input="q", target="a"), "b").value == 0.0


def test_unknown_escape_when_unparseable():
    j = LLMJudge(judge_model=FakeJudge("I have no idea honestly"),
                 choices={"correct": 1.0, "incorrect": 0.0}, prompt="{output}",
                 unknown_score=0.0)
    s = j.score(Sample(input="q", target="a"), "b")
    assert s.metadata.get("unknown") is True
    assert s.value == 0.0
    assert "UNKNOWN" in s.reason      # explicit, not a silent midpoint


def test_fallback_ignores_incidental_label_word_in_earlier_reasoning():
    """CoT reasoning that never states a real verdict shouldn't be matched just
    because an ordinary word (here "no") happens to equal a choice label."""
    reply = ("Let us think about this step by step. There is no clear issue "
              "with the phrasing, and no further context is needed to "
              "understand it. The reasoning seems sound overall.")
    j = LLMJudge(judge_model=FakeJudge(reply), choices={"yes": 1.0, "no": 0.0},
                 prompt="{output}", use_cot=True)
    s = j.score(Sample(input="q", target="a"), "b")
    assert s.metadata.get("unknown") is True


def test_fallback_still_matches_genuine_markerless_verdict_on_last_line():
    """A real verdict with no CHOICE: marker, on its own final line, should
    still match -- the fix narrows the scan window, it doesn't remove it."""
    j = LLMJudge(judge_model=FakeJudge("Some reasoning about the answer.\nno."),
                 choices={"yes": 1.0, "no": 0.0}, prompt="{output}", use_cot=True)
    s = j.score(Sample(input="q", target="a"), "b")
    assert s.metadata.get("choice") == "no"
    assert s.value == 0.0


# --- numeric mode ---------------------------------------------------------

def test_numeric_scale_normalizes():
    j = LLMJudge(judge_model=FakeJudge("SCORE: 4"), scale=(1.0, 5.0), prompt="{output}")
    s = j.score(Sample(input="q", target="a"), "x")
    assert s.value == 0.75              # (4-1)/(5-1)
    assert s.metadata["raw_score"] == 4.0


def test_numeric_clamps_out_of_range():
    j = LLMJudge(judge_model=FakeJudge("SCORE: 9"), scale=(1.0, 5.0), prompt="{output}")
    assert j.score(Sample(input="q", target="a"), "x").value == 1.0


# --- template + config validation ----------------------------------------

def test_template_substitution_reaches_judge():
    fake = FakeJudge("CHOICE: yes")
    j = LLMJudge(judge_model=fake, choices={"yes": 1.0, "no": 0.0},
                 prompt="Question: {input}\nExpected: {expected}\nAnswer: {output}")
    j.score(Sample(input="capital of France?", target="Paris"), "Paris")
    sent = fake.seen[0]
    assert "capital of France?" in sent and "Paris" in sent


def test_choices_and_scale_are_mutually_exclusive():
    import pytest
    with pytest.raises(ValueError, match="either choices or scale"):
        LLMJudge(judge_model=FakeJudge("x"), choices={"a": 1.0}, scale=(0.0, 1.0))


def test_missing_judge_model_raises():
    import pytest
    j = LLMJudge(judge_model=None, choices={"a": 1.0}, prompt="{output}")
    with pytest.raises(NotImplementedError, match="judge_model"):
        j.score(Sample(input="q"), "out")


def test_runs_through_evaluate():
    j = LLMJudge(judge_model=FakeJudge("SCORE: 1"), scale=(0.0, 1.0), prompt="{output}", name="qual")
    r = ak.evaluate([Sample(input="q", target="a")], model="echo", scorers=[j])
    assert "qual" in r.headline


# --- GEval reworked onto LLMJudge -----------------------------------------

def test_geval_now_produces_real_score():
    g = GEval(rubric=[RubricItem("correctness", 2.0), RubricItem("fluency")],
              judge_model=FakeJudge("Reasoning...\nSCORE: 5"), name="g-eval")
    s = g.score(Sample(input="q", target="a"), "a")
    assert s.value == 1.0               # 5 on a 1-5 scale
    assert s.reason


# --- Phase 0: identity in the fingerprint ---------------------------------

def _fp(adapter=None, metrics=None):
    return RunSpec(
        scenario=ListScenario([Sample(input="q", target="a")]),
        model=EchoModel(),
        adapter=adapter or GenerationAdapter(),
        metrics=metrics or [ExactMatch()],
        config=RunConfig(),
    ).fingerprint()


def test_fingerprint_differs_for_different_prompts():
    a = _fp(adapter=TemplateAdapter("{input}"))
    b = _fp(adapter=TemplateAdapter("{input}\nThink step by step."))
    assert a != b                       # the prompt-collision bug is fixed


def test_fingerprint_stable_for_same_prompt():
    assert _fp(adapter=TemplateAdapter("{input}")) == _fp(adapter=TemplateAdapter("{input}"))


def test_fingerprint_differs_for_different_judges():
    j1 = LLMJudge(judge_model="openai:gpt-4o", choices={"a": 1.0}, prompt="v1 {output}", name="j")
    j2 = LLMJudge(judge_model="openai:gpt-4o", choices={"a": 1.0}, prompt="v2 {output}", name="j")
    assert _fp(metrics=[j1]) != _fp(metrics=[j2])


# --- judge-side generation settings (temperature/max_tokens/top_p) --------
#
# The judge call never goes through an Adapter/RunConfig (there's no sample
# being adapted, just someone else's output being graded), so unlike the main
# pipeline there was never another way to control the judge model's own
# generation settings. Before this fix, judge_model_args={"temperature": ...}
# went straight into the judge model's constructor -- which stopped working
# (and started raising) once model constructors were cleaned up to reject
# generation-behavior kwargs project-wide. These dedicated kwargs restore that
# control through the *request* (resolve_params()), the same channel every
# real backend already uses for the main pipeline.

def test_temperature_max_tokens_top_p_reach_the_judge_models_request():
    class RecordingJudge:
        name = "recording"
        def generate(self, requests):
            self.last_params = requests[0].params
            return [Result_(completions=[Generated(text="CHOICE: correct")])]
    rj = RecordingJudge()
    j = LLMJudge(judge_model=rj, choices={"correct": 1.0, "incorrect": 0.0},
                 prompt="{output}", temperature=0.2, max_tokens=150, top_p=0.9)
    j.judge(Sample(input="q", target="a"), "a")
    assert rj.last_params == {"temperature": 0.2, "max_tokens": 150, "top_p": 0.9}


def test_no_gen_params_means_empty_request_params():
    class RecordingJudge:
        name = "recording"
        def generate(self, requests):
            self.last_params = requests[0].params
            return [Result_(completions=[Generated(text="CHOICE: correct")])]
    rj = RecordingJudge()
    j = LLMJudge(judge_model=rj, choices={"correct": 1.0, "incorrect": 0.0}, prompt="{output}")
    j.judge(Sample(input="q", target="a"), "a")
    assert rj.last_params == {}


def test_judge_model_args_no_longer_accepts_generation_kwargs():
    # This is the exact regression this fix addresses: judge_model_args used
    # to be the only way to set the judge model's temperature/max_tokens, by
    # forwarding them straight into the model constructor. Model constructors
    # now reject those kwargs project-wide -- confirms judge_model_args is
    # connection-level only going forward (api_key/api_base/device/...), not
    # a place to smuggle generation settings through anymore.
    j = LLMJudge(judge_model="openai:gpt-4o", judge_model_args={"api_key": "fake", "temperature": 0.5},
                 choices={"a": 1.0}, prompt="{output}")
    try:
        j._model()
        assert False, "expected AuditKitError"
    except ak.AuditKitError as e:
        assert "temperature" in str(e)


def test_real_backend_resolution_with_connection_args_plus_new_gen_kwargs_works():
    # The actual notebook scenario this fix restores: judge_model_args carries
    # only connection-level settings; temperature/max_tokens go through the
    # new dedicated kwargs instead -- construction and resolution both work.
    j = LLMJudge(judge_model="groq:llama-3.1-8b-instant", judge_model_args={"api_key": "fake"},
                 choices={"a": 1.0}, prompt="{output}", temperature=0.0, max_tokens=300)
    model = j._model()  # would have raised before this fix
    assert type(model).__name__ == "GroqModel"


def test_gen_params_included_in_identity_and_fingerprint():
    j1 = LLMJudge(judge_model="openai:gpt-4o", choices={"a": 1.0}, prompt="v {output}", name="j",
                  temperature=0.0)
    j2 = LLMJudge(judge_model="openai:gpt-4o", choices={"a": 1.0}, prompt="v {output}", name="j",
                  temperature=0.9)
    assert j1.identity()["gen_params"] != j2.identity()["gen_params"]
    assert _fp(metrics=[j1]) != _fp(metrics=[j2])


def test_geval_and_prebuilt_judges_accept_the_same_gen_kwargs():
    from auditkit.metrics.judge import Factuality, ClosedQA, Relevance

    rj_calls = []

    class RecordingJudge:
        name = "recording"
        def generate(self, requests):
            rj_calls.append(requests[0].params)
            return [Result_(completions=[Generated(text="SCORE: 3")])]

    GEval(rubric=[RubricItem("accuracy")], judge_model=RecordingJudge(),
          temperature=0.1, max_tokens=50).judge(Sample(input="q", target="a"), "a")
    Factuality(judge_model=RecordingJudge(), temperature=0.1).judge(
        Sample(input="q", target="a"), "a")
    ClosedQA(judge_model=RecordingJudge(), max_tokens=64).judge(Sample(input="q"), "a")
    Relevance(judge_model=RecordingJudge(), top_p=0.8).judge(Sample(input="q"), "a")

    assert rj_calls[0] == {"temperature": 0.1, "max_tokens": 50}
    assert rj_calls[1] == {"temperature": 0.1}
    assert rj_calls[2] == {"max_tokens": 64}
    assert rj_calls[3] == {"top_p": 0.8}


def test_match_choice_picks_the_label_at_the_last_text_position_not_dict_order():
    # Previously: when more than one candidate label appeared in the judge's
    # final line, the fallback picked whichever key happened to iterate last
    # in self.choices (dict order), not whichever the model actually placed
    # last in its own text. Here "irrelevant" iterates last in choices but
    # "relevant" is the label that actually appears last in the text (the
    # real verdict, per "final line" convention) -- confirmed previously
    # broken: it returned "irrelevant".
    j = LLMJudge(choices={"relevant": 1.0, "partially_relevant": 0.5, "irrelevant": 0.0})
    label = j._match_choice("the answer is not irrelevant, it is relevant.")
    assert label == "relevant"
