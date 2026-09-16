"""Phase 2 (prebuilt judges) + Phase 3 (generate -> score) tests. Offline."""

from __future__ import annotations

import auditkit as ak
from auditkit.metrics.judge import Factuality, ClosedQA, Relevance, LLMJudge
from auditkit.model import Generated, Result_, Model
from auditkit.sample import Sample


class Canned(Model):
    """Judge model that replies with a fixed string."""
    name = "canned"

    def __init__(self, reply: str):
        self.reply = reply

    def generate(self, requests):
        return [Result_(completions=[Generated(text=self.reply)]) for _ in requests]


# --- Phase 2: prebuilt judges --------------------------------------------

def test_factuality_maps_letters():
    f = Factuality(judge_model=Canned("Reasoning.\nCHOICE: C"))
    s = f.score(Sample(input="capital of France?", target="Paris"), "Paris is the capital.")
    assert s.value == 1.0            # C -> full agreement
    assert s.name == "factuality"


def test_factuality_disagree_scores_zero():
    f = Factuality(judge_model=Canned("CHOICE: D"))
    assert f.score(Sample(input="q", target="Paris"), "London").value == 0.0


def test_closed_qa_yes_no():
    qa = ClosedQA(judge_model=Canned("CHOICE: yes"))
    assert qa.score(Sample(input="Is water wet?"), "Yes, water is wet.").value == 1.0
    qa_no = ClosedQA(judge_model=Canned("CHOICE: no"))
    assert qa_no.score(Sample(input="Is water wet?"), "No.").value == 0.0


def test_relevance_three_way():
    r = Relevance(judge_model=Canned("CHOICE: partially_relevant"))
    assert r.score(Sample(input="Tell me about cats"), "Cats and dogs...").value == 0.5


def test_prebuilt_judges_are_registered():
    for name in ("factuality", "closed_qa", "relevance", "llm_judge", "g_eval"):
        assert name in ak.METRICS.names()


# --- API key plumbing for closed-source judge models ---------------------

def test_judge_model_args_forwarded_to_resolve(monkeypatch):
    captured = {}

    def fake_resolve(spec, **opts):
        captured["spec"] = spec
        captured["opts"] = opts
        return Canned("CHOICE: yes")

    monkeypatch.setattr("auditkit.model.AutoModel.resolve", staticmethod(fake_resolve))
    j = LLMJudge(judge_model="openai:gpt-4o-mini", choices={"yes": 1.0, "no": 0.0},
                 prompt="{output}", judge_model_args={"api_key": "sk-test", "api_base": "https://x"})
    j.score(Sample(input="q"), "out")
    assert captured["spec"] == "openai:gpt-4o-mini"
    assert captured["opts"]["api_key"] == "sk-test"
    assert captured["opts"]["api_base"] == "https://x"


# --- Phase 3: generate -> score ------------------------------------------

def test_generate_fills_actual_output():
    data = [Sample(input="a"), Sample(input="b")]
    answers = ak.generate(data, model=lambda ps: [p.upper() for p in ps])
    assert [s.actual_output for s in answers] == ["A", "B"]
    # originals untouched (generate returns copies)
    assert data[0].actual_output is None


def test_precomputed_scores_stored_outputs_without_a_model():
    # answers already generated (or uploaded); score them, no generation
    samples = [Sample(input="q1", target="4", actual_output="4"),
               Sample(input="q2", target="6", actual_output="7")]
    r = ak.evaluate(samples, model="precomputed", scorers=["exact_match"])
    assert r.headline["exact_match"] == 0.5
    assert [p.raw_output for p in r.predictions] == ["4", "7"]


def test_generate_then_score_two_stage_flow():
    data = [Sample(input="say hi", target="HI")]
    answers = ak.generate(data, model=lambda ps: ["HI"])
    # score the same answers two different ways without regenerating
    r1 = ak.evaluate(answers, model="precomputed", scorers=["exact_match"])
    judge = LLMJudge(judge_model=Canned("CHOICE: correct"),
                     choices={"correct": 1.0, "incorrect": 0.0}, prompt="{output}", name="j")
    r2 = ak.evaluate(answers, model="precomputed", scorers=[judge])
    assert r1.headline["exact_match"] == 1.0
    assert r2.headline["j"] == 1.0


def test_load_csv_output_col(tmp_path):
    from auditkit.loaders import load_csv
    p = tmp_path / "d.csv"
    p.write_text("input,target,answer\nq1,4,4\nq2,6,7\n", encoding="utf-8")
    samples = load_csv(str(p), input_col="input", target_col="target", output_col="answer")
    assert samples[0].actual_output == "4"
    assert samples[1].actual_output == "7"
