"""Tests for auditkit.metrics.rag_judge with scripted stub judges."""

from __future__ import annotations

import pytest

import auditkit as ak
from auditkit.metrics.rag_judge import (
    ContextPrecision, ContextRecall, Faithfulness, average_precision, parse_verdicts,
)
from auditkit.model import CallableModel
from auditkit.sample import Sample

FAITH = ("SUPPORTED", "UNSUPPORTED", "CONTRADICTED")
ATTR = ("ATTRIBUTED", "NOT_ATTRIBUTED")


def stub(claims="", verdicts="", precision="", recall=""):
    """A judge that answers by prompt type; returns (model, prompts seen)."""
    prompts: list[str] = []

    def fn(ps):
        out = []
        for p in ps:
            prompts.append(p)
            if "<claims>" in p:
                out.append(verdicts)
            elif "Break the ANSWER" in p:
                out.append(claims)
            elif "<reference_sentences>" in p:
                out.append(recall)
            else:
                out.append(precision)
        return out

    return CallableModel(fn), prompts


def rag_sample(**kw):
    kw.setdefault("input", "Tell me about Paris.")
    kw.setdefault("retrieval_context", ["Paris is the capital of France.", "Paris has 2.1 million people."])
    return Sample(**kw)


# -- parse_verdicts -----------------------------------------------------------

def test_parse_plain_and_decorated_verdict_lines():
    text = "**1: SUPPORTED**\n2) unsupported\n- 3. Contradicted\n4 - SUPPORTED\n5. **Supported**\n6:UNSUPPORTED"
    assert parse_verdicts(text, 6, FAITH) == ["SUPPORTED", "UNSUPPORTED", "CONTRADICTED",
                                              "SUPPORTED", "SUPPORTED", "UNSUPPORTED"]


def test_parse_underscore_and_space_labels():
    assert parse_verdicts("1: NOT_ATTRIBUTED\n2: not attributed\n3: Attributed", 3, ATTR) == \
        ["NOT_ATTRIBUTED", "NOT_ATTRIBUTED", "ATTRIBUTED"]


def test_parse_ignores_prose_and_out_of_order_lines():
    text = ("Let me check each claim. There are 2 claims.\n"
            "Claim 1 is backed by 2 sources.\n"
            "2: SUPPORTED\n"
            "1: UNSUPPORTED\n"
            "Overall 1 of 2 is supported.")
    assert parse_verdicts(text, 2, FAITH) == ["UNSUPPORTED", "SUPPORTED"]


def test_prose_with_numbers_does_not_create_verdicts():
    text = ("There are 3 claims.\n2 of them are supported.\n1. The context mentions Paris.\n"
            "2019: a good year\n3.5 supported\nSUPPORTED")
    with pytest.raises(ValueError, match="missing"):
        parse_verdicts(text, 1, FAITH)


def test_missing_verdict_raises():
    with pytest.raises(ValueError, match=r"missing \[2\]"):
        parse_verdicts("1: SUPPORTED\n3: SUPPORTED", 3, FAITH)


def test_extra_verdicts_raise():
    # More verdicts than items means the judge's numbering is off.
    with pytest.raises(ValueError):
        parse_verdicts("1: SUPPORTED\n2: SUPPORTED\n3: UNSUPPORTED", 2, FAITH)
    with pytest.raises(ValueError):
        parse_verdicts("0: SUPPORTED\n1: SUPPORTED", 1, FAITH)


def test_unknown_label_is_rejected():
    for text in ("1: MAYBE", "1: SUPPORTEDISH", "1: the claim is SUPPORTED"):
        with pytest.raises(ValueError):
            parse_verdicts(text, 1, FAITH)


def test_trailing_reason_after_label_is_accepted():
    # Weak judges often explain after the verdict; the label still leads the line.
    for text in ("1: SUPPORTED - the context says so", "1: SUPPORTED (see chunk 2)", "1. **Supported**: stated"):
        assert parse_verdicts(text, 1, FAITH) == ["SUPPORTED"]
    assert parse_verdicts("1: UNSUPPORTED, not mentioned", 1, FAITH) == ["UNSUPPORTED"]
    assert parse_verdicts("1: NOT ATTRIBUTED - missing", 1, ("ATTRIBUTED", "NOT_ATTRIBUTED")) == ["NOT_ATTRIBUTED"]


def test_extra_verdict_numbers_are_rejected():
    with pytest.raises(ValueError):
        parse_verdicts("1: SUPPORTED\n2: SUPPORTED\n3: SUPPORTED", 2, FAITH)


def test_parse_markdown_decorated_number():
    # Markdown decoration placed on the NUMBER itself is ignored (finding [11]).
    for text in ("**1**: SUPPORTED", "**1**. SUPPORTED", "`1`: SUPPORTED"):
        assert parse_verdicts(text, 1, FAITH) == ["SUPPORTED"], text


def test_broadened_regex_still_rejects_out_of_range_and_prose():
    # The strict <number>: <VERDICT> contract holds: 'Claim N:', restated claims
    # and out-of-range numbers are still rejected.
    with pytest.raises(ValueError):
        parse_verdicts("Claim 1: SUPPORTED", 1, FAITH)
    with pytest.raises(ValueError):
        parse_verdicts("**2**: SUPPORTED", 1, FAITH)  # out of range


def test_average_precision_helper():
    assert average_precision([True, False, True]) == pytest.approx((1 + 2 / 3) / 2)
    assert average_precision([False, True]) == 0.5
    assert average_precision([False, False]) == 0.0 and average_precision([]) == 0.0


# -- Faithfulness -------------------------------------------------------------

def test_faithfulness_mixed_verdicts():
    judge, prompts = stub(
        claims="Here are the claims:\n1. Paris is the capital of France.\n2. Paris has 9 million people.\n"
               "3. Paris is in Spain.",
        verdicts="1: SUPPORTED\n2: UNSUPPORTED\n3: CONTRADICTED")
    (s,) = Faithfulness(judge_model=judge).score(rag_sample(), "Paris is France's capital, ...")
    assert s.name == "faithfulness" and s.value == pytest.approx(1 / 3)
    assert s.metadata["claims"] == ["Paris is the capital of France.", "Paris has 9 million people.",
                                    "Paris is in Spain."]
    assert s.metadata["verdicts"] == ["SUPPORTED", "UNSUPPORTED", "CONTRADICTED"]
    assert s.metadata["n_contradicted"] == 1
    assert "UNSUPPORTED: Paris has 9 million people." in s.reason and "CONTRADICTED" in s.reason
    assert len(prompts) == 2
    assert "Paris is France's capital" in prompts[0] and "Tell me about Paris." in prompts[0]
    assert "1. Paris is the capital of France." in prompts[1] and "3. Paris is in Spain." in prompts[1]
    assert "2. Paris has 2.1 million people." in prompts[1]  # numbered contexts


def test_faithfulness_all_supported():
    judge, _ = stub(claims="1. A.\n2. B.", verdicts="1: SUPPORTED\n2: SUPPORTED")
    (s,) = Faithfulness(judge_model=judge).score(rag_sample(), "A. B.")
    assert s.value == 1.0 and s.reason is None


def test_faithfulness_no_claims_is_skipped():
    judge, prompts = stub(claims="NONE")
    assert Faithfulness(judge_model=judge).score(rag_sample(), "I can't answer that.") == []
    assert len(prompts) == 1


def test_faithfulness_strips_think_reasoning_before_extracting_claims():
    # qwen3 served without a reasoning parser: numbered lines inside <think> are
    # drafts, not claims; the real answer after </think> is NONE, so skip.
    judge, prompts = stub(
        claims="<think>\n1. It says it cannot help.\n2. No facts are stated.\n</think>\nNONE")
    assert Faithfulness(judge_model=judge).score(rag_sample(), "I can't help.") == []
    assert len(prompts) == 1  # no verdict call for a skipped sample


def test_faithfulness_ignores_numbered_lines_in_think_and_numbered_none():
    judge, _ = stub(
        claims="<think>\n1. First, identify what the answer states.\n2. It mentions 3 years.\n"
               "</think>\n1. Laptops are replaced every 3 years.\n2. NONE",
        verdicts="1: SUPPORTED")
    (s,) = Faithfulness(judge_model=judge).score(
        rag_sample(retrieval_context=["Laptops are replaced every 3 years."]),
        "Laptops are replaced every 3 years.")
    assert s.metadata["claims"] == ["Laptops are replaced every 3 years."]
    assert s.value == 1.0


def test_faithfulness_skips_numbered_none_claim():
    # command-r7b appends "4. NONE" after the real claims (BUG D).
    judge, _ = stub(
        claims="1. Radium was discovered in 1898.\n2. Polonium was discovered in 1898.\n"
               "3. Marie Curie was born in Warsaw.\n4. NONE",
        verdicts="1: SUPPORTED\n2: SUPPORTED\n3: UNSUPPORTED")
    (s,) = Faithfulness(judge_model=judge).score(rag_sample(), "Marie Curie facts.")
    assert s.metadata["claims"] == ["Radium was discovered in 1898.",
                                    "Polonium was discovered in 1898.",
                                    "Marie Curie was born in Warsaw."]
    assert s.value == pytest.approx(2 / 3)


def test_faithfulness_numbered_none_only_is_skipped():
    # After dropping the numbered NONE, no claims remain -> skip, not raise.
    judge, _ = stub(claims="1. NONE")
    assert Faithfulness(judge_model=judge).score(rag_sample(), "I decline.") == []


def test_faithfulness_skips_without_context_or_answer():
    judge, prompts = stub()
    assert Faithfulness(judge_model=judge).score(rag_sample(retrieval_context=None), "x") == []
    assert Faithfulness(judge_model=judge).score(rag_sample(retrieval_context=[]), "x") == []
    assert Faithfulness(judge_model=judge).score(rag_sample(), "   ") == []
    assert prompts == []


def test_faithfulness_unreadable_claims_raise():
    judge, _ = stub(claims="Paris is the capital and it is big.")
    with pytest.raises(ValueError, match="could not read claims"):
        Faithfulness(judge_model=judge).score(rag_sample(), "Paris is the capital.")


def test_faithfulness_verdict_count_mismatch_raises():
    judge, _ = stub(claims="1. A.\n2. B.\n3. C.", verdicts="1: SUPPORTED\n2: SUPPORTED")
    with pytest.raises(ValueError):
        Faithfulness(judge_model=judge).score(rag_sample(), "A. B. C.")


def test_faithfulness_strips_think_from_the_answer_before_extracting_claims():
    # The answer's own <think> reasoning is not part of the answer (finding 12):
    # it must not reach the claim-extraction prompt.
    seen: list[str] = []

    def judge(ps):
        seen.extend(ps)
        return ["1. Paris is the capital." if p.startswith("Break the ANSWER") else "1: SUPPORTED" for p in ps]

    ans = "<think>\n1. secret draft reasoning\n</think>\nParis is the capital."
    (s,) = Faithfulness(judge_model=judge).score(rag_sample(), ans)
    assert s.value == 1.0
    assert "secret draft reasoning" not in seen[0] and "Paris is the capital." in seen[0]


def test_faithfulness_uses_trace_contexts():
    judge, prompts = stub(claims="1. A.", verdicts="1: SUPPORTED")
    Faithfulness(judge_model=judge).score(rag_sample(), "A.", {"trace": {"retrieved_contexts": ["RUNTIME-CHUNK"]}})
    assert "RUNTIME-CHUNK" in prompts[1] and "capital of France" not in prompts[1]


def test_max_context_chars_truncates():
    judge, prompts = stub(claims="1. A.", verdicts="1: SUPPORTED")
    Faithfulness(judge_model=judge, max_context_chars=20).score(rag_sample(retrieval_context=["x" * 500]), "A.")
    assert "...(truncated)" in prompts[1] and "x" * 30 not in prompts[1]


def test_missing_judge_raises():
    with pytest.raises(NotImplementedError):
        Faithfulness().score(rag_sample(), "A.")


def test_identity_tracks_judge_and_settings():
    a, b = Faithfulness(judge_model="openai:a").identity(), Faithfulness(judge_model="openai:b").identity()
    assert a != b and a["name"] == "faithfulness"
    assert Faithfulness(judge_model="openai:a", max_context_chars=100).identity() != a
    assert ContextRecall(judge_model="openai:a").identity()["name"] == "context_recall"


# -- ContextPrecision ---------------------------------------------------------

def test_context_precision_ap_and_relevance():
    judge, prompts = stub(precision="1: RELEVANT\n2: IRRELEVANT\n3: RELEVANT")
    s = rag_sample(retrieval_context=["c1", "c2", "c3"], target="Paris")
    got = {x.name: x.value for x in ContextPrecision(judge_model=judge).score(s, "ans")}
    assert got == pytest.approx({"context_precision": (1 / 1 + 2 / 3) / 2, "context_relevance": 2 / 3})
    assert "<reference_answer>\nParis\n</reference_answer>" in prompts[0]
    assert "REFERENCE ANSWER" in prompts[0] and "3. c3" in prompts[0]


def test_context_precision_without_target_and_all_irrelevant():
    judge, prompts = stub(precision="1: IRRELEVANT\n2: IRRELEVANT")
    got = {x.name: x.value for x in ContextPrecision(judge_model=judge).score(rag_sample(), "ans")}
    assert got == {"context_precision": 0.0, "context_relevance": 0.0}
    assert "<reference_answer>" not in prompts[0] and "{with_ref}" not in prompts[0]


def test_context_precision_relevant_first_is_perfect():
    judge, _ = stub(precision="1: RELEVANT\n2: RELEVANT\n3: IRRELEVANT")
    out = ContextPrecision(judge_model=judge).score(rag_sample(retrieval_context=["a", "b", "c"]), "x")
    assert out[0].value == 1.0 and out[1].value == pytest.approx(2 / 3)


# -- ContextRecall ------------------------------------------------------------

def test_context_recall_sentence_split_and_attribution():
    judge, prompts = stub(recall="1: ATTRIBUTED\n2: NOT_ATTRIBUTED\n3: attributed\n4: Attributed")
    s = rag_sample(target="Paris is the capital. It has 2.1 million people! Is it big? Yes")
    (score,) = ContextRecall(judge_model=judge).score(s, "")
    assert score.metadata["sentences"] == ["Paris is the capital.", "It has 2.1 million people!",
                                           "Is it big?", "Yes"]
    assert score.value == 0.75 and "It has 2.1 million people!" in score.reason
    assert "4. Yes" in prompts[0]


def test_context_recall_needs_target():
    m = ContextRecall(judge_model=stub()[0])
    assert not m.applicable(rag_sample())
    assert m.score(rag_sample(target="   "), "") == []
    assert m.score(rag_sample(target="A.", retrieval_context=[]), "") == []


# -- end to end ---------------------------------------------------------------

def test_evaluate_precomputed_rag_judges(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    judge, _ = stub(claims="1. A.\n2. B.", verdicts="1: SUPPORTED\n2: UNSUPPORTED",
                    precision="1: RELEVANT\n2: IRRELEVANT", recall="1: ATTRIBUTED")
    samples = [rag_sample(target="Paris is the capital.", actual_output="A. B.",
                          actual_trace={"retrieved_contexts": ["r1", "r2"]})]
    r = ak.evaluate(samples, model="precomputed",
                    scorers=[Faithfulness(judge_model=judge), ContextPrecision(judge_model=judge),
                             ContextRecall(judge_model=judge)])
    assert r.errors == []
    assert r.headline == pytest.approx({"faithfulness": 0.5, "context_precision": 1.0,
                                        "context_relevance": 0.5, "context_recall": 1.0})


def test_evaluate_records_parse_failure_instead_of_scoring_zero(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    judge, _ = stub(claims="1. A.\n2. B.\n3. C.", verdicts="1: SUPPORTED\n2: SUPPORTED")
    samples = [rag_sample(actual_output="A. B. C.")]
    r = ak.evaluate(samples, model="precomputed", scorers=[Faithfulness(judge_model=judge)])
    assert len(r.errors) == 1 and r.errors[0]["metric"] == "faithfulness"
    assert "missing [3]" in r.errors[0]["error"]
    assert "faithfulness" not in r.stats and "faithfulness" not in r.headline


def test_none_word_inside_prose_does_not_skip_and_bare_function_judge_works():
    from auditkit.metrics.rag_judge import Faithfulness
    from auditkit.sample import Sample
    m = Faithfulness(judge_model=lambda prompts: ["Nonetheless, I cannot list claims." for _ in prompts])
    with pytest.raises(ValueError):
        m.score(Sample(input="q", retrieval_context=["c"]), "Paris is in France.")
    m = Faithfulness(judge_model=lambda prompts: ["NONE" for _ in prompts])
    assert m.score(Sample(input="q", retrieval_context=["c"]), "I don't know.") == []


# -- stress regressions -------------------------------------------------------

def _lenient_judge(prompts):
    return ["1. Claim one." if p.startswith("Break the ANSWER") else "1: SUPPORTED" for p in prompts]


def _strict_judge(prompts):
    return ["1. Claim one." if p.startswith("Break the ANSWER") else "1: UNSUPPORTED" for p in prompts]


def test_cache_does_not_serve_another_judges_scores(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    samples = [rag_sample(actual_output="A.")]
    a = ak.evaluate(samples, model="precomputed", scorers=[Faithfulness(judge_model=_lenient_judge)])
    b = ak.evaluate(samples, model="precomputed", scorers=[Faithfulness(judge_model=_strict_judge)])
    assert a.headline == {"faithfulness": 1.0} and b.headline == {"faithfulness": 0.0}
    assert a.fingerprint != b.fingerprint
    # The same function again is a stable identity, so a repeat run can replay from cache.
    assert Faithfulness(judge_model=_lenient_judge).identity() == Faithfulness(judge_model=_lenient_judge).identity()


def test_identity_hashes_judge_model_args():
    a = Faithfulness(judge_model="api:x", judge_model_args={"api_base": "http://h/lenient"}).identity()
    b = Faithfulness(judge_model="api:x", judge_model_args={"api_base": "http://h/strict"}).identity()
    assert a != b


def test_callable_judge_without_stable_identity_is_never_cached():
    class Judge:
        def __call__(self, prompts):
            return ["1: SUPPORTED"] * len(prompts)

    j = Judge()
    assert Faithfulness(judge_model=j).identity() != Faithfulness(judge_model=j).identity()
    client = object()  # an opaque closure capture can't be identified

    def via_client(prompts):
        return [str(client)] * len(prompts)

    assert Faithfulness(judge_model=via_client).identity() != Faithfulness(judge_model=via_client).identity()


def test_parsers_are_linear_on_long_whitespace_runs():
    import time
    judge, _ = stub(claims="1. Placeholder claim" + " " * 100_000 + "end.", verdicts="1:" + " " * 100_000 + "#")
    t0 = time.perf_counter()
    with pytest.raises(ValueError):
        Faithfulness(judge_model=judge).score(rag_sample(), "A.")
    assert time.perf_counter() - t0 < 1.0


def test_conflicting_duplicate_verdicts_raise():
    with pytest.raises(ValueError, match="conflicting verdicts for item 2"):
        parse_verdicts("1: SUPPORTED\n2: UNSUPPORTED\n3: SUPPORTED\n2: SUPPORTED", 3, FAITH)
    assert parse_verdicts("1: SUPPORTED\n1: SUPPORTED", 1, FAITH) == ["SUPPORTED"]


def test_context_precision_with_max_context_chars_keeps_every_chunk():
    judge, prompts = stub(precision="1: RELEVANT\n2: IRRELEVANT\n3: RELEVANT")
    out = ContextPrecision(judge_model=judge, max_context_chars=60).score(
        rag_sample(retrieval_context=["a" * 200, "b" * 200, "c" * 200]), "")
    assert {s.name: s.value for s in out}["context_relevance"] == pytest.approx(2 / 3)
    assert all(f"{i}. " in prompts[0] for i in (1, 2, 3)) and "a" * 30 not in prompts[0]


# --- the RAG verdict judges share the 600 s api: judge timeout ------------------
#
# The LLM judges got a 600 s default for api: models; the RAG verdict judges
# resolved their model from the raw args and kept the backend's 120 s, which
# timed out a reasoning judge on faithfulness.

@pytest.mark.parametrize("cls", [ak.Faithfulness, ak.ContextPrecision, ak.ContextRecall])
def test_rag_judges_get_the_long_default_timeout(cls, monkeypatch):
    monkeypatch.delenv("AUDITKIT_JUDGE_TIMEOUT", raising=False)
    m = cls(judge_model="api:x", judge_model_args={"api_base": "http://localhost:8000/v1", "api_key": "k"})
    assert m._model().timeout == 600.0


def test_rag_judge_timeout_env_and_explicit_override(monkeypatch):
    monkeypatch.setenv("AUDITKIT_JUDGE_TIMEOUT", "30")
    assert ak.Faithfulness(judge_model="api:x", judge_model_args={"api_key": "k"})._model().timeout == 30.0
    m = ak.Faithfulness(judge_model="api:x", judge_model_args={"api_key": "k", "timeout": 45})
    assert m._model().timeout == 45


def test_rag_judge_timeout_is_not_identity():
    a = ak.Faithfulness(judge_model="api:x", judge_model_args={"api_base": "http://h/v1"}).identity()
    b = ak.Faithfulness(judge_model="api:x", judge_model_args={"api_base": "http://h/v1", "timeout": 5}).identity()
    assert a == b
