"""Tests for RegexAnnotator and the extraction-wiring it needs to actually
reach scoring: Runner.score_one()'s extracted_by, the 3 Runner.run()
branches synthesizing real `results` for annotators, _to_annotators()
coercion, and RunSpec.fingerprint() sensitivity to annotators/extracted_by.
"""
from __future__ import annotations

import auditkit as ak
from auditkit.adapter import GenerationAdapter
from auditkit.annotator import Annotator, LLMAnnotator, RegexAnnotator, ThinkingStripAnnotator
from auditkit.api import _to_annotators
from auditkit.errors import AuditKitError
from auditkit.metric import ExactMatch
from auditkit.model import CallableModel, Generated, Result_
from auditkit.runner import Runner
from auditkit.runspec import RunSpec
from auditkit.sample import Sample
from auditkit.scenario import ListScenario


# ---------------------------------------------------------------------------
# RegexAnnotator in isolation
# ---------------------------------------------------------------------------
class TestRegexAnnotatorInIsolation:
    def _results(self, text: str) -> list[Result_]:
        return [Result_(completions=[Generated(text=text)])]

    def test_default_group_extracts_whole_match(self):
        ann = RegexAnnotator(r"\d+")
        ctx = ann.annotate(Sample(input="q"), self._results("the answer is 42 today"))
        assert ctx == {"extracted": "42", "matched": True, "raw": "the answer is 42 today"}

    def test_capture_group_extracts_just_the_group(self):
        ann = RegexAnnotator(r"FINAL ANSWER:\s*(\d+)", group=1)
        ctx = ann.annotate(Sample(input="q"), self._results("reasoning...\nFINAL ANSWER: 42"))
        assert ctx["extracted"] == "42"
        assert ctx["matched"] is True

    def test_no_match_falls_back_to_on_no_match(self):
        ann = RegexAnnotator(r"FINAL ANSWER:\s*(\d+)", group=1, on_no_match="UNKNOWN")
        ctx = ann.annotate(Sample(input="q"), self._results("I have no idea"))
        assert ctx == {"extracted": "UNKNOWN", "matched": False, "raw": "I have no idea"}

    def test_no_match_default_on_no_match_is_empty_string(self):
        ann = RegexAnnotator(r"\d+")
        ctx = ann.annotate(Sample(input="q"), self._results("no digits here"))
        assert ctx["extracted"] == ""
        assert ctx["matched"] is False

    def test_empty_results_treated_as_empty_text(self):
        ann = RegexAnnotator(r"\d+", on_no_match="none")
        ctx = ann.annotate(Sample(input="q"), [])
        assert ctx == {"extracted": "none", "matched": False, "raw": ""}

    def test_flags_are_honored(self):
        import re
        ann = RegexAnnotator(r"final answer:\s*(\w+)", group=1, flags=re.IGNORECASE)
        ctx = ann.annotate(Sample(input="q"), self._results("FINAL ANSWER: yes"))
        assert ctx["extracted"] == "yes"

    def test_matched_group_that_did_not_participate_falls_back(self):
        ann = RegexAnnotator(r"(a)|(b)", group=2, on_no_match="MISSING")
        ctx = ann.annotate(Sample(input="q"), self._results("a"))
        assert ctx["matched"] is True
        assert ctx["extracted"] == "MISSING"

    def test_custom_name(self):
        ann = RegexAnnotator(r"\d+", name="my_extractor")
        assert ann.name == "my_extractor"

    def test_identity_reflects_config_not_just_type(self):
        a = RegexAnnotator(r"\d+", group=0)
        b = RegexAnnotator(r"[a-z]+", group=0)
        assert a.identity() != b.identity()

    def test_registered_in_annotators_registry(self):
        from auditkit.registry import ANNOTATORS
        assert "regex" in ANNOTATORS.names()
        assert isinstance(ANNOTATORS.get("regex")(pattern=r"\d+"), RegexAnnotator)


# ---------------------------------------------------------------------------
# ThinkingStripAnnotator in isolation
#
# The fixture text below (REAL_QWEN_TRIVIA_RESPONSE) is the actual raw text
# returned by Groq's qwen/qwen3.6-27b for "capital of France" with a
# "write ANSWER: <value> on the final line" system prompt (captured live
# during real testing) -- reasoning models like this one embed their whole
# <think>...</think> draft inline in the same content field as the real
# answer, and that draft mentions the target pattern several times before
# the real final line. A plain RegexAnnotator's re.search() would match the
# FIRST "ANSWER:" occurrence -- still inside the reasoning -- not the real
# one after </think>.
# ---------------------------------------------------------------------------
REAL_QWEN_TRIVIA_RESPONSE = (
    "\n<think>\nThinking Process:\n\n1.  **Identify the user's question:** "
    "The user is asking for the \"capital of France\".\n2.  **Retrieve "
    "knowledge:** Access geographical knowledge about France.\n3.  "
    "**Identify the capital:** The capital of France is Paris.\n4.  "
    "**Format the output:** The user requested the answer directly and "
    "specifically asked for the final line to be exactly `ANSWER: <value>`."
    "\n5.  **Draft the response:**\n    *   Direct answer: Paris.\n    *   "
    "Final line: ANSWER: Paris\n6.  **Final Review:** Does the response "
    "meet all constraints? Yes.\n\n*Self-Correction/Refinement:* The user "
    "said \"Answer directly.\" so I should just provide the name and then "
    "the formatted line.\n\nDraft:\nParis\n\nANSWER: Paris\n\nWait, usually "
    "\"Answer directly\" implies a short sentence or just the fact. I will "
    "provide the fact and then the required footer.\n\nFinal Output "
    "Construction:\nParis\n\nANSWER: Paris\n</think>\n\nParis\n\nANSWER: Paris"
)


class TestThinkingStripAnnotator:
    def _results(self, text: str) -> list[Result_]:
        return [Result_(completions=[Generated(text=text)])]

    def test_plain_regex_would_still_match_first_occurrence_inside_thinking(self):
        # Establishes the bug this annotator exists to fix: a plain
        # RegexAnnotator's re.search() finds the FIRST "ANSWER:" -- which
        # happens to also say "Paris" here (by luck, since every draft in
        # this transcript agrees), so this doesn't itself prove a wrong
        # answer, only that it's searching inside the reasoning at all.
        ann = RegexAnnotator(r"ANSWER:\s*([^\n]+)", group=1)
        ctx = ann.annotate(Sample(input="q"), self._results(REAL_QWEN_TRIVIA_RESPONSE))
        assert ctx["matched"] is True
        # The FIRST match is the one inside "4. **Format the output:**"'s
        # description, several lines before the real final line.
        first_ANSWER_pos = REAL_QWEN_TRIVIA_RESPONSE.index("ANSWER:")
        last_ANSWER_pos = REAL_QWEN_TRIVIA_RESPONSE.rindex("ANSWER:")
        assert first_ANSWER_pos != last_ANSWER_pos  # multiple occurrences exist to disambiguate

    def test_strips_thinking_and_takes_the_real_final_answer(self):
        ann = ThinkingStripAnnotator(r"ANSWER:\s*([^\n]+)", group=1)
        ctx = ann.annotate(Sample(input="q"), self._results(REAL_QWEN_TRIVIA_RESPONSE))
        assert ctx == {"extracted": "Paris", "matched": True, "raw": REAL_QWEN_TRIVIA_RESPONSE}

    def test_picks_last_match_when_answer_changes_after_thinking(self):
        # A case where stripping <think> actually changes the outcome (not
        # just "happens to agree" like the real transcript above): the draft
        # explores a wrong answer, self-corrects, and the real final line
        # differs from what's mentioned mid-reasoning.
        text = (
            "<think>\nMaybe it's 70000... no wait, let me recompute.\n"
            "ANSWER: 70000\nActually that's wrong, recalculating: 75 * 1000 = 75000.\n"
            "ANSWER: 75000\n</think>\nANSWER: 75000"
        )
        ann = ThinkingStripAnnotator(r"ANSWER:\s*([^\n]+)", group=1)
        ctx = ann.annotate(Sample(input="q"), self._results(text))
        assert ctx["extracted"] == "75000"
        # Confirm a plain RegexAnnotator would have grabbed the wrong,
        # early draft value instead -- this is the concrete bug fixed.
        plain = RegexAnnotator(r"ANSWER:\s*([^\n]+)", group=1)
        wrong_ctx = plain.annotate(Sample(input="q"), self._results(text))
        assert wrong_ctx["extracted"] == "70000"

    def test_trims_trailing_backtick_and_parenthetical_noise(self):
        # Real noise pattern seen live: a model restates the value with
        # commentary after it, past the actual final answer.
        text = "ANSWER: 18` (or `$18`, but usually just the number is fine.)"
        ann = ThinkingStripAnnotator(r"ANSWER:\s*([^\n]+)", group=1)
        ctx = ann.annotate(Sample(input="q"), self._results(text))
        assert ctx["extracted"] == "18"

    def test_trim_trailing_noise_can_be_disabled(self):
        text = "ANSWER: 18` (extra)"
        ann = ThinkingStripAnnotator(r"ANSWER:\s*([^\n]+)", group=1, trim_trailing_noise=False)
        ctx = ann.annotate(Sample(input="q"), self._results(text))
        assert ctx["extracted"] == "18` (extra)"

    def test_no_match_falls_back_to_on_no_match(self):
        ann = ThinkingStripAnnotator(r"ANSWER:\s*([^\n]+)", group=1, on_no_match="UNKNOWN")
        ctx = ann.annotate(Sample(input="q"), self._results("<think>no final answer given</think>"))
        assert ctx == {"extracted": "UNKNOWN", "matched": False, "raw": "<think>no final answer given</think>"}

    def test_no_thinking_block_still_works_like_plain_regex(self):
        ann = ThinkingStripAnnotator(r"ANSWER:\s*([^\n]+)", group=1)
        ctx = ann.annotate(Sample(input="q"), self._results("Just a plain reply.\nANSWER: 42"))
        assert ctx["extracted"] == "42"

    def test_custom_strip_pattern(self):
        ann = ThinkingStripAnnotator(r"ANSWER:\s*([^\n]+)", group=1, strip_pattern=r"\[reasoning\].*?\[/reasoning\]")
        ctx = ann.annotate(Sample(input="q"), self._results("[reasoning]ANSWER: wrong[/reasoning]\nANSWER: right"))
        assert ctx["extracted"] == "right"

    def test_identity_reflects_config(self):
        a = ThinkingStripAnnotator(r"\d+")
        b = ThinkingStripAnnotator(r"[a-z]+")
        assert a.identity() != b.identity()

    def test_registered_in_annotators_registry(self):
        from auditkit.registry import ANNOTATORS
        assert "thinking_strip" in ANNOTATORS.names()
        assert isinstance(ANNOTATORS.get("thinking_strip")(pattern=r"\d+"), ThinkingStripAnnotator)


class TestRegexAnnotatorCast:
    def _results(self, text: str) -> list[Result_]:
        return [Result_(completions=[Generated(text=text)])]

    def test_no_cast_keeps_string_unchanged_default_behavior(self):
        ann = RegexAnnotator(r"\d+")
        ctx = ann.annotate(Sample(input="q"), self._results("the answer is 42"))
        assert ctx["extracted"] == "42"
        assert isinstance(ctx["extracted"], str)
        assert "cast_failed" not in ctx  # key absent, not present-and-False

    def test_cast_int_success(self):
        ann = RegexAnnotator(r"FINAL ANSWER:\s*(\d+)", group=1, cast=int)
        ctx = ann.annotate(Sample(input="q"), self._results("reasoning... FINAL ANSWER: 42"))
        assert ctx["extracted"] == 42
        assert isinstance(ctx["extracted"], int)

    def test_cast_float_success(self):
        ann = RegexAnnotator(r"score:\s*([\d.]+)", group=1, cast=float)
        ctx = ann.annotate(Sample(input="q"), self._results("score: 3.14"))
        assert ctx["extracted"] == 3.14
        assert isinstance(ctx["extracted"], float)

    def test_cast_custom_callable(self):
        ann = RegexAnnotator(r"\w+", cast=str.upper)
        ctx = ann.annotate(Sample(input="q"), self._results("hello"))
        assert ctx["extracted"] == "HELLO"

    def test_cast_failure_degrades_to_original_string_not_strict_by_default(self):
        ann = RegexAnnotator(r"ANSWER:\s*(\w+)", group=1, cast=int)
        ctx = ann.annotate(Sample(input="q"), self._results("ANSWER: banana"))
        assert ctx["extracted"] == "banana"  # original string, not crashed
        assert isinstance(ctx["extracted"], str)
        assert ctx["cast_failed"] is True
        assert ctx["matched"] is True  # the regex DID match -- only the cast failed

    def test_cast_failure_raises_when_strict(self):
        ann = RegexAnnotator(r"ANSWER:\s*(\w+)", group=1, cast=int, strict=True)
        try:
            ann.annotate(Sample(input="q"), self._results("ANSWER: banana"))
            assert False, "expected ValueError"
        except ValueError as e:
            assert "banana" in str(e)

    def test_no_match_never_attempts_cast(self):
        # on_no_match fallback should not be run through cast -- it's a
        # sentinel string, not extracted data.
        ann = RegexAnnotator(r"\d+", cast=int, on_no_match="N/A")
        ctx = ann.annotate(Sample(input="q"), self._results("no digits here"))
        assert ctx["extracted"] == "N/A"
        assert ctx["matched"] is False
        assert "cast_failed" not in ctx

    def test_identity_reflects_cast_and_strict(self):
        a = RegexAnnotator(r"\d+", cast=int)
        b = RegexAnnotator(r"\d+", cast=float)
        c = RegexAnnotator(r"\d+")
        assert a.identity()["cast"] == "int"
        assert b.identity()["cast"] == "float"
        assert c.identity()["cast"] is None
        assert a.identity() != b.identity() != c.identity()

    def test_cast_value_reaches_scoring_through_extract_with(self):
        # End-to-end: a cast int must still reach the metric correctly via
        # the extract_with wiring (Runner never re-stringifies it).
        class IntEqualsFive(ExactMatch):
            def score(self, sample, output, context=None):
                from auditkit.score import Score
                return Score(name="int_equals_five", value=1.0 if output == 5 else 0.0)

        model = CallableModel(lambda prompts: ["reasoning... FINAL ANSWER: 5" for _ in prompts])
        result = ak.evaluate(
            [Sample(input="q", target="5")], model=model, scorers=[IntEqualsFive()],
            annotators=RegexAnnotator(r"FINAL ANSWER:\s*(\d+)", group=1, cast=int, name="regex"),
            extract_with="regex",
        )
        pred = result.predictions[0]
        assert pred.parsed_answer == 5
        assert isinstance(pred.parsed_answer, int)
        assert result.headline["int_equals_five"] == 1.0


# ---------------------------------------------------------------------------
# Runner.score_one() -- the extraction reaching scoring
# ---------------------------------------------------------------------------
class TestScoreOneExtractionWiring:
    def test_no_extracted_by_scores_raw_output_unchanged(self):
        # Regression guard: default behavior must be bit-for-bit identical
        # to before this feature existed.
        runner = Runner()
        sample = Sample(input="q", target="42")
        scores, pred = runner.score_one(
            [ExactMatch()], sample, "42", context={}, run_id="r",
        )
        assert pred.raw_output == "42"
        assert pred.parsed_answer == "42"
        assert scores[0].value == 1.0

    def test_extracted_by_scores_the_extracted_value_not_raw(self):
        runner = Runner()
        sample = Sample(input="q", target="42")
        context = {"regex": {"extracted": "42", "matched": True, "raw": "..."}}
        scores, pred = runner.score_one(
            [ExactMatch()], sample, "long reasoning text FINAL ANSWER: 42",
            context=context, run_id="r", extracted_by="regex",
        )
        assert pred.raw_output == "long reasoning text FINAL ANSWER: 42"
        assert pred.parsed_answer == "42"
        assert scores[0].value == 1.0  # scored against "42", not the raw text

    def test_prediction_context_exposes_annotator_output_for_post_run_inspection(self):
        # Prediction.context existed as a field but score_one() never
        # populated it -- same class of dead-field issue as parsed_answer
        # was before extraction wiring. Now it lets you inspect, per sample
        # after a full run, exactly what each annotator saw/extracted --
        # e.g. finding every sample where a RegexAnnotator's pattern didn't
        # match, alongside the real raw_output, without re-running anything.
        runner = Runner()
        sample = Sample(input="q", target="42")
        context = {"regex": {"extracted": "42", "matched": True, "raw": "..."}}
        _, pred = runner.score_one(
            [ExactMatch()], sample, "42", context=context, run_id="r",
        )
        assert pred.context == context
        assert pred.context["regex"]["matched"] is True

    def test_unknown_extracted_by_name_falls_back_to_raw_no_crash(self):
        runner = Runner()
        sample = Sample(input="q", target="42")
        scores, pred = runner.score_one(
            [ExactMatch()], sample, "42", context={}, run_id="r",
            extracted_by="does_not_exist",
        )
        assert pred.parsed_answer == "42"
        assert scores[0].value == 1.0

    def test_annotator_entry_without_extracted_key_falls_back_to_raw(self):
        runner = Runner()
        sample = Sample(input="q", target="42")
        context = {"some_other_annotator": {"n_results": 1}}
        scores, pred = runner.score_one(
            [ExactMatch()], sample, "42", context=context, run_id="r",
            extracted_by="some_other_annotator",
        )
        assert pred.parsed_answer == "42"


# ---------------------------------------------------------------------------
# Runner.run() -- all 3 branches actually deliver real text to annotators
# and honor extracted_by end to end
# ---------------------------------------------------------------------------
class _MarkerCallableModel(CallableModel):
    """Always answers with a reasoning chain ending in a FINAL ANSWER: marker."""

    def __init__(self):
        super().__init__(lambda prompts: ["Let's see... FINAL ANSWER: 42" for _ in prompts])


class TestRunnerRunExtractionAllBranches:
    def _regex(self):
        return RegexAnnotator(r"FINAL ANSWER:\s*(\d+)", group=1, name="regex")

    def test_generative_branch(self):
        spec = RunSpec(
            scenario=ListScenario([Sample(input="q", target="42")]),
            model=_MarkerCallableModel(),
            adapter=GenerationAdapter(),
            metrics=[ExactMatch()],
            annotators=[self._regex()],
            extracted_by="regex",
        )
        result = Runner().run(spec)
        pred = result.predictions[0]
        assert "FINAL ANSWER: 42" in pred.raw_output
        assert pred.parsed_answer == "42"
        assert pred.correct is True

    def test_precomputed_reads_actual_output_branch(self):
        from auditkit.model import PrecomputedModel
        from dataclasses import replace

        sample = replace(Sample(input="q", target="42"),
                          actual_output="Let's see... FINAL ANSWER: 42")
        spec = RunSpec(
            scenario=ListScenario([sample]),
            model=PrecomputedModel(),
            adapter=GenerationAdapter(),
            metrics=[ExactMatch()],
            annotators=[self._regex()],
            extracted_by="regex",
        )
        result = Runner().run(spec)
        pred = result.predictions[0]
        assert "FINAL ANSWER: 42" in pred.raw_output
        assert pred.parsed_answer == "42"
        assert pred.correct is True

    def test_loglikelihood_branch_annotator_sees_the_picked_choice_text(self):
        from auditkit.adapter import MCQAdapter
        from tests.test_benchmark import PreferentialLogLikelihoodModel

        # Neutral prompt (no "right" substring) -- PreferentialLogLikelihoodModel
        # also prefers whichever choice's own request.params["target"] literally
        # equals "correct", which is what actually disambiguates the two
        # choices here (both choice requests share the same sample.input as
        # their prompt, so a prompt-based trigger can't distinguish them).
        sample = Sample(input="pick one", choices=["wrong", "correct"], target=1)
        spec = RunSpec(
            scenario=ListScenario([sample]),
            model=PreferentialLogLikelihoodModel(),
            adapter=MCQAdapter(method="mcq_loglikelihood"),
            metrics=[],
            annotators=[RegexAnnotator(r"correct", name="regex")],
        )
        result = Runner().run(spec)
        # Real assertion, now that Prediction.context is populated: confirm
        # the annotator actually received and matched against the real
        # picked-choice text ("correct"), not an empty string.
        pred = result.predictions[0]
        assert pred.raw_output == "correct"
        assert pred.context["regex"]["matched"] is True
        assert pred.context["regex"]["extracted"] == "correct"


# ---------------------------------------------------------------------------
# _to_annotators() coercion
# ---------------------------------------------------------------------------
class TestToAnnotatorsCoercion:
    def test_none_returns_empty_list(self):
        assert _to_annotators(None) == []

    def test_single_instance_wrapped_in_list(self):
        ann = RegexAnnotator(r"\d+")
        assert _to_annotators(ann) == [ann]

    def test_registered_name_string_resolves_zero_arg_annotator(self):
        # RegexAnnotator requires `pattern` (no sensible default), so a bare
        # name string can never construct it -- string resolution only works
        # for annotators with a zero-arg-friendly constructor, same
        # limitation _to_adapter()/_to_metrics() have for their registries.
        class _EchoAnnotator(Annotator):
            name = "echo_test_annotator"
            def annotate(self, sample, results):
                return {"n_results": len(results)}
        from auditkit.registry import ANNOTATORS
        ANNOTATORS.register("echo_test_annotator")(_EchoAnnotator)
        result = _to_annotators("echo_test_annotator")
        assert len(result) == 1
        assert isinstance(result[0], _EchoAnnotator)

    def test_regex_annotator_resolves_by_bare_name(self):
        # `pattern` used to be mandatory, which made every bare registry name
        # unusable -- and because _to_annotators() runs inside the run, it raised
        # only after the job had been submitted. It now defaults to match-all, so
        # the bare name resolves to an identity extractor.
        result = _to_annotators("regex")
        assert len(result) == 1
        assert isinstance(result[0], RegexAnnotator)

    def test_bare_regex_extracts_the_whole_output(self):
        ann = RegexAnnotator()
        out = ann.annotate(Sample(input="x"), [Result_(completions=[Generated(text="hello world")])])
        assert out["matched"] is True
        assert out["extracted"] == "hello world"

    def test_bare_thinking_strip_removes_reasoning_and_keeps_the_rest(self):
        # The useful default: strip <think>...</think>, keep everything after it.
        ann = ThinkingStripAnnotator()
        out = ann.annotate(
            Sample(input="x"), [Result_(completions=[Generated(text="<think>12 times 12 is 144</think>144")])]
        )
        assert out["matched"] is True
        assert out["extracted"] == "144"

    def test_bare_thinking_strip_spans_newlines(self):
        ann = ThinkingStripAnnotator()
        out = ann.annotate(
            Sample(input="x"), [Result_(completions=[Generated(text="<think>a\nb</think>final answer")])]
        )
        assert out["extracted"] == "final answer"

    def test_explicit_pattern_still_overrides_the_default(self):
        ann = RegexAnnotator(r"Answer:\s*(.+)", group=1)
        out = ann.annotate(Sample(input="x"), [Result_(completions=[Generated(text="Answer: 42")])])
        assert out["extracted"] == "42"
        assert ann.identity()["pattern"] == r"Answer:\s*(.+)"

    def test_llm_annotator_still_cannot_be_resolved_by_bare_name(self):
        # There is no sensible default model, so this one stays unresolvable --
        # and it must keep failing loudly rather than picking a model silently.
        try:
            _to_annotators("llm")
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_list_of_mixed_items_flattens(self):
        ann = RegexAnnotator(r"\d+")
        ann2 = RegexAnnotator(r"[a-z]+")
        result = _to_annotators([ann, ann2])
        assert len(result) == 2
        assert all(isinstance(a, Annotator) for a in result)

    def test_unknown_name_raises_auditkit_error(self):
        try:
            _to_annotators("does_not_exist")
            assert False, "expected AuditKitError"
        except AuditKitError as e:
            assert "does_not_exist" in str(e)


# ---------------------------------------------------------------------------
# evaluate() actually reaches this -- the real public-API gap being closed
# ---------------------------------------------------------------------------
class TestEvaluateAnnotatorsReachable:
    def test_annotators_and_extract_with_work_through_evaluate(self):
        samples = [Sample(input="q", target="42")]
        result = ak.evaluate(
            samples, model=_MarkerCallableModel(), scorers="exact_match",
            annotators=RegexAnnotator(r"FINAL ANSWER:\s*(\d+)", group=1, name="regex"),
            extract_with="regex",
        )
        pred = result.predictions[0]
        assert "FINAL ANSWER: 42" in pred.raw_output
        assert pred.parsed_answer == "42"
        assert result.headline["exact_match"] == 1.0

    def test_no_annotators_still_works_as_before(self):
        samples = [Sample(input="q", target="q")]
        result = ak.evaluate(samples, model="echo", scorers="exact_match")
        assert result.headline["exact_match"] == 1.0


# ---------------------------------------------------------------------------
# RunSpec.fingerprint() sensitivity
# ---------------------------------------------------------------------------
class TestFingerprintSensitivity:
    def _fp(self, annotators=None, extracted_by=None):
        return RunSpec(
            scenario=ListScenario([Sample(input="q", target="a")]),
            model=CallableModel(lambda prompts: prompts),
            adapter=GenerationAdapter(),
            metrics=[ExactMatch()],
            annotators=annotators or [],
            extracted_by=extracted_by,
        ).fingerprint()

    def test_different_annotators_differ(self):
        a = self._fp(annotators=[RegexAnnotator(r"\d+")])
        b = self._fp(annotators=[RegexAnnotator(r"[a-z]+")])
        assert a != b

    def test_presence_of_annotator_differs_from_none(self):
        a = self._fp()
        b = self._fp(annotators=[RegexAnnotator(r"\d+")])
        assert a != b

    def test_different_extracted_by_differs(self):
        a = self._fp(extracted_by="regex")
        b = self._fp(extracted_by="other")
        assert a != b

    def test_same_config_same_fingerprint(self):
        assert self._fp(annotators=[RegexAnnotator(r"\d+")], extracted_by="regex") == \
            self._fp(annotators=[RegexAnnotator(r"\d+")], extracted_by="regex")


# ---------------------------------------------------------------------------
# LLMAnnotator -- same model-resolution machinery as LLMJudge, but for
# extraction instead of scoring
# ---------------------------------------------------------------------------
class TestLLMAnnotatorInIsolation:
    def _results(self, text: str) -> list[Result_]:
        return [Result_(completions=[Generated(text=text)])]

    def test_calls_model_with_rendered_prompt_and_returns_reply(self):
        seen = {}

        def fake(prompts):
            seen["prompt"] = prompts[0]
            return ["42"]

        ann = LLMAnnotator(model=CallableModel(fake), prompt="Extract from: {output}", name="llm_ex")
        ctx = ann.annotate(Sample(input="q", target="x"), self._results("reasoning... FINAL ANSWER: 42"))
        assert ctx == {"extracted": "42", "matched": True, "raw": "42"}
        assert "reasoning... FINAL ANSWER: 42" in seen["prompt"]

    def test_prompt_fields_are_filled(self):
        seen = {}

        def fake(prompts):
            seen["prompt"] = prompts[0]
            return ["ok"]

        ann = LLMAnnotator(
            model=CallableModel(fake), name="llm_ex",
            prompt="in={input} out={output} exp={expected} ctx={context}",
        )
        sample = Sample(input="q1", target="t1", retrieval_context=["doc1", "doc2"])
        ann.annotate(sample, self._results("model said this"))
        assert seen["prompt"] == "in=q1 out=model said this exp=t1 ctx=doc1\ndoc2"

    def test_system_prompt_is_prepended(self):
        seen = {}

        def fake(prompts):
            seen["prompt"] = prompts[0]
            return ["ok"]

        ann = LLMAnnotator(model=CallableModel(fake), system_prompt="You are a strict extractor.",
                            prompt="{output}", name="llm_ex")
        ann.annotate(Sample(input="q"), self._results("text"))
        assert seen["prompt"].startswith("You are a strict extractor.")

    def test_empty_reply_falls_back_to_on_empty(self):
        ann = LLMAnnotator(model=CallableModel(lambda p: [""]), prompt="{output}",
                            name="llm_ex", on_empty="NONE")
        ctx = ann.annotate(Sample(input="q"), self._results("text"))
        assert ctx == {"extracted": "NONE", "matched": False, "raw": ""}

    def test_reply_is_stripped(self):
        ann = LLMAnnotator(model=CallableModel(lambda p: ["  42  \n"]), prompt="{output}", name="llm_ex")
        ctx = ann.annotate(Sample(input="q"), self._results("text"))
        assert ctx["extracted"] == "42"

    def test_string_model_spec_resolves_via_auto_model(self, monkeypatch):
        # Doesn't hit a real backend -- just confirms the same
        # AutoModel.resolve(spec, **model_args) path LLMJudge uses is taken
        # for a string model spec, without needing network/HF access.
        from auditkit.model import EchoModel

        import auditkit.model as model_mod
        monkeypatch.setattr(model_mod.AutoModel, "resolve", staticmethod(lambda spec, **kw: EchoModel()))
        ann = LLMAnnotator(model="fake:spec", prompt="{output}", name="llm_ex")
        ctx = ann.annotate(Sample(input="q"), self._results("echo this"))
        assert ctx["extracted"] == "echo this"  # EchoModel echoes the rendered prompt back
        assert ctx["matched"] is True


class TestLLMAnnotatorCast:
    def _results(self, text: str) -> list[Result_]:
        return [Result_(completions=[Generated(text=text)])]

    def test_cast_converts_reply(self):
        ann = LLMAnnotator(model=CallableModel(lambda p: ["42"]), prompt="{output}",
                            name="llm_ex", cast=int)
        ctx = ann.annotate(Sample(input="q"), self._results("text"))
        assert ctx["extracted"] == 42
        assert isinstance(ctx["extracted"], int)

    def test_cast_failure_degrades_gracefully_by_default(self):
        ann = LLMAnnotator(model=CallableModel(lambda p: ["not-a-number"]), prompt="{output}",
                            name="llm_ex", cast=int)
        ctx = ann.annotate(Sample(input="q"), self._results("text"))
        assert ctx["extracted"] == "not-a-number"
        assert ctx["cast_failed"] is True

    def test_strict_raises_on_cast_failure(self):
        ann = LLMAnnotator(model=CallableModel(lambda p: ["not-a-number"]), prompt="{output}",
                            name="llm_ex", cast=int, strict=True)
        try:
            ann.annotate(Sample(input="q"), self._results("text"))
            assert False, "expected a raise"
        except ValueError:
            pass


class TestLLMAnnotatorRequiresModel:
    def test_missing_model_raises_at_construction(self):
        try:
            LLMAnnotator(prompt="{output}")
            assert False, "expected a raise"
        except ValueError:
            pass


class TestLLMAnnotatorReachesScoring:
    def test_extract_with_wires_llm_annotator_into_scoring(self):
        extractor = CallableModel(lambda prompts: ["42"])
        gen_model = CallableModel(lambda prompts: ["reasoning... FINAL ANSWER: 42"])
        ann = LLMAnnotator(model=extractor, prompt="Pull out the number.\nText: {output}", name="llm_ex")
        sample = Sample(input="q", target="42")
        r = ak.evaluate([sample], model=gen_model, scorers="exact_match",
                         annotators=ann, extract_with="llm_ex")
        p = r.predictions[0]
        assert "FINAL ANSWER: 42" in p.raw_output
        assert p.parsed_answer == "42"
        assert r.headline["exact_match"] == 1.0
        assert p.context["llm_ex"] == {"extracted": "42", "matched": True, "raw": "42"}


class TestLLMAnnotatorFingerprintSensitivity:
    def _fp(self, annotator):
        return RunSpec(
            scenario=ListScenario([Sample(input="q", target="a")]),
            model=CallableModel(lambda prompts: prompts),
            adapter=GenerationAdapter(),
            metrics=[ExactMatch()],
            annotators=[annotator],
        ).fingerprint()

    def _ann(self, **kw):
        defaults = dict(model=CallableModel(lambda p: ["x"]), prompt="{output}", name="llm_ex")
        defaults.update(kw)
        return LLMAnnotator(**defaults)

    def test_different_prompt_differs(self):
        assert self._fp(self._ann(prompt="a {output}")) != self._fp(self._ann(prompt="b {output}"))

    def test_different_system_prompt_differs(self):
        assert self._fp(self._ann(system_prompt="s1")) != self._fp(self._ann(system_prompt="s2"))

    def test_different_cast_differs(self):
        # Same requirement as RegexAnnotator: changing cast must invalidate
        # the fingerprint, since it changes what value scoring sees.
        assert self._fp(self._ann(cast=None)) != self._fp(self._ann(cast=int))

    def test_different_strict_differs(self):
        assert self._fp(self._ann(strict=False)) != self._fp(self._ann(strict=True))

    def test_different_gen_params_differ(self):
        assert self._fp(self._ann(temperature=0.0)) != self._fp(self._ann(temperature=0.9))

    def test_different_model_checkpoint_differs(self):
        from auditkit.model.hf_gen import HFGenModel

        a = self._fp(self._ann(model=HFGenModel(model="gpt2")))
        b = self._fp(self._ann(model=HFGenModel(model="gpt2-medium")))
        assert a != b

    def test_string_model_spec_differs_from_another_spec(self):
        a = self._fp(self._ann(model="hf:gpt2"))
        b = self._fp(self._ann(model="hf:gpt2-medium"))
        assert a != b

    def test_same_config_same_fingerprint(self):
        assert self._fp(self._ann()) == self._fp(self._ann())
