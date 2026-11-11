"""Tests for the MultiJail, XSTest and Global-MMLU loaders. Offline: the Hub is faked."""
from __future__ import annotations

import sys
import types

import pytest

import auditkit as ak
from auditkit.benchmark_loaders import GLOBAL_MMLU, GLOBAL_MMLU_LITE, MULTIJAIL, MULTIJAIL_LANGUAGES, XSTEST
from auditkit.errors import ExtraNotInstalled


def _jail_row(n, **cells):
    row = {"id": n, "source": "x", "tags": "['Category A', 'Category B']"}
    row.update({lang: f"prompt {n} in {lang}" for lang in MULTIJAIL_LANGUAGES})
    row.update(cells)
    return row


def _mmlu_row(n, answer="B"):
    return {"sample_id": f"topic/test/{n}", "subject": "topic", "subject_category": "STEM",
            "question": f"question {n}", "option_a": "first", "option_b": "second",
            "option_c": "third", "option_d": "fourth", "answer": answer,
            "cultural_sensitivity_label": "CA"}


XSTEST_ROWS = [{"id": "v2-1", "type": "homonyms", "prompt": "safe one", "completion": ""},
               {"id": "v2-2", "type": "contrast_homonyms", "prompt": "unsafe one", "completion": ""},
               {"id": "v2-3", "type": "privacy_public", "prompt": "safe two", "completion": ""}]


@pytest.fixture
def hub(monkeypatch):
    """Serve ``hub.rows[(repo, config)]`` as ``datasets.load_dataset``; record each call."""
    state = types.SimpleNamespace(calls=[], rows={})

    def load_dataset(repo, config=None, *, split=None, revision=None, **kwargs):
        state.calls.append({"repo": repo, "config": config, "split": split, "revision": revision,
                            **kwargs})
        return state.rows[(repo, config)]

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset))
    return state


class TestMultiJail:
    def test_one_sample_per_prompt_and_language_with_task_set_to_the_language(self, hub):
        hub.rows[(MULTIJAIL, None)] = [_jail_row(1), _jail_row(2)]
        samples = ak.load_multijail(["en", "bn"])
        assert [(s.task, s.input) for s in samples] == [
            ("en", "prompt 1 in en"), ("bn", "prompt 1 in bn"),
            ("en", "prompt 2 in en"), ("bn", "prompt 2 in bn")]
        assert len({s.id for s in samples}) == 4
        assert all(s.target is None and s.tags == ["harmful"] for s in samples)

    def test_defaults_to_every_language(self, hub):
        hub.rows[(MULTIJAIL, None)] = [_jail_row(1)]
        assert [s.task for s in ak.load_multijail()] == list(MULTIJAIL_LANGUAGES)

    def test_translations_of_a_prompt_share_a_group(self, hub):
        hub.rows[(MULTIJAIL, None)] = [_jail_row(7)]
        en, sw = ak.load_multijail(["en", "sw"])
        assert en.metadata["group"] == sw.metadata["group"] == "multijail-7"
        assert (en.metadata["translated"], sw.metadata["translated"]) == (False, True)

    def test_categories_are_parsed_from_their_python_literal_text(self, hub):
        hub.rows[(MULTIJAIL, None)] = [_jail_row(1)]
        assert ak.load_multijail("en")[0].metadata["categories"] == ["Category A", "Category B"]

    @pytest.mark.parametrize("raw,expected", [('["a"]', ["a"]), (["a", "b"], ["a", "b"]),
                                              ("plain text", ["plain text"]), ("", []), (None, [])])
    def test_other_category_shapes(self, hub, raw, expected):
        hub.rows[(MULTIJAIL, None)] = [_jail_row(1, tags=raw)]
        assert ak.load_multijail("en")[0].metadata["categories"] == expected

    def test_a_single_language_string_is_not_read_as_letters(self, hub):
        hub.rows[(MULTIJAIL, None)] = [_jail_row(1)]
        assert [s.task for s in ak.load_multijail("bn")] == ["bn"]

    def test_a_blank_cell_is_skipped(self, hub):
        hub.rows[(MULTIJAIL, None)] = [_jail_row(1, bn="   "), _jail_row(2, bn=None)]
        assert [s.task for s in ak.load_multijail(["en", "bn"])] == ["en", "en"]

    def test_limit_counts_prompts_not_samples(self, hub):
        hub.rows[(MULTIJAIL, None)] = [_jail_row(n) for n in range(5)]
        assert len(ak.load_multijail(["en", "bn"], limit=2)) == 4

    def test_a_language_the_set_lacks_is_an_error(self, hub):
        with pytest.raises(ValueError, match=r"no \['hi'\]"):
            ak.load_multijail(["en", "hi"])
        assert hub.calls == []


class TestXSTest:
    def test_labels_come_from_the_contrast_prefix(self, hub):
        hub.rows[(XSTEST, None)] = XSTEST_ROWS
        samples = ak.load_xstest()
        assert [s.metadata["label"] for s in samples] == ["safe", "unsafe", "safe"]
        assert [s.tags for s in samples] == [["safe"], ["unsafe"], ["safe"]]
        assert samples[1].metadata["type"] == "contrast_homonyms"
        assert {s.task for s in samples} == {"en"}

    def test_reads_only_the_prompt_file_and_keeps_no_completions(self, hub):
        hub.rows[(XSTEST, None)] = XSTEST_ROWS
        samples = ak.load_xstest()
        assert hub.calls[0]["split"] == "prompts"
        assert hub.calls[0]["data_files"] == {"prompts": "data/prompts-00000-of-00001.parquet"}
        assert all("completion" not in s.metadata for s in samples)

    def test_another_repo_gets_its_test_split_and_no_data_files(self, hub):
        hub.rows[("walledai/XSTest", None)] = XSTEST_ROWS
        ak.load_xstest(repo="walledai/XSTest")
        assert hub.calls[0]["split"] == "test" and "data_files" not in hub.calls[0]

    def test_limit(self, hub):
        hub.rows[(XSTEST, None)] = XSTEST_ROWS
        assert len(ak.load_xstest(limit=2)) == 2


class TestGlobalMMLU:
    def test_an_mcq_sample_per_row_with_the_answer_as_an_index(self, hub):
        hub.rows[(GLOBAL_MMLU, "hi")] = [_mmlu_row(1, "C")]
        (sample,) = ak.load_global_mmlu("hi")
        assert sample.kind == ak.TaskKind.MCQ
        assert sample.task == "hi"
        assert sample.input == "question 1"
        assert sample.choices == ["first", "second", "third", "fourth"]
        assert sample.target == 2
        assert sample.metadata["subject"] == "topic"
        assert sample.metadata["cultural_sensitivity"] == "CA"

    def test_one_config_per_language_and_a_group_across_them(self, hub):
        hub.rows[(GLOBAL_MMLU, "hi")] = [_mmlu_row(1)]
        hub.rows[(GLOBAL_MMLU, "sw")] = [_mmlu_row(1)]
        hi, sw = ak.load_global_mmlu(["hi", "sw"])
        assert [c["config"] for c in hub.calls] == ["hi", "sw"]
        assert hi.id != sw.id
        assert hi.metadata["group"] == sw.metadata["group"] == "topic/test/1"

    def test_the_split_and_the_lite_repo_are_chosen_by_the_caller(self, hub):
        hub.rows[(GLOBAL_MMLU_LITE, "bn")] = [_mmlu_row(1)]
        ak.load_global_mmlu("bn", repo=GLOBAL_MMLU_LITE, split="dev")
        assert hub.calls[0]["repo"] == GLOBAL_MMLU_LITE and hub.calls[0]["split"] == "dev"

    def test_limit_applies_per_language(self, hub):
        hub.rows[(GLOBAL_MMLU, "hi")] = [_mmlu_row(n) for n in range(4)]
        hub.rows[(GLOBAL_MMLU, "sw")] = [_mmlu_row(n) for n in range(4)]
        assert len(ak.load_global_mmlu(["hi", "sw"], limit=3)) == 6

    @pytest.mark.parametrize("answer,index", [("a", 0), (" D ", 3)])
    def test_answers_are_read_loosely(self, hub, answer, index):
        hub.rows[(GLOBAL_MMLU, "hi")] = [_mmlu_row(1, answer)]
        assert ak.load_global_mmlu("hi")[0].target == index

    @pytest.mark.parametrize("answer", ["E", "", None])
    def test_a_bad_answer_is_an_error_not_a_silent_skip(self, hub, answer):
        hub.rows[(GLOBAL_MMLU, "hi")] = [_mmlu_row(1, answer)]
        with pytest.raises(ValueError, match="topic/test/1"):
            ak.load_global_mmlu("hi")

    def test_the_target_scores_through_acc(self, hub):
        hub.rows[(GLOBAL_MMLU, "hi")] = [_mmlu_row(1, "B"), _mmlu_row(2, "D")]
        samples = ak.load_global_mmlu("hi")
        for sample in samples:
            sample.actual_output = "1"  # the MCQ adapter labels choices from 0
        result = ak.evaluate(samples, "precomputed", ["acc"])
        assert result.headline["acc"] == pytest.approx(0.5)


class TestRevisions:
    def test_a_pinned_revision_is_used_and_recorded(self, hub):
        hub.rows[(MULTIJAIL, None)] = [_jail_row(1)]
        sample = ak.load_multijail("en")[0]
        assert len(hub.calls[0]["revision"]) == 40
        assert sample.metadata["revision"] == hub.calls[0]["revision"]

    def test_the_caller_can_choose_another(self, hub):
        hub.rows[(XSTEST, None)] = XSTEST_ROWS
        sample = ak.load_xstest(revision="abc123")[0]
        assert hub.calls[0]["revision"] == "abc123"
        assert sample.metadata["revision"] == "abc123"

    def test_an_unpinned_source_records_none(self, hub):
        hub.rows[("/local/folder", None)] = [_jail_row(1)]
        sample = ak.load_multijail("en", repo="/local/folder")[0]
        assert hub.calls[0]["revision"] is None
        assert "revision" not in sample.metadata


def test_results_break_down_by_language(hub):
    """The reason for task=lang: a per-language delta, which a flat task can't give."""
    hub.rows[(MULTIJAIL, None)] = [_jail_row(1), _jail_row(2)]

    def run(reply_by_language):
        samples = ak.load_multijail(["en", "sw"])
        for sample in samples:
            sample.actual_output = reply_by_language[sample.task]
        return ak.evaluate(samples, "precomputed", [ak.Contains("no")])

    deltas = ak.compare(run({"en": "no", "sw": "no"}), run({"en": "no", "sw": "yes"})).per_task_deltas()
    assert {d.task: d.delta for d in deltas} == {"en": 0.0, "sw": -1.0}


@pytest.mark.parametrize("load", [lambda: ak.load_multijail("en"), ak.load_xstest,
                                  lambda: ak.load_global_mmlu("hi")])
def test_without_the_extra_it_says_what_to_install(monkeypatch, load):
    monkeypatch.setitem(sys.modules, "datasets", None)
    with pytest.raises(ExtraNotInstalled):
        load()
