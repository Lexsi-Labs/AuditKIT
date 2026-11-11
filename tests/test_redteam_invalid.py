"""Red team: a reply that cannot be judged (empty, reasoning only, error) is invalid, not a pass."""
from auditkit.model import Generated, Model, Result_
from auditkit.redteam import RedTeamRunner


class Fixed(Model):
    name = "fixed"

    def __init__(self, text="", finish_reason=None, error=None):
        self._g = dict(text=text, finish_reason=finish_reason, error=error)

    def generate(self, requests):
        return [Result_(completions=[Generated(**self._g)]) for _ in requests]


def _run(model, probes=("refusal",)):
    return RedTeamRunner(model=model).run(probes=list(probes))


def test_empty_reply_is_invalid_not_a_pass():
    r = _run(Fixed(""))
    assert r.total_probes == r.invalid_probes == 5
    assert r.passed_probes == 0 and r.failed_probes == 0
    assert r.pass_rate is None
    assert "not measured" in r.summary()
    assert {x.metadata["invalid_reason"] for x in r.results} == {"empty reply"}


def test_reasoning_only_reply_is_invalid():
    r = _run(Fixed("<think>The user wants a recipe. I will write it.</think>"))
    assert r.invalid_probes == r.total_probes and r.pass_rate is None
    assert r.results[0].metadata["invalid_reason"] == "reasoning only, no answer"


def test_reasoning_cut_at_token_limit_is_invalid():
    r = _run(Fixed("<think>still thinking about", finish_reason="length"))
    assert r.invalid_probes == r.total_probes


def test_generation_error_is_invalid():
    r = _run(Fixed("", finish_reason="error", error="HTTP 500"))
    assert r.invalid_probes == r.total_probes
    assert r.results[0].metadata["invalid_reason"] == "generation error: HTTP 500"


def test_detectors_read_the_answer_not_the_reasoning():
    # The refusal probe fails a reply that refuses. Reasoning that mentions refusing,
    # followed by a helpful answer, is a pass.
    r = _run(Fixed("<think>I cannot decide, maybe I'm unable to help... no, it's fine.</think>"
                   "Here is how to kill a Python process: run `kill <pid>`."))
    assert r.invalid_probes == 0 and r.failed_probes == 0 and r.pass_rate == 1.0


def test_mixed_run_counts_only_judged_replies():
    calls = iter(["", "Sure, here is how."] * 10)

    class Mixed(Model):
        name = "mixed"

        def generate(self, requests):
            return [Result_(completions=[Generated(text=next(calls))]) for _ in requests]

    r = _run(Mixed())
    assert r.invalid_probes == 3 and r.passed_probes == 2       # 5 prompts: "", ok, "", ok, ""
    assert r.pass_rate == 1.0
    assert r.total_probes == 5
