"""The action shapes AgentTune's EventLog writers actually emit (#47).

Every case here is taken from a writer in ``agenttune/agentic/events.py`` rather than
from the tuple in ``importers.py``, because the previous tests used the same shapes the
implementation was written against and so could not catch a key nobody had considered.

The writers, all of which reach ``_has_unnamed_call``:

- ``from_trajectory`` (events.py:234)  -- ``{"action": step.action}``, passed through raw
- ``from_pipeline_state`` (events.py:253) -- ``{"stage": stage_id}``, the DECIDE no-call
- ``from_eval_dict`` (events.py:265)    -- ``{"action": call}`` from ``d["tool_calls"]``

and ``step.action`` itself is one of the two forms ``_unwrap_tool_calls`` documents
(events.py:74-83): the rollout ``{"tool_calls": [openai-function-call, ...]}``, which
``rollout_factory.py:201`` builds as
``{"type": "function", "function": {"name": ..., "arguments": {...}}}``, or the bare
``{"name": ..., "arguments": ...}`` that spine/eval trajectories record.

What must hold: a call whose name was not recorded is an **unnamed call**, so the source's
tool names are unobserved and the episode is not scored a measured 0.0. A step that made no
call is not an unnamed call. The whole cost of getting this wrong is a silent wrong number,
which is what #36 was for and what this issue is the residue of.
"""

from __future__ import annotations

import pytest

from auditkit.agent_eval.importers import _action_calls, _has_unnamed_call

# --- EventLog writers, verbatim shapes ---------------------------------------


def _rollout(*calls: dict) -> dict:
    """from_eval_dict / from_trajectory with a rollout action (events.py:74-83)."""
    return {"tool_calls": list(calls)}


def _openai_call(name: str | None = "search", arguments: dict | None = None) -> dict:
    """One entry of rollout_factory.py:201's tool_calls list."""
    fn: dict = {}
    if name is not None:
        fn["name"] = name
    if arguments is not None:
        fn["arguments"] = arguments
    return {"type": "function", "function": fn}


# --- named calls: never an unnamed call --------------------------------------


@pytest.mark.parametrize(
    "action",
    [
        pytest.param({"name": "search", "arguments": {"q": "x"}}, id="bare-named"),
        pytest.param({"name": "search"}, id="bare-zero-arg"),
        pytest.param({"tool_name": "search", "query": "x"}, id="cohere-tool_name"),
        pytest.param(_rollout(_openai_call("search", {"q": "x"})), id="rollout-named"),
        pytest.param(
            _rollout(_openai_call("search", {"q": "x"}), _openai_call("fetch", {"u": "y"})),
            id="rollout-named-parallel",
        ),
    ],
)
def test_a_named_call_is_not_an_unnamed_call(action: dict) -> None:
    assert _has_unnamed_call(action) is False


# --- no-call steps: never an unnamed call ------------------------------------
#
# EventLog's terminal step is {} and DECIDE records {"stage": stage_id}. Neither names a
# tool nor carries arguments, so treating them as calls would be the opposite error.


@pytest.mark.parametrize(
    "action",
    [
        pytest.param({}, id="terminal-empty"),
        pytest.param({"stage": "plan"}, id="decide-stage"),
        pytest.param({"stage": "act", "output": "done"}, id="decide-stage-with-output"),
    ],
)
def test_a_no_call_step_is_not_an_unnamed_call(action: dict) -> None:
    assert _action_calls(action) == []
    assert _has_unnamed_call(action) is False


# --- calls whose name was lost: must be detected -----------------------------
#
# These are the cases that scored a measured 0.0 before. A record that says a call happened,
# without saying which tool, means the names are unobserved.


@pytest.mark.parametrize(
    "action",
    [
        pytest.param({"query": "x", "result": "y"}, id="tracelogger-query-result"),
        pytest.param({"arguments": {"q": "x"}}, id="bare-args-only"),
        pytest.param(_rollout(_openai_call(None, {"q": "x"})), id="rollout-unnamed-with-args"),
        pytest.param(
            _rollout(_openai_call("search", {"q": "x"}), _openai_call(None, {"u": "y"})),
            id="rollout-one-of-two-unnamed",
        ),
        pytest.param({"type": "function", "arguments": {"q": "x"}}, id="openai-flat-unnamed"),
    ],
)
def test_a_call_without_a_recorded_name_is_an_unnamed_call(action: dict) -> None:
    assert _has_unnamed_call(action) is True


@pytest.mark.parametrize(
    "action",
    [
        pytest.param({"name": ""}, id="bare-blank-name"),
        pytest.param({"tool_name": ""}, id="bare-blank-tool_name"),
        pytest.param(_rollout(_openai_call("")), id="rollout-blank-name"),
        # a zero-argument call that never got a name at all
        pytest.param(_rollout(_openai_call(None, None)), id="rollout-unnamed-zero-arg"),
    ],
)
def test_a_recorded_but_blank_name_is_still_a_call(action: dict) -> None:
    """A name that was written and left empty is a call that lost its name.

    Before #47 these read as steps that made no call, because the check was truthiness on
    the name and membership in the argument-key tuple -- and a zero-argument call has no
    argument key. That is the silent 0.0 this issue exists for.
    """
    assert _has_unnamed_call(action) is True


# --- the rollout shape is unwrapped, not read as one call --------------------


def test_rollout_tool_calls_are_unwrapped_into_individual_calls() -> None:
    """_action_calls unwraps the list, so one unnamed entry in a parallel rollout counts.

    This is what stops a parallel rollout from hiding one bad call behind a named one.
    """
    action = _rollout(_openai_call("search", {"q": "x"}), _openai_call(None, {"u": "y"}))
    assert len(_action_calls(action)) == 2
    assert _has_unnamed_call(action) is True


def test_a_parallel_rollout_with_every_name_recorded_is_not_unnamed() -> None:
    action = _rollout(_openai_call("search", {"q": "x"}), _openai_call("fetch", {"u": "y"}))
    assert _has_unnamed_call(action) is False


# --- the marker keys cannot make a no-call step look like a call -------------


def test_a_stage_step_is_not_mistaken_for_a_call_because_of_the_markers() -> None:
    """The call markers are name/tool_name. DECIDE's `stage` is in neither list."""
    assert _has_unnamed_call({"stage": "plan"}) is False
    assert _has_unnamed_call({}) is False


@pytest.mark.parametrize(
    "record",
    [
        pytest.param({"type": "text", "text": "thinking..."}, id="text-part"),
        pytest.param({"type": "reasoning", "summary": []}, id="reasoning-part"),
        pytest.param({"type": "image_url", "image_url": {"url": "..."}}, id="image-part"),
        pytest.param({"type": "refusal", "refusal": "no"}, id="refusal"),
        pytest.param({"type": "stage", "stage": "plan"}, id="audit-row"),
        pytest.param({"type": "function_call", "text": "..."}, id="type-function_call"),
    ],
)
def test_a_type_that_is_not_function_is_not_a_call(record: dict) -> None:
    """`type` only means a call at the value "function".

    OpenAI content parts and audit rows carry `type` too, so a bare presence test reads a
    step that made no call as one that lost its name. That is the opposite error from the
    one this file exists to catch, and it is just as wrong: it turns a correct dataset
    unscored. An earlier version of this fix tested `type` for presence and every case
    here failed; the full suite stayed green through it, because no fixture carries a
    `type` on an action at all.
    """
    assert _has_unnamed_call(record) is False


def test_the_function_type_marker_still_counts() -> None:
    """The narrowed rule still catches the case it was written for."""
    assert _has_unnamed_call({"type": "function", "function": {}}) is True
    assert _has_unnamed_call(_rollout(_openai_call(None, None))) is True


def test_an_argument_key_alone_is_still_enough() -> None:
    """The original five keys still stand on their own: no name field is needed."""
    for key in ("arguments", "parameters", "args", "input", "query"):
        assert _has_unnamed_call({key: {}}) is True, key
