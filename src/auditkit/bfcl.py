"""BFCL v3 single-turn categories as AuditKIT samples (#45, phase 2).

Reads the Berkeley Function Calling Leaderboard files as published on the Hub
(``gorilla-llm/Berkeley-Function-Calling-Leaderboard``): a question file
``BFCL_v3_<category>.json`` and its ``possible_answer/BFCL_v3_<category>.json``.
Both are JSON Lines despite the ``.json`` name. Stdlib only; download the files
yourself, pinned to a revision so a result reproduces::

    from huggingface_hub import snapshot_download
    root = snapshot_download("gorilla-llm/Berkeley-Function-Calling-Leaderboard", repo_type="dataset",
                             revision="61fc0608cfd831fcfbbaa676ebdfef0ed963eeda")
    samples = ak.load_bfcl(f"{root}/BFCL_v3_simple.json", revision="61fc060")
    ak.evaluate(samples, model=..., adapter=ak.ToolCallAdapter(),
                scorers=[ak.ToolCallF1(arg_match=ak.bfcl_arg_match)])

A possible answer lists the accepted values of each argument (``{"unit": ["units", ""]}``),
and ``""`` in that list means the argument may be left out. The reference keeps those
lists as its arguments, so score with :func:`bfcl_arg_match`, which reads them; the
default exact matching would compare against the lists themselves.

Multi-turn categories write their answers as call strings (``"cd(folder='x')"``) and
need a parser: a separate piece of work, so they are refused here. So are the
categories with no possible answers (``exec_*``, ``rest``, ``live_relevance``,
``chatable``). The two irrelevance categories load with an empty reference: calling
no tool is the correct answer.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

from .sample import Sample

_log = logging.getLogger(__name__)

SINGLE_TURN = ("simple", "multiple", "parallel", "parallel_multiple", "live_simple", "live_multiple",
               "live_parallel", "live_parallel_multiple", "java", "javascript", "sql")
IRRELEVANCE = ("irrelevance", "live_irrelevance")
_LANGUAGE = {"java": "Java", "javascript": "JavaScript"}
_TYPES = {"dict": "object", "float": "number", "tuple": "array"}

# BFCL's ast_checker.standardize_string: case, spaces and these marks never decide a match
_STD = re.compile(r"[ ,./\-_*^]")


def _std(s: str) -> str:
    return _STD.sub("", s).lower().replace("'", '"')


def _value_eq(pred: Any, alt: Any) -> bool:
    """One predicted value against one accepted value, as BFCL's AST checker compares them."""
    if isinstance(alt, bool) or isinstance(pred, bool):
        return type(pred) is type(alt) and pred == alt
    if alt is None or pred is None:
        return pred is alt
    if isinstance(alt, str):
        return isinstance(pred, str) and _std(pred) == _std(alt)
    if isinstance(alt, (int, float)):
        return isinstance(pred, (int, float)) and pred == alt
    if isinstance(alt, list):
        return (isinstance(pred, (list, tuple)) and len(pred) == len(alt)
                and all(_value_eq(p, a) for p, a in zip(pred, alt)))
    if isinstance(alt, dict):
        # a nested dict lists its own accepted values per key, like the top level
        return _args_ok(pred, {k: v if isinstance(v, list) else [v] for k, v in alt.items()})
    return pred == alt


def _args_ok(pred: Any, ref: dict[str, list[Any]]) -> bool:
    if not isinstance(pred, dict) or any(k not in ref for k in pred):
        return False                                      # BFCL: "Unexpected parameter"
    for key, alts in ref.items():
        if key not in pred:
            if "" not in alts:
                return False                              # required, or no "" marker
        elif not any(_value_eq(pred[key], a) for a in alts):
            return False
    return True


def bfcl_arg_match(tool_name: str, pred_args: dict[str, Any], ref_args: dict[str, Any]) -> bool:
    """``arg_match`` for references loaded by :func:`load_bfcl`: each predicted argument
    must equal one of its accepted values (strings compared as BFCL does: case, spaces
    and ``, . / - _ * ^`` ignored), an argument may be left out only when ``""`` is
    accepted, and an argument the answer doesn't list fails the call.

    Numbers compare by value (``5 == 5.0``); BFCL additionally rejects a float where
    the schema says integer, which this matcher, having no schema, does not."""
    return _args_ok(pred_args, ref_args)


def _jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    if text.lstrip().startswith("["):                     # a re-saved file as one array
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _category(path: str) -> str:
    m = re.match(r"BFCL_v\d+_(.+)\.jsonl?$", os.path.basename(path))
    if not m:
        raise ValueError(f"not a BFCL question file (BFCL_v3_<category>.json): {path!r}")
    return m.group(1)


def _schema(node: Any, language: Optional[str]) -> Any:
    """A BFCL parameter schema as JSON Schema: dict/float/tuple renamed, "any" untyped.
    Java/JavaScript arguments are passed as source strings, so they are typed string
    with the language type named in the description (as BFCL prompts them)."""
    if not isinstance(node, dict):
        return node
    out = {k: _schema(v, language) if k in ("items",) else v for k, v in node.items() if k != "properties"}
    if "properties" in node:
        out["properties"] = {k: _schema(v, language) for k, v in node["properties"].items()}
    t = node.get("type")
    if language and t not in (None, "dict") and "properties" not in node:
        out["type"] = "string"
        out["description"] = f"{node.get('description', '')} This parameter should be in {language} {t} type.".strip()
        out.pop("items", None)
    elif t == "any":
        out.pop("type")
    elif t in _TYPES:
        out["type"] = _TYPES[t]
    return out


def _tool(fn: dict[str, Any], language: Optional[str], rename: dict[str, str]) -> dict[str, Any]:
    return {"type": "function", "function": {
        "name": rename.get(fn["name"], fn["name"]), "description": fn.get("description", ""),
        "parameters": _schema(fn.get("parameters") or {"type": "dict", "properties": {}}, language)}}


def _messages(question: Any) -> list[dict[str, Any]]:
    # single-turn: [[msg, ...]]; sql ships the flat form [msg, ...]
    if question and isinstance(question[0], list):
        if len(question) != 1:
            raise ValueError(f"expected one turn, got {len(question)}")
        return question[0]
    return list(question or [])


def _index_key(bfcl_id: str) -> str:
    return bfcl_id.split("-", 1)[0]                        # live_multiple_1052-79-0 -> live_multiple_1052


def load_bfcl(path: str, answers: Optional[str] = None, *, revision: Optional[str] = None,
              limit: Optional[int] = None, dots_to_underscores: bool = False) -> list[Sample]:
    """Load one BFCL v3 single-turn category as samples.

    Parameters
    ----------
    path
        The question file, ``BFCL_v3_<category>.json``.
    answers
        Its possible answers; default ``possible_answer/<same name>`` next to *path*.
        Not read for the irrelevance categories.
    revision
        The dataset revision the files came from, recorded on every sample
        (``metadata["bfcl_revision"]``) so a reported number names its data.
    limit
        Keep the first *limit* samples.
    dots_to_underscores
        Rename ``math.factorial`` to ``math_factorial`` in tools and references, for
        providers that reject dots in function names (OpenAI).

    Each sample carries ``tools``, ``expected_tool_calls`` (one turn; its arguments are
    the accepted-value lists, see :func:`bfcl_arg_match`), the conversation in
    ``metadata["messages"]`` when it has a system prompt, and ``metadata["bfcl_id"]``.
    An answer whose id doesn't match its question's, but whose index does, is paired
    by index, logged, and recorded as ``metadata["bfcl_answer_id"]``. An argument with
    no accepted value at all is recorded in ``metadata["bfcl_unsatisfiable"]``.
    """
    category = _category(path)
    if category.startswith("multi_turn"):
        raise ValueError(f"BFCL {category!r} is multi-turn: its answers are call strings and need a "
                         f"parser, which load_bfcl doesn't have. Supported: {', '.join(SINGLE_TURN + IRRELEVANCE)}")
    if category not in SINGLE_TURN + IRRELEVANCE:
        raise ValueError(f"BFCL {category!r} has no possible answers to score against. "
                         f"Supported: {', '.join(SINGLE_TURN + IRRELEVANCE)}")
    questions = _jsonl(path)
    by_id: dict[str, dict] = {}
    if category in SINGLE_TURN:
        answers = answers or os.path.join(os.path.dirname(path), "possible_answer", os.path.basename(path))
        if not os.path.exists(answers):
            raise FileNotFoundError(f"no possible answers for {category!r} at {answers!r}; pass answers=")
        by_id = {a["id"]: a for a in _jsonl(answers)}
        unpaired_q = [q["id"] for q in questions if q["id"] not in by_id]
        spare = {_index_key(i): i for i in set(by_id) - {q["id"] for q in questions}}
        for qid in unpaired_q:
            aid = spare.get(_index_key(qid))
            if aid is None:
                continue
            _log.warning("BFCL %s: answer %r paired with question %r by index (ids differ)", category, aid, qid)
            by_id[qid] = {**by_id[aid], "_answer_id": aid}
    language = _LANGUAGE.get(category)
    samples: list[Sample] = []
    missing: list[str] = []
    for q in questions:
        fns = q.get("function") or []
        rename = {f["name"]: f["name"].replace(".", "_") for f in fns} if dots_to_underscores else {}
        messages = _messages(q.get("question"))
        user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), None)
        if user is None:
            raise ValueError(f"BFCL {q['id']}: no user message")
        meta: dict[str, Any] = {"bfcl_id": q["id"], "bfcl_category": category}
        if revision:
            meta["bfcl_revision"] = revision
        if len(messages) > 1:
            meta["messages"] = messages
        if category in IRRELEVANCE:
            reference: list = []
        else:
            answer = by_id.get(q["id"])
            if answer is None:
                missing.append(q["id"])
                continue
            if "_answer_id" in answer:
                meta["bfcl_answer_id"] = answer["_answer_id"]
            calls = [{"name": rename.get(name, name), "arguments": args}
                     for call in answer["ground_truth"] for name, args in call.items()]
            unsatisfiable = [f"{c['name']}.{k}" for c in calls for k, alts in c["arguments"].items() if alts == []]
            if unsatisfiable:
                meta["bfcl_unsatisfiable"] = unsatisfiable
            reference = [calls]
        samples.append(Sample(id=q["id"], input=user, task=f"bfcl_{category}",
                              tools=[_tool(f, language, rename) for f in fns],
                              expected_tool_calls=reference, metadata=meta))
        if limit is not None and len(samples) >= limit:
            break
    if missing:
        _log.warning("BFCL %s: %d question(s) have no possible answer and were left out: %s",
                     category, len(missing), ", ".join(missing[:10]))
    return samples
