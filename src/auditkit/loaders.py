"""Data loaders that produce lists of :class:`~auditkit.sample.Sample`.

T0 provides ``load_csv``, ``load_jsonl`` and ``load_agenttune`` (stdlib only).
Optional loaders (``load_hf``, ``load_croissant``) are deferred to their own
cycles and raise :class:`ExtraNotInstalled` when their extra is absent.
"""

from __future__ import annotations

import csv
import json
import re
from typing import Any, Iterator, Optional

from .errors import ExtraNotInstalled
from .sample import Sample
from .trace import is_call_payload, to_reference_paths
from .types import TaskKind


def load_csv(
    path: str,
    *,
    input_col: str = "input",
    target_col: Optional[str] = "target",
    output_col: Optional[str] = None,
    **kwargs: Any,
) -> list[Sample]:
    """Load samples from a CSV file.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    input_col : str
        Column name for the sample input (default ``"input"``).
    target_col : str or None
        Column name for the sample target (default ``"target"``);
        ``None`` means no target column is expected.
    output_col : str or None
        Column name for a pre-generated answer (default ``None``); when set, it
        fills ``Sample.actual_output`` so the rows can be scored directly with
        ``model="precomputed"`` (no generation step).
    **kwargs
        Extra keyword arguments forwarded to ``csv.DictReader``.
    """
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, **kwargs)
        samples: list[Sample] = []
        for row in reader:
            target = row.get(target_col) if target_col else None  # type: ignore[arg-type]
            output = row.get(output_col) if output_col else None
            samples.append(
                Sample(
                    input=row[input_col],
                    target=target,
                    actual_output=output,
                )
            )
    return samples


def load_hf(
    path: str,
    *,
    split: str = "test",
    input_col: str = "question",
    target_col: str = "answer",
    **kwargs: Any,
) -> list[Sample]:
    try:
        import datasets
    except ImportError:
        raise ExtraNotInstalled("interop", "HuggingFace datasets (install auditkit[interop])") from None
    records = datasets.load_dataset(path, split=split, **kwargs)
    return [Sample(input=row[input_col], target=row.get(target_col)) for row in records]


_SPLIT_ORDER = ("test", "validation", "train")


def _row_sample(row: dict[str, Any], input_col: Optional[str], target_col: Optional[str]) -> Sample:
    """One dataset row -> Sample. Named columns win; otherwise the layout is
    read from the columns: CuratorKIT ``sft_alpaca`` (instruction/input/output),
    ``sft_sharegpt`` (conversations) or chat ``messages``, ``dpo``
    (prompt/chosen: ``chosen`` is the target), ``grpo``/``ppo`` (prompt only),
    then ``input``/``target`` and ``question``/``answer``. Columns named like
    a Sample field (``tools``, ``expected_tool_calls``, ``id``, ...) fill it,
    as in :func:`load_jsonl`; the rest go to ``metadata``."""
    if input_col:
        inp, target, used = row[input_col], row.get(target_col) if target_col else None, {input_col, target_col}
    elif "instruction" in row:
        inp = row["instruction"] + (f"\n\n{row['input']}" if row.get("input") else "")
        target, used = row.get("output"), {"instruction", "input", "output"}
    elif isinstance(row.get("conversations") or row.get("messages"), list):
        # ponytail: last user turn -> input, final assistant turn -> target;
        # earlier turns are dropped (single-turn eval).
        key = "conversations" if isinstance(row.get("conversations"), list) else "messages"
        turns = [(t.get("from") or t.get("role"), t.get("value", t.get("content"))) for t in row[key]]
        target = turns.pop()[1] if turns and turns[-1][0] in ("gpt", "assistant") else None
        inp = next((v for r, v in reversed(turns) if r in ("human", "user")), None)
        used = {key}
    elif "prompt" in row:
        inp, target, used = row["prompt"], row.get("chosen"), {"prompt", "chosen"}
    else:
        i, t = next(((i, t) for i, t in (("input", "target"), ("question", "answer")) if i in row),
                    (None, None))
        if i is None:
            raise ValueError(f"cannot tell the input column from {sorted(row)}; pass input_col=")
        inp, target, used = row[i], row.get(t), {i, t}
    if inp is None:
        raise ValueError(f"row has no input: {row!r}")
    rest = {k: v for k, v in row.items() if k not in used}
    kw = {k: rest.pop(k) for k in _JSONL_FIELDS - {"input", "target", "metadata"} if k in rest}
    if kw.get("id") is not None:
        kw["id"] = str(kw["id"])
    return Sample(input=inp, target=target, metadata=rest, **kw)


def load_dataset(
    path: str,
    config: Optional[str] = None,
    *,
    split: Optional[str] = None,
    input_col: Optional[str] = None,
    target_col: Optional[str] = None,
    **kwargs: Any,
):
    """Load a dataset folder, file or Hub id as a scenario ``ak.evaluate`` takes.

    *path* is a ``.jsonl``/``.csv`` file (stdlib), or anything
    ``datasets.load_dataset(path, config)`` reads: a CuratorKIT export folder
    whose README ``configs:`` name each format (``sft_alpaca``, ``dpo``, ...)
    or a Hub id (``auditkit[interop]``). *split* defaults to the first of
    test / validation / train present. Columns are read per
    :func:`_row_sample`, unless ``input_col``/``target_col`` name them.

    The folder's (or file's folder's) ``lexsi_provenance.json`` is recorded as
    the run's dataset input (``RunResult.metadata["inputs"]``).
    ``**kwargs`` go to ``datasets.load_dataset``.
    """
    import os

    from .provenance import read_provenance
    from .scenario import ListScenario

    is_file = os.path.isfile(path)
    if is_file and path.endswith(".csv"):
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    elif is_file and path.endswith(".jsonl"):
        rows = [r for _, r in _jsonl_records(path)]
    else:
        try:
            import datasets
        except ImportError:
            raise ExtraNotInstalled("interop", "HuggingFace datasets (install auditkit[interop])") from None
        ds = datasets.load_dataset(path, config, split=split, **kwargs)
        if split is None:
            split = next((s for s in _SPLIT_ORDER if s in ds), next(iter(ds)))
            ds = ds[split]
        rows = list(ds)
    samples = [_with_kind(_row_sample(r, input_col, target_col)) for r in rows]
    folder = os.path.dirname(os.path.abspath(path)) if is_file else path  # a Hub id reads as None
    entry = {"kind": "dataset", "ref": path, "config": config, "split": split,
             "provenance": read_provenance(folder)}
    return ListScenario(samples, metadata={"lexsi_input": entry})


def load_croissant(
    path: str,
    record_set: str,
    *,
    input_col: str = "input",
    target_col: str = "target",
    **kwargs: Any,
) -> list[Sample]:
    """Load samples from a Croissant (JSON-LD) dataset descriptor.

    Parameters
    ----------
    path : str
        Path or URL to the Croissant JSON-LD file (e.g. a HuggingFace
        dataset's ``.../croissant`` endpoint).
    record_set : str
        The Croissant record set to read (a Croissant file can describe
        several; there's no universally correct default). Inspect
        ``mlcroissant.Dataset(jsonld=path).metadata.record_sets`` to see
        what's available for a given dataset.
    input_col : str
        Field name for the sample input, *without* the record-set prefix
        Croissant adds (e.g. ``"question"``, not ``"question-answer/question"``
        -- the prefix is added automatically).
    target_col : str
        Field name for the sample target, same convention as ``input_col``.
    **kwargs
        Extra keyword arguments forwarded to ``mlcroissant.Dataset(...)``.
    """
    try:
        import mlcroissant
    except ImportError:
        raise ExtraNotInstalled("interop", "mlcroissant (install auditkit[interop])") from None

    def _decode(value: Any) -> Any:
        return value.decode("utf-8") if isinstance(value, bytes) else value

    dataset = mlcroissant.Dataset(jsonld=path, **kwargs)
    input_key = f"{record_set}/{input_col}"
    target_key = f"{record_set}/{target_col}"
    samples = []
    for row in dataset.records(record_set):
        samples.append(Sample(
            input=_decode(row[input_key]),
            target=_decode(row.get(target_key)),
        ))
    return samples


_JSONL_FIELDS = frozenset({
    "input", "target", "id", "choices", "retrieval_context", "tools", "expected_tool_calls",
    "reference_contexts", "actual_output", "actual_trace", "tags", "metadata",
})


def _jsonl_records(path: str) -> Iterator[tuple[int, dict[str, Any]]]:
    """``(line_number, object)`` per non-blank line; ValueError names the bad line."""
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as e:
                raise ValueError(f"{path}: line {n}: invalid JSON ({e})") from None
            if not isinstance(row, dict):
                raise ValueError(f"{path}: line {n}: expected a JSON object, got {type(row).__name__}")
            yield n, row


def _with_kind(sample: Sample) -> Sample:
    if sample.tools is not None or sample.expected_tool_calls is not None:
        sample.kind = TaskKind.AGENT
    elif sample.retrieval_context is not None or sample.reference_contexts is not None:
        sample.kind = TaskKind.RAG
    return sample


def load_jsonl(path: str, *, field_map: Optional[dict[str, str]] = None) -> list[Sample]:
    """Load one :class:`Sample` per line of a JSONL file.

    Keys named like a Sample field (``input``, ``target``, ``id``, ``choices``,
    ``retrieval_context``, ``tools``, ``expected_tool_calls``,
    ``reference_contexts``, ``actual_output``, ``actual_trace``, ``tags``,
    ``metadata``) fill that field; ``field_map`` renames source keys first
    (``{"question": "input", "answer": "target"}``). Every other key lands in
    ``metadata``. ``kind`` is ``AGENT`` when tools/expected_tool_calls are set,
    else ``RAG`` when retrieval_context/reference_contexts are set. Rows with
    ``actual_output`` (and optionally ``actual_trace``) score with
    ``model="precomputed"``. Blank lines are skipped; invalid JSON or a row
    without ``input`` raises ValueError naming the line.
    """
    field_map = field_map or {}
    samples: list[Sample] = []
    for n, row in _jsonl_records(path):
        kw: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        for key, val in row.items():
            name = field_map.get(key, key)
            if name in _JSONL_FIELDS:
                kw[name] = val
            else:
                extra[key] = val
        if kw.get("input") is None:
            raise ValueError(f"{path}: line {n}: missing 'input' (keys: {sorted(row)}; map one with field_map=)")
        if kw.get("id") is not None:
            kw["id"] = str(kw["id"])
        if kw.get("expected_tool_calls") is not None:
            # Read the reference while the line number is still known. Left to
            # scoring time it is a per-sample metric error, and a file whose every
            # reference is malformed looks like a run that scored nothing.
            try:
                to_reference_paths(kw["expected_tool_calls"])
            except ValueError as e:
                raise ValueError(f"{path}: line {n}: expected_tool_calls: {e}") from e
        md = kw.get("metadata")
        if md and not isinstance(md, dict):
            raise ValueError(f"{path}: line {n}: 'metadata' must be a JSON object, "
                             f"got {type(md).__name__}")
        kw["metadata"] = {**extra, **(md or {})}
        samples.append(_with_kind(Sample(**kw)))
    return samples


def _first(value: Any) -> Any:
    return (value[0] if value else None) if isinstance(value, list) else value


def _as_list(x: Any) -> list:
    """A JSON list stays as-is; a scalar becomes a one-element list. A lone
    string is NOT char-split (``"doc-1"`` -> ``["doc-1"]``, not ``list("doc-1")``)."""
    return x if isinstance(x, list) else [x]


def _flat_ids(ids: Any) -> Optional[list[str]]:
    # AgentTune stringifies id lists (build_dataset writes gold_path as
    # json.dumps(gold_chunk_ids); QASample.to_row does the same): decode first.
    # Trust json.loads only when it yields a list or str -- a bare numeric or
    # keyword string ('1.50', '1e3', 'null') must keep its raw form, never be
    # reformatted into a value that can no longer match a retrieved id by
    # string equality. JSON null ("null") means "no gold", not a phantom id.
    if isinstance(ids, str):
        try:
            dec = json.loads(ids)
        except ValueError:
            dec = ids
        if dec is None:
            return None
        ids = dec if isinstance(dec, list) else [dec if isinstance(dec, str) else ids]
    if ids is None:
        return None
    if not isinstance(ids, list):  # a scalar id field (int/float/...): one id
        ids = [ids]
    out = [str(i) for item in ids for i in (item if isinstance(item, list) else [item])]
    return out or None


def _last_user(msgs: Any) -> Optional[str]:
    """The content of the last ``user`` message in a chat list, else None."""
    if not isinstance(msgs, list):
        return None
    return next((m.get("content") for m in reversed(msgs)
                 if isinstance(m, dict) and m.get("role") == "user"), None)


_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.I | re.S)


def _agenttune_answer(text: Any) -> str:
    """AgentTune's RAG/Enron prompts force the answer inside <answer></answer>,
    and every AgentTune answer metric strips the tag before comparing (see
    eval/agent_eval._extract_answer). Mirror it, else keep the full text."""
    # Coerce to str first: a numeric final_answer/final_response/predicted is
    # truthy, so `text or ""` would keep the int and blow up in the regex. Only
    # None becomes "" -- a genuine 0/False answer keeps its stringified value.
    text = "" if text is None else str(text)
    m = _ANSWER_TAG.search(text)
    return m.group(1).strip() if m else text.strip()


def _agenttune_row(path: str, n: int, row: dict[str, Any]) -> Sample:
    """A run_eval / RAG-GRPO dataset row: something to run, not a result."""
    prompt = row["prompt"]
    meta = {k: v for k, v in row.items() if k not in (
        "prompt", "answer", "gold_answer", "question_id", "message_ids",
        "gold_chunk_ids", "gold_path")}
    if isinstance(prompt, list):
        user = _last_user(prompt)
        if user is None:
            raise ValueError(f"{path}: line {n}: 'prompt' has no user message")
        meta["messages"] = prompt
    else:
        user = str(prompt)
    target = _first(row.get("answer"))
    if target is None:
        target = row.get("gold_answer")
    qid = row.get("question_id")
    return _with_kind(Sample(
        input=user, target=target, id=None if qid is None else str(qid), metadata=meta,
        reference_contexts=(_flat_ids(row.get("message_ids"))
                            or _flat_ids(row.get("gold_chunk_ids"))
                            or _flat_ids(row.get("gold_path"))),
    ))


def _agenttune_trace(row: dict[str, Any]) -> Sample:
    """A train_grpo TraceLogger record (trace.jsonl) or TrajectoryStore export:
    answer + retrieval ids, no per-turn tool structure. Its ``tool_calls`` stay
    in metadata, so tool-call metrics don't apply; answer metrics +
    RetrievalMetrics do. (A TraceLogger record's tool_calls are ``{query,
    result}`` pairs with no tool name; a TrajectoryStore export's carry a
    ``tool_name`` but still stay in metadata.) An export's ``trajectory_id``
    becomes the Sample ``id`` (a store-minted uuid4, not the rollout's
    trajectory id); a trace.jsonl record has none, so id stays None."""
    trace: dict[str, Any] = {}
    rc = row.get("retrieved_chunk_ids")
    if rc is not None:
        trace["retrieved_contexts"] = [str(c) for c in _as_list(rc)]
    # A TrajectoryStore export (trajectory_store.export_jsonl) carries a
    # trajectory_id; a train_grpo trace.jsonl record does not, so id stays None
    # there. Preserve it as the Sample id for provenance when present.
    tid = row.get("trajectory_id")
    return _with_kind(Sample(
        input=str(row["question"]), target=row.get("gold_answer"),
        id=None if tid is None else str(tid),
        actual_output=_agenttune_answer(row.get("final_answer")), actual_trace=trace or None,
        reference_contexts=_flat_ids(row.get("gold_chunk_ids")),
        metadata={k: row[k] for k in (
            "reward", "reward_components", "n_tool_calls", "has_answer_tag", "tool_calls") if k in row},
    ))


def _agenttune_trajectory(row: dict[str, Any]) -> Sample:
    """A recorded TrajectoryDataset run: scored as is with model="precomputed"."""
    tmeta = row.get("metadata") or {}
    trace: dict[str, Any] = {}
    # Steps record every executed turn; the rollout can truncate or rewrite
    # (MEM1/recent-k) the conversation, so build the trace from steps and keep
    # the conversation for render_trace only. A step's action is either the
    # rollout's {"tool_calls": [...]} or the flat {"name", "arguments"} shape
    # (both documented in AgentTune's dataset.py / trajectory_utils.py); a flat
    # action is one call in its own turn. The terminal step has no calls.
    actions = [(s.get("action") or {}) for s in row.get("steps") or []]
    turns = [a["tool_calls"] if a.get("tool_calls") else [a]
             for a in actions if a.get("tool_calls") or is_call_payload(a)]
    if turns or not tmeta.get("conversation"):
        trace["tool_calls"] = turns
    if tmeta.get("conversation"):
        trace["messages"] = tmeta["conversation"]
    trc = tmeta.get("retrieved_chunk_ids")
    if trc is not None:
        trace["retrieved_contexts"] = [str(c) for c in _as_list(trc)]
    meta = {k: v for k, v in tmeta.items() if k not in ("conversation", "retrieved_chunk_ids")}
    meta["reward"] = row.get("reward")
    # Trajectory.task is a plain question string, or during real GRPO the full
    # chat prompt list (extract_question_text unwraps it to the last user turn).
    task = row.get("task")
    user = _last_user(task) if isinstance(task, list) else task
    fr = row.get("final_response")  # coerce: a numeric final_response must not crash raw.strip()
    raw = "" if fr is None else str(fr)
    answer = _agenttune_answer(raw)
    if answer != raw.strip():  # keep the tagged original only when we stripped a tag
        meta["raw_output"] = raw
    tid = row.get("trajectory_id")
    return Sample(
        input=str(user or ""), id=None if tid is None else str(tid), kind=TaskKind.AGENT,
        actual_output=answer, actual_trace=trace, metadata=meta,
    )


def load_agenttune(path: str) -> list[Sample]:
    """Load AgentTune data as Samples; the shape is detected per record.

    **run_eval / RAG-GRPO dataset rows** (JSONL, ``{"prompt", "answer",
    "message_ids"?}`` or ``{"prompt", "gold_answer", "question_id",
    "gold_path"}``): tasks to *run*. ``input`` is the last user message
    (or the plain-string prompt), ``target`` the answer (first if a list),
    ``reference_contexts`` the flattened gold evidence ids (from
    ``message_ids``, ``gold_chunk_ids``, or ``gold_path`` -- a JSON-string
    list, as build_dataset writes to dataset_grpo.jsonl), and
    ``metadata["messages"]`` the full prompt so ``ToolCallAdapter`` sends the
    original system prompt. Run them against a live agent, e.g.
    ``ak.evaluate(samples, model="agent:https://.../run",
    adapter=ak.ToolCallAdapter(), scorers=[...])``. Feeds answer metrics
    (``"exact_match"``, ``"quasi_exact_match"``, LLM judges,
    ``TaskCompletion``) and,
    when the endpoint returns retrieved ids as ``contexts``,
    ``RetrievalMetrics``. Tool-call metrics need ``expected_tool_calls``,
    which these rows don't carry.

    **TrajectoryDataset records** (JSONL, ``{"task", "steps",
    "final_response", "metadata": {"conversation", "retrieved_chunk_ids"}}``):
    recorded runs, scored offline with ``model="precomputed"``. Per-turn tool
    calls come from ``steps`` (each step's ``action`` -- both the
    ``{"tool_calls": [...]}`` and flat ``{"name", "arguments"}`` shapes -- as
    ``actual_trace["tool_calls"]``); the ``conversation`` is kept only as
    ``actual_trace["messages"]`` because the rollout truncates/rewrites it.
    ``input`` is the last user message when ``task`` is a chat list, and
    ``actual_output`` has any ``<answer>...</answer>`` wrapper stripped (like
    every AgentTune answer metric; ``target`` is left raw, mirroring
    agent_eval, and the raw output is kept in ``metadata["raw_output"]`` only
    when a tag was stripped).
    ``actual_trace["retrieved_contexts"]`` holds the retrieved chunk ids. Feeds
    ``ToolCallValidity`` (after attaching ``tools``), ``RedundantToolCalls``,
    ``TaskCompletion``, ``ToolCallF1`` / ``TrajectoryMatch`` /
    ``ParallelToolCalls`` (after attaching ``expected_tool_calls``) and
    ``RetrievalMetrics`` (after attaching ``reference_contexts``). The
    retrieved contexts are ids, not text, so the judge RAG metrics
    (``Faithfulness``, ``ContextPrecision``, ``ContextRecall``) don't apply.

    **trace.jsonl records** (JSONL, ``{"question", "final_answer",
    "gold_answer"?, "retrieved_chunk_ids"?, "gold_chunk_ids"?, ...}`` from
    train_grpo's TraceLogger, or a TrajectoryStore export): the answer plus
    retrieval ids, with no per-turn tool structure. ``actual_output`` is the
    ``<answer>``-unwrapped ``final_answer``, ``reference_contexts`` the gold
    chunk ids, ``actual_trace["retrieved_contexts"]`` the retrieved ids. Feeds
    answer metrics and ``RetrievalMetrics``; its ``tool_calls`` are query/result
    pairs kept in metadata only (a TraceLogger record's carry no tool name; a
    TrajectoryStore export's do, but still stay in metadata). A TrajectoryStore
    export's ``trajectory_id`` is preserved as the Sample ``id``; it is a
    store-minted uuid4, not the rollout's ``Trajectory.trajectory_id``, so it
    joins nothing outside that store.

    **run_eval report JSON** (one object from ``Report.save`` with a
    ``"samples"`` list): precomputed answers (``input=question``,
    ``target=gold``, ``actual_output=`` the ``<answer>``-unwrapped
    ``predicted``) for answer-level metrics only. Its tool calls are flattened
    names with no turn structure, so the tool-call and parallel metrics don't
    apply.
    """
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    try:
        doc = json.loads(text)
    except ValueError:
        doc = None
    if isinstance(doc, dict) and isinstance(doc.get("samples"), list):
        out = []
        for i, r in enumerate(doc["samples"]):
            if not isinstance(r, dict):
                raise ValueError(f"{path}: samples[{i}] is not a JSON object, "
                                 f"got {type(r).__name__}")
            out.append(Sample(
                input=str(r.get("question") or ""), target=_first(r.get("gold")),
                id=str(r.get("idx", i)), actual_output=_agenttune_answer(r.get("predicted")),
                metadata={k: r[k] for k in ("n_tools", "tool_calls", "scores", "error") if k in r},
            ))
        return out
    samples = []
    for n, row in _jsonl_records(path):
        try:
            if "stage_type" in row and "state_snapshot" in row:
                # EventLog.to_audit_records (Project.heal): one record per call,
                # rollout tool names lost as 'unknown', a phantom call per
                # terminal step. Not a trajectory; never route it as an empty one.
                raise ValueError(
                    f"{path}: line {n}: AgentTune EventLog audit record (Project.heal / "
                    f"to_audit_records), a lossy failure-detector projection, not a trajectory. "
                    f"Import the EventLog itself with auditkit.agent_eval.episode_from_eventlog.")
            if isinstance(row.get("events"), list):
                raise ValueError(
                    f"{path}: line {n}: a serialized AgentTune EventLog; load it with "
                    f"auditkit.agent_eval.episodes_from_agenttune or episode_from_eventlog.")
            if "question" in row and "final_answer" in row:
                samples.append(_agenttune_trace(row))
            elif "steps" in row or "trajectory_id" in row or "final_response" in row:
                samples.append(_agenttune_trajectory(row))
            elif "prompt" in row:
                samples.append(_agenttune_row(path, n, row))
            else:
                raise ValueError(f"{path}: line {n}: not an AgentTune record (expected 'prompt' for a "
                                 f"run_eval row, 'task'/'steps' for a trajectory, or 'question'+'final_answer' "
                                 f"for a trace.jsonl record; keys: {sorted(row)})")
        except (TypeError, AttributeError) as e:
            # A structural mismatch with no sane coercion (steps as a dict, a
            # step that isn't a dict, trajectory metadata as a list, ...) surfaces
            # as a raw TypeError/AttributeError from a helper; the module contract
            # (like the ValueErrors above) is to name the bad line. Tradeoff: this
            # can mask a genuine helper bug, but malformed input is far likelier.
            # _agenttune_row's own ValueErrors (and the else) are ValueErrors, so
            # they pass through unwrapped.
            raise ValueError(f"{path}: line {n}: malformed AgentTune record ({e})") from None
    return samples
