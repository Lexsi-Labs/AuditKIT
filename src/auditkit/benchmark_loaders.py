"""MultiJail, XSTest and Global-MMLU as samples, with one task per language.

Each sample's ``task`` is its language, so ``RunComparison.per_task_deltas()`` reports
every metric per language. Translations of one item share ``metadata["group"]``, for
splits and paired tests that must keep them together.

These read from the Hub and need ``auditkit[interop]``. Each defaults to a pinned
revision, recorded as ``metadata["revision"]`` so a reported number names its data;
pass ``revision=`` to change it. ``repo=`` also takes a local dataset folder.

MultiJail and the unsafe half of XSTest are harmful prompts by design. Handle the
prompts and any model answers to them accordingly.
"""

from __future__ import annotations

import ast
import json
from typing import Any, Iterable, Optional, Union

from .errors import ExtraNotInstalled
from .sample import Sample
from .types import TaskKind

MULTIJAIL = "DAMO-NLP-SG/MultiJail"
XSTEST = "natolambert/xstest-v2-copy"
GLOBAL_MMLU = "CohereLabs/Global-MMLU"
GLOBAL_MMLU_LITE = "CohereLabs/Global-MMLU-Lite"

MULTIJAIL_LANGUAGES = ("en", "zh", "it", "vi", "ar", "ko", "th", "bn", "sw", "jv")

_REVISIONS = {
    MULTIJAIL: "93d65c778925d55973b9ff30c7bccc8545494eb6",
    XSTEST: "b71afe2a6d10e5a6254ea8bcb006c48b095a15d5",
    GLOBAL_MMLU: "0e619dbeb34206cd48705a1a0ea7fb21cae09993",
    GLOBAL_MMLU_LITE: "36c2fd756f19ccf13a9a96c8e53ccecc02192b8b",
}
# XSTest's prompts are one split of six; the other five hold model completions.
_XSTEST_PROMPTS = {"prompts": "data/prompts-00000-of-00001.parquet"}
_XSTEST_UNSAFE_PREFIX = "contrast_"


def _rows(repo: str, config: Optional[str], split: str, revision: Optional[str],
          **kwargs: Any) -> tuple[Any, Optional[str]]:
    """The dataset's rows and the revision they were read at."""
    try:
        import datasets
    except ImportError:
        raise ExtraNotInstalled("interop", "HuggingFace datasets (install auditkit[interop])") from None
    revision = revision or _REVISIONS.get(repo)
    return datasets.load_dataset(repo, config, split=split, revision=revision, **kwargs), revision


def _languages(languages: Union[str, Iterable[str]]) -> list[str]:
    return [languages] if isinstance(languages, str) else list(languages)


def _categories(raw: Any) -> list[str]:
    """A category list stored as a list, JSON text or Python-literal text."""
    if not raw:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(item) for item in raw]
    for parse in (json.loads, ast.literal_eval):
        try:
            value = parse(raw)
        except (ValueError, SyntaxError):
            continue
        return [str(item) for item in value] if isinstance(value, (list, tuple)) else [str(value)]
    return [str(raw)]


def _meta(group: str, revision: Optional[str], **fields: Any) -> dict[str, Any]:
    return {"group": group, **fields, **({"revision": revision} if revision else {})}


def load_multijail(languages: Optional[Union[str, Iterable[str]]] = None, *, repo: str = MULTIJAIL,
                   revision: Optional[str] = None, limit: Optional[int] = None) -> list[Sample]:
    """MultiJail (Deng et al., ICLR 2024): 315 harmful prompts in ten languages.

    Returns one sample per prompt and language, with ``task`` set to the language.
    ``languages`` defaults to all of :data:`MULTIJAIL_LANGUAGES`. ``limit`` keeps the
    first *limit* prompts in each language. The set has no reference answer, so
    ``target`` is ``None``. ``metadata["translated"]`` is False only for English.
    """
    selected = _languages(languages) if languages is not None else list(MULTIJAIL_LANGUAGES)
    unknown = sorted(set(selected) - set(MULTIJAIL_LANGUAGES))
    if unknown:
        raise ValueError(f"MultiJail has no {unknown}; it has {list(MULTIJAIL_LANGUAGES)}")
    rows, revision = _rows(repo, None, "train", revision)
    samples: list[Sample] = []
    for n, row in enumerate(rows):
        if limit is not None and n >= limit:
            break
        group = f"multijail-{row['id']}"
        for lang in selected:
            text = (row.get(lang) or "").strip()
            if text:
                samples.append(Sample(
                    id=f"{group}-{lang}", input=text, task=lang, tags=["harmful"],
                    metadata=_meta(group, revision, categories=_categories(row.get("tags")),
                                   translated=lang != "en")))
    return samples


def load_xstest(*, repo: str = XSTEST, revision: Optional[str] = None,
                limit: Optional[int] = None) -> list[Sample]:
    """XSTest v2 (Röttger et al., NAACL 2024): 250 safe prompts that look unsafe, and
    200 unsafe contrast prompts.

    The set is English, so ``task`` is ``"en"``. ``metadata["label"]`` and ``tags`` say
    ``safe`` or ``unsafe``, and ``metadata["type"]`` keeps the prompt type. A refusal is
    the failure on safe prompts and the goal on unsafe ones, so score the two apart.
    """
    data_files = _XSTEST_PROMPTS if repo == XSTEST else None
    rows, revision = _rows(repo, None, "prompts" if data_files else "test", revision,
                           **({"data_files": data_files} if data_files else {}))
    samples: list[Sample] = []
    for row in rows:
        if limit is not None and len(samples) >= limit:
            break
        label = "unsafe" if row["type"].startswith(_XSTEST_UNSAFE_PREFIX) else "safe"
        group = f"xstest-{row['id']}"
        samples.append(Sample(id=group, input=row["prompt"], task="en", tags=[label],
                              metadata=_meta(group, revision, label=label, type=row["type"])))
    return samples


def load_global_mmlu(languages: Union[str, Iterable[str]], *, repo: str = GLOBAL_MMLU,
                     split: str = "test", revision: Optional[str] = None,
                     limit: Optional[int] = None) -> list[Sample]:
    """Global-MMLU (Singh et al., ACL 2025): MMLU in 42 languages, one config per language.

    Returns multiple-choice samples: ``choices`` are the four options, ``target`` is the
    0-based index of the right one, and ``task`` is the language. Use
    ``repo=GLOBAL_MMLU_LITE`` for the 400-question set. ``limit`` applies per language.
    Keep ``split="dev"`` for tuning, so choices made there don't touch the test items.
    Raises ``ValueError`` on an answer that isn't A to D.
    """
    samples: list[Sample] = []
    for lang in _languages(languages):
        rows, used = _rows(repo, lang, split, revision)
        for n, row in enumerate(rows):
            if limit is not None and n >= limit:
                break
            answer = str(row["answer"]).strip().upper()
            if answer not in ("A", "B", "C", "D"):
                raise ValueError(f"Global-MMLU {lang} {row['sample_id']}: answer {row['answer']!r}")
            samples.append(Sample(
                id=f"{row['sample_id']}-{lang}", input=row["question"], task=lang, kind=TaskKind.MCQ,
                choices=[row[f"option_{letter}"] for letter in "abcd"], target="ABCD".index(answer),
                metadata=_meta(row["sample_id"], used, subject=row.get("subject"),
                               subject_category=row.get("subject_category"),
                               cultural_sensitivity=row.get("cultural_sensitivity_label"))))
    return samples
