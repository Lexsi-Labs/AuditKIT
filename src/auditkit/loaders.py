"""Data loaders that produce lists of :class:`~auditkit.sample.Sample`.

T0 provides ``load_csv`` (stdlib only). Optional loaders (``load_hf``,
``load_croissant``) are deferred to their own cycles and raise
:class:`ExtraNotInstalled` when their extra is absent.
"""

from __future__ import annotations

import csv
from typing import Any, Optional

from .errors import ExtraNotInstalled
from .sample import Sample


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
