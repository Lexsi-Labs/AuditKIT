"""ExtraNotInstalled message construction.

Previously always prefixed a separately-generated "install auditkit[{extra}]"
in front of whatever hint the caller passed -- every real call site already
passes a complete, ready-to-run instruction as that hint, so the result was
a redundant, confusing message like "install auditkit[vllm] for pip install
auditkit[vllm]". Fixed to use the hint verbatim.
"""
from __future__ import annotations

from auditkit.errors import ExtraNotInstalled


def test_hint_used_verbatim_no_redundant_prefix():
    err = ExtraNotInstalled("vllm", "pip install auditkit[vllm]")
    assert str(err) == "pip install auditkit[vllm]"


def test_falls_back_to_default_message_when_no_hint_given():
    err = ExtraNotInstalled("vision")
    assert str(err) == "install auditkit[vision]"


def test_extra_attribute_preserved():
    err = ExtraNotInstalled("bert-score", "pip install bert-score")
    assert err.extra == "bert-score"
    assert str(err) == "pip install bert-score"
