"""Small text helpers shared by the metrics and the red-team detectors."""
from __future__ import annotations

#: Apostrophe look-alikes a model can write for ``'``: right and left single quotes, modifier letter apostrophe, prime.
_APOSTROPHES = {ord(c): "'" for c in "’‘ʼ′"}


def fold_apostrophes(text: str) -> str:
    """Straighten curly apostrophes (U+2019, U+2018, U+02BC, U+2032) and nothing else.

    Case is kept, so it can sit in front of a case-sensitive matcher: ``"can’t"`` matches a marker
    written ``"can't"``.
    """
    return (text or "").translate(_APOSTROPHES)
