"""Shared class-definition-time guard against forgotten ``identity()``
overrides, used by :class:`Annotator`, :class:`Adapter`, and :class:`Metric`.

The bug this catches: a custom subclass takes real constructor config but
never overrides the base ``identity()`` (which only ever returns
``{"name": self.name}``) -- two differently-configured instances then
silently produce the same ``RunSpec.fingerprint()``, and ``Runner.run()``
can return a stale cached ``RunResult`` scored under a different config,
with no error anywhere. Warning at class-definition time (via
``__init_subclass__``) catches this as soon as the offending class is
*defined* -- before anyone even instantiates it, let alone runs anything.

One real false-positive this file exists to avoid: several built-in metrics
(``Contains``, ``StartsWith``, ``KeywordDetector``, ...) never override
``identity()`` either, but they're not buggy -- they assign a
parameter-derived string to ``self.name`` inside ``__init__``
(e.g. ``self.name = f"contains({substring})"``), which already makes the
base ``identity()`` (``{"name": self.name}``) parameter-aware. Blindly
warning on "no identity() override" alone would fire on every one of those.
Detecting this safely means checking whether ``__init__`` assigns
``self.name`` *without* instantiating the class (constructing an arbitrary
user class as a side effect of merely defining it is unacceptable -- some
constructors have required args with no defaults, and even when they don't,
running arbitrary user `__init__` code at import time is not something a
library should ever do implicitly). AST inspection of the already-loaded
source is the only way to check this without executing anything.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import warnings
from typing import Any


def _assigns_self_attr(init_func: Any, attr: str) -> bool:
    """True iff *init_func*'s body contains ``self.<attr> = ...`` anywhere
    (a plain assignment, or a property setter reached via ``self.<attr> =``
    -- both look identical in the AST).

    Best-effort: classes defined somewhere ``inspect.getsource`` can't reach
    (a REPL, ``exec``, certain notebook cells) fail open (return True, i.e.
    "assume it's fine") rather than risk a false-positive warning on code
    this check has no reliable way to inspect.
    """
    try:
        source = textwrap.dedent(inspect.getsource(init_func))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError):
        return True
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and target.attr == attr
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                return True
    return False


def warn_if_identity_incomplete(
    cls: type, base_cls: type, main_method: str, name_attr: str = "name",
) -> None:
    """Warn if *cls* (a fresh subclass of *base_cls*) takes real constructor
    config but neither overrides ``identity()`` nor derives ``self.<name_attr>``
    from it -- called from ``base_cls.__init_subclass__``. *name_attr* is
    ``"name"`` for :class:`Metric`/:class:`Annotator`, ``"method"`` for
    :class:`Adapter` (whichever attribute that base's ``identity()`` keys off).
    """
    if cls.identity is not base_cls.identity:
        return  # this class (or a parent) already overrides identity()
    init = cls.__dict__.get("__init__")
    if init is None:
        return  # no new __init__ defined here -- nothing new to guard
    params = [
        p for pname, p in inspect.signature(init).parameters.items()
        if pname != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]
    if not params:
        return
    if _assigns_self_attr(init, name_attr):
        return  # self.<name_attr> is parameter-derived -- base identity() is already fine
    warnings.warn(
        f"{cls.__name__} defines __init__ parameters "
        f"({', '.join(p.name for p in params)}) but does not override "
        f"{base_cls.__name__}.identity() (and doesn't derive self.{name_attr} "
        f"from them either). Differently-configured instances will silently "
        f"produce the same RunSpec.fingerprint() and can reuse a stale "
        f"cached RunResult scored under a different config. Add an "
        f"identity() override that includes every constructor argument "
        f"affecting {main_method}()'s output.",
        stacklevel=3,
    )
