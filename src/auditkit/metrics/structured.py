"""Structured-output metrics: does the output follow a JSON Schema, and does it carry the right fields.

- :class:`JsonSchemaValid` - 1.0 iff the output parses as JSON and validates against a JSON Schema.
- :class:`FieldF1` - per-field precision / recall / F1 of an extracted JSON object against the target object,
  with path flattening, list alignment and normalisation hooks for numbers, dates and case.

Both are deterministic and need no third-party package. ``JsonSchemaValid`` uses the ``jsonschema`` package when
it is installed (``pip install auditkit[jsonschema]``) and otherwise a small built-in validator for the common
keyword subset; the built-in one refuses a schema that uses a keyword it cannot check instead of passing it.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Union

from auditkit.metric import Metric
from auditkit.registry import METRICS
from auditkit.sample import Sample
from auditkit.score import Score
from auditkit.types import Direction, ScoreKind

_FENCE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def _parse_json(output: str, strip_fences: bool) -> tuple[Any, Optional[str]]:
    """(value, None) or (None, error). Strict ``json.loads`` unless ``strip_fences`` removes one Markdown fence."""
    text = output.strip()
    if strip_fences:
        m = _FENCE.match(text)
        if m:
            text = m.group(1).strip()
    try:
        return json.loads(text), None
    except (json.JSONDecodeError, ValueError) as e:
        return None, f"output is not valid JSON: {e}"


# ---------------------------------------------------------------- JSON Schema: built-in validator

_SUPPORTED = frozenset({
    "type", "enum", "const", "required", "properties", "additionalProperties", "patternProperties", "items",
    "minItems", "maxItems", "uniqueItems", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "multipleOf", "minLength", "maxLength", "pattern", "anyOf", "oneOf", "allOf", "not", "minProperties",
    "maxProperties", "$ref", "$defs", "definitions",
    # annotations and keywords that do not assert anything here
    "$schema", "$id", "$comment", "title", "description", "default", "examples", "format", "deprecated",
    "readOnly", "writeOnly",
})
_MAP_KEYWORDS = ("properties", "patternProperties", "$defs", "definitions")
_SCHEMA_KEYWORDS = ("additionalProperties", "items", "not")
_LIST_KEYWORDS = ("anyOf", "oneOf", "allOf")


def _unsupported(schema: Any, found: Optional[set] = None) -> set[str]:
    """Keywords in ``schema`` (recursively, at schema positions only) the built-in validator cannot check."""
    found = set() if found is None else found
    if not isinstance(schema, dict):
        return found
    found |= {k for k in schema if k not in _SUPPORTED}
    for k in _MAP_KEYWORDS:
        for sub in (schema.get(k) or {}).values():
            _unsupported(sub, found)
    for k in _SCHEMA_KEYWORDS:
        _unsupported(schema.get(k), found)
    for k in _LIST_KEYWORDS:
        for sub in schema.get(k) or []:
            _unsupported(sub, found)
    return found


_TYPES: dict[str, Callable[[Any], bool]] = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: (isinstance(v, int) and not isinstance(v, bool)) or (isinstance(v, float) and v.is_integer()),
}


def _same(a: Any, b: Any) -> bool:
    """JSON equality: 1 == 1.0, but True != 1."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def _resolve(ref: str, root: dict) -> Any:
    if not ref.startswith("#"):
        raise ValueError(f"the built-in validator only resolves local $ref ('#/...'), got {ref!r}")
    node: Any = root
    for part in [p for p in ref[1:].split("/") if p]:
        node = node[part.replace("~1", "/").replace("~0", "~")]
    return node


def _errors(inst: Any, schema: Any, root: dict, path: str, first: bool = True) -> list[str]:
    """Validation errors of ``inst`` against the supported keyword subset; stops after the first when ``first``."""
    if schema is True or schema == {}:
        return []
    if schema is False:
        return [f"{path}: no value is allowed here"]
    out: list[str] = []

    def bad(msg: str) -> bool:
        out.append(f"{path}: {msg}")
        return first

    if "$ref" in schema:
        out += _errors(inst, _resolve(schema["$ref"], root), root, path, first)
        if out and first:
            return out
    if "type" in schema:
        types = [schema["type"]] if isinstance(schema["type"], str) else schema["type"]
        if not any(_TYPES[t](inst) for t in types):
            if bad(f"{inst!r} is not of type {', '.join(map(repr, types))}"):
                return out
    if "enum" in schema and not any(_same(inst, e) for e in schema["enum"]):
        if bad(f"{inst!r} is not one of {schema['enum']!r}"):
            return out
    if "const" in schema and not _same(inst, schema["const"]):
        if bad(f"{inst!r} is not {schema['const']!r}"):
            return out
    if "anyOf" in schema and not any(not _errors(inst, s, root, path) for s in schema["anyOf"]):
        if bad("is not valid under any of the anyOf schemas"):
            return out
    if "oneOf" in schema and sum(not _errors(inst, s, root, path) for s in schema["oneOf"]) != 1:
        if bad("is not valid under exactly one of the oneOf schemas"):
            return out
    for s in schema.get("allOf", []):
        out += _errors(inst, s, root, path, first)
        if out and first:
            return out
    if "not" in schema and not _errors(inst, schema["not"], root, path):
        if bad("is valid under the 'not' schema"):
            return out
    checks: list[tuple[bool, str]] = []
    if isinstance(inst, dict):
        props, patterns = schema.get("properties", {}), schema.get("patternProperties", {})
        checks += [(k not in inst, f"{k!r} is a required property") for k in schema.get("required", [])]
        if "minProperties" in schema:
            checks.append((len(inst) < schema["minProperties"], f"has fewer than {schema['minProperties']} properties"))
        if "maxProperties" in schema:
            checks.append((len(inst) > schema["maxProperties"], f"has more than {schema['maxProperties']} properties"))
        for msg in [c[1] for c in checks if c[0]]:
            if bad(msg):
                return out
        for k, v in inst.items():
            subs = ([props[k]] if k in props else []) + [s for p, s in patterns.items() if re.search(p, k)]
            if not subs and "additionalProperties" in schema:
                subs = [schema["additionalProperties"]]
            for s in subs:
                out += _errors(v, s, root, f"{path}.{k}", first)
                if out and first:
                    return out
        return out
    if isinstance(inst, list):
        if "minItems" in schema:
            checks.append((len(inst) < schema["minItems"], f"has fewer than {schema['minItems']} items"))
        if "maxItems" in schema:
            checks.append((len(inst) > schema["maxItems"], f"has more than {schema['maxItems']} items"))
        if schema.get("uniqueItems"):
            checks.append((any(_same(a, b) for i, a in enumerate(inst) for b in inst[i + 1:]), "has duplicate items"))
        for msg in [c[1] for c in checks if c[0]]:
            if bad(msg):
                return out
        if "items" in schema:
            for i, v in enumerate(inst):
                out += _errors(v, schema["items"], root, f"{path}[{i}]", first)
                if out and first:
                    return out
        return out
    if isinstance(inst, str):
        if "minLength" in schema:
            checks.append((len(inst) < schema["minLength"], f"is shorter than {schema['minLength']} characters"))
        if "maxLength" in schema:
            checks.append((len(inst) > schema["maxLength"], f"is longer than {schema['maxLength']} characters"))
        if "pattern" in schema:
            checks.append((not re.search(schema["pattern"], inst), f"{inst!r} does not match {schema['pattern']!r}"))
    elif _TYPES["number"](inst):
        for kw, bad_if, text in (("minimum", lambda v, b: v < b, "less than"), ("maximum", lambda v, b: v > b, "greater than"),
                                 ("exclusiveMinimum", lambda v, b: v <= b, "not greater than"),
                                 ("exclusiveMaximum", lambda v, b: v >= b, "not less than")):
            if kw in schema:
                checks.append((bad_if(inst, schema[kw]), f"{inst!r} is {text} the {kw} {schema[kw]!r}"))
        if "multipleOf" in schema:
            q = inst / schema["multipleOf"]
            checks.append((not math.isclose(q, round(q), abs_tol=1e-9), f"{inst!r} is not a multiple of {schema['multipleOf']!r}"))
    for msg in [c[1] for c in checks if c[0]]:
        if bad(msg):
            return out
    return out


# ---------------------------------------------------------------- JsonSchemaValid


@METRICS.register("json_schema_valid")
class JsonSchemaValid(Metric):
    """1.0 iff the output parses as JSON and validates against a JSON Schema.

    The schema is a constructor argument (a dict, or a path to a ``.json`` file) or, when ``schema`` is ``None``,
    read per sample from ``sample.metadata[schema_key]``; samples without one are skipped. The reason on a 0.0
    score names the first failure (``$.items[2].price: 'x' is not of type 'number'``).

    ``backend`` is ``"jsonschema"`` (the ``jsonschema`` package: every draft, ``$ref`` resolution, all keywords),
    ``"minimal"`` (built-in; type, enum, const, required, properties, additionalProperties, patternProperties,
    items, array/string/number bounds, anyOf/oneOf/allOf/not, local ``$ref``; ``format`` is not asserted) or
    ``"auto"`` (default: ``jsonschema`` if importable, else ``minimal``). The minimal backend raises ``ValueError``
    for a schema using any other keyword, so a check is never silently skipped. ``strip_fences=True`` accepts one
    ```json fence around the object, as chat models often emit; the default is strict, like ``is_json``.
    """

    kind = ScoreKind.CODE
    direction = Direction.MAXIMIZE

    def __init__(self, schema: Union[dict, str, Path, None] = None, *, backend: str = "auto",
                 strip_fences: bool = False, schema_key: str = "schema") -> None:
        if backend not in ("auto", "jsonschema", "minimal"):
            raise ValueError("backend must be 'auto', 'jsonschema' or 'minimal'")
        if isinstance(schema, (str, Path)):
            schema = json.loads(Path(schema).read_text())
        self._schema, self._schema_key, self._strip = schema, schema_key, strip_fences
        self._backend = self._resolve_backend(backend)
        self._validators: dict[str, Any] = {}
        if schema is not None:
            self._validator(schema)  # fail at construction on a bad schema, not mid-run
        self.name = "json_schema_valid"
        self.required_fields = frozenset()

    @staticmethod
    def _resolve_backend(backend: str) -> str:
        if backend == "minimal":
            return "minimal"
        try:
            import jsonschema  # noqa: F401
        except ImportError:
            if backend == "jsonschema":
                from auditkit.errors import ExtraNotInstalled
                raise ExtraNotInstalled("jsonschema", "pip install auditkit[jsonschema]") from None
            return "minimal"
        return "jsonschema"

    def identity(self) -> dict:
        digest = hashlib.sha256(json.dumps(self._schema, sort_keys=True).encode()).hexdigest()[:16] if self._schema is not None else None
        return {"name": self.name, "schema_sha256": digest, "schema_key": self._schema_key, "backend": self._backend,
                "strip_fences": self._strip}

    def _schema_for(self, sample: Sample) -> Optional[dict]:
        return self._schema if self._schema is not None else (sample.metadata or {}).get(self._schema_key)

    def applicable(self, sample: Sample) -> bool:
        return self._schema_for(sample) is not None

    def _validator(self, schema: dict) -> Callable[[Any], Optional[str]]:
        """A function returning None for a valid instance, else the first error message."""
        key = json.dumps(schema, sort_keys=True)
        if key not in self._validators:
            if self._backend == "jsonschema":
                import jsonschema

                cls = jsonschema.validators.validator_for(schema)
                cls.check_schema(schema)
                v = cls(schema)

                def run(inst: Any) -> Optional[str]:
                    err = next(iter(v.iter_errors(inst)), None)
                    return None if err is None else f"${''.join(f'[{p}]' if isinstance(p, int) else f'.{p}' for p in err.absolute_path)}: {err.message}"
            else:
                missing = _unsupported(schema)
                if missing:
                    raise ValueError(f"the minimal JSON Schema validator cannot check {sorted(missing)}; "
                                     f"pip install auditkit[jsonschema] and use backend='jsonschema'")

                def run(inst: Any) -> Optional[str]:
                    errs = _errors(inst, schema, schema, "$")
                    return errs[0] if errs else None
            self._validators[key] = run
        return self._validators[key]

    def score(self, sample: Sample, output: str, context: Any = None) -> Score:
        schema = self._schema_for(sample)
        if schema is None:
            raise ValueError(f"no schema: pass schema= or set sample.metadata[{self._schema_key!r}]")
        value, err = _parse_json(output, self._strip)
        if err is None:
            err = self._validator(schema)(value)
        return Score(name=self.name, value=0.0 if err else 1.0, kind=self.kind, reason=err,
                     metadata={"backend": self._backend})


# ---------------------------------------------------------------- FieldF1

Normalizer = Callable[[Any], Any]


def _to_number(s: str) -> Optional[float]:
    t = s.strip().replace(",", "")
    t = re.sub(r"^([+-]?)\s*[$€£₹]\s*", r"\1", t)
    try:
        x = float(t)
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def _leaf(v: Any) -> bool:
    return not isinstance(v, (dict, list))


def _gen(path: str) -> str:
    return re.sub(r"\[\d+\]", "[]", path)


def _glob(pattern: str, path: str) -> bool:
    return re.fullmatch(".*".join(map(re.escape, pattern.split("*"))), path) is not None


@METRICS.register("field_f1")
class FieldF1(Metric):
    """Per-field precision, recall and F1 of an extracted JSON object against ``sample.target`` (a JSON object).

    Both sides are flattened to scalar leaves addressed by path (``customer.name``, ``items[0].price``). A
    predicted leaf is a true positive when the target has a leaf at the same path whose normalised value is equal;
    precision = TP / predicted leaves, recall = TP / target leaves, micro-averaged over the sample. Extra keys
    cost precision, missing keys cost recall, a wrong value costs both. Empty containers hold no leaves. Emits
    ``field_f1`` (the headline), ``field_precision`` and ``field_recall``; the f1 score's ``reason`` lists up to
    five wrong or missing paths. Output that is not JSON scores 0 on all three; a target that is not JSON raises.

    Lists are aligned before comparing, never by position alone: lists of scalars as multisets; lists of objects
    by a key field when ``list_keys`` names one (``"id"``, or ``{"items": "sku"}`` mapping a path pattern such as
    ``"orders[].lines"`` to its key field), otherwise greedily, pairing the item pairs with the most matching leaves
    first. A predicted object whose key is absent from the target stays unmatched.

    Normalisation (applied to both sides): whitespace is collapsed (``strip``); ``case_sensitive=False`` casefolds
    strings; ``normalize_numbers`` reads numeric strings (``"1,234.50"``, ``"$5"``; a ``Rs`` prefix is not
    understood, pass a ``normalizers`` hook) as numbers and compares within ``number_tol`` (absolute); ``date_formats`` lists ``strptime`` formats whose
    parsed dates compare equal across formats. ``normalizers`` maps a path pattern (``*`` wildcard, list indices
    written ``[]``, e.g. ``"items[].price"``) to a function applied to the value instead of the defaults.
    ``ignore`` lists path patterns whose leaves are dropped from both sides.
    """

    kind = ScoreKind.CODE
    direction = Direction.MAXIMIZE
    required_fields = frozenset({"target"})

    def __init__(self, *, normalize_numbers: bool = False, number_tol: float = 0.0, case_sensitive: bool = True,
                 strip: bool = True, date_formats: Optional[Sequence[str]] = None,
                 normalizers: Optional[Mapping[str, Normalizer]] = None,
                 list_keys: Union[str, Mapping[str, str], None] = None, ignore: Sequence[str] = (),
                 strip_fences: bool = False) -> None:
        self._num, self._tol, self._cs, self._strip_ws = normalize_numbers, number_tol, case_sensitive, strip
        self._dates = tuple(date_formats or ())
        self._hooks = dict(normalizers or {})
        self._keys = list_keys
        self._ignore = tuple(ignore)
        self._fences = strip_fences
        self.name = "field_f1"

    def identity(self) -> dict:
        hooks = sorted(f"{p}={getattr(fn, '__module__', '')}.{getattr(fn, '__qualname__', repr(fn))}" for p, fn in self._hooks.items())
        return {"name": self.name, "normalize_numbers": self._num, "number_tol": self._tol, "case_sensitive": self._cs,
                "strip": self._strip_ws, "date_formats": list(self._dates), "normalizers": hooks,
                "list_keys": self._keys if isinstance(self._keys, (str, type(None))) else dict(self._keys),
                "ignore": list(self._ignore), "strip_fences": self._fences}

    # -- normalisation and equality
    def _norm(self, gpath: str, v: Any) -> Any:
        for pat, fn in self._hooks.items():
            if _glob(pat, gpath):
                return fn(v)
        if isinstance(v, str):
            s = " ".join(v.split()) if self._strip_ws else v
            if not self._cs:
                s = s.casefold()
            for fmt in self._dates:
                try:
                    return ("date", datetime.strptime(s.strip(), fmt).date().isoformat())
                except ValueError:
                    pass
            if self._num and (n := _to_number(s)) is not None:
                return n
            return s
        if self._num and isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        return v

    def _eq(self, gpath: str, a: Any, b: Any) -> bool:
        na, nb = self._norm(gpath, a), self._norm(gpath, b)
        if isinstance(na, float) and isinstance(nb, float):
            return abs(na - nb) <= self._tol
        return _same(na, nb) if _leaf(na) and _leaf(nb) else na == nb

    def _ignored(self, gpath: str) -> bool:
        return any(_glob(p, gpath) for p in self._ignore)

    # -- counting
    def _leaves(self, v: Any, path: str = "") -> int:
        if isinstance(v, dict):
            return sum(self._leaves(x, f"{path}.{k}" if path else str(k)) for k, x in v.items())
        if isinstance(v, list):
            return sum(self._leaves(x, f"{path}[{i}]") for i, x in enumerate(v))
        return 0 if self._ignored(_gen(path)) else 1

    def _key_field(self, gpath: str) -> Optional[str]:
        if isinstance(self._keys, str):
            return self._keys
        for pat, field in (self._keys or {}).items():
            if _glob(pat, gpath):
                return field
        return None

    def _align(self, pred: list, truth: list, path: str, gpath: str) -> list[tuple[int, int]]:
        item_path = gpath + "[]"
        if all(_leaf(x) for x in pred + truth):  # scalars: multiset matching
            used: set[int] = set()
            pairs = []
            for i, p in enumerate(pred):
                j = next((j for j, t in enumerate(truth) if j not in used and self._eq(item_path, p, t)), None)
                if j is not None:
                    used.add(j)
                    pairs.append((i, j))
            return pairs
        pairs, used_p, used_t = [], set(), set()
        key = self._key_field(gpath)
        if key:
            queues: dict[Any, list[int]] = defaultdict(list)
            for j, t in enumerate(truth):
                if isinstance(t, dict) and key in t and _leaf(t[key]):
                    queues[self._norm(f"{item_path}.{key}", t[key])].append(j)
            for i, p in enumerate(pred):
                if isinstance(p, dict) and key in p and _leaf(p[key]):
                    q = queues.get(self._norm(f"{item_path}.{key}", p[key]))
                    if q:
                        j = q.pop(0)
                        pairs.append((i, j))
                        used_p.add(i)
                        used_t.add(j)
            # keyed items never fall back to greedy: a wrong id is a miss, not a near match
            keyed_p = {i for i, p in enumerate(pred) if isinstance(p, dict) and key in p}
            keyed_t = {j for j, t in enumerate(truth) if isinstance(t, dict) and key in t}
            used_p |= keyed_p
            used_t |= keyed_t
        cands = sorted(((self._tp(pred[i], truth[j], f"{path}[{i}]", item_path), i, j)
                        for i in range(len(pred)) if i not in used_p for j in range(len(truth)) if j not in used_t),
                       key=lambda c: (-c[0], c[1], c[2]))
        for s, i, j in cands:
            if i not in used_p and j not in used_t:
                used_p.add(i)
                used_t.add(j)
                pairs.append((i, j))
        return pairs

    def _tp(self, pred: Any, truth: Any, path: str, gpath: str, wrong: Optional[list] = None) -> int:
        """Matching leaves between two aligned values; mismatching paths are appended to ``wrong`` when given."""
        if isinstance(pred, dict) and isinstance(truth, dict):
            return sum(self._tp(pred[k], truth[k], f"{path}.{k}" if path else str(k), f"{gpath}.{k}" if gpath else str(k), wrong)
                       for k in pred.keys() & truth.keys())
        if isinstance(pred, list) and isinstance(truth, list):
            return sum(self._tp(pred[i], truth[j], f"{path}[{i}]", gpath + "[]", wrong) for i, j in self._align(pred, truth, path, gpath))
        if _leaf(pred) and _leaf(truth):
            if self._ignored(gpath):
                return 0
            ok = self._eq(gpath, pred, truth)
            if not ok and wrong is not None:
                wrong.append(path)
            return int(ok)
        return 0

    def _missing(self, truth: Any, pred: Any, path: str = "") -> list[str]:
        """Target leaf paths absent from the prediction's structure (by path, for the reason text only)."""
        if isinstance(truth, dict):
            out = []
            for k, v in truth.items():
                p = f"{path}.{k}" if path else str(k)
                if isinstance(pred, dict) and k in pred:
                    out += self._missing(v, pred[k], p)
                elif self._leaves(v, p):
                    out.append(p)
            return out
        return []

    def score(self, sample: Sample, output: str, context: Any = None) -> list[Score]:
        target = sample.target
        if isinstance(target, str):
            try:
                target = json.loads(target)
            except ValueError as e:
                raise ValueError(f"FieldF1 needs sample.target to be a JSON object or array: {e}") from None
        if not isinstance(target, (dict, list)):
            raise ValueError("FieldF1 needs sample.target to be a JSON object or array")
        pred, err = _parse_json(output, self._fences)
        if err is None and not isinstance(pred, (dict, list)):
            err = "output JSON is not an object or array"
        if err:
            return [Score(name=n, value=0.0, kind=self.kind, reason=err if n == "field_f1" else None)
                    for n in ("field_f1", "field_precision", "field_recall")]
        n_pred, n_true = self._leaves(pred), self._leaves(target)
        wrong: list[str] = []
        tp = self._tp(pred, target, "", "", wrong)
        if n_pred == 0 and n_true == 0:
            p = r = f1 = 1.0
        else:
            p = tp / n_pred if n_pred else 0.0
            r = tp / n_true if n_true else 0.0
            f1 = 2 * p * r / (p + r) if p + r else 0.0
        missing = self._missing(target, pred)
        reason = None if not (wrong or missing) else "; ".join(
            [f"wrong: {', '.join(wrong[:5])}"] * bool(wrong) + [f"missing: {', '.join(missing[:5])}"] * bool(missing))
        meta = {"tp": tp, "n_pred": n_pred, "n_target": n_true}
        return [Score(name="field_f1", value=f1, kind=self.kind, reason=reason, metadata=meta),
                Score(name="field_precision", value=p, kind=self.kind),
                Score(name="field_recall", value=r, kind=self.kind)]
