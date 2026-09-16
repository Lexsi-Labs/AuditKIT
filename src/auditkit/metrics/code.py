from __future__ import annotations

import json
import re
from typing import Any

from auditkit.metric import Metric
from auditkit.registry import METRICS
from auditkit.sample import Sample
from auditkit.score import Score
from auditkit.types import Direction, ScoreKind


@METRICS.register("equals")
class Equals(Metric):
    kind = ScoreKind.CODE
    direction = Direction.MAXIMIZE

    def __init__(self, ignore_case: bool = False) -> None:
        self._ignore_case = ignore_case
        self.name = "equals_ci" if ignore_case else "equals"
        self.required_fields = frozenset({"target"})

    def score(self, sample: Sample, output: str, context: Any = None) -> Score:
        a = output.strip()
        b = (sample.target or "").strip()
        if self._ignore_case:
            hit = a.lower() == b.lower()
        else:
            hit = a == b
        return Score(name=self.name, value=1.0 if hit else 0.0, kind=self.kind)


@METRICS.register("contains")
class Contains(Metric):
    kind = ScoreKind.CODE
    direction = Direction.MAXIMIZE

    def __init__(self, substring: str, ignore_case: bool = False) -> None:
        self._substring = substring
        self._ignore_case = ignore_case
        prefix = "contains_ci" if ignore_case else "contains"
        self.name = f"{prefix}({substring})"
        self.required_fields = frozenset()

    def score(self, sample: Sample, output: str, context: Any = None) -> Score:
        if self._ignore_case:
            hit = self._substring.lower() in output.lower()
        else:
            hit = self._substring in output
        return Score(name=self.name, value=1.0 if hit else 0.0, kind=self.kind)


@METRICS.register("starts_with")
class StartsWith(Metric):
    kind = ScoreKind.CODE
    direction = Direction.MAXIMIZE

    def __init__(self, prefix: str, ignore_case: bool = False) -> None:
        self._prefix = prefix
        self._ignore_case = ignore_case
        prefix_label = "startswith_ci" if ignore_case else "startswith"
        self.name = f"{prefix_label}({prefix})"
        self.required_fields = frozenset()

    def score(self, sample: Sample, output: str, context: Any = None) -> Score:
        if self._ignore_case:
            hit = output.lower().startswith(self._prefix.lower())
        else:
            hit = output.startswith(self._prefix)
        return Score(name=self.name, value=1.0 if hit else 0.0, kind=self.kind)


@METRICS.register("ends_with")
class EndsWith(Metric):
    kind = ScoreKind.CODE
    direction = Direction.MAXIMIZE

    def __init__(self, suffix: str, ignore_case: bool = False) -> None:
        self._suffix = suffix
        self._ignore_case = ignore_case
        self.name = f"endswith({suffix})"
        self.required_fields = frozenset()

    def score(self, sample: Sample, output: str, context: Any = None) -> Score:
        if self._ignore_case:
            hit = output.lower().endswith(self._suffix.lower())
        else:
            hit = output.endswith(self._suffix)
        return Score(name=self.name, value=1.0 if hit else 0.0, kind=self.kind)


@METRICS.register("regex")
class Regex(Metric):
    kind = ScoreKind.CODE
    direction = Direction.MAXIMIZE

    def __init__(self, pattern: str) -> None:
        self._compiled = re.compile(pattern)
        self.name = f"regex({pattern})"
        self.required_fields = frozenset()

    def score(self, sample: Sample, output: str, context: Any = None) -> Score:
        hit = bool(self._compiled.search(output))
        return Score(name=self.name, value=1.0 if hit else 0.0, kind=self.kind)


@METRICS.register("levenshtein")
class Levenshtein(Metric):
    kind = ScoreKind.CODE
    direction = Direction.MAXIMIZE

    def __init__(self) -> None:
        self.name = "levenshtein"
        self.required_fields = frozenset()

    def score(self, sample: Sample, output: str, context: Any = None) -> Score:
        ref = sample.target or ""
        if not output and not ref:
            return Score(name=self.name, value=0.0, kind=self.kind)
        distance = self._edit_distance(output, ref)
        max_len = max(len(output), len(ref))
        score_val = 1.0 - distance / max_len
        return Score(name=self.name, value=score_val, kind=self.kind)

    @staticmethod
    def _edit_distance(a: str, b: str) -> int:
        m, n = len(a), len(b)
        prev = list(range(n + 1))
        for i in range(1, m + 1):
            curr = [i] * (n + 1)
            for j in range(1, n + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                curr[j] = min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + cost,
                )
            prev = curr
        return prev[n]


@METRICS.register("word_count")
class WordCount(Metric):
    kind = ScoreKind.CODE
    direction = Direction.MAXIMIZE

    def __init__(self, min_words: int = 0, max_words: int | None = None) -> None:
        self._min = min_words
        self._max = max_words
        if min_words > 0 and max_words is not None:
            self.name = f"wordcount({min_words}-{max_words})"
        elif min_words > 0:
            self.name = f"wordcount({min_words}-)"
        elif max_words is not None:
            self.name = f"wordcount(-{max_words})"
        else:
            self.name = f"wordcount({min_words}-)"
        self.required_fields = frozenset()

    def score(self, sample: Sample, output: str, context: Any = None) -> Score:
        count = len(output.split())
        if count < self._min:
            hit = False
        elif self._max is not None and count > self._max:
            hit = False
        else:
            hit = True
        return Score(name=self.name, value=1.0 if hit else 0.0, kind=self.kind)


@METRICS.register("is_json")
class IsJson(Metric):
    kind = ScoreKind.CODE
    direction = Direction.MAXIMIZE

    def __init__(self, require_keys: list[str] | None = None) -> None:
        self._require_keys = require_keys
        if require_keys:
            keys_str = ",".join(require_keys)
            self.name = f"is_json(keys={keys_str})"
        else:
            self.name = "is_json"
        self.required_fields = frozenset()

    def score(self, sample: Sample, output: str, context: Any = None) -> Score:
        try:
            parsed = json.loads(output)
        except (json.JSONDecodeError, ValueError):
            return Score(name=self.name, value=0.0, kind=self.kind)
        if self._require_keys is not None:
            if not isinstance(parsed, dict):
                return Score(name=self.name, value=0.0, kind=self.kind)
            if not all(k in parsed for k in self._require_keys):
                return Score(name=self.name, value=0.0, kind=self.kind)
        return Score(name=self.name, value=1.0, kind=self.kind)


@METRICS.register("f1_score")
class F1Score(Metric):
    kind = ScoreKind.CODE
    direction = Direction.MAXIMIZE

    def __init__(self) -> None:
        self.name = "f1_score"
        self.required_fields = frozenset({"target"})

    def score(self, sample: Sample, output: str, context: Any = None) -> Score:
        output_tokens = output.split()
        target_tokens = (sample.target or "").split()
        if not output_tokens and not target_tokens:
            return Score(name=self.name, value=0.0, kind=self.kind)
        out_set = set(output_tokens)
        tgt_set = set(target_tokens)
        intersection = out_set & tgt_set
        precision = len(intersection) / len(output_tokens) if output_tokens else 0.0
        recall = len(intersection) / len(target_tokens) if target_tokens else 0.0
        if precision == 0.0 and recall == 0.0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        return Score(name=self.name, value=f1, kind=self.kind)
