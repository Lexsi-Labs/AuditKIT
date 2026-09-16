"""Pre-built benchmark scenarios (optional extra: auditkit[interop])."""
from __future__ import annotations

from . import arc, gsm8k, hellaswag, humaneval, mmlu, truthfulqa  # noqa: F401 -- triggers @SCENARIOS.register decorators
