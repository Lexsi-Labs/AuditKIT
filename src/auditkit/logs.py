"""Logging setup for AuditKIT."""
from __future__ import annotations

import logging
import sys


def configure_logging(*, level: int | str = logging.INFO, fmt: str | None = None) -> None:
    """Configure auditkit logging with a structured format."""
    logger = logging.getLogger("auditkit")
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(
            fmt or "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
