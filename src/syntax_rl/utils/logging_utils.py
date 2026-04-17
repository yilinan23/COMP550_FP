"""Logging setup helpers."""

from __future__ import annotations

import logging
from pathlib import Path

from .io import ensure_dir, resolve_project_path

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(
    level: int = logging.INFO,
    log_file: str | Path | None = None,
) -> None:
    """Configure root logging for scripts and notebooks."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        input_path = Path(log_file)
        log_path = input_path if input_path.is_absolute() else resolve_project_path(input_path)
        ensure_dir(log_path.parent)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format=DEFAULT_LOG_FORMAT,
        handlers=handlers,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a logger with the given name."""
    return logging.getLogger(name)
