"""Placeholder preprocessing entry points."""

from __future__ import annotations

from pathlib import Path


def preprocess_raw_data(raw_dir: str | Path, processed_dir: str | Path) -> None:
    """Preprocess raw benchmark data into a project-local format."""
    raise NotImplementedError("Data preprocessing will be implemented in phase 2.")
