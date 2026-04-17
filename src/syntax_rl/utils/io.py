"""Path helpers that resolve locations relative to the project root."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the repository root for this project."""
    return Path(__file__).resolve().parents[3]


def resolve_project_path(*parts: str | Path) -> Path:
    """Resolve path parts relative to the repository root."""
    return project_root().joinpath(*parts).resolve()


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return its resolved path."""
    input_path = Path(path)
    directory = input_path if input_path.is_absolute() else resolve_project_path(input_path)
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory
