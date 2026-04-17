"""Load BLiMP and BLiMP-style minimal-pair files for agreement evaluation.

Expected input formats:
- Local JSONL/JSON/CSV records with fields:
  ``uid``, ``phenomenon``, ``sentence_good``, and ``sentence_bad``.
- Official BLiMP JSONL records with fields:
  ``UID``, ``pairID``, ``field``, ``linguistics_term``, ``sentence_good``,
  and ``sentence_bad``.

Additional fields are preserved as metadata. Duplicate ``uid`` values are
treated as invalid input because they make pair-level results ambiguous. For
official BLiMP files, the pair id is normalized to ``UID-pairID`` because
``UID`` names a paradigm and repeats across the 1000 examples in that file.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MinimalPair:
    """A grammatical/ungrammatical sentence pair with optional metadata."""

    grammatical: str
    ungrammatical: str
    phenomenon: str
    pair_id: str | None = None
    metadata: dict[str, Any] | None = None


SUPPORTED_EXTENSIONS = {".csv", ".json", ".jsonl"}
SENTENCE_FIELDS = ("sentence_good", "sentence_bad")
UID_ALIASES = ("uid", "UID", "pair_id", "pairID")
PHENOMENON_ALIASES = ("phenomenon", "linguistics_term", "linguistic_term", "field")
SUBTYPE_ALIASES = ("subtype", "UID", "uid", "linguistics_term", "linguistic_term", "field")


def load_blimp_subset(path: str | Path, phenomenon: str = "agreement") -> list[MinimalPair]:
    """Load minimal pairs matching an agreement-focused phenomenon.

    Args:
        path: A JSONL/JSON/CSV file or a directory containing supported files.
        phenomenon: Case-insensitive substring used to filter records/files.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If no supported files or matching pairs are found.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"BLiMP input path does not exist: {source}")

    files = _discover_files(source, phenomenon)
    if not files:
        raise ValueError(f"No supported BLiMP files found under: {source}")

    pairs: list[MinimalPair] = []
    for file_path in files:
        records = _read_records(file_path)
        for line_number, record in records:
            _validate_record(record, file_path, line_number)
            pairs.append(_record_to_pair(record, file_path))

    if not pairs:
        raise ValueError(f"No minimal-pair records found in {source}")

    _validate_unique_uids(pairs)

    filtered = [pair for pair in pairs if _matches_phenomenon(pair, phenomenon)]
    if not filtered:
        raise ValueError(f"No minimal pairs matched phenomenon '{phenomenon}' in {source}")
    return filtered


def _discover_files(path: Path, phenomenon: str) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_EXTENSIONS else []

    all_files = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS and not candidate.name.startswith("._")
    )
    phenomenon_files = [file_path for file_path in all_files if phenomenon.lower() in file_path.stem.lower()]
    return phenomenon_files or all_files


def _read_records(path: Path) -> list[tuple[int, dict[str, Any]]]:
    extension = path.suffix.lower()
    if extension == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [(index, record) for index, record in enumerate(csv.DictReader(handle), start=2)]
    if extension == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            records = []
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON on line {line_number} in {path}: {error}") from error
                if not isinstance(payload, dict):
                    raise ValueError(f"Line {line_number} in {path} must be a JSON object.")
                records.append((line_number, payload))
            return records
    if extension == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return [(index, record) for index, record in enumerate(payload, start=1)]
        if isinstance(payload, dict) and isinstance(payload.get("pairs"), list):
            return [(index, record) for index, record in enumerate(payload["pairs"], start=1)]
    raise ValueError(f"Unsupported or malformed BLiMP file: {path}")


def _validate_record(record: dict[str, Any], source_path: Path, line_number: int) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"Record/line {line_number} in {source_path} must be an object with BLiMP fields.")
    missing = [field for field in SENTENCE_FIELDS if _is_missing(record.get(field))]
    if missing:
        fields = ", ".join(missing)
        raise ValueError(f"Missing required field(s) {fields} in {source_path} at record/line {line_number}.")
    if _is_missing(_first_present(record, UID_ALIASES)) and _is_missing(record.get("pairID")):
        raise ValueError(f"Missing required uid/UID/pairID field in {source_path} at record/line {line_number}.")
    if _is_missing(_first_present(record, PHENOMENON_ALIASES)):
        raise ValueError(
            f"Missing required phenomenon/linguistics_term/field field in {source_path} at record/line {line_number}."
        )


def _record_to_pair(record: dict[str, Any], source_path: Path) -> MinimalPair:
    metadata = {
        key: value
        for key, value in record.items()
        if key not in {"uid", "phenomenon", "sentence_good", "sentence_bad"}
    }
    metadata["source_file"] = str(source_path)
    subtype = _first_present(record, SUBTYPE_ALIASES)
    if subtype is not None:
        metadata.setdefault("subtype", str(subtype))

    return MinimalPair(
        grammatical=str(record["sentence_good"]),
        ungrammatical=str(record["sentence_bad"]),
        phenomenon=_normalize_phenomenon(record),
        pair_id=_normalize_pair_id(record),
        metadata=metadata,
    )


def _validate_unique_uids(pairs: list[MinimalPair]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for pair in pairs:
        if pair.pair_id in seen:
            duplicates.add(str(pair.pair_id))
        seen.add(str(pair.pair_id))
    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates))
        raise ValueError(f"Duplicate BLiMP uid value(s) found: {duplicate_list}")


def _is_missing(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def _first_present(record: dict[str, Any], aliases: tuple[str, ...]) -> Any:
    for alias in aliases:
        value = record.get(alias)
        if not _is_missing(value):
            return value
    return None


def _normalize_pair_id(record: dict[str, Any]) -> str:
    uid = _first_present(record, ("uid", "UID", "pair_id"))
    pair_index = record.get("pairID")
    if not _is_missing(uid) and not _is_missing(pair_index):
        return f"{uid}-{pair_index}"
    if not _is_missing(uid):
        return str(uid)
    return str(pair_index)


def _normalize_phenomenon(record: dict[str, Any]) -> str:
    value = _first_present(record, ("phenomenon", "linguistics_term", "linguistic_term"))
    if not _is_missing(value):
        return str(value)
    return str(record["field"])


def _matches_phenomenon(pair: MinimalPair, phenomenon: str) -> bool:
    query = phenomenon.lower()
    metadata = pair.metadata or {}
    haystack = [
        pair.phenomenon,
        str(metadata.get("subtype", "")),
        str(metadata.get("UID", "")),
        str(metadata.get("linguistics_term", "")),
        str(metadata.get("linguistic_term", "")),
        str(metadata.get("field", "")),
    ]
    return any(query in value.lower() for value in haystack)
