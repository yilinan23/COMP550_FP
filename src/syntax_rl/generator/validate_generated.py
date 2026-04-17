"""Validate generated subject-verb agreement JSONL files."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from syntax_rl.generator.grammar_checks import compute_dependency_distance
from syntax_rl.utils import resolve_project_path

REQUIRED_FIELDS = (
    "uid",
    "phenomenon",
    "sentence_good",
    "sentence_bad",
    "dependency_distance",
    "attractor_count",
    "clause_depth",
    "template_type",
    "subtype",
)
NUMERIC_METADATA_FIELDS = ("dependency_distance", "attractor_count", "clause_depth")
KNOWN_TEMPLATE_TYPES = {"simple_agreement", "pp_attractor", "relative_clause"}
KNOWN_VERB_PAIRS = {
    ("runs", "run"),
    ("smiles", "smile"),
    ("arrives", "arrive"),
    ("waits", "wait"),
    ("sleeps", "sleep"),
    ("laughs", "laugh"),
    ("speaks", "speak"),
    ("listens", "listen"),
    ("works", "work"),
    ("dances", "dance"),
    ("travels", "travel"),
    ("returns", "return"),
    ("studies", "study"),
    ("notices", "notice"),
    ("answers", "answer"),
    ("opens", "open"),
    ("closes", "close"),
    ("moves", "move"),
    ("reads", "read"),
    ("writes", "write"),
}
EXPECTED_METADATA_BY_TEMPLATE = {
    "simple_agreement": {"dependency_distance": {1}, "attractor_count": {0}, "clause_depth": {0}},
    "pp_attractor": {"dependency_distance": {4}, "attractor_count": {1}, "clause_depth": {0}},
    "relative_clause": {"dependency_distance": {5}, "attractor_count": {0, 1}, "clause_depth": {1}},
}


@dataclass(frozen=True)
class ValidationError:
    """One validation error for a generated JSONL record."""

    line_number: int
    uid: str
    message: str


@dataclass(frozen=True)
class ValidationReport:
    """Validation result for a generated JSONL file."""

    path: Path
    total_records: int
    errors: list[ValidationError]

    @property
    def is_valid(self) -> bool:
        """Return whether no validation errors were found."""
        return not self.errors


def validate_generated_file(path: str | Path) -> ValidationReport:
    """Validate every JSON object in a generated agreement JSONL file."""
    jsonl_path = resolve_project_path(path)
    errors: list[ValidationError] = []
    total_records = 0

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            total_records += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                errors.append(ValidationError(line_number, "<unknown>", f"Invalid JSON: {error}"))
                continue
            if not isinstance(record, dict):
                errors.append(ValidationError(line_number, "<unknown>", "Record must be a JSON object."))
                continue
            errors.extend(validate_generated_record(record, line_number=line_number))

    return ValidationReport(path=jsonl_path, total_records=total_records, errors=errors)


def validate_generated_record(record: dict[str, Any], line_number: int = 0) -> list[ValidationError]:
    """Validate one generated agreement minimal-pair record."""
    uid = str(record.get("uid") or "<missing uid>")
    errors: list[ValidationError] = []

    for field_name in REQUIRED_FIELDS:
        if _is_missing(record.get(field_name)):
            errors.append(ValidationError(line_number, uid, f"Missing or empty required field: {field_name}"))

    if errors:
        return errors

    good_tokens = _sentence_tokens(str(record["sentence_good"]))
    bad_tokens = _sentence_tokens(str(record["sentence_bad"]))
    differing_indices = [
        index
        for index, (good_token, bad_token) in enumerate(zip(good_tokens, bad_tokens))
        if good_token != bad_token
    ]
    if len(good_tokens) != len(bad_tokens):
        errors.append(ValidationError(line_number, uid, "Good/bad sentences must have the same token length."))
    elif len(differing_indices) != 1:
        errors.append(
            ValidationError(
                line_number,
                uid,
                f"Good/bad sentences must differ in exactly one token; found {len(differing_indices)}.",
            )
        )
    elif differing_indices[0] != len(good_tokens) - 1:
        errors.append(ValidationError(line_number, uid, "The differing token must be the final main verb."))
    elif not _is_known_agreement_contrast(good_tokens[-1], bad_tokens[-1]):
        errors.append(
            ValidationError(
                line_number,
                uid,
                f"Differing final tokens are not a known agreement verb pair: {good_tokens[-1]} / {bad_tokens[-1]}",
            )
        )

    errors.extend(_validate_numeric_metadata(record, line_number, uid))
    errors.extend(_validate_template_metadata(record, line_number, uid))
    errors.extend(_validate_dependency_distance(record, line_number, uid))
    return errors


def main() -> None:
    """CLI entry point for generated-file validation."""
    parser = argparse.ArgumentParser(description="Validate generated agreement JSONL records.")
    parser.add_argument("path", help="Path to generated JSONL, relative to project root by default.")
    args = parser.parse_args()

    report = validate_generated_file(args.path)
    if report.is_valid:
        print(f"Validation passed: {report.total_records} records in {report.path}")
        return

    print(f"Validation failed: {len(report.errors)} error(s) across {report.total_records} records in {report.path}")
    for error in report.errors:
        print(f"line {error.line_number} uid={error.uid}: {error.message}")
    raise SystemExit(1)


def _validate_numeric_metadata(record: dict[str, Any], line_number: int, uid: str) -> list[ValidationError]:
    errors: list[ValidationError] = []
    for field_name in NUMERIC_METADATA_FIELDS:
        try:
            value = int(record[field_name])
        except (TypeError, ValueError):
            errors.append(ValidationError(line_number, uid, f"Metadata field must be an integer: {field_name}"))
            continue
        if value < 0:
            errors.append(ValidationError(line_number, uid, f"Metadata field must be non-negative: {field_name}"))
    return errors


def _validate_template_metadata(record: dict[str, Any], line_number: int, uid: str) -> list[ValidationError]:
    template_type = str(record["template_type"])
    if template_type not in KNOWN_TEMPLATE_TYPES:
        return [ValidationError(line_number, uid, f"Unknown template_type: {template_type}")]

    errors: list[ValidationError] = []
    expected = EXPECTED_METADATA_BY_TEMPLATE[template_type]
    for field_name, expected_values in expected.items():
        if int(record[field_name]) not in expected_values:
            expected_display = ", ".join(str(value) for value in sorted(expected_values))
            errors.append(
                ValidationError(
                    line_number,
                    uid,
                    f"{field_name}={record[field_name]} is inconsistent with template_type={template_type}; "
                    f"expected one of {{{expected_display}}}.",
                )
            )
    return errors


def _validate_dependency_distance(record: dict[str, Any], line_number: int, uid: str) -> list[ValidationError]:
    try:
        computed_distance = compute_dependency_distance(str(record["sentence_good"]))
    except ValueError as error:
        return [ValidationError(line_number, uid, str(error))]

    declared_distance = int(record["dependency_distance"])
    if declared_distance != computed_distance:
        return [
            ValidationError(
                line_number,
                uid,
                f"dependency_distance={declared_distance} does not match computed distance {computed_distance}.",
            )
        ]
    return []


def _sentence_tokens(sentence: str) -> list[str]:
    return sentence.rstrip(".").split()


def _is_known_agreement_contrast(good_token: str, bad_token: str) -> bool:
    return (good_token, bad_token) in KNOWN_VERB_PAIRS or (bad_token, good_token) in KNOWN_VERB_PAIRS


def _is_missing(value: Any) -> bool:
    return value is None or str(value).strip() == ""


if __name__ == "__main__":
    main()
