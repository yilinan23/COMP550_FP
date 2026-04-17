"""Sanity checks and error analysis for benchmark-vs-RL comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from syntax_rl.generator.realize import DEFAULT_LEXICON, VERB_PAIRS
from syntax_rl.generator.validate_generated import ValidationError, validate_generated_record
from syntax_rl.utils import ensure_dir, resolve_project_path

PREPOSITIONS = {"near", "behind", "beside", "with"}
RELATIVE_PRONOUNS = {"that", "who"}
MODEL_TINY = "hf_tiny_gpt2"
MODEL_GPT2 = "hf_gpt2"
STRONGER_MODELS = ("hf_distilgpt2", "hf_gpt2")


def run_sanity_check(config_path: str | Path) -> dict[str, Path]:
    """Run dataset, lexical, validation, and model-disagreement diagnostics."""
    config = _load_config(resolve_project_path(config_path))
    output_dir = ensure_dir(config.get("outputs", {}).get("sanity_dir", "outputs/sanity_check"))
    experiment_name = config.get("experiment", {}).get("name", "sanity_check")
    sample_size = int(config.get("analysis", {}).get("manual_sample_size", 20))

    inputs = config["inputs"]
    benchmark_records = _load_jsonl(resolve_project_path(inputs["benchmark_dataset"]))
    rl_records = _load_jsonl(resolve_project_path(inputs["rl_dataset"]))
    comparison_rows = _load_csv(resolve_project_path(inputs["comparison_results"]))

    dataset_summary = _dataset_summary(benchmark_records, rl_records)
    validation_rows = _validation_rows(rl_records)
    duplicate_summary, duplicate_rows = _duplicate_analysis(rl_records)
    lexical_summary, lexical_rows = _lexical_analysis(benchmark_records, rl_records)
    disagreement_rows, disagreement_summary = _model_disagreement_analysis(comparison_rows)
    extreme_rows = _tiny_extreme_failure_rows(comparison_rows)
    manual_tables = _manual_inspection_tables(comparison_rows, sample_size=sample_size)
    final_summary = _final_summary(
        dataset_summary=dataset_summary,
        validation_rows=validation_rows,
        duplicate_summary=duplicate_summary,
        lexical_summary=lexical_summary,
        disagreement_summary=disagreement_summary,
        comparison_rows=comparison_rows,
    )

    paths = {
        "dataset_summary_json": output_dir / f"{experiment_name}_dataset_summary.json",
        "validation_csv": output_dir / f"{experiment_name}_pair_validation.csv",
        "duplicate_summary_json": output_dir / f"{experiment_name}_duplicate_analysis.json",
        "duplicate_examples_csv": output_dir / f"{experiment_name}_duplicate_examples.csv",
        "lexical_summary_json": output_dir / f"{experiment_name}_lexical_diversity.json",
        "lexical_by_subtype_csv": output_dir / f"{experiment_name}_lexical_by_subtype.csv",
        "tiny_extreme_failures_csv": output_dir / f"{experiment_name}_tiny_extreme_failures.csv",
        "model_disagreement_csv": output_dir / f"{experiment_name}_model_disagreement.csv",
        "manual_tiny_hardest_csv": output_dir / f"{experiment_name}_manual_tiny_hardest_rl.csv",
        "manual_gpt2_hardest_csv": output_dir / f"{experiment_name}_manual_gpt2_hardest_rl.csv",
        "manual_shared_subtype_csv": output_dir / f"{experiment_name}_manual_shared_subtype_examples.csv",
        "manual_examples_json": output_dir / f"{experiment_name}_manual_examples.json",
        "summary_json": output_dir / f"{experiment_name}_summary.json",
        "report_md": output_dir / f"{experiment_name}_report.md",
    }

    _write_json(dataset_summary, paths["dataset_summary_json"])
    _write_csv(validation_rows, paths["validation_csv"])
    _write_json(duplicate_summary, paths["duplicate_summary_json"])
    _write_csv(duplicate_rows, paths["duplicate_examples_csv"])
    _write_json(lexical_summary, paths["lexical_summary_json"])
    _write_csv(lexical_rows, paths["lexical_by_subtype_csv"])
    _write_csv(extreme_rows, paths["tiny_extreme_failures_csv"])
    _write_csv(disagreement_rows, paths["model_disagreement_csv"])
    _write_csv(manual_tables["tiny_hardest_rl"], paths["manual_tiny_hardest_csv"])
    _write_csv(manual_tables["gpt2_hardest_rl"], paths["manual_gpt2_hardest_csv"])
    _write_csv(manual_tables["shared_subtype_examples"], paths["manual_shared_subtype_csv"])
    _write_json(manual_tables, paths["manual_examples_json"])
    _write_json(
        {
            "dataset_summary": dataset_summary,
            "duplicate_summary": duplicate_summary,
            "lexical_summary": lexical_summary,
            "disagreement_summary": disagreement_summary,
            "final_summary": final_summary,
            "config": config,
        },
        paths["summary_json"],
    )
    _write_markdown_report(final_summary, paths["report_md"])
    return paths


def main() -> None:
    """CLI entry point for sanity-check diagnostics."""
    parser = argparse.ArgumentParser(description="Run sanity checks for benchmark-vs-RL comparison outputs.")
    parser.add_argument("--config", default="configs/sanity_check.yaml")
    args = parser.parse_args()
    outputs = run_sanity_check(args.config)
    for label, path in outputs.items():
        print(f"Wrote {label} to {path}")


def _dataset_summary(benchmark_records: list[dict[str, Any]], rl_records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "benchmark": _record_distribution_summary(benchmark_records),
        "rl": _record_distribution_summary(rl_records),
        "subtype_coverage": _coverage_summary(benchmark_records, rl_records, "subtype"),
        "template_type_coverage": _coverage_summary(benchmark_records, rl_records, "template_type"),
    }


def _record_distribution_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(records),
        "subtype_counts": dict(Counter(str(record.get("subtype")) for record in records)),
        "template_type_counts": dict(Counter(str(record.get("template_type")) for record in records)),
        "dependency_distance_bucket_counts": dict(Counter(_distance_bucket(record.get("dependency_distance")) for record in records)),
        "clause_depth_counts": dict(Counter(str(record.get("clause_depth")) for record in records)),
    }


def _coverage_summary(
    benchmark_records: list[dict[str, Any]],
    rl_records: list[dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    benchmark_values = {str(record.get(field)) for record in benchmark_records}
    rl_values = {str(record.get(field)) for record in rl_records}
    return {
        "shared": sorted(benchmark_values & rl_values),
        "benchmark_only": sorted(benchmark_values - rl_values),
        "rl_only": sorted(rl_values - benchmark_values),
        "matched": benchmark_values == rl_values,
    }


def _validation_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        validation_errors = validate_generated_record(record, line_number=index)
        heuristic_errors = _heuristic_subtype_errors(record, index)
        errors = validation_errors + heuristic_errors
        rows.append(
            {
                "uid": record.get("uid"),
                "subtype": record.get("subtype"),
                "template_type": record.get("template_type"),
                "valid": not errors,
                "error_count": len(errors),
                "errors": " | ".join(error.message for error in errors),
                "sentence_good": record.get("sentence_good"),
                "sentence_bad": record.get("sentence_bad"),
            }
        )
    return rows


def _heuristic_subtype_errors(record: dict[str, Any], line_number: int) -> list[ValidationError]:
    uid = str(record.get("uid", "<missing uid>"))
    subtype = str(record.get("subtype", ""))
    template_type = str(record.get("template_type", ""))
    good = str(record.get("sentence_good", ""))
    tokens = _tokens(good)
    errors: list[ValidationError] = []
    if template_type == "pp_attractor":
        prepositions = [token for token in tokens if token in PREPOSITIONS]
        if not prepositions:
            errors.append(ValidationError(line_number, uid, "PP subtype has no recognized preposition."))
        elif not subtype.endswith(prepositions[0]):
            errors.append(ValidationError(line_number, uid, f"Subtype {subtype} does not match preposition {prepositions[0]}."))
        if "plural_attractor" in subtype and int(record.get("attractor_count", 0)) != 1:
            errors.append(ValidationError(line_number, uid, "PP attractor subtype should have attractor_count=1."))
    if template_type == "relative_clause":
        relative_pronouns = [token for token in tokens if token in RELATIVE_PRONOUNS]
        if not relative_pronouns:
            errors.append(ValidationError(line_number, uid, "Relative-clause subtype has no that/who token."))
        elif subtype.endswith("that") and "that" not in relative_pronouns:
            errors.append(ValidationError(line_number, uid, "Subtype ends with that but sentence does not use that."))
        elif subtype.endswith("who") and "who" not in relative_pronouns:
            errors.append(ValidationError(line_number, uid, "Subtype ends with who but sentence does not use who."))
        if "plural_embedded" in subtype and int(record.get("attractor_count", 0)) != 1:
            errors.append(ValidationError(line_number, uid, "Plural-embedded relative subtype should have attractor_count=1."))
    return errors


def _duplicate_analysis(records: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pair_counts = Counter((record.get("sentence_good"), record.get("sentence_bad")) for record in records)
    good_counts = Counter(record.get("sentence_good") for record in records)
    bad_counts = Counter(record.get("sentence_bad") for record in records)
    uid_counts = Counter(record.get("uid") for record in records)
    rows: list[dict[str, Any]] = []
    for (good, bad), count in pair_counts.items():
        if count > 1:
            rows.append({"kind": "duplicate_pair", "count": count, "sentence_good": good, "sentence_bad": bad})
    for uid, count in uid_counts.items():
        if count > 1:
            rows.append({"kind": "duplicate_uid", "count": count, "uid": uid})
    summary = {
        "duplicate_pair_count": sum(1 for count in pair_counts.values() if count > 1),
        "duplicate_good_sentence_count": sum(1 for count in good_counts.values() if count > 1),
        "duplicate_bad_sentence_count": sum(1 for count in bad_counts.values() if count > 1),
        "duplicate_uid_count": sum(1 for count in uid_counts.values() if count > 1),
        "max_pair_repetition": max(pair_counts.values(), default=0),
        "max_good_sentence_repetition": max(good_counts.values(), default=0),
    }
    return summary, rows


def _lexical_analysis(
    benchmark_records: list[dict[str, Any]],
    rl_records: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = {
        "benchmark": _lexical_summary_for_records(benchmark_records),
        "rl": _lexical_summary_for_records(rl_records),
    }
    rows: list[dict[str, Any]] = []
    for dataset, records in (("benchmark", benchmark_records), ("rl", rl_records)):
        for subtype in sorted({str(record.get("subtype")) for record in records}):
            subtype_records = [record for record in records if str(record.get("subtype")) == subtype]
            lex = _lexical_summary_for_records(subtype_records)
            rows.append(
                {
                    "dataset": dataset,
                    "subtype": subtype,
                    "total": len(subtype_records),
                    "unique_nouns": lex["unique_noun_count"],
                    "unique_verbs": lex["unique_verb_count"],
                    "unique_prepositions": lex["unique_preposition_count"],
                    "top_nouns": json.dumps(lex["top_nouns"][:5]),
                    "top_verbs": json.dumps(lex["top_verbs"][:5]),
                    "top_patterns": json.dumps(lex["top_patterns"][:5]),
                }
            )
    return summary, rows


def _lexical_summary_for_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    nouns: Counter[str] = Counter()
    verbs: Counter[str] = Counter()
    prepositions: Counter[str] = Counter()
    patterns: Counter[str] = Counter()
    for record in records:
        features = _lexical_features(record)
        nouns.update(features["nouns"])
        verbs.update(features["verbs"])
        prepositions.update(features["prepositions"])
        patterns[features["pattern"]] += 1
    total = max(len(records), 1)
    return {
        "total": len(records),
        "unique_noun_count": len(nouns),
        "unique_verb_count": len(verbs),
        "unique_preposition_count": len(prepositions),
        "top_nouns": nouns.most_common(10),
        "top_verbs": verbs.most_common(10),
        "top_prepositions": prepositions.most_common(10),
        "top_patterns": patterns.most_common(10),
        "top_pattern_share": patterns.most_common(1)[0][1] / total if patterns else 0.0,
    }


def _lexical_features(record: dict[str, Any]) -> dict[str, Any]:
    tokens = _tokens(str(record.get("sentence_good", "")))
    noun_vocab = _noun_vocab()
    verb_vocab = _verb_vocab()
    nouns = [token for token in tokens if token in noun_vocab]
    verbs = [token for token in tokens if token in verb_vocab]
    prepositions = [token for token in tokens if token in PREPOSITIONS]
    pattern_parts = [
        str(record.get("subtype")),
        "+".join(nouns[:3]),
        "+".join(verbs[-2:]),
        "+".join(prepositions),
    ]
    return {
        "nouns": nouns,
        "verbs": verbs,
        "prepositions": prepositions,
        "pattern": "|".join(pattern_parts),
    }


def _model_disagreement_analysis(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_example: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_example[(row["dataset"], row["pair_id"])][row["model_id"]] = row
    output_rows: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    for (dataset, pair_id), model_rows in sorted(by_example.items()):
        if MODEL_TINY not in model_rows:
            continue
        failed_models = sorted(model_id for model_id, row in model_rows.items() if not _as_bool(row.get("correct")))
        category = _disagreement_category(failed_models)
        category_counts[f"{dataset}:{category}"] += 1
        base = next(iter(model_rows.values()))
        output_rows.append(
            {
                "dataset": dataset,
                "pair_id": pair_id,
                "subtype": base.get("subtype"),
                "template_type": base.get("template_type"),
                "category": category,
                "failed_models": ",".join(failed_models),
                "tiny_correct": _as_bool(model_rows.get(MODEL_TINY, {}).get("correct")),
                "distilgpt2_correct": _as_bool(model_rows.get("hf_distilgpt2", {}).get("correct")),
                "gpt2_correct": _as_bool(model_rows.get(MODEL_GPT2, {}).get("correct")),
                "tiny_margin": model_rows.get(MODEL_TINY, {}).get("preference_margin"),
                "gpt2_margin": model_rows.get(MODEL_GPT2, {}).get("preference_margin"),
                "sentence_good": base.get("grammatical"),
                "sentence_bad": base.get("ungrammatical"),
            }
        )
    return output_rows, dict(category_counts)


def _disagreement_category(failed_models: list[str]) -> str:
    failed = set(failed_models)
    stronger = set(STRONGER_MODELS)
    if failed == {MODEL_TINY}:
        return "tiny_fails_stronger_succeed"
    if MODEL_TINY in failed and stronger.issubset(failed):
        return "all_models_fail"
    if failed and MODEL_TINY not in failed and len(failed & stronger) == 1:
        return "only_one_stronger_model_fails"
    if not failed:
        return "all_models_succeed"
    return "mixed_disagreement"


def _tiny_extreme_failure_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    tiny_rows = [row for row in rows if row.get("model_id") == MODEL_TINY]
    benchmark_success_subtypes = {
        row["subtype"]
        for row in tiny_rows
        if row.get("dataset") == "benchmark" and _as_bool(row.get("correct"))
    }
    extreme = [
        row
        for row in tiny_rows
        if row.get("dataset") == "rl" and not _as_bool(row.get("correct")) and row.get("subtype") in benchmark_success_subtypes
    ]
    output = []
    for row in sorted(extreme, key=lambda item: (float(item["preference_margin"]), item["subtype"]))[:100]:
        features = _lexical_features(_record_from_result_row(row))
        output.append(
            {
                "pair_id": row["pair_id"],
                "subtype": row["subtype"],
                "template_type": row["template_type"],
                "preference_margin": row["preference_margin"],
                "grammatical_score": row["grammatical_score"],
                "ungrammatical_score": row["ungrammatical_score"],
                "lexical_pattern": features["pattern"],
                "sentence_good": row["grammatical"],
                "sentence_bad": row["ungrammatical"],
            }
        )
    return output


def _manual_inspection_tables(rows: list[dict[str, str]], sample_size: int) -> dict[str, list[dict[str, Any]]]:
    return {
        "tiny_hardest_rl": _hardest_examples(rows, MODEL_TINY, "rl", sample_size),
        "gpt2_hardest_rl": _hardest_examples(rows, MODEL_GPT2, "rl", sample_size),
        "shared_subtype_examples": _shared_subtype_examples(rows, sample_size),
    }


def _hardest_examples(rows: list[dict[str, str]], model_id: str, dataset: str, sample_size: int) -> list[dict[str, Any]]:
    model_rows = [row for row in rows if row.get("model_id") == model_id and row.get("dataset") == dataset]
    ranked = sorted(model_rows, key=lambda row: (float(row["preference_margin"]), _as_bool(row.get("correct"))))
    return [_inspection_row(row) for row in ranked[:sample_size]]


def _shared_subtype_examples(rows: list[dict[str, str]], sample_size: int) -> list[dict[str, Any]]:
    rl_gpt2 = _hardest_examples(rows, MODEL_GPT2, "rl", max(sample_size // 2, 1))
    subtypes = {row["subtype"] for row in rl_gpt2}
    benchmark_rows = [
        row for row in rows if row.get("model_id") == MODEL_GPT2 and row.get("dataset") == "benchmark" and row.get("subtype") in subtypes
    ]
    ranked_benchmark = sorted(benchmark_rows, key=lambda row: float(row["preference_margin"]))[: max(sample_size - len(rl_gpt2), 0)]
    return rl_gpt2 + [_inspection_row(row) for row in ranked_benchmark]


def _inspection_row(row: dict[str, str]) -> dict[str, Any]:
    return {
        "dataset": row.get("dataset"),
        "model_id": row.get("model_id"),
        "uid": row.get("pair_id"),
        "subtype": row.get("subtype"),
        "template_type": row.get("template_type"),
        "grammatical": row.get("grammatical"),
        "ungrammatical": row.get("ungrammatical"),
        "grammatical_score": row.get("grammatical_score"),
        "ungrammatical_score": row.get("ungrammatical_score"),
        "preference_margin": row.get("preference_margin"),
        "correct": row.get("correct"),
    }


def _final_summary(
    dataset_summary: dict[str, Any],
    validation_rows: list[dict[str, Any]],
    duplicate_summary: dict[str, Any],
    lexical_summary: dict[str, Any],
    disagreement_summary: dict[str, Any],
    comparison_rows: list[dict[str, str]],
) -> dict[str, Any]:
    invalid_count = sum(1 for row in validation_rows if not row["valid"])
    rl_top_pattern_share = float(lexical_summary["rl"]["top_pattern_share"])
    benchmark_top_pattern_share = float(lexical_summary["benchmark"]["top_pattern_share"])
    subtype_counts = dataset_summary["rl"]["subtype_counts"]
    top_subtype_share = max(subtype_counts.values(), default=0) / max(dataset_summary["rl"]["total"], 1)
    stronger_hard_subtypes = _stronger_model_hard_subtypes(comparison_rows)
    return {
        "subtype_collapse_evidence": top_subtype_share > 0.25,
        "top_rl_subtype_share": top_subtype_share,
        "lexical_collapse_evidence": rl_top_pattern_share > max(benchmark_top_pattern_share * 2, 0.15),
        "rl_top_pattern_share": rl_top_pattern_share,
        "benchmark_top_pattern_share": benchmark_top_pattern_share,
        "duplicate_or_near_duplicate_evidence": duplicate_summary["duplicate_pair_count"] > 0
        or duplicate_summary["max_good_sentence_repetition"] > 2,
        "validation_error_count": invalid_count,
        "grammar_or_metadata_issue_evidence": invalid_count > 0,
        "tiny_artifact_exploitation_evidence": disagreement_summary.get("rl:tiny_fails_stronger_succeed", 0) > 0,
        "genuinely_harder_subtypes_across_stronger_models": stronger_hard_subtypes,
        "disagreement_summary": disagreement_summary,
    }


def _stronger_model_hard_subtypes(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("model_id") in STRONGER_MODELS and row.get("dataset") == "rl":
            grouped[row["subtype"]][row["model_id"]].append(row)
    hard_subtypes = []
    for subtype, by_model in grouped.items():
        model_failure_rates = {}
        for model_id in STRONGER_MODELS:
            model_rows = by_model.get(model_id, [])
            if not model_rows:
                continue
            failures = sum(not _as_bool(row.get("correct")) for row in model_rows)
            model_failure_rates[model_id] = failures / len(model_rows)
        if len(model_failure_rates) == len(STRONGER_MODELS) and all(rate >= 0.2 for rate in model_failure_rates.values()):
            hard_subtypes.append({"subtype": subtype, "failure_rates": model_failure_rates})
    return sorted(hard_subtypes, key=lambda row: (-sum(row["failure_rates"].values()), row["subtype"]))


def _write_markdown_report(summary: dict[str, Any], path: Path) -> None:
    final = summary
    lines = ["# Sanity Check Report", ""]
    lines.append("## Bottom Line")
    lines.append(f"- Subtype collapse evidence: {final['subtype_collapse_evidence']} (top RL subtype share {final['top_rl_subtype_share']:.3f})")
    lines.append(f"- Lexical collapse evidence: {final['lexical_collapse_evidence']} (RL top pattern share {final['rl_top_pattern_share']:.3f})")
    lines.append(f"- Duplicate or near-duplicate evidence: {final['duplicate_or_near_duplicate_evidence']}")
    lines.append(f"- Grammar/metadata issue evidence: {final['grammar_or_metadata_issue_evidence']} ({final['validation_error_count']} validation errors)")
    lines.append(f"- TinyGPT-2 artifact exploitation evidence: {final['tiny_artifact_exploitation_evidence']}")
    lines.extend(["", "## Model Disagreement"])
    for category, count in sorted(final["disagreement_summary"].items()):
        lines.append(f"- {category}: {count}")
    lines.extend(["", "## Stronger-Model Hard Subtypes"])
    hard = final["genuinely_harder_subtypes_across_stronger_models"]
    if hard:
        for row in hard:
            lines.append(f"- {row['subtype']}: {row['failure_rates']}")
    else:
        lines.append("No subtype crossed the default stronger-model failure-rate threshold.")
    lines.extend(["", "## Files To Inspect"])
    lines.append("- sanity_check_pair_validation.csv")
    lines.append("- sanity_check_tiny_extreme_failures.csv")
    lines.append("- sanity_check_model_disagreement.csv")
    lines.append("- sanity_check_manual_tiny_hardest_rl.csv")
    lines.append("- sanity_check_manual_gpt2_hardest_rl.csv")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _record_from_result_row(row: dict[str, str]) -> dict[str, Any]:
    return {
        "sentence_good": row.get("grammatical", ""),
        "sentence_bad": row.get("ungrammatical", ""),
        "subtype": row.get("subtype", ""),
        "template_type": row.get("template_type", ""),
    }


def _tokens(sentence: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z]+", sentence)]


def _noun_vocab() -> set[str]:
    noun_slots = [
        "subject_singular",
        "subject_plural",
        "person_singular",
        "attractor_singular",
        "attractor_plural",
        "object_singular",
        "object_plural",
    ]
    return {item.lower() for slot in noun_slots for item in DEFAULT_LEXICON[slot]}


def _verb_vocab() -> set[str]:
    verbs = {verb for pair in VERB_PAIRS for verb in pair}
    verbs.update(item.lower() for slot in ("embedded_verb_singular", "embedded_verb_plural") for item in DEFAULT_LEXICON[slot])
    return verbs


def _distance_bucket(value: Any) -> str:
    distance = int(value)
    if distance <= 1:
        return "short_1"
    if distance <= 4:
        return "medium_2_4"
    return "long_5_plus"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _write_json(payload: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return loaded


if __name__ == "__main__":
    main()
