"""Compare benchmark and RL-generated agreement datasets under shared scorers."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import yaml

from syntax_rl.data.blimp_loader import MinimalPair, load_blimp_subset
from syntax_rl.evaluation.metrics import evaluate_minimal_pairs
from syntax_rl.models.scoring import build_scorer
from syntax_rl.utils import configure_logging, ensure_dir, get_logger, resolve_project_path, seed_everything

DATASETS = ("benchmark", "rl")
DEFAULT_GROUP_BY = ["template_type", "subtype", "dependency_distance_bucket", "clause_depth"]
SHARED_SUBTYPE_METRICS = ("accuracy", "failure_rate", "mean_preference_margin")
HARDNESS_DEFINITION = (
    "RL is harder only when RL accuracy is lower than benchmark accuracy "
    "and/or RL failure_rate is higher than benchmark failure_rate. "
    "Mean preference margin is reported separately as a signed confidence or boundary-proximity measure."
)
LOGGER = get_logger(__name__)


def run_benchmark_vs_rl(config_path: str | Path) -> dict[str, Path]:
    """Evaluate benchmark and RL datasets with the same configured models."""
    config = _load_config(resolve_project_path(config_path))
    seed_everything(int(config.get("experiment", {}).get("seed", 42)))

    experiment_name = config.get("experiment", {}).get("name", "benchmark_vs_rl")
    output_dir = ensure_dir(config.get("outputs", {}).get("comparison_dir", "outputs/benchmark_vs_rl"))
    figures_dir = ensure_dir(config.get("outputs", {}).get("figures_dir", output_dir / "figures"))
    log_dir = ensure_dir(config.get("outputs", {}).get("log_dir", "outputs/logs"))
    configure_logging(log_file=log_dir / f"{experiment_name}.log")
    near_failure_margin = float(config.get("evaluation", {}).get("near_failure_margin", 0.01))
    continue_on_error = bool(config.get("evaluation", {}).get("continue_on_error", True))
    concentration_top_n = int(config.get("evaluation", {}).get("concentration_top_n", 3))
    concentration_threshold = float(config.get("evaluation", {}).get("concentration_failure_share_threshold", 0.6))
    distribution_delta = float(config.get("evaluation", {}).get("distribution_mismatch_share_delta", 0.2))
    hard_subtype_top_n = int(config.get("evaluation", {}).get("hard_subtype_top_n", 5))

    inputs = config.get("inputs", {})
    datasets = {
        "benchmark": _load_dataset(inputs["benchmark"], default_label="benchmark"),
        "rl": _load_dataset(inputs["rl"], default_label="rl"),
    }
    LOGGER.info(
        "Loaded benchmark-vs-RL datasets: benchmark=%d examples, rl=%d examples",
        len(datasets["benchmark"]),
        len(datasets["rl"]),
    )
    matched_subset_summary: dict[str, Any] | None = None
    matched_subset_config = config.get("evaluation", {}).get("matched_subset", {})
    if bool(matched_subset_config.get("enabled", False)):
        datasets, matched_subset_summary = _build_matched_subset(
            datasets,
            matched_subset_config,
            seed=int(config.get("experiment", {}).get("seed", 42)),
        )
        LOGGER.info(
            "Using matched subset: benchmark=%d examples, rl=%d examples, matching_field=%s, exact_subtype_match=%s",
            len(datasets["benchmark"]),
            len(datasets["rl"]),
            matched_subset_summary["matching_field"],
            matched_subset_summary["exact_subtype_match"],
        )

    result_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for model_config in config.get("models", []):
        if not model_config.get("enabled", True):
            continue
        model_id = _model_id(model_config)
        try:
            load_start = time.perf_counter()
            LOGGER.info(
                "Loading scorer model_id=%s provider=%s name=%s",
                model_id,
                model_config.get("provider", "length_normalized"),
                model_config.get("name"),
            )
            scorer = build_scorer(
                provider=model_config.get("provider", "length_normalized"),
                model_name=model_config.get("name"),
                **{key: value for key, value in model_config.items() if key not in {"id", "enabled", "label", "provider", "name"}},
            )
            LOGGER.info(
                "Loaded scorer model_id=%s device=%s elapsed=%.2fs",
                model_id,
                getattr(scorer, "device", "n/a"),
                time.perf_counter() - load_start,
            )
            max_examples = _max_examples_for_model(config, model_config)
            model_datasets = _limit_datasets(datasets, max_examples)
            for dataset_name, pairs in model_datasets.items():
                eval_start = time.perf_counter()
                LOGGER.info(
                    "Evaluating model_id=%s dataset=%s examples=%d max_examples_per_dataset=%s",
                    model_id,
                    dataset_name,
                    len(pairs),
                    max_examples if max_examples is not None else "all",
                )
                results = evaluate_minimal_pairs(pairs, scorer)
                for result, pair in zip(results, pairs):
                    metadata = pair.metadata or {}
                    margin = float(result.preference_margin)
                    result_rows.append(
                        {
                            "dataset": dataset_name,
                            "model_id": model_id,
                            "model_name": model_config.get("name"),
                            "model_provider": model_config.get("provider"),
                            "pair_id": result.pair_id,
                            "phenomenon": result.phenomenon,
                            "subtype": result.subtype,
                            "template_type": metadata.get("template_type"),
                            "dependency_distance": metadata.get("dependency_distance"),
                            "dependency_distance_bucket": _distance_bucket(metadata.get("dependency_distance")),
                            "clause_depth": metadata.get("clause_depth"),
                            "grammatical": result.grammatical,
                            "ungrammatical": result.ungrammatical,
                            "grammatical_score": result.grammatical_score,
                            "ungrammatical_score": result.ungrammatical_score,
                            "preference_margin": margin,
                            "correct": result.correct,
                            "failure": not result.correct,
                            "near_failure": abs(margin) <= near_failure_margin,
                        }
                    )
                LOGGER.info(
                    "Finished model_id=%s dataset=%s examples=%d elapsed=%.2fs",
                    model_id,
                    dataset_name,
                    len(pairs),
                    time.perf_counter() - eval_start,
                )
        except Exception as error:
            LOGGER.exception("Model evaluation failed for model_id=%s", model_id)
            errors.append({"model_id": model_id, "error": str(error)})
            if not continue_on_error:
                raise

    group_by = config.get("evaluation", {}).get("group_by", DEFAULT_GROUP_BY)
    overall_summary = _summarize_rows(result_rows, ["model_id", "dataset"])
    grouped_summary = _summarize_group_fields(result_rows, group_by)
    shared_subtype_comparison = _shared_subtype_comparison(grouped_summary)
    report_summary = _report_summary(
        overall_summary,
        grouped_summary,
        shared_subtype_comparison,
        concentration_top_n=concentration_top_n,
        concentration_threshold=concentration_threshold,
        distribution_delta=distribution_delta,
        matched_subset_summary=matched_subset_summary,
    )
    subtype_failure_summary = _subtype_failure_summary(
        grouped_summary,
        shared_subtype_comparison,
        top_n=hard_subtype_top_n,
    )
    cross_model_hardest_subtypes = _cross_model_hardest_subtypes(
        grouped_summary,
        config.get("evaluation", {}).get("cross_model_sources", []),
        top_n=hard_subtype_top_n,
    )

    paths = {
        "results_csv": output_dir / f"{experiment_name}_results.csv",
        "overall_summary_csv": output_dir / f"{experiment_name}_overall_summary.csv",
        "grouped_summary_csv": output_dir / f"{experiment_name}_grouped_summary.csv",
        "shared_subtype_comparison_csv": output_dir / f"{experiment_name}_shared_subtype_comparison.csv",
        "subtype_failure_summary_csv": output_dir / f"{experiment_name}_subtype_failure_summary.csv",
        "cross_model_hardest_subtypes_csv": output_dir / f"{experiment_name}_cross_model_hardest_subtypes.csv",
        "summary_json": output_dir / f"{experiment_name}_summary.json",
        "report_md": output_dir / f"{experiment_name}_report.md",
    }
    if matched_subset_summary is not None:
        paths.update(
            {
                "sampled_benchmark_jsonl": output_dir / f"{experiment_name}_sampled_benchmark.jsonl",
                "sampled_benchmark_csv": output_dir / f"{experiment_name}_sampled_benchmark.csv",
                "sampled_rl_jsonl": output_dir / f"{experiment_name}_sampled_rl.jsonl",
                "sampled_rl_csv": output_dir / f"{experiment_name}_sampled_rl.csv",
                "matched_subset_summary_json": output_dir / f"{experiment_name}_matched_subset_summary.json",
                "matched_subset_summary_csv": output_dir / f"{experiment_name}_matched_subset_summary.csv",
            }
        )
        _write_pairs_jsonl(datasets["benchmark"], paths["sampled_benchmark_jsonl"])
        _write_csv(_pairs_to_records(datasets["benchmark"]), paths["sampled_benchmark_csv"])
        _write_pairs_jsonl(datasets["rl"], paths["sampled_rl_jsonl"])
        _write_csv(_pairs_to_records(datasets["rl"]), paths["sampled_rl_csv"])
        _write_json(matched_subset_summary, paths["matched_subset_summary_json"])
        _write_csv(matched_subset_summary["group_rows"], paths["matched_subset_summary_csv"])
    _write_csv(result_rows, paths["results_csv"])
    _write_csv(overall_summary, paths["overall_summary_csv"])
    _write_csv(grouped_summary, paths["grouped_summary_csv"])
    _write_csv(shared_subtype_comparison, paths["shared_subtype_comparison_csv"])
    _write_csv(subtype_failure_summary["subtype_rows"], paths["subtype_failure_summary_csv"])
    _write_csv(cross_model_hardest_subtypes, paths["cross_model_hardest_subtypes_csv"])
    _write_json(
        {
            "overall_summary": overall_summary,
            "grouped_summary": grouped_summary,
            "shared_subtype_comparison": shared_subtype_comparison,
            "report_summary": report_summary,
            "subtype_failure_summary": subtype_failure_summary,
            "cross_model_hardest_subtypes": cross_model_hardest_subtypes,
            "matched_subset_summary": matched_subset_summary,
            "hardness_definition": HARDNESS_DEFINITION,
            "errors": errors,
            "config": config,
        },
        paths["summary_json"],
    )
    _write_markdown_report(
        report_summary,
        overall_summary,
        shared_subtype_comparison,
        paths["report_md"],
        subtype_failure_summary=subtype_failure_summary,
        cross_model_hardest_subtypes=cross_model_hardest_subtypes,
    )
    paths.update(_write_figures(overall_summary, shared_subtype_comparison, figures_dir, experiment_name, grouped_summary))
    return paths


def main() -> None:
    """CLI entry point for benchmark-vs-RL dataset comparison."""
    parser = argparse.ArgumentParser(description="Compare benchmark and RL-generated agreement datasets.")
    parser.add_argument("--config", default="configs/benchmark_vs_rl.yaml")
    args = parser.parse_args()
    outputs = run_benchmark_vs_rl(args.config)
    for label, path in outputs.items():
        print(f"Wrote {label} to {path}")


def _load_dataset(dataset_config: str | dict[str, Any], default_label: str) -> list[MinimalPair]:
    if isinstance(dataset_config, str):
        path = dataset_config
        phenomenon = "agreement"
    else:
        path = dataset_config["path"]
        phenomenon = dataset_config.get("phenomenon", "agreement")
    pairs = load_blimp_subset(resolve_project_path(path), phenomenon=phenomenon)
    return [
        MinimalPair(
            grammatical=pair.grammatical,
            ungrammatical=pair.ungrammatical,
            phenomenon=pair.phenomenon,
            pair_id=pair.pair_id,
            metadata={**(pair.metadata or {}), "dataset_label": default_label},
        )
        for pair in pairs
    ]


def _max_examples_for_model(config: dict[str, Any], model_config: dict[str, Any]) -> int | None:
    value = model_config.get(
        "max_examples_per_dataset",
        config.get("evaluation", {}).get("max_examples_per_dataset"),
    )
    if value in {None, ""}:
        return None
    max_examples = int(value)
    if max_examples <= 0:
        return None
    return max_examples


def _limit_datasets(
    datasets: dict[str, list[MinimalPair]],
    max_examples_per_dataset: int | None,
) -> dict[str, list[MinimalPair]]:
    if max_examples_per_dataset is None:
        return datasets
    return {
        dataset_name: pairs[:max_examples_per_dataset]
        for dataset_name, pairs in datasets.items()
    }


def _build_matched_subset(
    datasets: dict[str, list[MinimalPair]],
    subset_config: dict[str, Any],
    seed: int,
) -> tuple[dict[str, list[MinimalPair]], dict[str, Any]]:
    examples_per_group = int(subset_config.get("examples_per_shared_subtype", 2))
    max_total = subset_config.get("max_total_examples_per_dataset")
    max_total_examples = int(max_total) if max_total not in {None, ""} else None
    rng = random.Random(seed)

    subtype_plan = _matched_sampling_plan(
        datasets,
        group_field="subtype",
        examples_per_group=examples_per_group,
        max_total_examples=max_total_examples,
    )
    if subtype_plan["sample_counts"]:
        matching_field = "subtype"
        exact_subtype_match = True
        fallback_to_template_type = False
        fallback_reason = ""
        plan = subtype_plan
    else:
        matching_field = "template_type"
        exact_subtype_match = False
        fallback_to_template_type = True
        fallback_reason = "No shared subtypes with examples in both datasets; matched by template_type instead."
        plan = _matched_sampling_plan(
            datasets,
            group_field="template_type",
            examples_per_group=examples_per_group,
            max_total_examples=max_total_examples,
        )

    sampled = {
        dataset_name: _sample_from_plan(
            pairs,
            group_field=matching_field,
            sample_counts=plan["sample_counts"],
            rng=rng,
        )
        for dataset_name, pairs in datasets.items()
    }
    group_rows = _matched_subset_group_rows(
        datasets,
        plan["sample_counts"],
        group_field=matching_field,
        requested_per_group=examples_per_group,
    )
    dropped = [row for row in group_rows if row["sampled_per_dataset"] == 0]
    limited = [
        row
        for row in group_rows
        if 0 < int(row["sampled_per_dataset"]) < examples_per_group
    ]
    total_examples = len(sampled["benchmark"])
    return sampled, {
        "enabled": True,
        "seed": seed,
        "matching_field": matching_field,
        "exact_subtype_match": exact_subtype_match,
        "fallback_to_template_type": fallback_to_template_type,
        "fallback_reason": fallback_reason,
        "requested_examples_per_shared_subtype": examples_per_group,
        "max_total_examples_per_dataset": max_total_examples,
        "benchmark_total": len(sampled["benchmark"]),
        "rl_total": len(sampled["rl"]),
        "distribution_matched": len(sampled["benchmark"]) == len(sampled["rl"])
        and all(int(row["benchmark_sampled"]) == int(row["rl_sampled"]) for row in group_rows),
        "matched_groups": [
            row["group"]
            for row in group_rows
            if int(row["sampled_per_dataset"]) > 0
        ],
        "matched_group_count": sum(1 for row in group_rows if int(row["sampled_per_dataset"]) > 0),
        "examples_per_matched_group": {
            row["group"]: int(row["sampled_per_dataset"])
            for row in group_rows
            if int(row["sampled_per_dataset"]) > 0
        },
        "dropped_groups_due_to_insufficient_examples": [
            row["group"] for row in dropped if row["status"] == "insufficient_examples"
        ],
        "dropped_groups_due_to_total_cap": [
            row["group"] for row in dropped if row["status"] == "dropped_by_total_cap"
        ],
        "limited_groups_below_requested_count": [
            {
                "group": row["group"],
                "sampled_per_dataset": int(row["sampled_per_dataset"]),
                "requested_per_group": examples_per_group,
            }
            for row in limited
        ],
        "group_rows": group_rows,
        "note": (
            f"Matched benchmark and RL by {matching_field}; "
            f"{total_examples} examples per dataset were sampled."
        ),
    }


def _matched_sampling_plan(
    datasets: dict[str, list[MinimalPair]],
    group_field: str,
    examples_per_group: int,
    max_total_examples: int | None,
) -> dict[str, Any]:
    benchmark_groups = _group_pairs_by_metadata(datasets["benchmark"], group_field)
    rl_groups = _group_pairs_by_metadata(datasets["rl"], group_field)
    shared_groups = sorted(set(benchmark_groups) & set(rl_groups))
    sample_counts = {
        group: min(examples_per_group, len(benchmark_groups[group]), len(rl_groups[group]))
        for group in shared_groups
        if min(examples_per_group, len(benchmark_groups[group]), len(rl_groups[group])) > 0
    }
    if max_total_examples is not None:
        sample_counts = _cap_sample_counts(sample_counts, max_total_examples)
    return {
        "group_field": group_field,
        "sample_counts": sample_counts,
    }


def _group_pairs_by_metadata(pairs: list[MinimalPair], group_field: str) -> dict[str, list[MinimalPair]]:
    groups: dict[str, list[MinimalPair]] = {}
    for pair in pairs:
        metadata = pair.metadata or {}
        value = metadata.get(group_field)
        if value in {None, ""}:
            continue
        groups.setdefault(str(value), []).append(pair)
    return groups


def _cap_sample_counts(sample_counts: dict[str, int], max_total_examples: int) -> dict[str, int]:
    capped = {group: count for group, count in sample_counts.items() if count > 0}
    while sum(capped.values()) > max_total_examples and capped:
        largest_group = max(
            sorted(capped),
            key=lambda group: (capped[group], group),
        )
        capped[largest_group] -= 1
        if capped[largest_group] <= 0:
            del capped[largest_group]
    return capped


def _sample_from_plan(
    pairs: list[MinimalPair],
    group_field: str,
    sample_counts: dict[str, int],
    rng: random.Random,
) -> list[MinimalPair]:
    groups = _group_pairs_by_metadata(pairs, group_field)
    sampled: list[MinimalPair] = []
    for group, count in sorted(sample_counts.items()):
        group_pairs = list(groups.get(group, []))
        if len(group_pairs) <= count:
            chosen = group_pairs
        else:
            chosen = rng.sample(group_pairs, count)
        sampled.extend(sorted(chosen, key=lambda pair: pair.pair_id))
    return sampled


def _matched_subset_group_rows(
    datasets: dict[str, list[MinimalPair]],
    sample_counts: dict[str, int],
    group_field: str,
    requested_per_group: int,
) -> list[dict[str, Any]]:
    benchmark_groups = _group_pairs_by_metadata(datasets["benchmark"], group_field)
    rl_groups = _group_pairs_by_metadata(datasets["rl"], group_field)
    all_groups = sorted(set(benchmark_groups) | set(rl_groups) | set(sample_counts))
    rows: list[dict[str, Any]] = []
    for group in all_groups:
        benchmark_available = len(benchmark_groups.get(group, []))
        rl_available = len(rl_groups.get(group, []))
        sampled = int(sample_counts.get(group, 0))
        if sampled <= 0 and benchmark_available and rl_available:
            status = "dropped_by_total_cap"
        elif sampled <= 0:
            status = "insufficient_examples"
        elif sampled < requested_per_group:
            status = "limited_by_available_examples"
        else:
            status = "matched"
        rows.append(
            {
                "group_by": group_field,
                "group": group,
                "benchmark_available": benchmark_available,
                "rl_available": rl_available,
                "benchmark_sampled": sampled,
                "rl_sampled": sampled,
                "sampled_per_dataset": sampled,
                "requested_per_group": requested_per_group,
                "status": status,
            }
        )
    return rows


def _summarize_rows(rows: list[dict[str, Any]], group_fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in group_fields)
        groups.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for key, group_rows in sorted(groups.items()):
        total = len(group_rows)
        correct = sum(bool(row["correct"]) for row in group_rows)
        failures = sum(bool(row["failure"]) for row in group_rows)
        near_failures = sum(bool(row["near_failure"]) for row in group_rows)
        margins = [float(row["preference_margin"]) for row in group_rows]
        summary = {field: value for field, value in zip(group_fields, key)}
        summary.update(
            {
                "total": total,
                "correct": correct,
                "accuracy": correct / total if total else 0.0,
                "failure_rate": failures / total if total else 0.0,
                "near_failure_rate": near_failures / total if total else 0.0,
                "mean_preference_margin": sum(margins) / total if total else 0.0,
            }
        )
        summaries.append(summary)
    return summaries


def _summarize_group_fields(rows: list[dict[str, Any]], group_fields: list[str]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for group_field in group_fields:
        available_rows = [row for row in rows if row.get(group_field) not in {None, ""}]
        for summary in _summarize_rows(available_rows, ["model_id", "dataset", group_field]):
            group_value = summary.pop(group_field)
            summary["group_by"] = group_field
            summary["group"] = group_value
            summaries.append(summary)
    return summaries


def _shared_subtype_comparison(grouped_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    subtype_rows = [row for row in grouped_summary if row.get("group_by") == "subtype"]
    model_ids = sorted({str(row["model_id"]) for row in subtype_rows})
    comparison_rows: list[dict[str, Any]] = []
    for model_id in model_ids:
        model_rows = [row for row in subtype_rows if row["model_id"] == model_id]
        benchmark_subtypes = {str(row["group"]) for row in model_rows if row["dataset"] == "benchmark"}
        rl_subtypes = {str(row["group"]) for row in model_rows if row["dataset"] == "rl"}
        for subtype in sorted(benchmark_subtypes & rl_subtypes):
            benchmark_row = _find_summary_row(model_rows, "benchmark", subtype)
            rl_row = _find_summary_row(model_rows, "rl", subtype)
            comparison_rows.append(
                {
                    "model_id": model_id,
                    "subtype": subtype,
                    "benchmark_total": benchmark_row["total"],
                    "rl_total": rl_row["total"],
                    "benchmark_accuracy": benchmark_row["accuracy"],
                    "rl_accuracy": rl_row["accuracy"],
                    "accuracy_delta_rl_minus_benchmark": float(rl_row["accuracy"]) - float(benchmark_row["accuracy"]),
                    "benchmark_failure_rate": benchmark_row["failure_rate"],
                    "rl_failure_rate": rl_row["failure_rate"],
                    "failure_rate_delta_rl_minus_benchmark": float(rl_row["failure_rate"]) - float(benchmark_row["failure_rate"]),
                    "benchmark_mean_preference_margin": benchmark_row["mean_preference_margin"],
                    "rl_mean_preference_margin": rl_row["mean_preference_margin"],
                    "mean_preference_margin_delta_rl_minus_benchmark": float(rl_row["mean_preference_margin"])
                    - float(benchmark_row["mean_preference_margin"]),
                    "rl_boundary_closer_by_abs_margin": abs(float(rl_row["mean_preference_margin"]))
                    < abs(float(benchmark_row["mean_preference_margin"])),
                    "rl_harder": _is_rl_harder(benchmark_row, rl_row),
                }
            )
    return comparison_rows


def _find_summary_row(rows: list[dict[str, Any]], dataset: str, subtype: str) -> dict[str, Any]:
    for row in rows:
        if row["dataset"] == dataset and row["group"] == subtype:
            return row
    raise ValueError(f"Missing shared subtype row for dataset={dataset}, subtype={subtype}")


def _is_rl_harder(benchmark_row: dict[str, Any], rl_row: dict[str, Any]) -> bool:
    """Return whether RL is harder by accuracy/failure-rate criteria only."""
    return (
        float(rl_row["accuracy"]) < float(benchmark_row["accuracy"])
        or float(rl_row["failure_rate"]) > float(benchmark_row["failure_rate"])
    )


def _report_summary(
    overall_summary: list[dict[str, Any]],
    grouped_summary: list[dict[str, Any]],
    shared_subtype_comparison: list[dict[str, Any]],
    concentration_top_n: int,
    concentration_threshold: float,
    distribution_delta: float,
    matched_subset_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_ids = sorted({str(row["model_id"]) for row in overall_summary})
    overall: dict[str, Any] = {}
    for model_id in model_ids:
        benchmark_row = _find_overall_row(overall_summary, model_id, "benchmark")
        rl_row = _find_overall_row(overall_summary, model_id, "rl")
        overall[model_id] = {
            "rl_harder_overall": _is_rl_harder(benchmark_row, rl_row),
            "accuracy_delta_rl_minus_benchmark": float(rl_row["accuracy"]) - float(benchmark_row["accuracy"]),
            "failure_rate_delta_rl_minus_benchmark": float(rl_row["failure_rate"]) - float(benchmark_row["failure_rate"]),
            "mean_preference_margin_delta_rl_minus_benchmark": float(rl_row["mean_preference_margin"])
            - float(benchmark_row["mean_preference_margin"]),
            "rl_boundary_closer_by_abs_margin": abs(float(rl_row["mean_preference_margin"]))
            < abs(float(benchmark_row["mean_preference_margin"])),
        }

    harder_shared: dict[str, list[dict[str, Any]]] = {}
    for model_id in model_ids:
        rows = [
            row
            for row in shared_subtype_comparison
            if row["model_id"] == model_id and bool(row["rl_harder"])
        ]
        harder_shared[model_id] = sorted(
            rows,
            key=lambda row: (
                float(row["accuracy_delta_rl_minus_benchmark"]),
                -float(row["failure_rate_delta_rl_minus_benchmark"]),
                str(row["subtype"]),
            ),
        )

    concentration = _rl_failure_concentration(
        grouped_summary,
        top_n=concentration_top_n,
        threshold=concentration_threshold,
    )
    coverage = _coverage_mismatch(grouped_summary)
    distribution = _distribution_mismatch(
        grouped_summary,
        share_delta=distribution_delta,
    )
    return {
        "hardness_definition": HARDNESS_DEFINITION,
        "overall": overall,
        "shared_subtypes_harder_in_rl": harder_shared,
        "coverage_mismatch": coverage,
        "distribution_mismatch": distribution,
        "rl_difficulty_concentration": concentration,
        "matched_subset": matched_subset_summary,
    }


def _subtype_failure_summary(
    grouped_summary: list[dict[str, Any]],
    shared_subtype_comparison: list[dict[str, Any]],
    top_n: int,
) -> dict[str, Any]:
    subtype_rows = [row for row in grouped_summary if row.get("group_by") == "subtype"]
    template_rows = [row for row in grouped_summary if row.get("group_by") == "template_type"]
    model_ids = sorted({str(row["model_id"]) for row in subtype_rows})
    summary_rows: list[dict[str, Any]] = []
    hardest: dict[str, dict[str, list[dict[str, Any]]]] = {}
    easiest: dict[str, dict[str, list[dict[str, Any]]]] = {}
    concentration: dict[str, dict[str, Any]] = {}
    shared_harder: dict[str, list[str]] = {}

    for model_id in model_ids:
        hardest[model_id] = {}
        easiest[model_id] = {}
        for dataset in DATASETS:
            model_dataset_rows = [
                row
                for row in subtype_rows
                if row["model_id"] == model_id and row["dataset"] == dataset
            ]
            ranked_hard = sorted(
                model_dataset_rows,
                key=lambda row: (
                    float(row["accuracy"]),
                    -float(row["failure_rate"]),
                    float(row["mean_preference_margin"]),
                    str(row["group"]),
                ),
            )[:top_n]
            ranked_easy = sorted(
                model_dataset_rows,
                key=lambda row: (
                    -float(row["accuracy"]),
                    float(row["failure_rate"]),
                    -float(row["mean_preference_margin"]),
                    str(row["group"]),
                ),
            )[:top_n]
            hardest[model_id][dataset] = [_subtype_summary_row(row) for row in ranked_hard]
            easiest[model_id][dataset] = [_subtype_summary_row(row) for row in ranked_easy]
            for rank, row in enumerate(ranked_hard, start=1):
                summary_rows.append(
                    {
                        "model_id": model_id,
                        "dataset": dataset,
                        "rank": rank,
                        "subtype": row["group"],
                        "total": row["total"],
                        "accuracy": row["accuracy"],
                        "failure_rate": row["failure_rate"],
                        "mean_preference_margin": row["mean_preference_margin"],
                    }
                )

        model_template_rows = [row for row in template_rows if row["model_id"] == model_id]
        concentration[model_id] = {}
        for dataset in DATASETS:
            dataset_template_rows = [row for row in model_template_rows if row["dataset"] == dataset]
            total_failures = sum(int(row["total"]) - int(row["correct"]) for row in dataset_template_rows)
            template_concentration = []
            for row in sorted(dataset_template_rows, key=lambda row: (-float(row["failure_rate"]), str(row["group"]))):
                failures = int(row["total"]) - int(row["correct"])
                template_concentration.append(
                    {
                        "template_type": row["group"],
                        "failures": failures,
                        "total": int(row["total"]),
                        "failure_rate": float(row["failure_rate"]),
                        "failure_share": failures / total_failures if total_failures else 0.0,
                    }
                )
            concentration[model_id][dataset] = template_concentration

        shared_harder[model_id] = [
            str(row["subtype"])
            for row in shared_subtype_comparison
            if row["model_id"] == model_id and bool(row["rl_harder"])
        ]

    return {
        "top_n": top_n,
        "hardest_subtypes_by_model_dataset": hardest,
        "easiest_subtypes_by_model_dataset": easiest,
        "failure_concentration_by_template_type": concentration,
        "shared_subtypes_harder_in_rl": shared_harder,
        "subtype_rows": summary_rows,
    }


def _subtype_summary_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "subtype": row["group"],
        "total": int(row["total"]),
        "correct": int(row["correct"]),
        "accuracy": float(row["accuracy"]),
        "failure_rate": float(row["failure_rate"]),
        "mean_preference_margin": float(row["mean_preference_margin"]),
        "near_failure_rate": float(row["near_failure_rate"]),
    }


def _cross_model_hardest_subtypes(
    current_grouped_summary: list[dict[str, Any]],
    source_configs: list[dict[str, Any]],
    top_n: int,
) -> list[dict[str, Any]]:
    rows = _hardest_rows_from_grouped_summary(current_grouped_summary, source_label="current", top_n=top_n)
    for source_config in source_configs or []:
        path_value = source_config.get("path") if isinstance(source_config, dict) else None
        if not path_value:
            continue
        source_path = resolve_project_path(path_value)
        if not source_path.exists():
            rows.append(
                {
                    "source": source_config.get("label", str(path_value)),
                    "model_id": "",
                    "dataset": "",
                    "rank": "",
                    "subtype": "",
                    "accuracy": "",
                    "failure_rate": "",
                    "mean_preference_margin": "",
                    "note": f"missing source file: {source_path}",
                }
            )
            continue
        source_rows = _read_csv(source_path)
        rows.extend(
            _hardest_rows_from_grouped_summary(
                source_rows,
                source_label=source_config.get("label", source_path.stem),
                top_n=top_n,
            )
        )
    return rows


def _hardest_rows_from_grouped_summary(
    grouped_summary: list[dict[str, Any]],
    source_label: str,
    top_n: int,
) -> list[dict[str, Any]]:
    subtype_rows = [row for row in grouped_summary if row.get("group_by") == "subtype"]
    model_ids = sorted({str(row.get("model_id", "")) for row in subtype_rows})
    output_rows: list[dict[str, Any]] = []
    for model_id in model_ids:
        model_rows = [row for row in subtype_rows if str(row.get("model_id", "")) == model_id]
        datasets = sorted({str(row.get("dataset", "controlled")) for row in model_rows})
        for dataset in datasets:
            dataset_rows = [row for row in model_rows if str(row.get("dataset", "controlled")) == dataset]
            hardest = sorted(
                dataset_rows,
                key=lambda row: (
                    float(row.get("accuracy", 0.0)),
                    -float(row.get("failure_rate", 0.0)),
                    float(row.get("mean_preference_margin", 0.0)),
                    str(row.get("group", "")),
                ),
            )[:top_n]
            for rank, row in enumerate(hardest, start=1):
                output_rows.append(
                    {
                        "source": source_label,
                        "model_id": model_id,
                        "dataset": dataset,
                        "rank": rank,
                        "subtype": row.get("group"),
                        "accuracy": row.get("accuracy"),
                        "failure_rate": row.get("failure_rate"),
                        "mean_preference_margin": row.get("mean_preference_margin"),
                        "note": "",
                    }
                )
    return output_rows


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _find_overall_row(rows: list[dict[str, Any]], model_id: str, dataset: str) -> dict[str, Any]:
    for row in rows:
        if row["model_id"] == model_id and row["dataset"] == dataset:
            return row
    raise ValueError(f"Missing overall summary for model={model_id}, dataset={dataset}")


def _rl_failure_concentration(
    grouped_summary: list[dict[str, Any]],
    top_n: int,
    threshold: float,
) -> dict[str, Any]:
    subtype_rows = [
        row
        for row in grouped_summary
        if row.get("group_by") == "subtype" and row.get("dataset") == "rl"
    ]
    concentration: dict[str, Any] = {}
    for model_id in sorted({str(row["model_id"]) for row in subtype_rows}):
        model_rows = [row for row in subtype_rows if row["model_id"] == model_id]
        total_failures = sum(int(row["total"]) - int(row["correct"]) for row in model_rows)
        ranked = sorted(
            model_rows,
            key=lambda row: (
                -float(row["failure_rate"]),
                -(int(row["total"]) - int(row["correct"])),
                str(row["group"]),
            ),
        )
        top_rows = ranked[:top_n]
        top_failures = sum(int(row["total"]) - int(row["correct"]) for row in top_rows)
        share = top_failures / total_failures if total_failures else 0.0
        concentration[model_id] = {
            "top_n": top_n,
            "top_subtypes": [
                {
                    "subtype": row["group"],
                    "failure_rate": float(row["failure_rate"]),
                    "failures": int(row["total"]) - int(row["correct"]),
                    "total": int(row["total"]),
                }
                for row in top_rows
            ],
            "top_failure_share": share,
            "concentrated": share >= threshold if total_failures else False,
            "threshold": threshold,
        }
    return concentration


def _coverage_mismatch(grouped_summary: list[dict[str, Any]]) -> dict[str, Any]:
    coverage: dict[str, Any] = {}
    for group_field in ("template_type", "subtype", "dependency_distance_bucket", "clause_depth"):
        rows = [row for row in grouped_summary if row.get("group_by") == group_field]
        if not rows:
            continue
        benchmark_groups = {str(row["group"]) for row in rows if row["dataset"] == "benchmark"}
        rl_groups = {str(row["group"]) for row in rows if row["dataset"] == "rl"}
        coverage[group_field] = {
            "benchmark_count": len(benchmark_groups),
            "rl_count": len(rl_groups),
            "shared_count": len(benchmark_groups & rl_groups),
            "benchmark_only": sorted(benchmark_groups - rl_groups),
            "rl_only": sorted(rl_groups - benchmark_groups),
            "coverage_mismatch": benchmark_groups != rl_groups,
        }
    return coverage


def _distribution_mismatch(grouped_summary: list[dict[str, Any]], share_delta: float) -> dict[str, Any]:
    subtype_rows = [row for row in grouped_summary if row.get("group_by") == "subtype"]
    model_ids = sorted({str(row["model_id"]) for row in subtype_rows})
    mismatch: dict[str, Any] = {}
    for model_id in model_ids:
        model_rows = [row for row in subtype_rows if row["model_id"] == model_id]
        benchmark_total = sum(int(row["total"]) for row in model_rows if row["dataset"] == "benchmark")
        rl_total = sum(int(row["total"]) for row in model_rows if row["dataset"] == "rl")
        subtypes = sorted({str(row["group"]) for row in model_rows})
        subtype_deltas: list[dict[str, Any]] = []
        for subtype in subtypes:
            benchmark_count = sum(
                int(row["total"])
                for row in model_rows
                if row["dataset"] == "benchmark" and row["group"] == subtype
            )
            rl_count = sum(
                int(row["total"])
                for row in model_rows
                if row["dataset"] == "rl" and row["group"] == subtype
            )
            benchmark_share = benchmark_count / benchmark_total if benchmark_total else 0.0
            rl_share = rl_count / rl_total if rl_total else 0.0
            delta = rl_share - benchmark_share
            if abs(delta) >= share_delta or benchmark_count == 0 or rl_count == 0:
                subtype_deltas.append(
                    {
                        "subtype": subtype,
                        "benchmark_count": benchmark_count,
                        "rl_count": rl_count,
                        "benchmark_share": benchmark_share,
                        "rl_share": rl_share,
                        "share_delta_rl_minus_benchmark": delta,
                    }
                )
        subtype_deltas.sort(key=lambda row: (-abs(float(row["share_delta_rl_minus_benchmark"])), str(row["subtype"])))
        mismatch[model_id] = {
            "share_delta_threshold": share_delta,
            "distribution_matched": not subtype_deltas,
            "warnings": subtype_deltas,
        }
    return mismatch


def _write_figures(
    overall_summary: list[dict[str, Any]],
    shared_subtype_comparison: list[dict[str, Any]],
    figures_dir: Path,
    experiment_name: str,
    grouped_summary: list[dict[str, Any]] | None = None,
) -> dict[str, Path]:
    if not overall_summary:
        return {}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = {
        "overall_accuracy_figure": figures_dir / f"{experiment_name}_overall_accuracy_benchmark_vs_rl.png",
        "overall_failure_rate_figure": figures_dir / f"{experiment_name}_overall_failure_rate_benchmark_vs_rl.png",
        "overall_margin_figure": figures_dir / f"{experiment_name}_overall_mean_preference_margin_benchmark_vs_rl.png",
    }
    _plot_overall_metric(plt, overall_summary, "accuracy", "Overall Accuracy: Benchmark vs RL", paths["overall_accuracy_figure"])
    _plot_overall_metric(plt, overall_summary, "failure_rate", "Overall Failure Rate: Benchmark vs RL", paths["overall_failure_rate_figure"])
    _plot_overall_metric(
        plt,
        overall_summary,
        "mean_preference_margin",
        "Overall Mean Preference Margin: Benchmark vs RL",
        paths["overall_margin_figure"],
    )

    if shared_subtype_comparison:
        paths["shared_subtype_accuracy_figure"] = figures_dir / f"{experiment_name}_accuracy_by_shared_subtype.png"
        paths["shared_subtype_failure_rate_figure"] = figures_dir / f"{experiment_name}_failure_rate_by_shared_subtype.png"
        _plot_shared_subtype_metric(
            plt,
            shared_subtype_comparison,
            "accuracy",
            "Accuracy by Shared Subtype: Benchmark vs RL",
            paths["shared_subtype_accuracy_figure"],
        )
        _plot_shared_subtype_metric(
            plt,
            shared_subtype_comparison,
            "failure_rate",
            "Failure Rate by Shared Subtype: Benchmark vs RL",
            paths["shared_subtype_failure_rate_figure"],
        )
    if grouped_summary:
        grouped_paths = _write_grouped_metric_figures(plt, grouped_summary, figures_dir, experiment_name)
        paths.update(grouped_paths)
    plt.close("all")
    return paths


def _write_grouped_metric_figures(
    plt: Any,
    grouped_summary: list[dict[str, Any]],
    figures_dir: Path,
    experiment_name: str,
) -> dict[str, Path]:
    paths = {
        "accuracy_by_subtype_figure": figures_dir / f"{experiment_name}_accuracy_by_subtype.png",
        "failure_rate_by_subtype_figure": figures_dir / f"{experiment_name}_failure_rate_by_subtype.png",
        "margin_by_subtype_figure": figures_dir / f"{experiment_name}_mean_preference_margin_by_subtype.png",
        "accuracy_by_template_type_figure": figures_dir / f"{experiment_name}_accuracy_by_template_type.png",
        "failure_rate_by_template_type_figure": figures_dir / f"{experiment_name}_failure_rate_by_template_type.png",
    }
    _plot_grouped_metric(
        plt,
        [row for row in grouped_summary if row.get("group_by") == "subtype"],
        "accuracy",
        "Accuracy by Subtype",
        paths["accuracy_by_subtype_figure"],
    )
    _plot_grouped_metric(
        plt,
        [row for row in grouped_summary if row.get("group_by") == "subtype"],
        "failure_rate",
        "Failure Rate by Subtype",
        paths["failure_rate_by_subtype_figure"],
    )
    _plot_grouped_metric(
        plt,
        [row for row in grouped_summary if row.get("group_by") == "subtype"],
        "mean_preference_margin",
        "Mean Preference Margin by Subtype",
        paths["margin_by_subtype_figure"],
    )
    _plot_grouped_metric(
        plt,
        [row for row in grouped_summary if row.get("group_by") == "template_type"],
        "accuracy",
        "Accuracy by Template Type",
        paths["accuracy_by_template_type_figure"],
    )
    _plot_grouped_metric(
        plt,
        [row for row in grouped_summary if row.get("group_by") == "template_type"],
        "failure_rate",
        "Failure Rate by Template Type",
        paths["failure_rate_by_template_type_figure"],
    )
    return paths


def _plot_grouped_metric(
    plt: Any,
    rows: list[dict[str, Any]],
    metric: str,
    title: str,
    path: Path,
) -> None:
    if not rows:
        return
    labels = sorted({f"{row['model_id']}:{row['dataset']}" for row in rows})
    groups = _sorted_groups_by_metric(rows, metric)
    positions = list(range(len(groups)))
    width = min(0.8 / max(len(labels), 1), 0.22)
    fig, ax = plt.subplots(figsize=(max(8.0, len(groups) * 0.9), 5.0))
    for label_index, label in enumerate(labels):
        model_id, dataset = label.split(":", 1)
        offset = (label_index - (len(labels) - 1) / 2) * width
        values = [
            float(
                next(
                    (
                        row[metric]
                        for row in rows
                        if row["model_id"] == model_id and row["dataset"] == dataset and row["group"] == group
                    ),
                    0.0,
                )
            )
            for group in groups
        ]
        ax.bar([position + offset for position in positions], values, width=width, label=label)
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.set_xticks(positions)
    ax.set_xticklabels(groups, rotation=35, ha="right")
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _sorted_groups_by_metric(rows: list[dict[str, Any]], metric: str) -> list[str]:
    groups = sorted({str(row["group"]) for row in rows})
    scored: list[tuple[float, str]] = []
    for group in groups:
        values = [float(row[metric]) for row in rows if str(row["group"]) == group]
        scored.append((sum(values) / len(values) if values else 0.0, group))
    reverse = metric == "failure_rate"
    return [group for _, group in sorted(scored, key=lambda item: (item[0], item[1]), reverse=reverse)]


def _plot_overall_metric(plt, rows: list[dict[str, Any]], metric: str, title: str, path: Path) -> None:
    model_ids = sorted({str(row["model_id"]) for row in rows})
    positions = list(range(len(model_ids)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(6.5, len(model_ids) * 1.5), 4.0))
    for offset, dataset in [(-width / 2, "benchmark"), (width / 2, "rl")]:
        values = [
            float(next((row[metric] for row in rows if row["model_id"] == model_id and row["dataset"] == dataset), 0.0))
            for model_id in model_ids
        ]
        ax.bar([position + offset for position in positions], values, width=width, label=dataset)
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.set_xticks(positions)
    ax.set_xticklabels(model_ids, rotation=25, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_shared_subtype_metric(
    plt,
    rows: list[dict[str, Any]],
    metric: str,
    title: str,
    path: Path,
) -> None:
    model_ids = sorted({str(row["model_id"]) for row in rows})
    subtypes = _sorted_shared_subtypes(rows, metric)
    positions = list(range(len(subtypes)))
    width = min(0.8 / max(len(model_ids) * 2, 1), 0.18)
    fig, ax = plt.subplots(figsize=(max(8.0, len(subtypes) * 1.0), 5.0))
    for model_index, model_id in enumerate(model_ids):
        for dataset_index, dataset in enumerate(DATASETS):
            offset_index = model_index * 2 + dataset_index
            center = (len(model_ids) * 2 - 1) / 2
            offset = (offset_index - center) * width
            values = []
            for subtype in subtypes:
                row = next(
                    (
                        candidate
                        for candidate in rows
                        if candidate["model_id"] == model_id and candidate["subtype"] == subtype
                    ),
                    None,
                )
                values.append(float(row[f"{dataset}_{metric}"]) if row else 0.0)
            ax.bar(
                [position + offset for position in positions],
                values,
                width=width,
                label=f"{model_id} {dataset}",
            )
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.set_xticks(positions)
    ax.set_xticklabels(subtypes, rotation=35, ha="right")
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _sorted_shared_subtypes(rows: list[dict[str, Any]], metric: str) -> list[str]:
    subtypes = sorted({str(row["subtype"]) for row in rows})
    scored: list[tuple[float, str]] = []
    for subtype in subtypes:
        subtype_rows = [row for row in rows if row["subtype"] == subtype]
        deltas = [float(row[f"rl_{metric}"]) - float(row[f"benchmark_{metric}"]) for row in subtype_rows]
        mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
        scored.append((mean_delta, subtype))
    reverse = metric == "failure_rate"
    return [subtype for _, subtype in sorted(scored, key=lambda item: (item[0], item[1]), reverse=reverse)]


def _distance_bucket(value: Any) -> str | None:
    if value in {None, ""}:
        return None
    distance = int(value)
    if distance <= 1:
        return "short_1"
    if distance <= 4:
        return "medium_2_4"
    return "long_5_plus"


def _model_id(model_config: dict[str, Any]) -> str:
    if model_config.get("id"):
        return str(model_config["id"])
    return str(model_config.get("name", "model")).replace("/", "_").replace(":", "_")


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


def _write_pairs_jsonl(pairs: list[MinimalPair], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in _pairs_to_records(pairs):
            handle.write(json.dumps(record) + "\n")


def _pairs_to_records(pairs: list[MinimalPair]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for pair in pairs:
        metadata = pair.metadata or {}
        record = {
            "uid": pair.pair_id,
            "phenomenon": pair.phenomenon,
            "sentence_good": pair.grammatical,
            "sentence_bad": pair.ungrammatical,
        }
        for key, value in metadata.items():
            if key == "dataset_label":
                continue
            record[key] = value
        records.append(record)
    return records


def _write_markdown_report(
    report_summary: dict[str, Any],
    overall_summary: list[dict[str, Any]],
    shared_subtype_comparison: list[dict[str, Any]],
    path: Path,
    subtype_failure_summary: dict[str, Any] | None = None,
    cross_model_hardest_subtypes: list[dict[str, Any]] | None = None,
) -> None:
    lines = ["# Benchmark vs RL Dataset Comparison", ""]
    lines.append("## Hardness Definition")
    lines.append("")
    lines.append(HARDNESS_DEFINITION)
    lines.append("")
    lines.append(
        "Mean preference margin is signed: positive values mean the model prefers the grammatical sentence, "
        "while values closer to zero indicate boundary cases. It is not used by itself to mark RL as harder."
    )
    lines.extend(["", "## Distribution Warning", ""])
    distribution = report_summary["distribution_mismatch"]
    if any(not row["distribution_matched"] for row in distribution.values()):
        lines.append(
            "Benchmark and RL datasets are not distribution-matched. Interpret overall comparisons as descriptive, "
            "and use shared-subtype rows for the fairest within-category comparison."
        )
    else:
        lines.append("Benchmark and RL subtype distributions match at the configured threshold.")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append(
        "| Model | Benchmark Accuracy | RL Accuracy | Accuracy Delta | Benchmark Failure | RL Failure | Failure Delta | RL Harder | Boundary Closer |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for model_id in sorted(report_summary["overall"]):
        benchmark = _find_overall_row(overall_summary, model_id, "benchmark")
        rl = _find_overall_row(overall_summary, model_id, "rl")
        summary = report_summary["overall"][model_id]
        lines.append(
            "| {model} | {bench_acc:.3f} | {rl_acc:.3f} | {acc_delta:.3f} | {bench_fail:.3f} | {rl_fail:.3f} | {fail_delta:.3f} | {harder} | {boundary} |".format(
                model=model_id,
                bench_acc=float(benchmark["accuracy"]),
                rl_acc=float(rl["accuracy"]),
                acc_delta=float(summary["accuracy_delta_rl_minus_benchmark"]),
                bench_fail=float(benchmark["failure_rate"]),
                rl_fail=float(rl["failure_rate"]),
                fail_delta=float(summary["failure_rate_delta_rl_minus_benchmark"]),
                harder="yes" if summary["rl_harder_overall"] else "no",
                boundary="yes" if summary["rl_boundary_closer_by_abs_margin"] else "no",
            )
        )

    matched_subset = report_summary.get("matched_subset")
    if matched_subset:
        lines.extend(["", "## Matched Subset", ""])
        exact = "yes" if matched_subset["exact_subtype_match"] else "no"
        matched_distribution = "yes" if matched_subset["distribution_matched"] else "no"
        lines.append(
            "- Matching field: {field}; exact subtype match: {exact}; distribution matched: {distribution}".format(
                field=matched_subset["matching_field"],
                exact=exact,
                distribution=matched_distribution,
            )
        )
        lines.append(
            "- Sampled examples: benchmark {benchmark}; RL {rl}".format(
                benchmark=matched_subset["benchmark_total"],
                rl=matched_subset["rl_total"],
            )
        )
        lines.append(
            "- Requested examples per shared subtype: {requested}; matched groups: {count}".format(
                requested=matched_subset["requested_examples_per_shared_subtype"],
                count=matched_subset["matched_group_count"],
            )
        )
        if matched_subset["fallback_to_template_type"]:
            lines.append(f"- Fallback: {matched_subset['fallback_reason']}")
        if matched_subset["limited_groups_below_requested_count"]:
            rendered = ", ".join(
                "{group} ({sampled}/{requested})".format(
                    group=row["group"],
                    sampled=row["sampled_per_dataset"],
                    requested=row["requested_per_group"],
                )
                for row in matched_subset["limited_groups_below_requested_count"]
            )
            lines.append(f"- Groups below requested count: {rendered}")
        if matched_subset["dropped_groups_due_to_insufficient_examples"]:
            lines.append(
                "- Dropped for insufficient examples: "
                + ", ".join(matched_subset["dropped_groups_due_to_insufficient_examples"])
            )

    lines.extend(["", "## Shared Subtypes Harder In RL"])
    for model_id, rows in report_summary["shared_subtypes_harder_in_rl"].items():
        if rows:
            rendered = ", ".join(str(row["subtype"]) for row in rows)
        else:
            rendered = "none"
        lines.append(f"- {model_id}: {rendered}")

    lines.extend(["", "## Coverage Mismatch"])
    for group_field, coverage in report_summary["coverage_mismatch"].items():
        lines.append(
            "- {field}: shared {shared}; benchmark-only {benchmark_only}; RL-only {rl_only}".format(
                field=group_field,
                shared=coverage["shared_count"],
                benchmark_only=", ".join(coverage["benchmark_only"]) if coverage["benchmark_only"] else "none",
                rl_only=", ".join(coverage["rl_only"]) if coverage["rl_only"] else "none",
            )
        )

    lines.extend(["", "## Distribution Mismatch"])
    for model_id, mismatch in distribution.items():
        if mismatch["distribution_matched"]:
            lines.append(f"- {model_id}: matched at threshold {mismatch['share_delta_threshold']:.3f}")
            continue
        rendered = ", ".join(
            f"{row['subtype']} ({row['share_delta_rl_minus_benchmark']:+.3f})"
            for row in mismatch["warnings"][:6]
        )
        lines.append(f"- {model_id}: not matched; largest subtype share shifts: {rendered}")

    lines.extend(["", "## RL Difficulty Concentration"])
    for model_id, concentration in report_summary["rl_difficulty_concentration"].items():
        rendered = ", ".join(str(row["subtype"]) for row in concentration["top_subtypes"])
        status = "concentrated" if concentration["concentrated"] else "not concentrated"
        lines.append(
            f"- {model_id}: {status}; top failure share {concentration['top_failure_share']:.3f}; top subtypes: {rendered}"
        )

    if subtype_failure_summary:
        lines.extend(["", "## Subtype Failure Profile"])
        for model_id, datasets in subtype_failure_summary["hardest_subtypes_by_model_dataset"].items():
            for dataset, rows in datasets.items():
                rendered = ", ".join(f"{row['subtype']} ({row['accuracy']:.3f})" for row in rows)
                lines.append(f"- Hardest for {model_id} on {dataset}: {rendered}")
        lines.extend(["", "## Template Failure Concentration"])
        for model_id, datasets in subtype_failure_summary["failure_concentration_by_template_type"].items():
            for dataset, rows in datasets.items():
                rendered = ", ".join(
                    f"{row['template_type']} ({row['failure_share']:.3f} of failures)"
                    for row in rows
                )
                lines.append(f"- {model_id} on {dataset}: {rendered}")

    if cross_model_hardest_subtypes:
        lines.extend(["", "## Cross-Model Hardest Subtypes"])
        for row in cross_model_hardest_subtypes[:20]:
            if row.get("note"):
                lines.append(f"- {row['source']}: {row['note']}")
                continue
            lines.append(
                "- {source} / {model} / {dataset}: #{rank} {subtype} (accuracy {accuracy}, failure {failure})".format(
                    source=row.get("source"),
                    model=row.get("model_id"),
                    dataset=row.get("dataset"),
                    rank=row.get("rank"),
                    subtype=row.get("subtype"),
                    accuracy=row.get("accuracy"),
                    failure=row.get("failure_rate"),
                )
            )

    lines.extend(["", "## Shared Subtype Table"])
    lines.append("")
    lines.append("| Model | Subtype | Benchmark Accuracy | RL Accuracy | Accuracy Delta | Failure Delta | Margin Delta | RL Harder |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in shared_subtype_comparison:
        lines.append(
            "| {model} | {subtype} | {bench_acc:.3f} | {rl_acc:.3f} | {acc_delta:.3f} | {fail_delta:.3f} | {margin_delta:.6f} | {harder} |".format(
                model=row["model_id"],
                subtype=row["subtype"],
                bench_acc=float(row["benchmark_accuracy"]),
                rl_acc=float(row["rl_accuracy"]),
                acc_delta=float(row["accuracy_delta_rl_minus_benchmark"]),
                fail_delta=float(row["failure_rate_delta_rl_minus_benchmark"]),
                margin_delta=float(row["mean_preference_margin_delta_rl_minus_benchmark"]),
                harder="yes" if row["rl_harder"] else "no",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return loaded


if __name__ == "__main__":
    main()
