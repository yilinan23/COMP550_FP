"""Run baseline BLiMP-style minimal-pair evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from syntax_rl.data.blimp_loader import load_blimp_subset
from syntax_rl.evaluation.metrics import (
    GroupedEvaluationSummary,
    PairEvaluation,
    evaluate_minimal_pairs,
    summarize_grouped_evaluations,
    summarize_pair_evaluations,
)
from syntax_rl.models.scoring import build_scorer
from syntax_rl.utils import configure_logging, ensure_dir, get_logger, resolve_project_path, seed_everything

LOGGER = get_logger(__name__)


def run_baseline(config_path: str | Path) -> dict[str, Path]:
    """Run baseline evaluation from a config file and save CSV/JSON outputs."""
    config_file = resolve_project_path(config_path)
    config = _load_config(config_file)

    seed = int(config.get("experiment", {}).get("seed", 42))
    seed_everything(seed)

    output_config = config.get("outputs", {})
    log_dir = ensure_dir(output_config.get("log_dir", "outputs/logs"))
    configure_logging(log_file=log_dir / "baseline.log")

    experiment_name = config.get("experiment", {}).get("name", "baseline")
    data_config = config.get("data", {})
    phenomenon = data_config.get("phenomenon") or data_config.get("blimp_subset", "agreement")
    data_path = _resolve_data_path(data_config)

    model_config = config.get("model", {})
    scorer = build_scorer(
        provider=model_config.get("provider", "length_normalized"),
        model_name=model_config.get("name"),
        device=model_config.get("device"),
        normalize_by_token_count=model_config.get("normalize_by_token_count", True),
    )

    LOGGER.info("Loading %s data from %s", phenomenon, data_path)
    pairs = load_blimp_subset(data_path, phenomenon=phenomenon)
    results = evaluate_minimal_pairs(pairs, scorer)
    summary = summarize_pair_evaluations(results)
    group_by = config.get("evaluation", {}).get("group_by", ["phenomenon", "subtype"])
    grouped_summary = summarize_grouped_evaluations(results, group_by=group_by)

    results_dir = ensure_dir(output_config.get("results_dir", "outputs/tables"))
    csv_path = results_dir / f"{experiment_name}_results.csv"
    grouped_csv_path = results_dir / f"{experiment_name}_grouped_summary.csv"
    json_path = results_dir / f"{experiment_name}_summary.json"
    _write_results_csv(results, csv_path)
    _write_grouped_summary_csv(grouped_summary, grouped_csv_path)
    _write_summary_json(
        summary=asdict(summary),
        grouped_summary=[asdict(group) for group in grouped_summary],
        config=config,
        json_path=json_path,
    )

    LOGGER.info("Evaluated %s pairs with accuracy %.3f", summary.total, summary.accuracy)
    return {"csv": csv_path, "grouped_csv": grouped_csv_path, "json": json_path}


def main() -> None:
    """CLI entry point for baseline evaluation."""
    parser = argparse.ArgumentParser(description="Run baseline BLiMP agreement evaluation.")
    parser.add_argument(
        "--config",
        default="configs/baseline.yaml",
        help="Path to a YAML config file, relative to the project root by default.",
    )
    args = parser.parse_args()
    output_paths = run_baseline(args.config)
    print(f"Wrote CSV results to {output_paths['csv']}")
    print(f"Wrote grouped CSV summary to {output_paths['grouped_csv']}")
    print(f"Wrote JSON summary to {output_paths['json']}")


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return loaded


def _resolve_data_path(data_config: dict[str, Any]) -> Path:
    if data_config.get("path"):
        return resolve_project_path(data_config["path"])
    raw_dir = data_config.get("raw_dir", "data/raw")
    blimp_subset = data_config.get("blimp_subset", "agreement")
    candidate = resolve_project_path(raw_dir, f"{blimp_subset}.jsonl")
    if candidate.exists():
        return candidate
    return resolve_project_path(raw_dir)


def _write_results_csv(results: list[PairEvaluation], csv_path: Path) -> None:
    fieldnames = [
        "pair_id",
        "phenomenon",
        "subtype",
        "grammatical",
        "ungrammatical",
        "grammatical_score",
        "ungrammatical_score",
        "preference_margin",
        "correct",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def _write_grouped_summary_csv(grouped_summary: list[GroupedEvaluationSummary], csv_path: Path) -> None:
    fieldnames = [
        "group_by",
        "group",
        "total",
        "correct",
        "accuracy",
        "mean_preference_margin",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group in grouped_summary:
            writer.writerow(asdict(group))


def _write_summary_json(
    summary: dict[str, Any],
    grouped_summary: list[dict[str, Any]],
    config: dict[str, Any],
    json_path: Path,
) -> None:
    payload = {"summary": summary, "grouped_summary": grouped_summary, "config": config}
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
