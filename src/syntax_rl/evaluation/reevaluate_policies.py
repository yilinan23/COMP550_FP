"""Re-evaluate trained and random syntax-RL outputs with configurable scorers."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

from syntax_rl.data.blimp_loader import MinimalPair
from syntax_rl.evaluation.metrics import evaluate_minimal_pair
from syntax_rl.models.scoring import build_scorer
from syntax_rl.utils import configure_logging, ensure_dir, get_logger, resolve_project_path, seed_everything

LOGGER = get_logger(__name__)
GROUP_FIELDS = ("phenomenon", "subtype", "template_type")


def run_policy_reevaluation(config_path: str | Path) -> dict[str, Path]:
    """Run trained-vs-random final-pair re-evaluation from YAML config."""
    config = _load_config(resolve_project_path(config_path))
    return run_policy_reevaluation_from_config(config)


def run_policy_reevaluation_from_config(config: dict[str, Any]) -> dict[str, Path]:
    """Run trained-vs-random re-evaluation from an already-loaded config."""
    seed_everything(int(config.get("experiment", {}).get("seed", 42)))

    output_config = config.get("outputs", {})
    log_dir = ensure_dir(output_config.get("log_dir", "outputs/logs"))
    configure_logging(log_file=log_dir / "policy_reevaluation.log")

    inputs = config.get("inputs", {})
    trained_records = _load_jsonl(resolve_project_path(inputs["trained_pairs"]))
    random_records = _load_jsonl(resolve_project_path(inputs["random_pairs"]))

    model_config = config.get("model", {})
    scorer = build_scorer(
        provider=model_config.get("provider", "length_normalized"),
        model_name=model_config.get("name"),
        **{key: value for key, value in model_config.items() if key not in {"provider", "name"}},
    )

    rows = []
    for policy, records in {"trained": trained_records, "random": random_records}.items():
        rows.extend(_evaluate_policy_records(policy, records, scorer))

    summary = _summarize_rows(rows)
    grouped_summary = _grouped_summary(rows, config.get("evaluation", {}).get("group_by", list(GROUP_FIELDS)))

    output_dir = ensure_dir(output_config.get("reevaluation_dir", "outputs/reevaluation"))
    experiment_name = config.get("experiment", {}).get("name", "syntax_rl_real_model_reevaluation")
    paths = {
        "results_csv": output_dir / f"{experiment_name}_results.csv",
        "summary_json": output_dir / f"{experiment_name}_summary.json",
        "grouped_csv": output_dir / f"{experiment_name}_grouped_summary.csv",
    }
    _write_rows_csv(rows, paths["results_csv"])
    _write_summary_json(summary, grouped_summary, config, paths["summary_json"])
    _write_grouped_csv(grouped_summary, paths["grouped_csv"])
    LOGGER.info("Re-evaluated %s final pairs with %s", len(rows), getattr(scorer, "name", scorer.__class__.__name__))
    return paths


def main() -> None:
    """CLI entry point for policy re-evaluation."""
    parser = argparse.ArgumentParser(description="Re-evaluate trained and random syntax-RL final pairs.")
    parser.add_argument("--config", default="configs/reevaluate_hf.yaml")
    args = parser.parse_args()
    outputs = run_policy_reevaluation(args.config)
    for label, path in outputs.items():
        print(f"Wrote {label} to {path}")


def _evaluate_policy_records(policy: str, records: list[dict[str, Any]], scorer) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        pair = _record_to_pair(record)
        result = evaluate_minimal_pair(pair, scorer, index=index)
        row = result.__dict__.copy()
        row["policy"] = policy
        row["template_type"] = record.get("template_type")
        row["dependency_distance"] = record.get("dependency_distance")
        row["attractor_count"] = record.get("attractor_count")
        row["clause_depth"] = record.get("clause_depth")
        rows.append(row)
    return rows


def _record_to_pair(record: dict[str, Any]) -> MinimalPair:
    metadata = {
        key: value
        for key, value in record.items()
        if key not in {"uid", "phenomenon", "sentence_good", "sentence_bad"}
    }
    return MinimalPair(
        grammatical=str(record["sentence_good"]),
        ungrammatical=str(record["sentence_bad"]),
        phenomenon=str(record.get("phenomenon", "agreement")),
        pair_id=str(record.get("uid", "")),
        metadata=metadata,
    )


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        policy: _summarize_policy_rows([row for row in rows if row["policy"] == policy])
        for policy in ("trained", "random")
    }


def _summarize_policy_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(bool(row["correct"]) for row in rows)
    margins = [float(row["preference_margin"]) for row in rows]
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "mean_preference_margin": sum(margins) / total if total else 0.0,
    }


def _grouped_summary(rows: list[dict[str, Any]], group_by: list[str]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for policy in ("trained", "random"):
        policy_rows = [row for row in rows if row["policy"] == policy]
        for field in group_by:
            if field not in GROUP_FIELDS:
                raise ValueError(f"Unsupported group field: {field}")
            groups = sorted({str(row.get(field)) for row in policy_rows if row.get(field) not in {None, ""}})
            for group in groups:
                group_rows = [row for row in policy_rows if str(row.get(field)) == group]
                summary = _summarize_policy_rows(group_rows)
                summaries.append({"policy": policy, "group_by": field, "group": group, **summary})
    return summaries


def _write_rows_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "policy",
        "pair_id",
        "phenomenon",
        "subtype",
        "template_type",
        "dependency_distance",
        "attractor_count",
        "clause_depth",
        "grammatical",
        "ungrammatical",
        "grammatical_score",
        "ungrammatical_score",
        "preference_margin",
        "correct",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _write_grouped_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = ["policy", "group_by", "group", "total", "correct", "accuracy", "mean_preference_margin"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_summary_json(
    summary: dict[str, Any],
    grouped_summary: list[dict[str, Any]],
    config: dict[str, Any],
    path: Path,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "grouped_summary": grouped_summary, "config": config}, handle, indent=2)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return loaded


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    main()
