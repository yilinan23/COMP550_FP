"""Subtype-level failure analysis for controlled agreement datasets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

from syntax_rl.data.blimp_loader import load_blimp_subset
from syntax_rl.evaluation.metrics import evaluate_minimal_pairs
from syntax_rl.models.scoring import build_scorer
from syntax_rl.utils import ensure_dir, resolve_project_path, seed_everything

DEFAULT_GROUP_BY = ["template_type", "subtype", "dependency_distance_bucket", "clause_depth"]
COMPARISON_GROUPS = ("subtype", "template_type")
WIDE_COMPARISONS = (
    ("subtype", "accuracy"),
    ("subtype", "failure_rate"),
    ("subtype", "mean_preference_margin"),
    ("template_type", "accuracy"),
    ("template_type", "failure_rate"),
)


def run_subtype_analysis(config_path: str | Path) -> dict[str, Path]:
    """Run model evaluation and subtype-level summaries for a generated dataset."""
    config = _load_config(resolve_project_path(config_path))
    seed_everything(int(config.get("experiment", {}).get("seed", 42)))
    output_dir = ensure_dir(config.get("outputs", {}).get("analysis_dir", "outputs/subtype_analysis"))
    figures_dir = ensure_dir(config.get("outputs", {}).get("figures_dir", output_dir / "figures"))
    experiment_name = config.get("experiment", {}).get("name", "subtype_analysis")
    near_failure_margin = float(config.get("evaluation", {}).get("near_failure_margin", 0.01))
    continue_on_error = bool(config.get("evaluation", {}).get("continue_on_error", True))
    hard_subtype_top_n = int(config.get("evaluation", {}).get("hard_subtype_top_n", 3))
    difficulty_change_delta = float(config.get("evaluation", {}).get("difficulty_change_accuracy_delta", 0.25))

    data_config = config.get("data", {})
    pairs = load_blimp_subset(
        resolve_project_path(data_config["path"]),
        phenomenon=data_config.get("phenomenon", "agreement"),
    )

    result_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for model_config in config.get("models", []):
        if not model_config.get("enabled", True):
            continue
        model_id = _model_id(model_config)
        try:
            scorer = build_scorer(
                provider=model_config.get("provider", "length_normalized"),
                model_name=model_config.get("name"),
                **{key: value for key, value in model_config.items() if key not in {"id", "enabled", "label", "provider", "name"}},
            )
            results = evaluate_minimal_pairs(pairs, scorer)
            for result, pair in zip(results, pairs):
                metadata = pair.metadata or {}
                margin = float(result.preference_margin)
                result_rows.append(
                    {
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
        except Exception as error:
            errors.append({"model_id": model_id, "error": str(error)})
            if not continue_on_error:
                raise

    group_by = config.get("evaluation", {}).get("group_by", DEFAULT_GROUP_BY)
    model_metadata = _model_metadata(config.get("models", []))
    model_summary = _with_model_metadata(_summarize_rows(result_rows, ["model_id"]), model_metadata)
    grouped_summary = _with_model_metadata(_summarize_group_fields(result_rows, group_by), model_metadata)
    cross_model_comparison = _cross_model_comparison_rows(grouped_summary)
    wide_comparisons = {
        f"{metric}_by_{group_field}": _wide_metric_rows(grouped_summary, group_field, metric)
        for group_field, metric in WIDE_COMPARISONS
    }
    summary = _difficulty_summary(
        grouped_summary,
        top_n=hard_subtype_top_n,
        difficulty_change_delta=difficulty_change_delta,
    )
    paths = {
        "results_csv": output_dir / f"{experiment_name}_results.csv",
        "model_summary_csv": output_dir / f"{experiment_name}_model_summary.csv",
        "grouped_summary_csv": output_dir / f"{experiment_name}_grouped_summary.csv",
        "cross_model_comparison_csv": output_dir / f"{experiment_name}_cross_model_comparison.csv",
        "accuracy_by_subtype_csv": output_dir / f"{experiment_name}_accuracy_by_subtype.csv",
        "failure_rate_by_subtype_csv": output_dir / f"{experiment_name}_failure_rate_by_subtype.csv",
        "mean_preference_margin_by_subtype_csv": output_dir / f"{experiment_name}_mean_preference_margin_by_subtype.csv",
        "accuracy_by_template_type_csv": output_dir / f"{experiment_name}_accuracy_by_template_type.csv",
        "failure_rate_by_template_type_csv": output_dir / f"{experiment_name}_failure_rate_by_template_type.csv",
        "summary_json": output_dir / f"{experiment_name}_summary.json",
        "summary_md": output_dir / f"{experiment_name}_summary.md",
    }
    _write_csv(result_rows, paths["results_csv"])
    _write_csv(model_summary, paths["model_summary_csv"])
    _write_csv(grouped_summary, paths["grouped_summary_csv"])
    _write_csv(cross_model_comparison, paths["cross_model_comparison_csv"])
    _write_csv(wide_comparisons["accuracy_by_subtype"], paths["accuracy_by_subtype_csv"])
    _write_csv(wide_comparisons["failure_rate_by_subtype"], paths["failure_rate_by_subtype_csv"])
    _write_csv(wide_comparisons["mean_preference_margin_by_subtype"], paths["mean_preference_margin_by_subtype_csv"])
    _write_csv(wide_comparisons["accuracy_by_template_type"], paths["accuracy_by_template_type_csv"])
    _write_csv(wide_comparisons["failure_rate_by_template_type"], paths["failure_rate_by_template_type_csv"])
    _write_json(
        {
            "model_summary": model_summary,
            "grouped_summary": grouped_summary,
            "cross_model_comparison": cross_model_comparison,
            "difficulty_summary": summary,
            "errors": errors,
            "config": config,
        },
        paths["summary_json"],
    )
    _write_markdown_summary(summary, model_summary, paths["summary_md"])
    paths.update(_write_figures(grouped_summary, figures_dir, experiment_name))
    return paths


def main() -> None:
    """CLI entry point for subtype-level analysis."""
    parser = argparse.ArgumentParser(description="Run subtype-level failure analysis.")
    parser.add_argument("--config", default="configs/subtype_analysis.yaml")
    args = parser.parse_args()
    outputs = run_subtype_analysis(args.config)
    for label, path in outputs.items():
        print(f"Wrote {label} to {path}")


def _summarize_rows(rows: list[dict[str, Any]], group_fields: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in group_fields)
        groups.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for key, group_rows in sorted(groups.items()):
        total = len(group_rows)
        correct = sum(bool(row["correct"]) for row in group_rows)
        margins = [float(row["preference_margin"]) for row in group_rows]
        failures = [bool(row["failure"]) for row in group_rows]
        near_failures = [bool(row["near_failure"]) for row in group_rows]
        summary = {field: value for field, value in zip(group_fields, key)}
        summary.update(
            {
                "total": total,
                "correct": correct,
                "accuracy": correct / total if total else 0.0,
                "mean_preference_margin": sum(margins) / total if total else 0.0,
                "failure_rate": sum(failures) / total if total else 0.0,
                "near_failure_rate": sum(near_failures) / total if total else 0.0,
            }
        )
        summaries.append(summary)
    return summaries


def _summarize_group_fields(rows: list[dict[str, Any]], group_fields: list[str]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for group_field in group_fields:
        for summary in _summarize_rows(rows, ["model_id", group_field]):
            group_value = summary.pop(group_field)
            summary["group_by"] = group_field
            summary["group"] = group_value
            summaries.append(summary)
    return summaries


def _model_metadata(model_configs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for model_config in model_configs:
        if not model_config.get("enabled", True):
            continue
        model_id = _model_id(model_config)
        metadata[model_id] = {
            "model_name": model_config.get("name"),
            "model_provider": model_config.get("provider"),
            "model_label": model_config.get("label"),
        }
    return metadata


def _with_model_metadata(
    rows: list[dict[str, Any]],
    model_metadata: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        model_id = str(row.get("model_id", ""))
        enriched.append({**model_metadata.get(model_id, {}), **row})
    return enriched


def _cross_model_comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparison_rows = [
        row
        for row in rows
        if row.get("group_by") in COMPARISON_GROUPS
    ]
    return sorted(
        comparison_rows,
        key=lambda row: (
            str(row.get("group_by", "")),
            str(row.get("group", "")),
            str(row.get("model_id", "")),
        ),
    )


def _wide_metric_rows(rows: list[dict[str, Any]], group_field: str, metric: str) -> list[dict[str, Any]]:
    metric_rows = [row for row in rows if row.get("group_by") == group_field]
    model_ids = list(dict.fromkeys(str(row["model_id"]) for row in metric_rows))
    groups = sorted({str(row["group"]) for row in metric_rows})
    wide_rows: list[dict[str, Any]] = []
    for group in groups:
        row: dict[str, Any] = {group_field: group}
        values: list[float] = []
        for model_id in model_ids:
            value = next(
                (
                    float(metric_row[metric])
                    for metric_row in metric_rows
                    if metric_row["model_id"] == model_id and metric_row["group"] == group
                ),
                None,
            )
            row[model_id] = value
            if value is not None:
                values.append(value)
        row["mean_across_models"] = sum(values) / len(values) if values else 0.0
        wide_rows.append(row)
    reverse = metric == "failure_rate"
    return sorted(wide_rows, key=lambda row: float(row["mean_across_models"]), reverse=reverse)


def _difficulty_summary(
    rows: list[dict[str, Any]],
    top_n: int,
    difficulty_change_delta: float,
) -> dict[str, Any]:
    subtype_rows = [row for row in rows if row.get("group_by") == "subtype"]
    model_ids = list(dict.fromkeys(str(row["model_id"]) for row in subtype_rows))
    hardest_by_model: dict[str, list[dict[str, Any]]] = {}
    easiest_by_model: dict[str, list[dict[str, Any]]] = {}

    for model_id in model_ids:
        model_rows = [row for row in subtype_rows if row["model_id"] == model_id]
        hardest = sorted(
            model_rows,
            key=lambda row: (
                float(row["accuracy"]),
                -float(row["failure_rate"]),
                float(row["mean_preference_margin"]),
                str(row["group"]),
            ),
        )[:top_n]
        easiest = sorted(
            model_rows,
            key=lambda row: (
                -float(row["accuracy"]),
                float(row["failure_rate"]),
                -float(row["mean_preference_margin"]),
                str(row["group"]),
            ),
        )[:top_n]
        hardest_by_model[model_id] = [_summary_subtype_row(row) for row in hardest]
        easiest_by_model[model_id] = [_summary_subtype_row(row) for row in easiest]

    hard_sets = [
        {row["subtype"] for row in model_rows}
        for model_rows in hardest_by_model.values()
        if model_rows
    ]
    consistently_hard = sorted(set.intersection(*hard_sets)) if hard_sets else []
    subtype_names = sorted({str(row["group"]) for row in subtype_rows})
    changing_subtypes: list[dict[str, Any]] = []
    for subtype in subtype_names:
        rows_for_subtype = [row for row in subtype_rows if row["group"] == subtype]
        accuracies = [float(row["accuracy"]) for row in rows_for_subtype]
        if len(accuracies) < 2:
            continue
        accuracy_range = max(accuracies) - min(accuracies)
        if accuracy_range >= difficulty_change_delta:
            changing_subtypes.append(
                {
                    "subtype": subtype,
                    "accuracy_min": min(accuracies),
                    "accuracy_max": max(accuracies),
                    "accuracy_range": accuracy_range,
                    "by_model": {
                        str(row["model_id"]): {
                            "accuracy": float(row["accuracy"]),
                            "failure_rate": float(row["failure_rate"]),
                            "mean_preference_margin": float(row["mean_preference_margin"]),
                        }
                        for row in rows_for_subtype
                    },
                }
            )

    changing_subtypes.sort(key=lambda row: (-float(row["accuracy_range"]), str(row["subtype"])))
    return {
        "hard_subtype_top_n": top_n,
        "difficulty_change_accuracy_delta": difficulty_change_delta,
        "hardest_subtypes_by_model": hardest_by_model,
        "easiest_subtypes_by_model": easiest_by_model,
        "consistently_hard_subtypes": consistently_hard,
        "subtypes_with_substantial_difficulty_change": changing_subtypes,
    }


def _summary_subtype_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "subtype": row.get("group"),
        "total": int(row.get("total", 0)),
        "correct": int(row.get("correct", 0)),
        "accuracy": float(row.get("accuracy", 0.0)),
        "failure_rate": float(row.get("failure_rate", 0.0)),
        "mean_preference_margin": float(row.get("mean_preference_margin", 0.0)),
        "near_failure_rate": float(row.get("near_failure_rate", 0.0)),
    }


def _write_figures(rows: list[dict[str, Any]], figures_dir: Path, experiment_name: str) -> dict[str, Path]:
    subtype_rows = [row for row in rows if row.get("group_by") == "subtype"]
    template_rows = [row for row in rows if row.get("group_by") == "template_type"]
    if not rows:
        return {}
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: dict[str, Path] = {}
    if subtype_rows:
        paths["accuracy_by_subtype_figure"] = figures_dir / f"{experiment_name}_accuracy_by_subtype_across_models.png"
        paths["margin_by_subtype_figure"] = figures_dir / f"{experiment_name}_margin_by_subtype_across_models.png"
        paths["failure_by_subtype_figure"] = figures_dir / f"{experiment_name}_failure_rate_by_subtype_across_models.png"
        _plot_group_metric(plt, subtype_rows, "accuracy", "Accuracy by Subtype Across Models", paths["accuracy_by_subtype_figure"])
        _plot_group_metric(
            plt,
            subtype_rows,
            "mean_preference_margin",
            "Mean Preference Margin by Subtype Across Models",
            paths["margin_by_subtype_figure"],
        )
        _plot_group_metric(plt, subtype_rows, "failure_rate", "Failure Rate by Subtype Across Models", paths["failure_by_subtype_figure"])
    if template_rows:
        paths["accuracy_by_template_figure"] = figures_dir / f"{experiment_name}_accuracy_by_template_type_across_models.png"
        paths["failure_by_template_figure"] = figures_dir / f"{experiment_name}_failure_rate_by_template_type_across_models.png"
        _plot_group_metric(
            plt,
            template_rows,
            "accuracy",
            "Accuracy by Template Type Across Models",
            paths["accuracy_by_template_figure"],
        )
        _plot_group_metric(
            plt,
            template_rows,
            "failure_rate",
            "Failure Rate by Template Type Across Models",
            paths["failure_by_template_figure"],
        )
    plt.close("all")
    return paths


def _plot_group_metric(plt, rows: list[dict[str, Any]], metric: str, title: str, path: Path) -> None:
    model_ids = list(dict.fromkeys(str(row["model_id"]) for row in rows))
    groups = _sorted_groups_for_metric(rows, metric)
    x_positions = list(range(len(groups)))
    width = min(0.8 / max(len(model_ids), 1), 0.28)
    fig, ax = plt.subplots(figsize=(max(7, len(groups) * 0.95), 4.8))
    for model_index, model_id in enumerate(model_ids):
        offset = (model_index - (len(model_ids) - 1) / 2) * width
        values = [
            float(next((row[metric] for row in rows if row["model_id"] == model_id and row["group"] == group), 0.0))
            for group in groups
        ]
        ax.bar([position + offset for position in x_positions], values, width=width, label=model_id)
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(groups, rotation=35, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _sorted_groups_for_metric(rows: list[dict[str, Any]], metric: str) -> list[str]:
    groups = sorted({str(row["group"]) for row in rows})
    scored_groups: list[tuple[float, str]] = []
    for group in groups:
        values = [float(row[metric]) for row in rows if str(row["group"]) == group]
        mean_value = sum(values) / len(values) if values else 0.0
        scored_groups.append((mean_value, group))
    reverse = metric == "failure_rate"
    return [group for _, group in sorted(scored_groups, key=lambda item: (item[0], item[1]), reverse=reverse)]


def _distance_bucket(value: Any) -> str:
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


def _write_markdown_summary(summary: dict[str, Any], model_summary: list[dict[str, Any]], path: Path) -> None:
    lines = ["# Multi-Model Subtype Analysis Summary", ""]
    lines.append("## Model Summary")
    lines.append("")
    lines.append("| Model | Total | Accuracy | Failure Rate | Mean Margin |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in model_summary:
        lines.append(
            "| {model} | {total} | {accuracy:.3f} | {failure_rate:.3f} | {margin:.6f} |".format(
                model=row.get("model_id"),
                total=int(row.get("total", 0)),
                accuracy=float(row.get("accuracy", 0.0)),
                failure_rate=float(row.get("failure_rate", 0.0)),
                margin=float(row.get("mean_preference_margin", 0.0)),
            )
        )
    lines.extend(["", "## Hardest Subtypes"])
    for model_id, rows in summary["hardest_subtypes_by_model"].items():
        rendered = ", ".join(f"{row['subtype']} ({row['accuracy']:.3f})" for row in rows)
        lines.append(f"- {model_id}: {rendered}")
    lines.extend(["", "## Easiest Subtypes"])
    for model_id, rows in summary["easiest_subtypes_by_model"].items():
        rendered = ", ".join(f"{row['subtype']} ({row['accuracy']:.3f})" for row in rows)
        lines.append(f"- {model_id}: {rendered}")
    consistently_hard = summary["consistently_hard_subtypes"]
    lines.extend(["", "## Consistently Hard Subtypes"])
    lines.append(", ".join(consistently_hard) if consistently_hard else "None at the configured top-n threshold.")
    lines.extend(["", "## Substantial Difficulty Changes"])
    changing = summary["subtypes_with_substantial_difficulty_change"]
    if changing:
        for row in changing:
            lines.append(f"- {row['subtype']}: accuracy range {row['accuracy_range']:.3f}")
    else:
        lines.append("None at the configured accuracy-delta threshold.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return loaded


def _write_json(payload: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
