"""Run final trained-vs-random evaluation across multiple configured models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

from syntax_rl.evaluation.reevaluate_policies import run_policy_reevaluation_from_config
from syntax_rl.utils import ensure_dir, resolve_project_path, seed_everything

POLICIES = ("trained", "random")


def run_multi_model_evaluation(config_path: str | Path) -> dict[str, Path]:
    """Evaluate trained/random final pairs across enabled models."""
    config = _load_config(resolve_project_path(config_path))
    seed_everything(int(config.get("experiment", {}).get("seed", 42)))
    output_dir = ensure_dir(config.get("outputs", {}).get("final_eval_dir", "outputs/final_eval"))
    per_model_dir = ensure_dir(output_dir / "per_model")
    figures_dir = ensure_dir(output_dir / "figures")
    experiment_name = config.get("experiment", {}).get("name", "syntax_rl_final_eval")
    near_failure_margin = float(config.get("evaluation", {}).get("near_failure_margin", 0.01))
    continue_on_error = bool(config.get("evaluation", {}).get("continue_on_error", True))

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for model_config in config.get("models", []):
        if not model_config.get("enabled", True):
            continue
        model_id = _model_id(model_config)
        reeval_config = _single_model_config(config, model_config, per_model_dir, f"{experiment_name}_{model_id}")
        try:
            paths = run_policy_reevaluation_from_config(reeval_config)
            summary_payload = _load_json(paths["summary_json"])
            result_rows = _load_csv(paths["results_csv"])
            rows.extend(_comparison_rows(model_id, model_config, summary_payload, result_rows, near_failure_margin))
        except Exception as error:
            errors.append({"model_id": model_id, "error": str(error)})
            if not continue_on_error:
                raise

    paths = {
        "comparison_csv": output_dir / f"{experiment_name}_model_comparison.csv",
        "summary_json": output_dir / f"{experiment_name}_summary.json",
    }
    _write_comparison_csv(rows, paths["comparison_csv"])
    _write_json({"comparison": rows, "errors": errors, "config": config}, paths["summary_json"])
    figure_paths = _write_figures(rows, figures_dir, experiment_name)
    paths.update(figure_paths)
    return paths


def main() -> None:
    """CLI entry point for multi-model final evaluation."""
    parser = argparse.ArgumentParser(description="Run final syntax-RL evaluation across multiple models.")
    parser.add_argument("--config", default="configs/final_eval.yaml")
    args = parser.parse_args()
    outputs = run_multi_model_evaluation(args.config)
    for label, path in outputs.items():
        print(f"Wrote {label} to {path}")


def _single_model_config(
    base_config: dict[str, Any],
    model_config: dict[str, Any],
    output_dir: Path,
    experiment_name: str,
) -> dict[str, Any]:
    return {
        "experiment": {"name": experiment_name, "seed": base_config.get("experiment", {}).get("seed", 42)},
        "inputs": base_config["inputs"],
        "model": {key: value for key, value in model_config.items() if key not in {"id", "enabled", "label"}},
        "evaluation": {"group_by": base_config.get("evaluation", {}).get("group_by", ["subtype", "template_type"])},
        "outputs": {
            "reevaluation_dir": str(output_dir),
            "log_dir": base_config.get("outputs", {}).get("log_dir", "outputs/logs"),
        },
    }


def _comparison_rows(
    model_id: str,
    model_config: dict[str, Any],
    summary_payload: dict[str, Any],
    result_rows: list[dict[str, str]],
    near_failure_margin: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary = summary_payload.get("summary", {})
    for policy in POLICIES:
        policy_summary = summary.get(policy, {})
        policy_results = [row for row in result_rows if row.get("policy") == policy]
        margins = [float(row.get("preference_margin", 0.0)) for row in policy_results]
        failures = [str(row.get("correct", "")).lower() != "true" for row in policy_results]
        near_failures = [abs(margin) <= near_failure_margin for margin in margins]
        total = int(policy_summary.get("total", len(policy_results)) or 0)
        rows.append(
            {
                "model_id": model_id,
                "model_name": model_config.get("name"),
                "model_provider": model_config.get("provider"),
                "policy": policy,
                "total": total,
                "correct": int(policy_summary.get("correct", 0) or 0),
                "accuracy": float(policy_summary.get("accuracy", 0.0) or 0.0),
                "mean_preference_margin": float(policy_summary.get("mean_preference_margin", 0.0) or 0.0),
                "failure_rate": sum(failures) / total if total else 0.0,
                "near_failure_rate": sum(near_failures) / total if total else 0.0,
            }
        )
    return rows


def _write_figures(rows: list[dict[str, Any]], figures_dir: Path, experiment_name: str) -> dict[str, Path]:
    if not rows:
        return {}
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = {
        "accuracy_figure": figures_dir / f"{experiment_name}_accuracy_by_model.png",
        "margin_figure": figures_dir / f"{experiment_name}_margin_by_model.png",
    }
    _plot_policy_metric(plt, rows, "accuracy", "Accuracy Across Models", paths["accuracy_figure"])
    _plot_policy_metric(plt, rows, "mean_preference_margin", "Mean Preference Margin Across Models", paths["margin_figure"])
    plt.close("all")
    return paths


def _plot_policy_metric(plt, rows: list[dict[str, Any]], metric: str, title: str, path: Path) -> None:
    model_ids = list(dict.fromkeys(str(row["model_id"]) for row in rows))
    positions = list(range(len(model_ids)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(6.5, len(model_ids) * 1.4), 3.8))
    for offset, policy in [(-width / 2, "trained"), (width / 2, "random")]:
        values = [
            float(next((row[metric] for row in rows if row["model_id"] == model_id and row["policy"] == policy), 0.0))
            for model_id in model_ids
        ]
        ax.bar([position + offset for position in positions], values, width=width, label=policy)
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.set_xticks(positions)
    ax.set_xticklabels(model_ids, rotation=25, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _model_id(model_config: dict[str, Any]) -> str:
    if model_config.get("id"):
        return str(model_config["id"])
    return str(model_config.get("name", "model")).replace("/", "_").replace(":", "_")


def _write_comparison_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "model_id",
        "model_name",
        "model_provider",
        "policy",
        "total",
        "correct",
        "accuracy",
        "mean_preference_margin",
        "failure_rate",
        "near_failure_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return loaded


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(payload: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
