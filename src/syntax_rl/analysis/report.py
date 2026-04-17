"""Report-oriented analysis for syntax RL training outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from syntax_rl.data.blimp_loader import MinimalPair
from syntax_rl.evaluation.metrics import evaluate_minimal_pairs, summarize_grouped_evaluations, summarize_pair_evaluations
from syntax_rl.models.scoring import build_scorer
from syntax_rl.utils import ensure_dir, resolve_project_path

POLICIES = ("trained", "random")
SCALAR_METRICS = (
    "average_reward",
    "average_trajectory_length",
    "average_preference_margin",
    "failure_rate",
    "near_failure_rate",
)


def run_analysis(config_path: str | Path) -> dict[str, Path]:
    """Run report-oriented analysis from a YAML config."""
    config = _load_config(resolve_project_path(config_path))
    inputs = config.get("inputs", {})
    output_config = config.get("outputs", {})
    analysis_config = config.get("analysis", {})
    analysis_dir = ensure_dir(output_config.get("analysis_dir", "outputs/analysis"))
    figures_dir = ensure_dir(output_config.get("figures_dir", "outputs/figures"))

    policy_comparison = _load_json(resolve_project_path(inputs["policy_comparison"]))
    training_summary = _load_json(resolve_project_path(inputs["training_summary"]))
    q_values = _load_json(resolve_project_path(inputs["q_values"]))
    training_rewards = _load_csv(resolve_project_path(inputs["training_rewards"]))
    trained_trajectories = _load_json(resolve_project_path(inputs["trained_trajectories"]))
    random_trajectories = _load_json(resolve_project_path(inputs["random_trajectories"]))
    trained_pairs = _load_jsonl(resolve_project_path(inputs["trained_pairs"]))
    random_pairs = _load_jsonl(resolve_project_path(inputs["random_pairs"]))
    official_baseline = _load_optional_baseline_summary(inputs.get("official_baseline_summary"))

    collapse = analyze_policy_collapse(
        policy_comparison,
        threshold=float(analysis_config.get("collapse_threshold", 0.8)),
    )
    q_summary = summarize_q_values(q_values)
    examples = {
        "trained": trained_pairs[: int(analysis_config.get("example_count", 5))],
        "random": random_pairs[: int(analysis_config.get("example_count", 5))],
    }

    reevaluation = {}
    reevaluation_results: list[dict[str, Any]] = []
    if config.get("reevaluation", {}).get("enabled", True):
        reevaluation_results, reevaluation = reevaluate_final_pairs(
            trained_pairs=trained_pairs,
            random_pairs=random_pairs,
            model_config=config.get("reevaluation", {}).get("model", {"provider": "length_normalized"}),
        )

    summary = {
        "policy_comparison": policy_comparison,
        "official_baseline": official_baseline,
        "policy_collapse": collapse,
        "q_value_summary": q_summary,
        "reevaluation_summary": reevaluation,
        "training_summary": training_summary,
        "artifact_counts": {
            "training_reward_rows": len(training_rewards),
            "trained_trajectories": len(trained_trajectories),
            "random_trajectories": len(random_trajectories),
            "trained_pairs": len(trained_pairs),
            "random_pairs": len(random_pairs),
        },
    }

    paths = {
        "summary_json": analysis_dir / "analysis_summary.json",
        "summary_md": analysis_dir / "analysis_report.md",
        "comparison_csv": analysis_dir / "policy_comparison_table.csv",
        "q_summary_json": analysis_dir / "q_value_summary.json",
        "examples_json": analysis_dir / "example_cases.json",
        "baseline_reference_csv": analysis_dir / "baseline_reference_table.csv",
        "reevaluation_csv": analysis_dir / "reevaluation_results.csv",
        "reevaluation_json": analysis_dir / "reevaluation_summary.json",
    }
    _write_json(summary, paths["summary_json"])
    _write_markdown_report(summary, paths["summary_md"])
    _write_comparison_csv(policy_comparison, paths["comparison_csv"])
    _write_baseline_reference_csv(official_baseline, paths["baseline_reference_csv"])
    _write_json(q_summary, paths["q_summary_json"])
    _write_json(examples, paths["examples_json"])
    _write_reevaluation_csv(reevaluation_results, paths["reevaluation_csv"])
    _write_json(reevaluation, paths["reevaluation_json"])
    figure_paths = create_figures(policy_comparison, training_rewards, figures_dir, official_baseline, reevaluation)
    paths.update(figure_paths)
    return paths


def analyze_policy_collapse(policy_comparison: dict[str, Any], threshold: float = 0.8) -> dict[str, Any]:
    """Detect whether a policy is dominated by one template family or subtype."""
    result: dict[str, Any] = {}
    for policy in POLICIES:
        summary = policy_comparison.get(policy, {})
        result[policy] = {
            "template_type": _dominance(summary.get("template_type_distribution", {}), threshold),
            "subtype": _dominance(summary.get("subtype_distribution", {}), threshold),
        }
    return result


def summarize_q_values(q_values: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Summarize key learned Q-values for interpretation."""
    initial_key = "unset|0|0|0|0|0|0"
    relative_key = "relative_clause|5|0|1|0|0|0"
    initial_actions = _rank_actions(q_values.get(initial_key, {}))
    relative_actions = _rank_actions(q_values.get(relative_key, {}))
    all_ranked = [
        {"state": state, "action": action, "q_value": value}
        for state, actions in q_values.items()
        for action, value in actions.items()
    ]
    all_ranked.sort(key=lambda row: row["q_value"], reverse=True)
    return {
        "initial_state_key": initial_key,
        "initial_state_actions": initial_actions,
        "best_initial_action": initial_actions[0] if initial_actions else None,
        "relative_clause_state_key": relative_key,
        "relative_clause_actions": relative_actions,
        "best_relative_clause_action": relative_actions[0] if relative_actions else None,
        "favors_immediate_stop_initially": bool(initial_actions and initial_actions[0]["action"] == "stop"),
        "top_q_entries": all_ranked[:10],
        "state_count": len(q_values),
    }


def reevaluate_final_pairs(
    trained_pairs: list[dict[str, Any]],
    random_pairs: list[dict[str, Any]],
    model_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Re-score final pairs with a configurable scorer."""
    scorer = build_scorer(
        provider=model_config.get("provider", "length_normalized"),
        model_name=model_config.get("name"),
        device=model_config.get("device"),
        normalize_by_token_count=model_config.get("normalize_by_token_count", True),
    )
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for policy, records in {"trained": trained_pairs, "random": random_pairs}.items():
        pairs = [_record_to_pair(record) for record in records]
        results = evaluate_minimal_pairs(pairs, scorer)
        summary = summarize_pair_evaluations(results)
        grouped = summarize_grouped_evaluations(results, ["phenomenon", "subtype"])
        summaries[policy] = {
            "total": summary.total,
            "correct": summary.correct,
            "accuracy": summary.accuracy,
            "mean_preference_margin": summary.mean_preference_margin,
            "grouped": [group.__dict__ for group in grouped],
        }
        for result in results:
            row = result.__dict__.copy()
            row["policy"] = policy
            rows.append(row)
    return rows, summaries


def create_figures(
    policy_comparison: dict[str, Any],
    training_rewards: list[dict[str, str]],
    figures_dir: Path,
    official_baseline: dict[str, Any] | None = None,
    reevaluation_summary: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Create a small set of matplotlib figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: dict[str, Path] = {}
    paths["figure_average_reward"] = _plot_policy_metric(
        plt,
        policy_comparison,
        "average_reward",
        "Average Reward",
        figures_dir / "policy_average_reward.png",
    )
    paths["figure_average_margin"] = _plot_policy_metric(
        plt,
        policy_comparison,
        "average_preference_margin",
        "Average Preference Margin",
        figures_dir / "policy_average_preference_margin.png",
    )
    paths["figure_near_failure"] = _plot_policy_metric(
        plt,
        policy_comparison,
        "near_failure_rate",
        "Near-Failure Rate",
        figures_dir / "policy_near_failure_rate.png",
    )
    paths["figure_template_distribution"] = _plot_distribution(
        plt,
        policy_comparison,
        "template_type_distribution",
        "Template Type Distribution",
        figures_dir / "template_type_distribution.png",
    )
    paths["figure_subtype_distribution"] = _plot_distribution(
        plt,
        policy_comparison,
        "subtype_distribution",
        "Subtype Distribution",
        figures_dir / "subtype_distribution.png",
    )
    if training_rewards:
        paths["figure_training_curve"] = _plot_training_curve(
            plt,
            training_rewards,
            figures_dir / "training_reward_curve.png",
        )
    if official_baseline:
        paths["figure_official_baseline_accuracy"] = _plot_official_baseline_accuracy(
            plt,
            official_baseline,
            reevaluation_summary or {},
            figures_dir / "official_baseline_accuracy_comparison.png",
        )
    plt.close("all")
    return paths


def main() -> None:
    """CLI entry point for analysis reports."""
    parser = argparse.ArgumentParser(description="Analyze syntax RL training outputs.")
    parser.add_argument("--config", default="configs/analysis.yaml")
    args = parser.parse_args()
    outputs = run_analysis(args.config)
    for name, path in outputs.items():
        print(f"Wrote {name} to {path}")


def _record_to_pair(record: dict[str, Any]) -> MinimalPair:
    metadata = {key: value for key, value in record.items() if key not in {"uid", "phenomenon", "sentence_good", "sentence_bad"}}
    return MinimalPair(
        grammatical=str(record["sentence_good"]),
        ungrammatical=str(record["sentence_bad"]),
        phenomenon=str(record.get("phenomenon", "agreement")),
        pair_id=str(record.get("uid", "")),
        metadata=metadata,
    )


def _dominance(distribution: dict[str, int], threshold: float) -> dict[str, Any]:
    total = sum(int(count) for count in distribution.values())
    if total <= 0:
        return {"collapsed": False, "dominant": None, "share": 0.0}
    dominant, count = max(distribution.items(), key=lambda item: int(item[1]))
    share = int(count) / total
    return {"collapsed": share >= threshold, "dominant": dominant, "share": share, "threshold": threshold}


def _rank_actions(actions: dict[str, float]) -> list[dict[str, Any]]:
    return [
        {"action": action, "q_value": value}
        for action, value in sorted(actions.items(), key=lambda item: item[1], reverse=True)
    ]


def _plot_policy_metric(plt, comparison: dict[str, Any], metric: str, title: str, path: Path) -> Path:
    values = [float(comparison.get(policy, {}).get(metric, 0.0)) for policy in POLICIES]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(POLICIES, values, color=["#3569a8", "#7a7a7a"])
    ax.set_title(title)
    ax.set_ylabel(metric)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_distribution(plt, comparison: dict[str, Any], key: str, title: str, path: Path) -> Path:
    labels = sorted(
        set(comparison.get("trained", {}).get(key, {}))
        | set(comparison.get("random", {}).get(key, {}))
    )
    trained = [comparison.get("trained", {}).get(key, {}).get(label, 0) for label in labels]
    random_values = [comparison.get("random", {}).get(key, {}).get(label, 0) for label in labels]
    positions = list(range(len(labels)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 3.8))
    ax.bar([position - width / 2 for position in positions], trained, width=width, label="trained", color="#3569a8")
    ax.bar([position + width / 2 for position in positions], random_values, width=width, label="random", color="#7a7a7a")
    ax.set_title(title)
    ax.set_ylabel("count")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_training_curve(plt, rows: list[dict[str, str]], path: Path) -> Path:
    episodes = [int(row["episode"]) for row in rows]
    rewards = [float(row["total_reward"]) for row in rows]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(episodes, rewards, color="#3569a8", linewidth=1.5)
    ax.set_title("Training Reward Curve")
    ax.set_xlabel("episode")
    ax.set_ylabel("total_reward")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _write_markdown_report(summary: dict[str, Any], path: Path) -> None:
    comparison = summary["policy_comparison"]
    official_baseline = summary.get("official_baseline")
    collapse = summary["policy_collapse"]
    q_summary = summary["q_value_summary"]
    lines = [
        "# Syntax RL Analysis Report",
        "",
        "## Trained vs Random",
        "",
        "| Metric | Trained | Random |",
        "| --- | ---: | ---: |",
    ]
    for metric in SCALAR_METRICS:
        lines.append(
            f"| {metric} | {comparison.get('trained', {}).get(metric, '')} | {comparison.get('random', {}).get(metric, '')} |"
        )
    lines.extend(
        [
            "",
            "## Policy Collapse",
            "",
            f"- Trained template collapse: {collapse['trained']['template_type']}",
            f"- Trained subtype collapse: {collapse['trained']['subtype']}",
            f"- Random template collapse: {collapse['random']['template_type']}",
            f"- Random subtype collapse: {collapse['random']['subtype']}",
            "",
            "## Learned Q-Values",
            "",
            f"- Best initial action: {q_summary.get('best_initial_action')}",
            f"- Best relative-clause action: {q_summary.get('best_relative_clause_action')}",
            f"- Favors immediate stop initially: {q_summary.get('favors_immediate_stop_initially')}",
            "",
            "## Official BLiMP Baseline",
            "",
            _format_official_baseline(official_baseline),
            "",
            "Note: trained/random policy metrics are computed on generated final pairs, while the official BLiMP baseline is computed on the downloaded BLiMP benchmark. Treat this as an external reference rather than an identical-distribution comparison.",
            "",
            "## Re-Evaluation",
            "",
            json.dumps(summary.get("reevaluation_summary", {}), indent=2),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_comparison_csv(comparison: dict[str, Any], path: Path) -> None:
    fieldnames = ["policy", *SCALAR_METRICS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for policy in POLICIES:
            row = {"policy": policy}
            row.update({metric: comparison.get(policy, {}).get(metric) for metric in SCALAR_METRICS})
            writer.writerow(row)


def _write_baseline_reference_csv(official_baseline: dict[str, Any] | None, path: Path) -> None:
    fieldnames = ["source", "total", "correct", "accuracy", "mean_preference_margin", "model_provider", "model_name"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if not official_baseline:
            return
        summary = official_baseline.get("summary", {})
        model = official_baseline.get("config", {}).get("model", {})
        writer.writerow(
            {
                "source": official_baseline.get("name", "official_blimp"),
                "total": summary.get("total"),
                "correct": summary.get("correct"),
                "accuracy": summary.get("accuracy"),
                "mean_preference_margin": summary.get("mean_preference_margin"),
                "model_provider": model.get("provider"),
                "model_name": model.get("name"),
            }
        )


def _write_reevaluation_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "policy",
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


def _load_optional_baseline_summary(path_value: str | Path | None) -> dict[str, Any] | None:
    if not path_value:
        return None
    path = resolve_project_path(path_value)
    if not path.exists():
        return None
    payload = _load_json(path)
    payload["path"] = str(path)
    payload["name"] = Path(path).stem.replace("_summary", "")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(payload: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _plot_official_baseline_accuracy(
    plt,
    official_baseline: dict[str, Any],
    reevaluation_summary: dict[str, Any],
    path: Path,
) -> Path:
    labels = ["trained generated", "random generated", "official BLiMP"]
    positions = list(range(len(labels)))
    values = [
        float(reevaluation_summary.get("trained", {}).get("accuracy", 0.0)),
        float(reevaluation_summary.get("random", {}).get("accuracy", 0.0)),
        float(official_baseline.get("summary", {}).get("accuracy", 0.0)),
    ]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.bar(positions, values, color=["#3569a8", "#7a7a7a", "#4f8f5f"])
    ax.set_title("Generated Policies vs Official BLiMP Accuracy")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _format_official_baseline(official_baseline: dict[str, Any] | None) -> str:
    if not official_baseline:
        return "No official BLiMP baseline summary was configured."
    summary = official_baseline.get("summary", {})
    config = official_baseline.get("config", {})
    data = config.get("data", {})
    model = config.get("model", {})
    return "\n".join(
        [
            f"- Source: {official_baseline.get('name', 'official_blimp')}",
            f"- Data path: {data.get('path')}",
            f"- Phenomenon filter: {data.get('phenomenon')}",
            f"- Model: {model.get('provider')} / {model.get('name')}",
            f"- Total pairs: {summary.get('total')}",
            f"- Accuracy: {summary.get('accuracy')}",
            f"- Mean preference margin: {summary.get('mean_preference_margin')}",
        ]
    )


if __name__ == "__main__":
    main()
