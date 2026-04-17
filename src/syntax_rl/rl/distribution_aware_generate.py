"""Distribution-aware hard-case generation for RL-style agreement datasets."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from syntax_rl.data.blimp_loader import load_blimp_subset
from syntax_rl.generator.grammar_checks import validate_generated_pair
from syntax_rl.generator.realize import realize_minimal_pair, sample_lexical_values
from syntax_rl.generator.templates import GeneratedMinimalPair, TemplateSpec, select_template_families
from syntax_rl.models.scoring import build_scorer
from syntax_rl.utils import ensure_dir, resolve_project_path, seed_everything


@dataclass(frozen=True)
class Candidate:
    """One generated candidate plus interpretable scoring metadata."""

    pair: GeneratedMinimalPair
    grammatical_score: float
    ungrammatical_score: float
    preference_margin: float
    model_failed: bool
    boundary_score: float
    diversity_bonus: float
    over_quota_penalty: float
    hardness_score: float


def run_distribution_aware_generation(config_path: str | Path) -> dict[str, Path]:
    """Generate a larger RL-style dataset while matching target subtype quotas."""
    config = _load_config(resolve_project_path(config_path))
    seed = int(config.get("experiment", {}).get("seed", 42))
    seed_everything(seed)
    rng = random.Random(seed)

    generation_config = config.get("generation", {})
    output_config = config.get("outputs", {})
    experiment_name = config.get("experiment", {}).get("name", "syntax_rl_distribution_aware")
    output_dir = ensure_dir(output_config.get("generated_dir", "data/generated"))
    summary_dir = ensure_dir(output_config.get("summary_dir", "outputs/distribution_aware_rl"))

    templates = _select_templates(generation_config)
    target_counts = _target_counts(config, templates)
    scorer = _build_scorer(config.get("model", {}))
    scoring_config = config.get("scoring", {})
    attempts_per_target = int(generation_config.get("attempts_per_target", 8))
    max_attempts = int(generation_config.get("max_attempts", max(sum(target_counts.values()) * attempts_per_target, 1)))
    near_failure_margin = float(scoring_config.get("near_failure_margin", 0.01))

    pools = _generate_candidate_pools(
        templates_by_subtype={template.subtype: template for template in templates},
        target_counts=target_counts,
        scorer=scorer,
        rng=rng,
        near_failure_margin=near_failure_margin,
        max_attempts=max_attempts,
        attempts_per_target=attempts_per_target,
        scoring_config=scoring_config,
    )
    selected_candidates = _select_candidates_from_pools(pools, target_counts)
    records = []
    for index, candidate in enumerate(selected_candidates, start=1):
        record = candidate.pair.to_record()
        record["uid"] = f"rl_dist_{index:04d}"
        records.append(record)

    distribution_summary = _distribution_summary(records)
    target_summary = _target_summary(target_counts, templates)
    comparison_rows = _distribution_comparison_rows(target_counts, records)
    summary = {
        "total": len(records),
        "target_total": sum(target_counts.values()),
        "subtype_distribution": distribution_summary["subtype"],
        "template_type_distribution": distribution_summary["template_type"],
        "dependency_distance_bucket_distribution": distribution_summary["dependency_distance_bucket"],
        "clause_depth_distribution": distribution_summary["clause_depth"],
        "target_distribution": target_summary,
        "distribution_comparison": comparison_rows,
        "coverage_gaps": [row for row in comparison_rows if int(row["missing_count"]) > 0],
        "overfilled_subtypes": [row for row in comparison_rows if int(row["extra_count"]) > 0],
        "config": config,
    }

    paths = {
        "jsonl": output_dir / f"{experiment_name}.jsonl",
        "csv": output_dir / f"{experiment_name}.csv",
        "candidate_scores_csv": summary_dir / f"{experiment_name}_candidate_scores.csv",
        "distribution_comparison_csv": summary_dir / f"{experiment_name}_distribution_comparison.csv",
        "summary_json": summary_dir / f"{experiment_name}_summary.json",
        "summary_md": summary_dir / f"{experiment_name}_summary.md",
    }
    _write_jsonl(records, paths["jsonl"])
    _write_records_csv(records, paths["csv"])
    _write_candidate_scores(selected_candidates, paths["candidate_scores_csv"])
    _write_csv(comparison_rows, paths["distribution_comparison_csv"])
    _write_json(summary, paths["summary_json"])
    _write_markdown_summary(summary, paths["summary_md"])
    return paths


def main() -> None:
    """CLI entry point for distribution-aware RL-style generation."""
    parser = argparse.ArgumentParser(description="Generate a quota-aware RL-style hard-case dataset.")
    parser.add_argument("--config", default="configs/rl_distribution_aware.yaml")
    args = parser.parse_args()
    outputs = run_distribution_aware_generation(args.config)
    for label, path in outputs.items():
        print(f"Wrote {label} to {path}")


def _select_templates(generation_config: dict[str, Any]) -> list[TemplateSpec]:
    templates = select_template_families(generation_config.get("target_template_types"))
    target_subtypes = generation_config.get("target_subtypes")
    if target_subtypes:
        requested = set(str(subtype) for subtype in target_subtypes)
        templates = [template for template in templates if template.subtype in requested]
    if not templates:
        raise ValueError("No templates matched the requested subtype/template constraints.")
    return templates


def _target_counts(config: dict[str, Any], templates: list[TemplateSpec]) -> dict[str, int]:
    generation_config = config.get("generation", {})
    explicit_counts = generation_config.get("target_counts")
    if explicit_counts:
        return {str(subtype): int(count) for subtype, count in explicit_counts.items() if int(count) > 0}

    benchmark_config = config.get("inputs", {}).get("benchmark_distribution", {})
    benchmark_path = benchmark_config.get("path")
    if benchmark_path:
        pairs = load_blimp_subset(
            resolve_project_path(benchmark_path),
            phenomenon=benchmark_config.get("phenomenon", "agreement"),
        )
        counts = Counter(str((pair.metadata or {}).get("subtype")) for pair in pairs)
    else:
        counts = Counter(template.subtype for template in templates)

    allowed_subtypes = {template.subtype for template in templates}
    filtered_counts = {subtype: count for subtype, count in counts.items() if subtype in allowed_subtypes}
    target_total = int(generation_config.get("target_total", sum(filtered_counts.values())))
    if target_total <= 0:
        raise ValueError("target_total must be positive.")
    if not filtered_counts:
        raise ValueError("No target subtypes overlap with available templates.")
    return _scale_counts(filtered_counts, target_total)


def _scale_counts(counts: dict[str, int], target_total: int) -> dict[str, int]:
    source_total = sum(counts.values())
    raw = {key: value * target_total / source_total for key, value in counts.items()}
    scaled = {key: int(value) for key, value in raw.items()}
    remainder = target_total - sum(scaled.values())
    ranked = sorted(raw, key=lambda key: (raw[key] - scaled[key], key), reverse=True)
    for key in ranked[:remainder]:
        scaled[key] += 1
    return {key: value for key, value in scaled.items() if value > 0}


def _generate_candidate_pools(
    templates_by_subtype: dict[str, TemplateSpec],
    target_counts: dict[str, int],
    scorer,
    rng: random.Random,
    near_failure_margin: float,
    max_attempts: int,
    attempts_per_target: int,
    scoring_config: dict[str, Any],
) -> dict[str, list[Candidate]]:
    pools: dict[str, list[Candidate]] = {subtype: [] for subtype in target_counts}
    seen_sentences: set[tuple[str, str]] = set()
    attempts = 0
    while attempts < max_attempts and _needs_more_candidates(pools, target_counts, attempts_per_target):
        subtype = _choose_underrepresented_subtype(pools, target_counts, rng, attempts_per_target)
        template = templates_by_subtype.get(subtype)
        if template is None:
            attempts += 1
            continue
        values = sample_lexical_values(rng)
        uid = f"candidate-{attempts + 1:05d}"
        pair = realize_minimal_pair(template, values, uid)
        attempts += 1
        sentence_key = (pair.sentence_good, pair.sentence_bad)
        if sentence_key in seen_sentences:
            continue
        try:
            validate_generated_pair(pair)
        except ValueError:
            continue
        seen_sentences.add(sentence_key)
        candidate = _score_candidate(
            pair=pair,
            scorer=scorer,
            current_count=len(pools[subtype]),
            target_count=target_counts[subtype],
            near_failure_margin=near_failure_margin,
            scoring_config=scoring_config,
        )
        pools[subtype].append(candidate)
    return pools


def _needs_more_candidates(
    pools: dict[str, list[Candidate]],
    target_counts: dict[str, int],
    attempts_per_target: int,
) -> bool:
    return any(len(pools[subtype]) < target_counts[subtype] * attempts_per_target for subtype in target_counts)


def _choose_underrepresented_subtype(
    pools: dict[str, list[Candidate]],
    target_counts: dict[str, int],
    rng: random.Random,
    attempts_per_target: int,
) -> str:
    weights: list[tuple[str, float]] = []
    for subtype, target_count in target_counts.items():
        target_candidates = max(target_count * attempts_per_target, 1)
        progress = len(pools[subtype]) / target_candidates
        weights.append((subtype, max(1.0 - progress, 0.01)))
    total = sum(weight for _, weight in weights)
    threshold = rng.random() * total
    running = 0.0
    for subtype, weight in weights:
        running += weight
        if running >= threshold:
            return subtype
    return weights[-1][0]


def _score_candidate(
    pair: GeneratedMinimalPair,
    scorer,
    current_count: int,
    target_count: int,
    near_failure_margin: float,
    scoring_config: dict[str, Any],
) -> Candidate:
    good_score = scorer.score(pair.sentence_good)
    bad_score = scorer.score(pair.sentence_bad)
    margin = good_score - bad_score
    model_failed = bad_score > good_score
    boundary_score = max(0.0, near_failure_margin - abs(margin)) / near_failure_margin if near_failure_margin > 0 else 0.0
    progress = current_count / max(target_count, 1)
    diversity_bonus = max(0.0, 1.0 - progress) * float(scoring_config.get("underrepresented_subtype_bonus", 0.25))
    over_quota_penalty = max(0.0, progress - 1.0) * float(scoring_config.get("overrepresented_subtype_penalty", 0.5))
    hardness_score = (
        float(scoring_config.get("model_failure_weight", 1.0)) * float(model_failed)
        + float(scoring_config.get("boundary_weight", 0.5)) * boundary_score
        - float(scoring_config.get("margin_weight", 0.1)) * margin
        + diversity_bonus
        - over_quota_penalty
    )
    return Candidate(
        pair=pair,
        grammatical_score=good_score,
        ungrammatical_score=bad_score,
        preference_margin=margin,
        model_failed=model_failed,
        boundary_score=boundary_score,
        diversity_bonus=diversity_bonus,
        over_quota_penalty=over_quota_penalty,
        hardness_score=hardness_score,
    )


def _select_candidates_from_pools(
    pools: dict[str, list[Candidate]],
    target_counts: dict[str, int],
) -> list[Candidate]:
    selected: list[Candidate] = []
    for subtype in sorted(target_counts):
        ranked = sorted(
            pools.get(subtype, []),
            key=lambda candidate: (
                -candidate.hardness_score,
                candidate.preference_margin,
                candidate.pair.sentence_good,
            ),
        )
        selected.extend(ranked[: target_counts[subtype]])
    return selected


def _distribution_summary(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        "subtype": dict(Counter(str(record.get("subtype")) for record in records)),
        "template_type": dict(Counter(str(record.get("template_type")) for record in records)),
        "dependency_distance_bucket": dict(Counter(_distance_bucket(record.get("dependency_distance")) for record in records)),
        "clause_depth": dict(Counter(str(record.get("clause_depth")) for record in records)),
    }


def _target_summary(target_counts: dict[str, int], templates: list[TemplateSpec]) -> dict[str, Any]:
    template_by_subtype = {template.subtype: template for template in templates}
    records = []
    for subtype, count in target_counts.items():
        template = template_by_subtype[subtype]
        records.extend(
            {
                "subtype": subtype,
                "template_type": template.template_type,
                "dependency_distance": template.dependency_distance,
                "clause_depth": template.clause_depth,
            }
            for _ in range(count)
        )
    return _distribution_summary(records)


def _distribution_comparison_rows(target_counts: dict[str, int], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    achieved = Counter(str(record.get("subtype")) for record in records)
    target_total = sum(target_counts.values())
    achieved_total = sum(achieved.values())
    rows: list[dict[str, Any]] = []
    for subtype in sorted(set(target_counts) | set(achieved)):
        target_count = int(target_counts.get(subtype, 0))
        achieved_count = int(achieved.get(subtype, 0))
        rows.append(
            {
                "subtype": subtype,
                "target_count": target_count,
                "achieved_count": achieved_count,
                "missing_count": max(target_count - achieved_count, 0),
                "extra_count": max(achieved_count - target_count, 0),
                "target_share": target_count / target_total if target_total else 0.0,
                "achieved_share": achieved_count / achieved_total if achieved_total else 0.0,
            }
        )
    return rows


def _distance_bucket(value: Any) -> str:
    distance = int(value)
    if distance <= 1:
        return "short_1"
    if distance <= 4:
        return "medium_2_4"
    return "long_5_plus"


def _build_scorer(model_config: dict[str, Any]):
    return build_scorer(
        provider=model_config.get("provider", "length_normalized"),
        model_name=model_config.get("name"),
        **{key: value for key, value in model_config.items() if key not in {"provider", "name"}},
    )


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _write_records_csv(records: list[dict[str, Any]], path: Path) -> None:
    _write_csv(records, path)


def _write_candidate_scores(candidates: list[Candidate], path: Path) -> None:
    rows = []
    for candidate in candidates:
        record = candidate.pair.to_record()
        rows.append(
            {
                **record,
                "grammatical_score": candidate.grammatical_score,
                "ungrammatical_score": candidate.ungrammatical_score,
                "preference_margin": candidate.preference_margin,
                "model_failed": candidate.model_failed,
                "boundary_score": candidate.boundary_score,
                "diversity_bonus": candidate.diversity_bonus,
                "over_quota_penalty": candidate.over_quota_penalty,
                "hardness_score": candidate.hardness_score,
            }
        )
    _write_csv(rows, path)


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


def _write_markdown_summary(summary: dict[str, Any], path: Path) -> None:
    lines = ["# Distribution-Aware RL Dataset Summary", ""]
    lines.append(f"- Total generated: {summary['total']}")
    lines.append(f"- Target total: {summary['target_total']}")
    lines.extend(["", "## Subtype Distribution"])
    for subtype, count in sorted(summary["subtype_distribution"].items()):
        lines.append(f"- {subtype}: {count}")
    lines.extend(["", "## Template Type Distribution"])
    for template_type, count in sorted(summary["template_type_distribution"].items()):
        lines.append(f"- {template_type}: {count}")
    lines.extend(["", "## Coverage Gaps"])
    gaps = summary["coverage_gaps"]
    if gaps:
        for row in gaps:
            lines.append(f"- {row['subtype']}: missing {row['missing_count']} of target {row['target_count']}")
    else:
        lines.append("No subtype coverage gaps.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return loaded


if __name__ == "__main__":
    main()
