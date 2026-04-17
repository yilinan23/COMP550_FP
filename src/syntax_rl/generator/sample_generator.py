"""Generate controlled subject-verb agreement minimal pairs."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import yaml

from syntax_rl.generator.grammar_checks import validate_generated_pair
from syntax_rl.generator.realize import realize_minimal_pair, sample_lexical_values
from syntax_rl.generator.templates import GeneratedMinimalPair, TemplateSpec, select_template_families
from syntax_rl.utils import configure_logging, ensure_dir, get_logger, resolve_project_path, seed_everything

LOGGER = get_logger(__name__)


def generate_minimal_pairs(
    templates: list[TemplateSpec],
    count: int,
    seed: int = 42,
    balance_template_types: bool = False,
) -> list[GeneratedMinimalPair]:
    """Generate grammar-controlled minimal pairs from template families."""
    if count < 0:
        raise ValueError("count must be non-negative")
    if not templates and count > 0:
        raise ValueError("At least one template is required")

    rng = random.Random(seed)
    generated: list[GeneratedMinimalPair] = []
    scheduled_templates = _schedule_templates(templates, count, balance_template_types)
    for index in range(count):
        template = scheduled_templates[index]
        values = sample_lexical_values(rng)
        uid = f"{template.template_type}-{index + 1:04d}"
        pair = realize_minimal_pair(template=template, values=values, uid=uid)
        validate_generated_pair(pair)
        generated.append(pair)
    return generated


def run_generation(config_path: str | Path) -> dict[str, Path]:
    """Run controlled generation from a YAML config and save JSONL/CSV outputs."""
    config_file = resolve_project_path(config_path)
    config = _load_config(config_file)
    seed = int(config.get("experiment", {}).get("seed", 42))
    seed_everything(seed)

    output_config = config.get("outputs", {})
    log_dir = ensure_dir(output_config.get("log_dir", "outputs/logs"))
    configure_logging(log_file=log_dir / "generator.log")

    generator_config = config.get("generator", {})
    template_types = generator_config.get("template_types")
    templates = select_template_families(template_types)
    templates = _filter_templates_by_metadata(templates, generator_config)
    count = int(generator_config.get("count", len(templates)))

    pairs = generate_minimal_pairs(
        templates=templates,
        count=count,
        seed=seed,
        balance_template_types=bool(generator_config.get("balance_template_types", False)),
    )
    generated_dir = ensure_dir(output_config.get("generated_dir", "data/generated"))
    experiment_name = config.get("experiment", {}).get("name", "agreement_pair_generation")
    jsonl_path = generated_dir / f"{experiment_name}.jsonl"
    csv_path = generated_dir / f"{experiment_name}.csv"
    _write_jsonl(pairs, jsonl_path)
    _write_csv(pairs, csv_path)

    LOGGER.info("Generated %s agreement minimal pairs", len(pairs))
    return {"jsonl": jsonl_path, "csv": csv_path}


def main() -> None:
    """CLI entry point for controlled agreement generation."""
    parser = argparse.ArgumentParser(description="Generate controlled agreement minimal pairs.")
    parser.add_argument(
        "--config",
        default="configs/generator.yaml",
        help="Path to a YAML config file, relative to the project root by default.",
    )
    args = parser.parse_args()
    output_paths = run_generation(args.config)
    print(f"Wrote generated JSONL to {output_paths['jsonl']}")
    print(f"Wrote generated CSV to {output_paths['csv']}")


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return loaded


def _filter_templates_by_metadata(
    templates: list[TemplateSpec],
    generator_config: dict[str, Any],
) -> list[TemplateSpec]:
    return [
        template
        for template in templates
        if template.dependency_distance <= int(generator_config.get("max_dependency_distance", 999))
        and template.attractor_count <= int(generator_config.get("max_attractor_count", 999))
        and template.clause_depth <= int(generator_config.get("max_clause_depth", 999))
    ]


def _schedule_templates(
    templates: list[TemplateSpec],
    count: int,
    balance_template_types: bool,
) -> list[TemplateSpec]:
    if not balance_template_types:
        return [templates[index % len(templates)] for index in range(count)]

    by_type: dict[str, list[TemplateSpec]] = {}
    for template in templates:
        by_type.setdefault(template.template_type, []).append(template)
    template_types = sorted(by_type)
    scheduled: list[TemplateSpec] = []
    type_offsets = {template_type: 0 for template_type in template_types}
    for index in range(count):
        template_type = template_types[index % len(template_types)]
        family = by_type[template_type]
        offset = type_offsets[template_type]
        scheduled.append(family[offset % len(family)])
        type_offsets[template_type] = offset + 1
    return scheduled


def _write_jsonl(pairs: list[GeneratedMinimalPair], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair.to_record()) + "\n")


def _write_csv(pairs: list[GeneratedMinimalPair], path: Path) -> None:
    fieldnames = list(GeneratedMinimalPair("", "", "", "", 0, 0, 0, "", "").to_record().keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for pair in pairs:
            writer.writerow(pair.to_record())


if __name__ == "__main__":
    main()
