"""Metrics for grammatical-vs-ungrammatical minimal-pair evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from syntax_rl.data.blimp_loader import MinimalPair
from syntax_rl.models.scoring import SentenceScorer


@dataclass(frozen=True)
class PairEvaluation:
    """Evaluation result for one minimal pair."""

    pair_id: str
    phenomenon: str
    subtype: str | None
    grammatical: str
    ungrammatical: str
    grammatical_score: float
    ungrammatical_score: float
    correct: bool
    preference_margin: float


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate minimal-pair evaluation metrics."""

    total: int
    correct: int
    accuracy: float
    mean_preference_margin: float


@dataclass(frozen=True)
class GroupedEvaluationSummary:
    """Aggregate metrics for one result group."""

    group_by: str
    group: str
    total: int
    correct: int
    accuracy: float
    mean_preference_margin: float


SUBTYPE_METADATA_FIELDS = ("subtype", "linguistic_term", "linguistics_term", "field")


def accuracy(correct: int, total: int) -> float:
    """Compute accuracy with explicit handling for empty inputs."""
    if total <= 0:
        return 0.0
    return correct / total


def evaluate_minimal_pair(pair: MinimalPair, scorer: SentenceScorer, index: int = 0) -> PairEvaluation:
    """Evaluate whether a scorer prefers the grammatical sentence."""
    grammatical_score = scorer.score(pair.grammatical)
    ungrammatical_score = scorer.score(pair.ungrammatical)
    margin = grammatical_score - ungrammatical_score

    return PairEvaluation(
        pair_id=pair.pair_id or str(index),
        phenomenon=pair.phenomenon,
        subtype=_extract_subtype(pair),
        grammatical=pair.grammatical,
        ungrammatical=pair.ungrammatical,
        grammatical_score=grammatical_score,
        ungrammatical_score=ungrammatical_score,
        correct=margin > 0,
        preference_margin=margin,
    )


def evaluate_minimal_pairs(pairs: list[MinimalPair], scorer: SentenceScorer) -> list[PairEvaluation]:
    """Evaluate a list of minimal pairs."""
    return [evaluate_minimal_pair(pair, scorer, index=index) for index, pair in enumerate(pairs)]


def summarize_pair_evaluations(results: list[PairEvaluation]) -> EvaluationSummary:
    """Summarize pair-level preference results."""
    correct_count = sum(result.correct for result in results)
    return EvaluationSummary(
        total=len(results),
        correct=correct_count,
        accuracy=accuracy(correct_count, len(results)),
        mean_preference_margin=_mean_preference_margin(results),
    )


def summarize_grouped_evaluations(
    results: list[PairEvaluation],
    group_by: list[str],
) -> list[GroupedEvaluationSummary]:
    """Summarize results by supported grouping fields."""
    summaries: list[GroupedEvaluationSummary] = []
    for group_field in group_by:
        grouped = _group_results(results, group_field)
        for group_name, group_results in sorted(grouped.items()):
            correct_count = sum(result.correct for result in group_results)
            summaries.append(
                GroupedEvaluationSummary(
                    group_by=group_field,
                    group=group_name,
                    total=len(group_results),
                    correct=correct_count,
                    accuracy=accuracy(correct_count, len(group_results)),
                    mean_preference_margin=_mean_preference_margin(group_results),
                )
            )
    return summaries


def _extract_subtype(pair: MinimalPair) -> str | None:
    metadata = pair.metadata or {}
    for field_name in SUBTYPE_METADATA_FIELDS:
        value = metadata.get(field_name)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _group_results(results: list[PairEvaluation], group_field: str) -> dict[str, list[PairEvaluation]]:
    grouped: dict[str, list[PairEvaluation]] = {}
    for result in results:
        group_value = _group_value(result, group_field)
        if group_value is None:
            continue
        grouped.setdefault(group_value, []).append(result)
    return grouped


def _group_value(result: PairEvaluation, group_field: str) -> str | None:
    if group_field == "phenomenon":
        return result.phenomenon
    if group_field == "subtype":
        return result.subtype
    raise ValueError(f"Unsupported group field: {group_field}")


def _mean_preference_margin(results: list[PairEvaluation]) -> float:
    if not results:
        return 0.0
    return sum(result.preference_margin for result in results) / len(results)
