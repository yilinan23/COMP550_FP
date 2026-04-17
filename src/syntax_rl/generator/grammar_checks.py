"""Validation helpers for controlled agreement minimal pairs."""

from __future__ import annotations

from .templates import GeneratedMinimalPair


def is_valid_minimal_pair(grammatical: str, ungrammatical: str) -> bool:
    """Return whether two non-empty strings can be treated as a minimal pair."""
    return bool(grammatical.strip()) and bool(ungrammatical.strip()) and grammatical != ungrammatical


def differs_only_in_final_verb(pair: GeneratedMinimalPair) -> bool:
    """Return whether two generated sentences differ only in the final verb token."""
    good_tokens = pair.sentence_good.rstrip(".").split()
    bad_tokens = pair.sentence_bad.rstrip(".").split()
    if len(good_tokens) != len(bad_tokens) or not good_tokens:
        return False
    return good_tokens[:-1] == bad_tokens[:-1] and good_tokens[-1] != bad_tokens[-1]


def compute_dependency_distance(sentence: str) -> int:
    """Compute linear subject-head to main-verb distance for generated sentences.

    The controlled templates always place the subject head noun after the
    initial determiner and the main verb at the final token. Distance is the
    number of token steps from that subject head to the main verb.
    """
    tokens = sentence.rstrip(".").split()
    if len(tokens) < 3:
        raise ValueError(f"Sentence is too short to compute dependency distance: {sentence}")
    subject_head_index = 1
    main_verb_index = len(tokens) - 1
    return main_verb_index - subject_head_index


def validate_generated_pair(pair: GeneratedMinimalPair) -> None:
    """Validate the expected agreement-only contrast and metadata bounds."""
    if not is_valid_minimal_pair(pair.sentence_good, pair.sentence_bad):
        raise ValueError(f"Invalid minimal pair: {pair.uid}")
    if not differs_only_in_final_verb(pair):
        raise ValueError(f"Generated pair must differ only in the final verb: {pair.uid}")
    if pair.dependency_distance < 0 or pair.attractor_count < 0 or pair.clause_depth < 0:
        raise ValueError(f"Generated metadata must be non-negative: {pair.uid}")
    computed_distance = compute_dependency_distance(pair.sentence_good)
    if pair.dependency_distance != computed_distance:
        raise ValueError(
            f"Dependency distance mismatch for {pair.uid}: "
            f"metadata={pair.dependency_distance}, computed={computed_distance}"
        )
