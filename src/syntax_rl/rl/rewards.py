"""Reward helpers for structure-level syntax search."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    """Weights used by the RL environment reward function."""

    model_failure_weight: float = 1.0
    low_confidence_weight: float = 0.6
    low_confidence_margin: float = 0.2
    complexity_weight: float = 0.02
    dependency_distance_weight: float = 0.5
    attractor_weight: float = 1.0
    clause_depth_weight: float = 1.0
    invalid_structure_penalty: float = 1.0
    length_penalty: float = 0.002
    grammar_failure_penalty: float = 1.0
    template_switch_penalty: float = 0.15
    simple_stop_penalty: float = 0.1
    repeated_edit_penalty: float = 0.1


@dataclass(frozen=True)
class RewardBreakdown:
    """Detailed reward components for debugging rollouts."""

    reward: float
    model_failed: bool
    preference_margin: float
    model_failure_reward: float
    low_confidence_reward: float
    complexity_reward: float
    invalid_penalty: float
    length_penalty: float
    grammar_penalty: float
    switching_penalty: float
    simple_stop_penalty: float
    repeated_edit_penalty: float

    def as_dict(self) -> dict[str, float | bool]:
        """Return a JSON-friendly reward dictionary."""
        return {
            "reward": self.reward,
            "model_failed": self.model_failed,
            "preference_margin": self.preference_margin,
            "model_failure_reward": self.model_failure_reward,
            "low_confidence_reward": self.low_confidence_reward,
            "complexity_reward": self.complexity_reward,
            "invalid_penalty": self.invalid_penalty,
            "length_penalty": self.length_penalty,
            "grammar_penalty": self.grammar_penalty,
            "switching_penalty": self.switching_penalty,
            "simple_stop_penalty": self.simple_stop_penalty,
            "repeated_edit_penalty": self.repeated_edit_penalty,
        }


def compute_reward(
    model_failed: bool,
    penalty: float = 0.0,
    preference_margin: float = 0.0,
    token_count: int = 0,
    dependency_distance: int = 0,
    attractor_count: int = 0,
    clause_depth: int = 0,
    invalid_action: bool = False,
    grammar_valid: bool = True,
    switched_template: bool = False,
    stopped_simple: bool = False,
    repeated_edit: bool = False,
    config: RewardConfig | None = None,
) -> float:
    """Compute scalar reward while preserving the old simple API."""
    if (
        config is None
        and preference_margin == 0.0
        and token_count == 0
        and dependency_distance == 0
        and attractor_count == 0
        and clause_depth == 0
        and not invalid_action
        and grammar_valid
        and not switched_template
        and not stopped_simple
        and not repeated_edit
    ):
        return float(model_failed) - penalty

    reward_config = config or RewardConfig()
    return compute_reward_breakdown(
        model_failed=model_failed,
        preference_margin=preference_margin,
        token_count=token_count,
        dependency_distance=dependency_distance,
        attractor_count=attractor_count,
        clause_depth=clause_depth,
        invalid_action=invalid_action,
        grammar_valid=grammar_valid,
        switched_template=switched_template,
        stopped_simple=stopped_simple,
        repeated_edit=repeated_edit,
        config=reward_config,
        extra_penalty=penalty,
    ).reward


def compute_reward_breakdown(
    model_failed: bool,
    preference_margin: float,
    token_count: int,
    dependency_distance: int,
    attractor_count: int,
    clause_depth: int,
    invalid_action: bool,
    grammar_valid: bool,
    switched_template: bool,
    stopped_simple: bool,
    repeated_edit: bool,
    config: RewardConfig,
    extra_penalty: float = 0.0,
) -> RewardBreakdown:
    """Compute reward components for a final or invalid transition."""
    model_failure_reward = config.model_failure_weight if model_failed else 0.0
    low_confidence_reward = _low_confidence_reward(preference_margin, config)
    complexity_reward = _complexity_reward(
        preference_margin=preference_margin,
        dependency_distance=dependency_distance,
        attractor_count=attractor_count,
        clause_depth=clause_depth,
        model_failed=model_failed,
        config=config,
    )
    invalid_penalty = config.invalid_structure_penalty if invalid_action else 0.0
    length_penalty = config.length_penalty * token_count
    grammar_penalty = 0.0 if grammar_valid else config.grammar_failure_penalty
    switching_penalty = config.template_switch_penalty if switched_template else 0.0
    simple_stop_penalty = config.simple_stop_penalty if stopped_simple else 0.0
    repeated_edit_penalty = config.repeated_edit_penalty if repeated_edit else 0.0
    reward = (
        model_failure_reward
        + low_confidence_reward
        + complexity_reward
        - invalid_penalty
        - length_penalty
        - grammar_penalty
        - switching_penalty
        - simple_stop_penalty
        - repeated_edit_penalty
        - extra_penalty
    )
    return RewardBreakdown(
        reward=reward,
        model_failed=model_failed,
        preference_margin=preference_margin,
        model_failure_reward=model_failure_reward,
        low_confidence_reward=low_confidence_reward,
        complexity_reward=complexity_reward,
        invalid_penalty=invalid_penalty,
        length_penalty=length_penalty,
        grammar_penalty=grammar_penalty,
        switching_penalty=switching_penalty,
        simple_stop_penalty=simple_stop_penalty,
        repeated_edit_penalty=repeated_edit_penalty,
    )


def _low_confidence_reward(preference_margin: float, config: RewardConfig) -> float:
    """Reward examples close to the decision boundary.

    Small positive margins get full low-confidence reward because the model is
    barely preferring the grammatical sentence. Small negative margins already
    trigger model-failure reward, so they get a smaller additional bonus.
    """
    absolute_margin = abs(preference_margin)
    if absolute_margin >= config.low_confidence_margin:
        return 0.0
    closeness = 1.0 - (absolute_margin / config.low_confidence_margin)
    multiplier = 1.0 if preference_margin >= 0 else 0.5
    return config.low_confidence_weight * closeness * multiplier


def _complexity_reward(
    preference_margin: float,
    dependency_distance: int,
    attractor_count: int,
    clause_depth: int,
    model_failed: bool,
    config: RewardConfig,
) -> float:
    """Return a weak complexity bonus gated by model difficulty."""
    if not model_failed and abs(preference_margin) >= config.low_confidence_margin:
        return 0.0
    complexity_score = (
        config.dependency_distance_weight * dependency_distance
        + config.attractor_weight * attractor_count
        + config.clause_depth_weight * clause_depth
    )
    return config.complexity_weight * complexity_score
