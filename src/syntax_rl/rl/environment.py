"""Lightweight structure-level RL environment for agreement examples."""

from __future__ import annotations

import random
from dataclasses import replace

from syntax_rl.generator.grammar_checks import validate_generated_pair
from syntax_rl.generator.realize import realize_minimal_pair, sample_lexical_values
from syntax_rl.generator.templates import GeneratedMinimalPair, TemplateSpec
from syntax_rl.models.scoring import LengthNormalizedScorer, SentenceScorer

from .actions import (
    ADD_PLURAL_ATTRACTOR,
    CHANGE_PREPOSITION,
    CHANGE_RELATIVE_PRONOUN,
    INCREASE_DEPTH_IF_VALID,
    SELECT_PP_TEMPLATE,
    SELECT_RELATIVE_CLAUSE_TEMPLATE,
    SELECT_SIMPLE_TEMPLATE,
    STOP,
    SyntaxAction,
)
from .rewards import RewardConfig, compute_reward_breakdown
from .state import SyntaxState

PREPOSITIONS = ("near", "behind", "beside", "with")
RELATIVE_PRONOUNS = ("that", "who")


class SyntaxEnvironment:
    """Sequential environment over structure-level agreement edits.

    The environment never edits tokens directly. Actions choose or modify a
    compact syntactic structure, and final sentence pairs are realized through
    the existing controlled generator utilities.
    """

    def __init__(
        self,
        scorer: SentenceScorer | None = None,
        reward_config: RewardConfig | None = None,
        max_steps: int = 8,
        seed: int = 42,
    ) -> None:
        self.scorer = scorer or LengthNormalizedScorer()
        self.reward_config = reward_config or RewardConfig()
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.state = SyntaxState()
        self.last_pair: GeneratedMinimalPair | None = None

    def reset(self) -> SyntaxState:
        """Reset the environment."""
        self.state = SyntaxState()
        self.last_pair = None
        return self.state

    def step(self, action: SyntaxAction) -> tuple[SyntaxState, float, bool, dict[str, object]]:
        """Apply one action and return the standard RL transition tuple."""
        if self.state.stopped:
            return self.state, 0.0, True, {"valid": False, "reason": "episode already stopped"}

        valid_actions = self.valid_actions()
        if action.name not in {valid_action.name for valid_action in valid_actions}:
            next_state = replace(
                self.state,
                step_count=self.state.step_count + 1,
                invalid_action_count=self.state.invalid_action_count + 1,
            )
            self.state = next_state
            breakdown = compute_reward_breakdown(
                model_failed=False,
                preference_margin=0.0,
                token_count=self.state.token_count,
                dependency_distance=0,
                attractor_count=0,
                clause_depth=0,
                invalid_action=True,
                grammar_valid=True,
                switched_template=False,
                stopped_simple=False,
                repeated_edit=False,
                config=self.reward_config,
            )
            done = self.state.step_count >= self.max_steps
            return self.state, breakdown.reward, done, {
                "valid": False,
                "reason": f"invalid action for state: {action.name}",
                "reward_breakdown": breakdown.as_dict(),
            }

        if action.name == STOP:
            next_state = replace(self.state, stopped=True, step_count=self.state.step_count + 1)
            self.state = next_state
            pair, grammar_valid = self._realize_current_pair()
            self.last_pair = pair
            good_score = self.scorer.score(pair.sentence_good)
            bad_score = self.scorer.score(pair.sentence_bad)
            margin = good_score - bad_score
            breakdown = compute_reward_breakdown(
                model_failed=bad_score > good_score,
                preference_margin=margin,
                token_count=self.state.token_count,
                dependency_distance=self.state.dependency_distance,
                attractor_count=self.state.attractor_count,
                clause_depth=self.state.clause_depth,
                invalid_action=False,
                grammar_valid=grammar_valid,
                switched_template=False,
                stopped_simple=self.state.template_type == "simple_agreement",
                repeated_edit=False,
                config=self.reward_config,
            )
            return self.state, breakdown.reward, True, {
                "valid": True,
                "pair": pair.to_record(),
                "grammatical_score": good_score,
                "ungrammatical_score": bad_score,
                "reward_breakdown": breakdown.as_dict(),
            }

        self.state = self._transition(action.name)
        done = self.state.step_count >= self.max_steps
        return self.state, 0.0, done, {"valid": True}

    def valid_actions(self) -> list[SyntaxAction]:
        """Return structure-level actions valid in the current state."""
        if self.state.stopped:
            return []
        if self.state.template_type == "unset":
            return [
                SyntaxAction(SELECT_SIMPLE_TEMPLATE),
                SyntaxAction(SELECT_PP_TEMPLATE),
                SyntaxAction(SELECT_RELATIVE_CLAUSE_TEMPLATE),
            ]

        actions = [SyntaxAction(STOP)]
        if self.state.template_type == "simple_agreement":
            actions.extend(
                [
                    SyntaxAction(ADD_PLURAL_ATTRACTOR),
                    SyntaxAction(INCREASE_DEPTH_IF_VALID),
                ]
            )
        elif self.state.template_type == "pp_attractor":
            if self.state.preposition_change_count < 1:
                actions.append(SyntaxAction(CHANGE_PREPOSITION))
            actions.append(SyntaxAction(INCREASE_DEPTH_IF_VALID))
        elif self.state.template_type == "relative_clause":
            if self.state.relative_pronoun_change_count < 1:
                actions.append(SyntaxAction(CHANGE_RELATIVE_PRONOUN))
        return actions

    def current_pair(self) -> GeneratedMinimalPair:
        """Realize the current structural state without changing the environment."""
        pair, _ = self._realize_current_pair()
        return pair

    def _transition(self, action_name: str) -> SyntaxState:
        if action_name == SELECT_SIMPLE_TEMPLATE:
            return self._simple_state()
        if action_name in {SELECT_PP_TEMPLATE, ADD_PLURAL_ATTRACTOR}:
            return self._pp_state(preposition=self.state.preposition)
        if action_name in {SELECT_RELATIVE_CLAUSE_TEMPLATE, INCREASE_DEPTH_IF_VALID}:
            return self._relative_state(relative_pronoun=self.state.relative_pronoun)
        if action_name == CHANGE_PREPOSITION:
            return self._pp_state(preposition=_next_value(PREPOSITIONS, self.state.preposition))
        if action_name == CHANGE_RELATIVE_PRONOUN:
            return self._relative_state(relative_pronoun=_next_value(RELATIVE_PRONOUNS, self.state.relative_pronoun))
        return self.state

    def _simple_state(self) -> SyntaxState:
        return SyntaxState(
            template_type="simple_agreement",
            dependency_distance=1,
            attractor_count=0,
            clause_depth=0,
            token_count=3,
            preposition=self.state.preposition,
            relative_pronoun=self.state.relative_pronoun,
            step_count=self.state.step_count + 1,
            invalid_action_count=self.state.invalid_action_count,
            subtype="simple_singular",
        )

    def _pp_state(self, preposition: str) -> SyntaxState:
        preposition_changed = self.state.template_type == "pp_attractor" and preposition != self.state.preposition
        return SyntaxState(
            template_type="pp_attractor",
            dependency_distance=4,
            attractor_count=1,
            clause_depth=0,
            token_count=6,
            preposition=preposition,
            relative_pronoun=self.state.relative_pronoun,
            step_count=self.state.step_count + 1,
            invalid_action_count=self.state.invalid_action_count,
            preposition_change_count=self.state.preposition_change_count + int(preposition_changed),
            relative_pronoun_change_count=self.state.relative_pronoun_change_count,
            subtype=f"pp_plural_attractor_{preposition}",
        )

    def _relative_state(self, relative_pronoun: str) -> SyntaxState:
        relative_pronoun_changed = (
            self.state.template_type == "relative_clause"
            and relative_pronoun != self.state.relative_pronoun
        )
        return SyntaxState(
            template_type="relative_clause",
            dependency_distance=5,
            attractor_count=0,
            clause_depth=1,
            token_count=7,
            preposition=self.state.preposition,
            relative_pronoun=relative_pronoun,
            step_count=self.state.step_count + 1,
            invalid_action_count=self.state.invalid_action_count,
            preposition_change_count=self.state.preposition_change_count,
            relative_pronoun_change_count=self.state.relative_pronoun_change_count + int(relative_pronoun_changed),
            subtype=f"object_relative_clause_{relative_pronoun}",
        )

    def _realize_current_pair(self) -> tuple[GeneratedMinimalPair, bool]:
        template = _template_from_state(self.state)
        values = sample_lexical_values(self.rng)
        pair = realize_minimal_pair(template, values, uid=f"rl-{self.state.step_count:04d}")
        try:
            validate_generated_pair(pair)
        except ValueError:
            return pair, False
        return pair, True


def _template_from_state(state: SyntaxState) -> TemplateSpec:
    if state.template_type == "pp_attractor":
        return TemplateSpec(
            name=f"rl_pp_{state.preposition}",
            grammatical_template=f"The {{subject_singular}} {state.preposition} the {{attractor_plural}} {{verb_singular}}.",
            ungrammatical_template=f"The {{subject_singular}} {state.preposition} the {{attractor_plural}} {{verb_plural}}.",
            phenomenon=state.phenomenon,
            template_type="pp_attractor",
            subtype=state.subtype,
            dependency_distance=4,
            attractor_count=1,
            clause_depth=0,
        )
    if state.template_type == "relative_clause":
        subject_slot = "person_singular" if state.relative_pronoun == "who" else "subject_singular"
        return TemplateSpec(
            name=f"rl_relative_{state.relative_pronoun}",
            grammatical_template=(
                f"The {{{subject_slot}}} {state.relative_pronoun} "
                "the {object_singular} {embedded_verb_singular} {verb_singular}."
            ),
            ungrammatical_template=(
                f"The {{{subject_slot}}} {state.relative_pronoun} "
                "the {object_singular} {embedded_verb_singular} {verb_plural}."
            ),
            phenomenon=state.phenomenon,
            template_type="relative_clause",
            subtype=state.subtype,
            dependency_distance=5,
            attractor_count=0,
            clause_depth=1,
        )
    return TemplateSpec(
        name="rl_simple",
        grammatical_template="The {subject_singular} {verb_singular}.",
        ungrammatical_template="The {subject_singular} {verb_plural}.",
        phenomenon=state.phenomenon,
        template_type="simple_agreement",
        subtype="simple_singular",
        dependency_distance=1,
        attractor_count=0,
        clause_depth=0,
    )


def _next_value(values: tuple[str, ...], current: str) -> str:
    try:
        index = values.index(current)
    except ValueError:
        return values[0]
    return values[(index + 1) % len(values)]
