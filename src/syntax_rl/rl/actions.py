"""Structure-level actions for the syntax RL environment."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SyntaxAction:
    """A named structural edit considered by the RL agent."""

    name: str


SELECT_SIMPLE_TEMPLATE = "select_simple_template"
SELECT_PP_TEMPLATE = "select_pp_template"
SELECT_RELATIVE_CLAUSE_TEMPLATE = "select_relative_clause_template"
ADD_PLURAL_ATTRACTOR = "add_plural_attractor"
CHANGE_PREPOSITION = "change_preposition"
CHANGE_RELATIVE_PRONOUN = "change_relative_pronoun"
INCREASE_DEPTH_IF_VALID = "increase_depth_if_valid"
STOP = "stop"

ACTION_NAMES = (
    SELECT_SIMPLE_TEMPLATE,
    SELECT_PP_TEMPLATE,
    SELECT_RELATIVE_CLAUSE_TEMPLATE,
    ADD_PLURAL_ATTRACTOR,
    CHANGE_PREPOSITION,
    CHANGE_RELATIVE_PRONOUN,
    INCREASE_DEPTH_IF_VALID,
    STOP,
)

ALL_ACTIONS = tuple(SyntaxAction(name) for name in ACTION_NAMES)
