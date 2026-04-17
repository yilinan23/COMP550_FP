"""State structures for structure-level syntax search."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SyntaxState:
    """Compact structural state for agreement minimal-pair construction."""

    phenomenon: str = "agreement"
    template_type: str = "unset"
    dependency_distance: int = 0
    attractor_count: int = 0
    clause_depth: int = 0
    token_count: int = 0
    stopped: bool = False
    preposition: str = "near"
    relative_pronoun: str = "that"
    step_count: int = 0
    invalid_action_count: int = 0
    preposition_change_count: int = 0
    relative_pronoun_change_count: int = 0
    subtype: str = "unset"

    def as_dict(self) -> dict[str, str | int | bool]:
        """Return a JSON-friendly state dictionary."""
        return {
            "phenomenon": self.phenomenon,
            "template_type": self.template_type,
            "dependency_distance": self.dependency_distance,
            "attractor_count": self.attractor_count,
            "clause_depth": self.clause_depth,
            "token_count": self.token_count,
            "stopped": self.stopped,
            "preposition": self.preposition,
            "relative_pronoun": self.relative_pronoun,
            "step_count": self.step_count,
            "invalid_action_count": self.invalid_action_count,
            "preposition_change_count": self.preposition_change_count,
            "relative_pronoun_change_count": self.relative_pronoun_change_count,
            "subtype": self.subtype,
        }
