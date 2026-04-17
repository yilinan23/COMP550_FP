"""Placeholder API-backed model adapter."""


class ApiModelScorer:
    """Placeholder scorer for remote language-model APIs."""

    def score(self, sentence: str) -> float:
        """Score a sentence."""
        raise NotImplementedError("API scoring will be implemented in phase 2.")
