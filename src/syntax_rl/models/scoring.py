"""Shared model scoring interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class SentenceScorer(Protocol):
    """Protocol for objects that assign scores to sentences."""

    def score(self, sentence: str) -> float:
        """Return a numeric sentence score."""
        ...


@dataclass(frozen=True)
class Score:
    """A sentence score returned by a scorer implementation."""

    sentence: str
    value: float


class LengthNormalizedScorer:
    """Placeholder dry-run scorer for pipeline tests.

    This is not a real language model and should not be used for research
    conclusions. It gives a stable score based on mean token length so the
    evaluation pipeline can be exercised without downloads.
    """

    name = "length_normalized"

    def score(self, sentence: str) -> float:
        """Return a deterministic sentence score."""
        tokens = sentence.split()
        if not tokens:
            return 0.0
        character_count = sum(len(token) for token in tokens)
        return character_count / len(tokens)


def build_scorer(
    provider: str,
    model_name: str | None = None,
    **kwargs: Any,
) -> SentenceScorer:
    """Build a scorer from config values."""
    normalized_provider = provider.lower()
    if normalized_provider in {"length", "length_normalized", "dry_run"}:
        return LengthNormalizedScorer()
    if normalized_provider in {"hf", "huggingface", "huggingface_causal_lm"}:
        from syntax_rl.models.hf_model import DEFAULT_HF_MODEL, HuggingFaceCausalLMScorer

        return HuggingFaceCausalLMScorer(
            model_name=model_name or DEFAULT_HF_MODEL,
            device=kwargs.get("device"),
            normalize_by_token_count=kwargs.get("normalize_by_token_count", True),
            cache_dir=kwargs.get("cache_dir"),
            torch_dtype=kwargs.get("torch_dtype"),
            trust_remote_code=bool(kwargs.get("trust_remote_code", False)),
            low_cpu_mem_usage=bool(kwargs.get("low_cpu_mem_usage", False)),
            tokenizer_use_fast=kwargs.get("tokenizer_use_fast"),
            device_map=kwargs.get("device_map"),
            offload_folder=kwargs.get("offload_folder"),
            max_memory=kwargs.get("max_memory"),
        )
    if normalized_provider in {"openai", "openai_responses"}:
        from syntax_rl.models.openai_model import DEFAULT_OPENAI_MODEL, OpenAIResponsesScorer

        return OpenAIResponsesScorer(
            model_name=model_name or DEFAULT_OPENAI_MODEL,
            api_key=kwargs.get("api_key"),
            instructions=kwargs.get("instructions"),
            max_output_tokens=int(kwargs.get("max_output_tokens", 20)),
        )
    raise ValueError(
        f"Unsupported model provider '{provider}'. "
        "Supported providers: length_normalized, huggingface, openai."
    )
