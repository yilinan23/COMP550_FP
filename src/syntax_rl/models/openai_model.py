"""OpenAI Responses API scorer for grammaticality judgments.

Unlike the Hugging Face causal-LM scorer, the Responses API does not expose
input-sentence log probabilities. This scorer therefore uses a lightweight
judgment prompt: for each sentence, the model returns a grammaticality score in
``[0, 1]``. Higher scores mean the sentence is judged more grammatical. The
class still implements the shared ``SentenceScorer`` interface so the existing
minimal-pair evaluation code can compare ``sentence_good`` and ``sentence_bad``
without knowing which backend produced the score.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any


DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_INSTRUCTIONS = (
    "You are evaluating English grammaticality for subject-verb agreement. "
    "Return only JSON with a numeric key named score. The score must be between "
    "0 and 1, where 1 means fully grammatical and 0 means clearly ungrammatical."
)


@dataclass(frozen=True)
class OpenAIScoringConfig:
    """Configuration for OpenAI Responses API grammaticality scoring."""

    model_name: str = DEFAULT_OPENAI_MODEL
    api_key: str | None = None
    instructions: str = DEFAULT_INSTRUCTIONS
    max_output_tokens: int = 20


class OpenAIResponsesScorer:
    """Score sentences using OpenAI's Responses API as a grammaticality judge."""

    def __init__(
        self,
        model_name: str = DEFAULT_OPENAI_MODEL,
        api_key: str | None = None,
        instructions: str | None = None,
        max_output_tokens: int = 20,
        client: Any | None = None,
    ) -> None:
        """Create an API-backed scorer.

        The OpenAI client is imported lazily so local dry-run and Hugging Face
        evaluation do not require the optional ``openai`` package.
        """
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as error:
                raise ImportError(
                    "OpenAI scoring requires the optional OpenAI SDK. "
                    "Install it with: py -m pip install -r requirements-openai.txt"
                ) from error
            client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

        self.client = client
        self.model_name = model_name
        self.instructions = instructions or DEFAULT_INSTRUCTIONS
        self.max_output_tokens = max_output_tokens

    @property
    def name(self) -> str:
        """Return a human-readable scorer name."""
        return f"openai:{self.model_name}"

    def score(self, sentence: str) -> float:
        """Return the model's grammaticality score for one sentence."""
        if not sentence.strip():
            return 0.0

        response = self.client.responses.create(
            model=self.model_name,
            instructions=self.instructions,
            input=f"Sentence: {sentence}\nReturn JSON only.",
            max_output_tokens=self.max_output_tokens,
        )
        return self._parse_score(_response_text(response))

    @staticmethod
    def _parse_score(text: str) -> float:
        """Parse and clamp a score from a Responses API text output."""
        stripped = text.strip()
        try:
            payload = json.loads(stripped)
            value = payload["score"]
        except (json.JSONDecodeError, KeyError, TypeError):
            match = re.search(r"-?\d+(?:\.\d+)?", stripped)
            if match is None:
                raise ValueError(f"OpenAI scorer response did not contain a numeric score: {text!r}")
            value = match.group(0)
        return max(0.0, min(1.0, float(value)))


def _response_text(response: Any) -> str:
    """Extract output text from the OpenAI Python SDK response object."""
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)

    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(str(text))
    return "\n".join(chunks)
