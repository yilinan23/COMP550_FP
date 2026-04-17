"""Surface realization helpers for controlled agreement templates."""

from __future__ import annotations

import random

from .templates import GeneratedMinimalPair, TemplateSpec


DEFAULT_LEXICON: dict[str, list[str]] = {
    "subject_singular": [
        "dog",
        "boy",
        "teacher",
        "author",
        "pilot",
        "chef",
        "scientist",
        "musician",
        "student",
        "reporter",
        "robot",
        "bird",
        "judge",
        "nurse",
        "driver",
        "writer",
        "dancer",
        "guard",
        "actor",
        "manager",
    ],
    "subject_plural": [
        "dogs",
        "boys",
        "teachers",
        "authors",
        "pilots",
        "chefs",
        "scientists",
        "musicians",
        "students",
        "reporters",
        "robots",
        "birds",
        "judges",
        "nurses",
        "drivers",
        "writers",
        "dancers",
        "guards",
        "actors",
        "managers",
    ],
    "person_singular": [
        "boy",
        "teacher",
        "author",
        "pilot",
        "chef",
        "scientist",
        "musician",
        "student",
        "reporter",
        "judge",
        "nurse",
        "driver",
        "writer",
        "dancer",
        "guard",
        "actor",
        "manager",
    ],
    "attractor_singular": [
        "car",
        "cabinet",
        "book",
        "student",
        "key",
        "report",
        "painting",
        "computer",
        "garden",
        "instrument",
        "map",
        "letter",
    ],
    "attractor_plural": [
        "cars",
        "cabinets",
        "books",
        "students",
        "keys",
        "reports",
        "paintings",
        "computers",
        "gardens",
        "instruments",
        "maps",
        "letters",
    ],
    "object_singular": [
        "girl",
        "doctor",
        "artist",
        "farmer",
        "neighbor",
        "coach",
        "editor",
        "mechanic",
        "librarian",
        "visitor",
        "lawyer",
        "singer",
        "clerk",
        "broker",
    ],
    "object_plural": [
        "girls",
        "doctors",
        "artists",
        "farmers",
        "neighbors",
        "coaches",
        "editors",
        "mechanics",
        "librarians",
        "visitors",
        "lawyers",
        "singers",
        "clerks",
        "brokers",
    ],
    "embedded_verb_singular": [
        "likes",
        "knows",
        "helps",
        "sees",
        "calls",
        "follows",
        "admires",
        "teaches",
        "remembers",
        "guides",
        "questions",
        "supports",
        "observes",
        "trusts",
    ],
    "embedded_verb_plural": [
        "like",
        "know",
        "help",
        "see",
        "call",
        "follow",
        "admire",
        "teach",
        "remember",
        "guide",
        "question",
        "support",
        "observe",
        "trust",
    ],
}
VERB_PAIRS: list[tuple[str, str]] = [
    ("runs", "run"),
    ("smiles", "smile"),
    ("arrives", "arrive"),
    ("waits", "wait"),
    ("sleeps", "sleep"),
    ("laughs", "laugh"),
    ("speaks", "speak"),
    ("listens", "listen"),
    ("works", "work"),
    ("dances", "dance"),
    ("travels", "travel"),
    ("returns", "return"),
    ("studies", "study"),
    ("notices", "notice"),
    ("answers", "answer"),
    ("opens", "open"),
    ("closes", "close"),
    ("moves", "move"),
    ("reads", "read"),
    ("writes", "write"),
]


def realize_template(template: str, values: dict[str, str]) -> str:
    """Fill a simple Python format-string template."""
    return template.format(**values)


def sample_lexical_values(rng: random.Random, lexicon: dict[str, list[str]] | None = None) -> dict[str, str]:
    """Sample one lexical value for each grammar slot."""
    choices = lexicon or DEFAULT_LEXICON
    sampled = {slot: rng.choice(values) for slot, values in choices.items()}
    verb_singular, verb_plural = rng.choice(VERB_PAIRS)
    sampled["verb_singular"] = verb_singular
    sampled["verb_plural"] = verb_plural
    return sampled


def realize_minimal_pair(template: TemplateSpec, values: dict[str, str], uid: str) -> GeneratedMinimalPair:
    """Realize grammatical and ungrammatical sentences from one template."""
    return GeneratedMinimalPair(
        uid=uid,
        phenomenon=template.phenomenon,
        sentence_good=realize_template(template.grammatical_template, values),
        sentence_bad=realize_template(template.ungrammatical_template, values),
        dependency_distance=template.dependency_distance,
        attractor_count=template.attractor_count,
        clause_depth=template.clause_depth,
        template_type=template.template_type,
        subtype=template.subtype,
    )
