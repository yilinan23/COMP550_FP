"""Structural templates for controlled subject-verb agreement generation."""

from __future__ import annotations

from dataclasses import dataclass


SINGULAR = "singular"
PLURAL = "plural"
PHENOMENON = "agreement"


@dataclass(frozen=True)
class TemplateSpec:
    """Metadata for one syntactic generation template."""

    name: str
    grammatical_template: str
    ungrammatical_template: str
    phenomenon: str
    template_type: str = "simple_agreement"
    subtype: str = "simple_agreement"
    dependency_distance: int = 0
    attractor_count: int = 0
    clause_depth: int = 0
    subject_number: str = SINGULAR


@dataclass(frozen=True)
class GeneratedMinimalPair:
    """One generated agreement minimal pair in BLiMP-style format."""

    uid: str
    phenomenon: str
    sentence_good: str
    sentence_bad: str
    dependency_distance: int
    attractor_count: int
    clause_depth: int
    template_type: str
    subtype: str

    def to_record(self) -> dict[str, str | int]:
        """Return a JSONL/CSV-friendly record."""
        return {
            "uid": self.uid,
            "phenomenon": self.phenomenon,
            "sentence_good": self.sentence_good,
            "sentence_bad": self.sentence_bad,
            "dependency_distance": self.dependency_distance,
            "attractor_count": self.attractor_count,
            "clause_depth": self.clause_depth,
            "template_type": self.template_type,
            "subtype": self.subtype,
        }


def simple_agreement_templates() -> list[TemplateSpec]:
    """Return simple adjacent subject-verb agreement templates."""
    return [
        TemplateSpec(
            name="simple_singular",
            grammatical_template="The {subject_singular} {verb_singular}.",
            ungrammatical_template="The {subject_singular} {verb_plural}.",
            phenomenon=PHENOMENON,
            template_type="simple_agreement",
            subtype="simple_singular",
            dependency_distance=1,
            attractor_count=0,
            clause_depth=0,
            subject_number=SINGULAR,
        ),
        TemplateSpec(
            name="simple_plural",
            grammatical_template="The {subject_plural} {verb_plural}.",
            ungrammatical_template="The {subject_plural} {verb_singular}.",
            phenomenon=PHENOMENON,
            template_type="simple_agreement",
            subtype="simple_plural",
            dependency_distance=1,
            attractor_count=0,
            clause_depth=0,
            subject_number=PLURAL,
        ),
    ]


def agreement_template_families() -> list[TemplateSpec]:
    """Return grammar-controlled subject-verb agreement template families."""
    simple_templates = simple_agreement_templates()
    pp_templates = pp_attractor_templates()
    relative_templates = relative_clause_templates()
    ordered: list[TemplateSpec] = []
    max_length = max(len(simple_templates), len(pp_templates), len(relative_templates))
    for index in range(max_length):
        for templates in (simple_templates, pp_templates, relative_templates):
            if index < len(templates):
                ordered.append(templates[index])
    return ordered


def pp_attractor_templates() -> list[TemplateSpec]:
    """Return PP attractor variants with controlled prepositional heads."""
    templates: list[TemplateSpec] = []
    for preposition in ("near", "behind", "beside", "with"):
        templates.append(
            TemplateSpec(
                name=f"pp_attractor_{preposition}_singular",
                grammatical_template=f"The {{subject_singular}} {preposition} the {{attractor_plural}} {{verb_singular}}.",
                ungrammatical_template=f"The {{subject_singular}} {preposition} the {{attractor_plural}} {{verb_plural}}.",
                phenomenon=PHENOMENON,
                template_type="pp_attractor",
                subtype=f"pp_plural_attractor_{preposition}",
                dependency_distance=4,
                attractor_count=1,
                clause_depth=0,
                subject_number=SINGULAR,
            )
        )
        templates.append(
            TemplateSpec(
                name=f"pp_attractor_{preposition}_plural",
                grammatical_template=f"The {{subject_plural}} {preposition} the {{attractor_singular}} {{verb_plural}}.",
                ungrammatical_template=f"The {{subject_plural}} {preposition} the {{attractor_singular}} {{verb_singular}}.",
                phenomenon=PHENOMENON,
                template_type="pp_attractor",
                subtype=f"pp_singular_attractor_{preposition}",
                dependency_distance=4,
                attractor_count=1,
                clause_depth=0,
                subject_number=PLURAL,
            )
        )
    return templates


def relative_clause_templates() -> list[TemplateSpec]:
    """Return object-relative clause variants with sampled embedded verbs."""
    return [
        TemplateSpec(
            name="relative_clause_that_singular",
            grammatical_template="The {subject_singular} that the {object_singular} {embedded_verb_singular} {verb_singular}.",
            ungrammatical_template="The {subject_singular} that the {object_singular} {embedded_verb_singular} {verb_plural}.",
            phenomenon=PHENOMENON,
            template_type="relative_clause",
            subtype="object_relative_clause_that",
            dependency_distance=5,
            attractor_count=0,
            clause_depth=1,
            subject_number=SINGULAR,
        ),
        TemplateSpec(
            name="relative_clause_who_singular",
            grammatical_template="The {person_singular} who the {object_singular} {embedded_verb_singular} {verb_singular}.",
            ungrammatical_template="The {person_singular} who the {object_singular} {embedded_verb_singular} {verb_plural}.",
            phenomenon=PHENOMENON,
            template_type="relative_clause",
            subtype="object_relative_clause_who",
            dependency_distance=5,
            attractor_count=0,
            clause_depth=1,
            subject_number=SINGULAR,
        ),
        TemplateSpec(
            name="relative_clause_that_plural",
            grammatical_template="The {subject_plural} that the {object_singular} {embedded_verb_singular} {verb_plural}.",
            ungrammatical_template="The {subject_plural} that the {object_singular} {embedded_verb_singular} {verb_singular}.",
            phenomenon=PHENOMENON,
            template_type="relative_clause",
            subtype="plural_object_relative_clause_that",
            dependency_distance=5,
            attractor_count=0,
            clause_depth=1,
            subject_number=PLURAL,
        ),
        TemplateSpec(
            name="relative_clause_singular_plural_embedded_that",
            grammatical_template="The {subject_singular} that the {object_plural} {embedded_verb_plural} {verb_singular}.",
            ungrammatical_template="The {subject_singular} that the {object_plural} {embedded_verb_plural} {verb_plural}.",
            phenomenon=PHENOMENON,
            template_type="relative_clause",
            subtype="object_relative_clause_plural_embedded_that",
            dependency_distance=5,
            attractor_count=1,
            clause_depth=1,
            subject_number=SINGULAR,
        ),
        TemplateSpec(
            name="relative_clause_who_plural_embedded",
            grammatical_template="The {person_singular} who the {object_plural} {embedded_verb_plural} {verb_singular}.",
            ungrammatical_template="The {person_singular} who the {object_plural} {embedded_verb_plural} {verb_plural}.",
            phenomenon=PHENOMENON,
            template_type="relative_clause",
            subtype="object_relative_clause_plural_embedded_who",
            dependency_distance=5,
            attractor_count=1,
            clause_depth=1,
            subject_number=SINGULAR,
        ),
    ]


def select_template_families(template_types: list[str] | None = None) -> list[TemplateSpec]:
    """Select template families by ``template_type``."""
    families = agreement_template_families()
    if not template_types:
        return families
    requested = set(template_types)
    return [template for template in families if template.template_type in requested]
