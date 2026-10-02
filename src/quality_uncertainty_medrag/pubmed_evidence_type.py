"""Deterministic publication-type mapping into the existing evidence hierarchy.

Only explicit type labels are considered; article prose and model predictions
are never used. PubMed's broad Controlled Clinical Trial, Observational Study,
and Case Reports labels do not identify the more specific enum categories.
"""

from __future__ import annotations

from typing import Iterable

from .models import EvidenceType


# Ordered from highest to lowest in the existing Med-R²-inspired hierarchy.
# Explicit design aliases also support upstream metadata that names a design
# more precisely than PubMed's standard publication-type vocabulary does.
_HIERARCHY_TYPES = (
    (EvidenceType.META_ANALYSIS, frozenset({"meta-analysis", "network meta-analysis"})),
    (EvidenceType.SYSTEMATIC_REVIEW, frozenset({"systematic review"})),
    (EvidenceType.EVIDENCE_BASED_GUIDELINE, frozenset({
        "practice guideline", "guideline", "evidence-based guideline",
    })),
    (EvidenceType.RANDOMIZED_CONTROLLED_TRIAL, frozenset({"randomized controlled trial"})),
    (EvidenceType.NON_RANDOMIZED_CONTROLLED_TRIAL, frozenset({
        "non-randomized controlled trial", "nonrandomized controlled trial",
        "non-randomised controlled trial", "nonrandomised controlled trial",
    })),
    (EvidenceType.COHORT_STUDY, frozenset({
        "cohort study", "cohort studies", "prospective cohort study",
        "prospective cohort studies", "retrospective cohort study", "retrospective cohort studies",
    })),
    (EvidenceType.CASE_SERIES_OR_STUDY, frozenset({"case series", "case study", "case studies"})),
    (EvidenceType.INDIVIDUAL_CASE_REPORT, frozenset({"individual case report", "single case report"})),
    (EvidenceType.EXPERT_OPINION, frozenset({"editorial", "expert opinion"})),
)


def evidence_type_from_publication_types(publication_types: str | Iterable[str]) -> EvidenceType:
    """Return the highest category justified by an explicit metadata label.

    Accept one string or an iterable of strings. Matching ignores case and
    surrounding/repeated whitespace, but never uses substrings. Unsupported,
    ambiguous or empty labels yield OTHER. Invalid non-string values raise
    TypeError instead of being normalized into invented labels.

    Guideline tags use the project's existing guideline category; their
    metadata alone does not verify evidence appraisal or guideline methods.
    Controlled Clinical Trial alone does not prove nonrandomized allocation;
    Observational Study alone does not establish a cohort design; Case Reports
    alone does not establish an individual case or a case series. More precise
    explicit labels may qualify those categories. Other recognized tags in a
    mixed list still participate in the normal hierarchy precedence.
    """

    values = (publication_types,) if isinstance(publication_types, str) else publication_types
    normalized = set()
    for value in values:
        if not isinstance(value, str):
            raise TypeError("publication types must be strings")
        normalized.add(" ".join(value.casefold().split()))
    for evidence_type, aliases in _HIERARCHY_TYPES:
        if not normalized.isdisjoint(aliases):
            return evidence_type
    return EvidenceType.OTHER
