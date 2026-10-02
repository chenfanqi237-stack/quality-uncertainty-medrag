"""Deterministic publication-type mapping and offline retrieval integration."""

from itertools import combinations

import pytest

from quality_uncertainty_medrag.models import EvidenceType, MedicalQuestion
from quality_uncertainty_medrag.pubmed import PubMedRetriever, evidence_to_json_record
from quality_uncertainty_medrag.pubmed_evidence_type import (
    evidence_type_from_publication_types,
)
from quality_uncertainty_medrag.pubmed_records import (
    evidence_type_from_publication_types as legacy_mapper,
)
from quality_uncertainty_medrag.quality import ConfiguredEvidenceTypeScorer


KNOWN_TYPES = [
    ("Meta-Analysis", EvidenceType.META_ANALYSIS),
    ("Network Meta-Analysis", EvidenceType.META_ANALYSIS),
    ("Systematic Review", EvidenceType.SYSTEMATIC_REVIEW),
    ("Practice Guideline", EvidenceType.EVIDENCE_BASED_GUIDELINE),
    ("Guideline", EvidenceType.EVIDENCE_BASED_GUIDELINE),
    ("Randomized Controlled Trial", EvidenceType.RANDOMIZED_CONTROLLED_TRIAL),
    ("Non-Randomized Controlled Trial", EvidenceType.NON_RANDOMIZED_CONTROLLED_TRIAL),
    ("Nonrandomized Controlled Trial", EvidenceType.NON_RANDOMIZED_CONTROLLED_TRIAL),
    ("Non-Randomised Controlled Trial", EvidenceType.NON_RANDOMIZED_CONTROLLED_TRIAL),
    ("Nonrandomised Controlled Trial", EvidenceType.NON_RANDOMIZED_CONTROLLED_TRIAL),
    ("Cohort Study", EvidenceType.COHORT_STUDY),
    ("Cohort Studies", EvidenceType.COHORT_STUDY),
    ("Prospective Cohort Study", EvidenceType.COHORT_STUDY),
    ("Retrospective Cohort Study", EvidenceType.COHORT_STUDY),
    ("Case Series", EvidenceType.CASE_SERIES_OR_STUDY),
    ("Case Study", EvidenceType.CASE_SERIES_OR_STUDY),
    ("Individual Case Report", EvidenceType.INDIVIDUAL_CASE_REPORT),
    ("Single Case Report", EvidenceType.INDIVIDUAL_CASE_REPORT),
    ("Editorial", EvidenceType.EXPERT_OPINION),
    ("Expert Opinion", EvidenceType.EXPERT_OPINION),
]


@pytest.mark.parametrize("publication_type, expected", KNOWN_TYPES)
def test_exact_known_metadata_mapping(publication_type, expected):
    assert evidence_type_from_publication_types([publication_type]) is expected


@pytest.mark.parametrize("publication_type, expected", KNOWN_TYPES)
def test_case_and_whitespace_normalization(publication_type, expected):
    normalized_variant = " \t" + "   ".join(publication_type.swapcase().split()) + "\n"
    assert evidence_type_from_publication_types([normalized_variant]) is expected


@pytest.mark.parametrize("publication_type", [
    "Journal Article", "Review", "Clinical Trial", "Observational Study",
    "Prospective Study", "Retrospective Study", "Comparative Study", "Letter",
    "Comment", "Controlled Clinical Trial", "Case Reports",
    "Unknown Publication Type", "Meta-Analysis as Topic",
    "Systematic Review Protocol", "Randomized Controlled Trials as Topic",
    "Cohort Studies as Topic", "Editorial Board", "", "   ",
])
def test_ambiguous_unsupported_and_substring_matches_are_other(publication_type):
    assert evidence_type_from_publication_types([publication_type]) is EvidenceType.OTHER


def test_empty_publication_types_are_other():
    assert evidence_type_from_publication_types([]) is EvidenceType.OTHER
    assert evidence_type_from_publication_types(()) is EvidenceType.OTHER


def test_single_string_is_one_publication_type():
    assert evidence_type_from_publication_types("Systematic Review") is EvidenceType.SYSTEMATIC_REVIEW
    assert evidence_type_from_publication_types("Journal Article") is EvidenceType.OTHER
    assert evidence_type_from_publication_types("") is EvidenceType.OTHER
    assert evidence_type_from_publication_types("  \t\n") is EvidenceType.OTHER


@pytest.mark.parametrize("invalid", [None, 42, 1.0, True, [None], [42], [False], [object()]])
def test_non_string_inputs_are_rejected(invalid):
    with pytest.raises(TypeError):
        evidence_type_from_publication_types(invalid)


@pytest.mark.parametrize("publication_types, expected", [
    (["Controlled Clinical Trial", "Non-Randomized Controlled Trial"],
     EvidenceType.NON_RANDOMIZED_CONTROLLED_TRIAL),
    (["Controlled Clinical Trial", "Randomized Controlled Trial"],
     EvidenceType.RANDOMIZED_CONTROLLED_TRIAL),
    (["Controlled Clinical Trial", "Meta-Analysis"], EvidenceType.META_ANALYSIS),
    (["Case Reports", "Case Series"], EvidenceType.CASE_SERIES_OR_STUDY),
    (["Case Reports", "Individual Case Report"], EvidenceType.INDIVIDUAL_CASE_REPORT),
    (["Case Reports", "Systematic Review"], EvidenceType.SYSTEMATIC_REVIEW),
    (["Case Reports", "Controlled Clinical Trial"], EvidenceType.OTHER),
])
def test_ambiguous_tags_do_not_override_explicit_designs(publication_types, expected):
    assert evidence_type_from_publication_types(publication_types) is expected
    assert evidence_type_from_publication_types(reversed(publication_types)) is expected


def test_tuple_generator_duplicates_and_unknown_types_are_supported():
    publication_types = ("Review", "Guideline", "Guideline", "Journal Article")
    assert evidence_type_from_publication_types(publication_types) is EvidenceType.EVIDENCE_BASED_GUIDELINE
    assert evidence_type_from_publication_types(iter(publication_types)) is EvidenceType.EVIDENCE_BASED_GUIDELINE
    assert publication_types == ("Review", "Guideline", "Guideline", "Journal Article")


# Listed in the existing evidence hierarchy order; explicit aliases only.
HIERARCHY_TYPES = [
    ("Meta-Analysis", EvidenceType.META_ANALYSIS),
    ("Systematic Review", EvidenceType.SYSTEMATIC_REVIEW),
    ("Guideline", EvidenceType.EVIDENCE_BASED_GUIDELINE),
    ("Randomized Controlled Trial", EvidenceType.RANDOMIZED_CONTROLLED_TRIAL),
    ("Non-Randomized Controlled Trial", EvidenceType.NON_RANDOMIZED_CONTROLLED_TRIAL),
    ("Cohort Study", EvidenceType.COHORT_STUDY),
    ("Case Series", EvidenceType.CASE_SERIES_OR_STUDY),
    ("Individual Case Report", EvidenceType.INDIVIDUAL_CASE_REPORT),
    ("Editorial", EvidenceType.EXPERT_OPINION),
]


@pytest.mark.parametrize("higher, lower", list(combinations(HIERARCHY_TYPES, 2)))
def test_highest_applicable_category_wins_independently_of_input_order(higher, lower):
    higher_name, expected = higher
    lower_name, _ = lower
    assert evidence_type_from_publication_types([higher_name, lower_name]) is expected
    assert evidence_type_from_publication_types([lower_name, higher_name]) is expected


def test_all_known_categories_use_the_same_precedence():
    names = [name for name, _ in HIERARCHY_TYPES]
    for offset in range(len(names)):
        reordered = names[offset:] + names[:offset]
        assert evidence_type_from_publication_types(reordered) is EvidenceType.META_ANALYSIS
    assert evidence_type_from_publication_types(reversed(names)) is EvidenceType.META_ANALYSIS


def test_existing_pubmed_records_import_reuses_new_mapper():
    assert legacy_mapper is evidence_type_from_publication_types


def test_retriever_maps_metadata_for_existing_quality_scorer_without_network():
    question = MedicalQuestion(
        question_id="q-1", question="Diabetes mellitus treatment?",
        option_labels=("A", "B"), options=("First", "Second"), answer_index=0,
    )
    query = "diabetes mellitus"
    publication_types = ["Journal Article", "Guideline", "Editorial"]

    class MetadataOnlyClient:
        def search(self, received_query, *, top_k):
            assert received_query == query
            assert top_k == 1
            return {"idlist": ["123"], "count": "1"}

        def fetch(self, pmids):
            assert pmids == ["123"]
            return {"123": {
                "title": "Diabetes management guidance",
                "abstract": "A guidance abstract.",
                "journal": "Example Journal",
                "publication_date": "2025",
                "publication_types": publication_types,
                "pubmed_metadata": {"pmid": "123"},
            }}

    evidence, = PubMedRetriever(
        MetadataOnlyClient(), query_builder=lambda text: query,
    ).retrieve(question, top_k=1)
    assert evidence.evidence_type is EvidenceType.EVIDENCE_BASED_GUIDELINE
    assert evidence.metadata["publication_types"] == tuple(publication_types)
    assert evidence.metadata["query"] == query
    assert not evidence.annotated_stances
    assert evidence_to_json_record(evidence)["evidence_type"] == "evidence_based_guideline"

    scores = {kind.value: 0.0 for kind in EvidenceType}
    scores[EvidenceType.EVIDENCE_BASED_GUIDELINE.value] = 7 / 9
    scorer = ConfiguredEvidenceTypeScorer(scores, scorer_name="existing-hierarchy")
    quality = scorer.score(question, evidence)
    assert quality.value == pytest.approx(7 / 9)
    assert quality.components["hierarchy"] == pytest.approx(7 / 9)
