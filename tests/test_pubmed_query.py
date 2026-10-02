"""Deterministic medical concept selection without models or live network access."""

from __future__ import annotations

import pytest

from quality_uncertainty_medrag.pubmed import PubMedRetriever
from quality_uncertainty_medrag.pubmed_query import build_pubmed_query, extract_medical_concepts


GENERIC_WRAPPER = (
    "The patient and mother say she was well several weeks ago. The physician asks about "
    "the time and says the patient has pain. The examination was otherwise normal. "
    "Her mother says they arrived at the hospital some time ago. "
    "A physician reviews the medical history and asks which of the following is most likely. "
)


def test_generic_words_and_nonspecific_pain_are_not_query_concepts():
    concepts = extract_medical_concepts(
        GENERIC_WRAPPER + "She now has bilious vomiting, abdominal pain, and dehydration."
    )
    joined = " ".join(concepts)
    for word in ("patient", "mother", "says", "well", "several", "time", "ago", "physician"):
        assert word not in joined.split()
    assert "pain" not in concepts
    assert "abdominal pain" in concepts
    assert "bilious vomiting" in concepts
    assert "dehydration" in concepts


def test_query_is_substantially_shorter_and_allows_automatic_term_mapping():
    question = GENERIC_WRAPPER * 2 + "She has bilious vomiting, abdominal pain, and dehydration."
    query = build_pubmed_query(question)
    assert len(query) < len(question) / 3
    assert " OR " not in query
    assert "[Title/Abstract]" not in query
    assert query.endswith("AND hasabstract")
    assert 'NOT "pubmed books"[sb]' in query
    assert "AND NOT" not in query


def test_generic_context_does_not_change_the_clinical_query():
    clinical = "Bilious vomiting, abdominal pain, and dehydration."
    assert build_pubmed_query(GENERIC_WRAPPER + clinical) == build_pubmed_query(clinical)


@pytest.mark.parametrize(
    "text",
    [
        "A joint fluid culture shows bacteria that does not ferment maltose. "
        "The drug blocks cell wall synthesis.",
        "A child has bilious vomiting, abdominal pain, and dehydration.",
        "A woman has difficulty falling asleep, diminished appetite, and tiredness.",
        "Diabetes mellitus and chronic kidney disease treatment?",
    ],
)
def test_generation_and_concept_order_are_deterministic(text):
    concepts = extract_medical_concepts(text)
    query = build_pubmed_query(text)
    assert extract_medical_concepts(text) == concepts
    assert build_pubmed_query(text) == query
    assert 1 <= len(concepts) <= 8
    assert len(set(concepts)) == len(concepts)
    assert all(concept in query for concept in concepts)


def test_selected_concepts_respect_requested_limit():
    text = "Diabetes mellitus, chronic kidney disease, bilious vomiting, abdominal pain, and dehydration."
    concepts = extract_medical_concepts(text, max_terms=2)
    assert len(concepts) == 2
    assert build_pubmed_query(text, max_terms=2).count(" AND ") == 2


def test_requested_limit_cannot_create_a_giant_query():
    text = (
        "Diabetes mellitus, chronic kidney disease, bilious vomiting, abdominal pain, "
        "dehydration, insomnia, hematuria, thrombocytopenia, seizures, dyspnea, "
        "night sweats, and jaundice."
    )
    assert len(extract_medical_concepts(text, max_terms=100)) == 8


def test_explicitly_absent_findings_are_omitted():
    concepts = extract_medical_concepts(
        "No hematuria. She denies dyspnea. She has bilious vomiting and abdominal pain."
    )
    assert "hematuria" not in concepts
    assert "dyspnea" not in concepts
    assert "bilious vomiting" in concepts
    assert "abdominal pain" in concepts


def test_organism_clues_do_not_infer_a_species_or_drug():
    question = (
        "Pain during urination with joint fluid culture showing bacteria that does not "
        "ferment maltose and has no polysaccharide capsule. The drug blocks cell wall synthesis."
    )
    query = build_pubmed_query(question)
    assert "maltose" in query
    assert "neisseria" not in query.casefold()
    assert "ceftriaxone" not in query.casefold()
    assert 'NOT (maltose)' not in query


def test_overlapping_finding_variants_do_not_consume_multiple_slots():
    concepts = extract_medical_concepts(
        "Multiple episodes of nausea and vomiting, bilious vomiting, abdominal pain, and dehydration."
    )
    assert concepts == ("bilious vomiting", "abdominal pain", "dehydration")


def test_distinctive_demographics_are_retained_without_ordinary_age_or_gender():
    concepts = extract_medical_concepts("A 30-year-old pregnant woman has hematuria and fatigue.")
    assert "pregnancy" in concepts
    assert "woman" not in concepts
    assert "30" not in concepts


def test_negation_does_not_leak_into_a_new_positive_predicate():
    concepts = extract_medical_concepts("She has no fever and has abdominal pain and dehydration.")
    assert "abdominal pain" in concepts
    assert "dehydration" in concepts


def test_independent_organism_characteristics_are_not_synonym_deduplicated():
    concepts = extract_medical_concepts(
        "The bacteria do not ferment maltose and have no polysaccharide capsule.", max_terms=8
    )
    assert "maltose" in concepts
    assert "polysaccharide capsule" in concepts


def test_clinical_normalization_does_not_add_unstated_qualifiers():
    assert extract_medical_concepts("Lupus.") == ("lupus",)
    assert extract_medical_concepts("Diabetes insipidus.") == ("diabetes insipidus",)
    concepts = extract_medical_concepts("The drug binds beta receptors. She has abdominal pain.")
    assert "beta adrenergic receptor" not in concepts


@pytest.mark.parametrize("max_terms", [0, -1, True, 2.5, "4"])
def test_term_limit_requires_positive_integer(max_terms):
    with pytest.raises(ValueError):
        build_pubmed_query("Diabetes mellitus", max_terms=max_terms)


def test_generic_only_text_does_not_fall_back_to_a_broad_query():
    with pytest.raises(ValueError):
        build_pubmed_query(GENERIC_WRAPPER)


class QuestionTextOnly:
    """Fail immediately if retrieval tries to read any option or gold information."""

    question = "Diabetes mellitus and chronic kidney disease treatment?"
    question_id = "question-text-only"

    def __getattribute__(self, name):
        if name in {"question", "question_id"}:
            return object.__getattribute__(self, name)
        raise AssertionError(f"Query generation accessed forbidden question field: {name}")


class EmptySearchClient:
    def __init__(self):
        self.search_calls = []

    def search(self, query, *, top_k):
        self.search_calls.append((query, top_k))
        return {"idlist": [], "count": "0"}

    def fetch(self, pmids):
        assert pmids == []
        return {}


def test_retriever_never_reads_options_or_gold_fields():
    question = QuestionTextOnly()
    client = EmptySearchClient()
    retriever = PubMedRetriever(client)
    assert retriever.retrieve(question, top_k=15) == ()
    expected_query = build_pubmed_query(question.question)
    assert client.search_calls == [(expected_query, 15)]
    assert retriever.reports[0]["query"] == expected_query
