"""Deterministic, question-text-only clinical concept queries for PubMed.

This deliberately small phrase vocabulary is inspectable and extensible. It
normalizes stated clinical findings, never infers a diagnosis or reads options.
Unrecognized prose is omitted rather than turned into a token-OR fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class _Rule:
    term: str
    pattern: str
    priority: int
    family: str
    characteristic: bool = False


# Priorities select search clues, not evidence weights or research scores.
# A family keeps overlapping synonyms/variants from consuming several slots.
_RULES = (
    _Rule("maltose", r"maltose", 100, "sugar-fermentation", True),
    _Rule("polysaccharide capsule", r"polysaccharide capsules?", 80, "capsule", True),
    _Rule("gram negative", r"gram[ -]negative", 90, "gram-stain", True),
    _Rule("gram positive", r"gram[ -]positive", 90, "gram-stain", True),
    _Rule("acid fast", r"acid[ -]fast", 90, "staining", True),
    _Rule("coagulase", r"coagulase", 90, "coagulase", True),
    _Rule("oxidase", r"oxidase", 90, "oxidase", True),
    _Rule("catalase", r"catalase", 90, "catalase", True),
    _Rule("intracellular bacteria", r"intracellular bacteria", 90, "intracellular", True),
    _Rule("joint fluid", r"(?:joint|synovial) fluid", 90, "joint-finding"),
    _Rule("cell wall synthesis", r"cell[ -]wall synthesis", 85, "drug-mechanism", True),
    _Rule("protein synthesis", r"protein synthesis", 85, "drug-mechanism", True),
    _Rule("DNA gyrase", r"dna gyrase", 85, "drug-mechanism", True),
    _Rule("dihydrofolate reductase", r"dihydrofolate reductase", 85, "drug-mechanism", True),
    _Rule("angiotensin converting enzyme", r"angiotensin[ -]converting enzyme", 85, "drug-mechanism", True),
    _Rule("beta adrenergic receptor", r"beta[ -]adrenergic[ -]receptors?", 85, "drug-mechanism", True),
    _Rule("ST elevation", r"st[ -](?:segment[ -])?elevation", 95, "ecg"),
    _Rule("hyperkalemia", r"hyperkal(?:emia|aemia)", 90, "potassium"),
    _Rule("hypokalemia", r"hypokal(?:emia|aemia)", 90, "potassium"),
    _Rule("hyponatremia", r"hyponatr(?:emia|aemia)", 90, "sodium"),
    _Rule("hypernatremia", r"hypernatr(?:emia|aemia)", 90, "sodium"),
    _Rule("hypercalcemia", r"hypercalc(?:emia|aemia)", 90, "calcium"),
    _Rule("hypocalcemia", r"hypocalc(?:emia|aemia)", 90, "calcium"),
    _Rule("metabolic acidosis", r"metabolic acidosis", 90, "acid-base"),
    _Rule("metabolic alkalosis", r"metabolic alkalosis", 90, "acid-base"),
    _Rule("respiratory acidosis", r"respiratory acidosis", 90, "acid-base"),
    _Rule("respiratory alkalosis", r"respiratory alkalosis", 90, "acid-base"),
    _Rule("neutropenia", r"neutropenia", 90, "neutrophils"),
    _Rule("leukocytosis", r"leukocytosis", 85, "leukocytes"),
    _Rule("thrombocytopenia", r"thrombocytopenia", 90, "platelets"),
    _Rule("eosinophilia", r"eosinophilia", 90, "eosinophils"),
    _Rule("hematuria", r"hematuria|blood in (?:the )?urine", 85, "urinary-blood"),
    _Rule("proteinuria", r"proteinuria|protein in (?:the )?urine", 85, "urinary-protein"),
    _Rule("elevated troponin", r"(?:elevated|increased|high) troponin", 95, "troponin"),
    _Rule("elevated creatinine", r"(?:elevated|increased|high) (?:serum )?creatinine", 85, "creatinine"),
    _Rule("low hemoglobin", r"(?:low|decreased|reduced) (?:serum )?hemoglobin", 85, "hemoglobin"),
    _Rule("diabetes mellitus", r"diabetes mellitus", 85, "diabetes"),
    _Rule("diabetes insipidus", r"diabetes insipidus", 85, "diabetes"),
    _Rule("diabetes", r"diabetes", 80, "diabetes"),
    _Rule("chronic kidney disease", r"chronic (?:kidney|renal) disease", 85, "kidney-disease"),
    _Rule("acute kidney injury", r"acute (?:kidney injury|renal failure)", 85, "kidney-disease"),
    _Rule("migraine", r"migraine", 85, "headache"),
    _Rule("hypertension", r"hypertension", 85, "hypertension"),
    _Rule("myocardial infarction", r"myocardial infarction", 85, "cardiac-disease"),
    _Rule("heart failure", r"(?:congestive )?heart failure", 85, "cardiac-failure"),
    _Rule("rheumatoid arthritis", r"rheumatoid arthritis", 85, "joint-finding"),
    _Rule("systemic lupus erythematosus", r"systemic lupus erythematosus", 85, "lupus"),
    _Rule("lupus", r"lupus", 80, "lupus"),
    _Rule("tuberculosis", r"tuberculosis", 85, "tuberculosis"),
    _Rule("pneumonia", r"pneumonia", 85, "pneumonia"),
    _Rule("meningitis", r"meningitis", 85, "meningitis"),
    _Rule("asthma", r"asthma", 85, "asthma"),
    _Rule("chronic obstructive pulmonary disease", r"chronic obstructive pulmonary disease|copd", 85, "copd"),
    _Rule("pancreatitis", r"pancreatitis", 85, "pancreatitis"),
    _Rule("hepatitis", r"hepatitis", 85, "hepatitis"),
    _Rule("HIV", r"hiv|human immunodeficiency virus", 85, "hiv"),
    _Rule("sickle cell", r"sickle[ -]cell", 85, "sickle-cell"),
    _Rule("bilious vomiting", r"bilious vomiting|vomiting (?:of )?bile", 90, "vomiting"),
    _Rule("recurrent vomiting", r"(?:recurrent|repeated) vomiting|(?:multiple|recurrent|similar) episodes of (?:nausea and )?vomiting", 85, "vomiting"),
    _Rule("hematemesis", r"hematemesis|vomiting blood", 90, "vomiting"),
    _Rule("dysuria", r"dysuria|pain (?:during|with) urination|painful urination", 80, "urination"),
    _Rule("arthritis", r"arthritis|inflammation (?:and pain )?in (?:the )?(?:(?:right|left) )?(?:knee|joint)", 80, "joint-finding"),
    _Rule("abdominal pain", r"abdominal pain|pain in (?:the )?abdomen", 80, "abdominal-pain"),
    _Rule("chest pain", r"chest pain|pain in (?:the )?chest", 80, "chest-pain"),
    _Rule("flank pain", r"flank pain", 80, "flank-pain"),
    _Rule("dehydration", r"dehydration|dry mucous membranes", 75, "dehydration"),
    _Rule("insomnia", r"insomnia|difficulty (?:falling|staying) asleep|unable to fall (?:back )?asleep", 80, "sleep"),
    _Rule("appetite loss", r"(?:diminished|decreased|reduced|poor) appetite|loss of appetite|appetite loss|anorexia", 75, "appetite"),
    _Rule("hopelessness", r"hopeless(?:ness)?", 70, "hopelessness"),
    _Rule("fatigue", r"fatigue|tiredness", 55, "fatigue"),
    _Rule("weight loss", r"weight loss|lost \d+(?:\.\d+)? (?:kg|kilograms?|lb|pounds?)", 65, "weight"),
    _Rule("dyspnea", r"dyspnea|shortness of breath|difficulty breathing", 75, "breathing"),
    _Rule("hemoptysis", r"hemoptysis|coughing (?:up )?blood", 85, "hemoptysis"),
    _Rule("jaundice", r"jaundice|yellow(?:ing)? (?:of (?:the )?)?(?:skin|sclerae?)", 80, "jaundice"),
    _Rule("night sweats", r"night sweats", 75, "sweats"),
    _Rule("lymphadenopathy", r"lymphadenopathy|enlarged lymph nodes", 80, "lymph-nodes"),
    _Rule("petechiae", r"petechiae|petechial rash", 85, "rash"),
    _Rule("seizures", r"seizures?", 80, "seizure"),
    _Rule("syncope", r"syncope|fainting", 75, "syncope"),
    _Rule("diplopia", r"diplopia|double vision", 80, "vision"),
    _Rule("dysphagia", r"dysphagia|difficulty swallowing", 80, "swallowing"),
    _Rule("amenorrhea", r"amenorrhea", 80, "menstruation"),
    _Rule("pregnancy", r"pregnan(?:t|cy)", 65, "pregnancy"),
    _Rule("neonatal", r"neonat(?:al|e)|newborn", 65, "neonatal"),
    _Rule("immunosuppression", r"immunosuppress(?:ion|ed)|immunocompromised", 80, "immune-status"),
)


def _negated(text: str, start: int) -> bool:
    """Omit explicitly absent findings; no clinical NOT operators are emitted."""

    preceding_clause = re.split(
        r"[.;!?]|\bbut\b|\band (?:has|reports|develops|experiences)\b", text[:start]
    )[-1]
    recent_words = " ".join(preceding_clause.split()[-6:])
    return bool(re.search(r"\b(?:no|not|denies|denied|without|negative for)\b", recent_words))


def extract_medical_concepts(question_text: str, max_terms: int = 3) -> tuple[str, ...]:
    """Select at most eight stated concepts, ordered by priority then occurrence.

    Only question prose is accepted. Generic prose, nonspecific pain, ordinary
    ages and normal vital signs have no rules. Sparse stems may yield fewer
    than three concepts; unsupported stems fail rather than broaden silently.
    """

    if not isinstance(question_text, str) or not question_text.strip():
        raise ValueError("question_text must be a nonempty string")
    if isinstance(max_terms, bool) or not isinstance(max_terms, int) or max_terms < 1:
        raise ValueError("max_terms must be a positive integer")
    text = question_text.casefold().replace("–", "-").replace("—", "-")
    matches = []
    for index, rule in enumerate(_RULES):
        for match in re.finditer(r"\b(?:" + rule.pattern + r")\b", text):
            if rule.characteristic or not _negated(text, match.start()):
                matches.append((-rule.priority, match.start(), index, rule))
                break
    selected = []
    families = set()
    for _, _, _, rule in sorted(matches, key=lambda item: item[:3]):
        if rule.family not in families:
            families.add(rule.family)
            selected.append(rule.term)
            if len(selected) == min(max_terms, 8):
                break
    if not selected:
        raise ValueError("Question text contains no recognized medical concepts")
    return tuple(selected)


def build_pubmed_query(question_text: str, max_terms: int = 3) -> str:
    """Combine a few clinical concepts using AND and PubMed term mapping.

    Multiword concepts are parenthesized, without quotes or field tags, so
    PubMed can perform Automatic Term Mapping. Negated organism/mechanism
    clues are neutral topic searches, not positive assertions or diagnoses.
    """

    concepts = extract_medical_concepts(question_text, max_terms=max_terms)
    clinical_query = " AND ".join(f"({term})" for term in concepts)
    return f'(({clinical_query}) NOT "pubmed books"[sb]) AND hasabstract'
