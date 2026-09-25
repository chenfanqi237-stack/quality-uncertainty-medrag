"""Shared typed records for questions, evidence, and baseline outputs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


class EvidenceType(str, Enum):
    META_ANALYSIS = "meta_analysis"
    SYSTEMATIC_REVIEW = "systematic_review"
    EVIDENCE_BASED_GUIDELINE = "evidence_based_guideline"
    RANDOMIZED_CONTROLLED_TRIAL = "randomized_controlled_trial"
    NON_RANDOMIZED_CONTROLLED_TRIAL = "non_randomized_controlled_trial"
    COHORT_STUDY = "cohort_study"
    CASE_SERIES_OR_STUDY = "case_series_or_study"
    INDIVIDUAL_CASE_REPORT = "individual_case_report"
    EXPERT_OPINION = "expert_opinion"
    OTHER = "other"


class Stance(str, Enum):
    SUPPORT = "SUPPORT"
    CONTRADICT = "CONTRADICT"
    IRRELEVANT = "IRRELEVANT"


@dataclass(frozen=True)
class CandidateClaim:
    """One answer option interpreted as a candidate claim for its question."""

    question_id: str
    option_index: int
    option_label: str
    option_text: str

    def __post_init__(self) -> None:
        if not self.question_id or self.option_index < 0:
            raise ValueError("Candidate claims require a question id and nonnegative option index")
        if not self.option_label or not self.option_text:
            raise ValueError("Candidate claim option label and text must not be empty")


@dataclass(frozen=True)
class MedicalQuestion:
    question_id: str
    question: str
    option_labels: tuple[str, ...]
    options: tuple[str, ...]
    answer_index: int
    relevant_doc_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.question_id or not self.question:
            raise ValueError("Questions require nonempty identifiers and text")
        if len(self.option_labels) != len(self.options) or len(self.options) < 2:
            raise ValueError("Option labels and text must have the same length of at least two")
        if any(not isinstance(label, str) or not label.strip() for label in self.option_labels):
            raise ValueError("Option labels must be nonempty strings")
        if any(not isinstance(text, str) or not text.strip() for text in self.options):
            raise ValueError("Option text must be nonempty strings")
        if len(set(self.option_labels)) != len(self.option_labels):
            raise ValueError("Option labels must be unique")
        if not 0 <= self.answer_index < len(self.options):
            raise ValueError("answer_index is out of range")

    @property
    def answer_label(self) -> str:
        return self.option_labels[self.answer_index]

    @property
    def answer_text(self) -> str:
        return self.options[self.answer_index]

    @property
    def candidate_claims(self) -> tuple[CandidateClaim, ...]:
        return tuple(
            CandidateClaim(
                question_id=self.question_id,
                option_index=index,
                option_label=label,
                option_text=text,
            )
            for index, (label, text) in enumerate(zip(self.option_labels, self.options))
        )


@dataclass(frozen=True)
class RetrievedEvidence:
    schema_version: str
    question_id: str
    doc_id: str
    rank: int
    text: str
    source: str
    evidence_type: EvidenceType
    retrieval_score: float
    annotated_stances: Mapping[str, Stance] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QualityScore:
    value: float
    scorer_name: str
    rationale: str = ""
    components: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError("Quality scores must be in [0, 1]")
        copied: dict[str, float] = {}
        for name, raw_value in self.components.items():
            if not name:
                raise ValueError("Quality component names must not be empty")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError("Quality components must be finite numbers")
            copied[name] = value
        object.__setattr__(self, "components", MappingProxyType(copied))


@dataclass(frozen=True)
class StancePrediction:
    probabilities: Mapping[Stance, float]
    classifier_name: str
    rationale: str = ""

    def __post_init__(self) -> None:
        copied: dict[Stance, float] = {}
        for raw_stance, raw_value in self.probabilities.items():
            try:
                stance = Stance(raw_stance)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Stance probabilities must contain exactly all three stance labels"
                ) from exc
            value = float(raw_value)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("Stance probabilities must be finite and in [0, 1]")
            copied[stance] = value
        if set(copied) != set(Stance):
            raise ValueError("Stance probabilities must contain exactly all three stance labels")
        if not math.isclose(sum(copied.values()), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("Stance probabilities must sum to approximately 1")
        object.__setattr__(self, "probabilities", MappingProxyType(copied))

    @property
    def label(self) -> Stance:
        """Return the highest-probability label for hard-label baselines."""

        return max(Stance, key=lambda stance: self.probabilities[stance])

    @property
    def confidence(self) -> float:
        """Return the winning probability for compatibility with simple callers."""

        return self.probabilities[self.label]


@dataclass(frozen=True)
class ScoredEvidence:
    evidence: RetrievedEvidence
    quality: QualityScore
    stance: StancePrediction


@dataclass(frozen=True)
class AggregationResult:
    claim: CandidateClaim
    decision: Stance
    support_weight: float
    contradict_weight: float
    irrelevant_count: int
    evidence_count: int


@dataclass(frozen=True)
class PipelineResult:
    question: MedicalQuestion
    claim: CandidateClaim
    evidence: tuple[ScoredEvidence, ...]
    aggregation: AggregationResult


@dataclass(frozen=True)
class QuestionPrediction:
    """Final MCQ prediction plus comparable scores for every answer option."""

    question_id: str
    predicted_option_label: str
    option_scores: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.question_id or not self.predicted_option_label or len(self.option_scores) < 2:
            raise ValueError("Question predictions require an id, answer, and option scores")
        copied: dict[str, float] = {}
        for label, raw_value in self.option_scores.items():
            if not label:
                raise ValueError("Option score labels must not be empty")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError("Option scores must be finite")
            copied[label] = value
        if self.predicted_option_label not in copied:
            raise ValueError("predicted_option_label must identify one of the scored options")
        object.__setattr__(self, "option_scores", MappingProxyType(copied))
