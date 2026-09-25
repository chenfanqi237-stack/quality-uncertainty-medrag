# quality-uncertainty-medrag

A lightweight Python research scaffold for medical RAG experiments with conflicting evidence. Med-R² motivates the research setting, but this repository does **not** reproduce the Med-R² pipeline and does not implement the proposed uncertainty-aware method.

All example questions and evidence are fictional. The model-free fixture classifier reads optional stance annotations so the pipeline can be tested without an API key, large model, GPU, or network call.

## Included

- A MedQA-style JSONL question loader supporting option mappings and lists.
- A `CandidateClaim` record for each question-option pair.
- Versioned retrieved-evidence JSONL and JSON Schemas.
- Interchangeable interfaces for retrieval, quality scoring, stance classification, claim-level aggregation, answer generation, and text-generation backends.
- Full `SUPPORT`, `CONTRADICT`, and `IRRELEVANT` probability distributions on every stance prediction.
- Inspectable quality-score components, with a transparent metadata-only hierarchy component in the baseline.
- A simple claim-level quality-weighted vote baseline. Ties and cases without directional evidence use `IRRELEVANT` as an abstention state.
- A separate `QuestionPrediction` record for a final MCQ answer and per-option scores.
- Question-prediction accuracy, exact-answer matching, MRR, and Hit@k utilities.
- YAML configuration, random seeds, logs, effective configuration snapshots, outputs, and unit tests.

The quality score is a configurable baseline derived from supplied evidence-type metadata; it is not a clinical quality assessment. The weighted vote uses each stance distribution's highest-probability label and contains no uncertainty-aware method. The repository defines a final-answer interface and data model, but it intentionally does not yet add an option-selection algorithm.

## Repository separation

The upstream Med-RR clone is kept separately at `D:\Download\Med-RR-reference`. Its recorded commit and relationship to this project are documented in [REFERENCE.md](REFERENCE.md). This repository has no runtime dependency on Med-RR.

## Install

Python 3.9 or newer is required.

```bash
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -e ".[dev]"
```

The only runtime dependency is PyYAML.

## Run the baseline

From the repository root:

```bash
python -m quality_uncertainty_medrag --config configs/baseline.yaml
python -m pytest
```

The baseline evaluates every answer option as a separate candidate claim. It writes `config.yaml`, claim-level `predictions.jsonl`, retrieval `metrics.json`, and `run.log` under `outputs/baseline/`. Input paths are resolved relative to the YAML file; the saved effective config uses absolute paths for reproducibility on the same machine. The prediction rows follow [schemas/claim_prediction.schema.json](schemas/claim_prediction.schema.json).

## Data formats

Question records follow a compact MedQA-style shape:

```json
{"id":"q1","question":"...","options":{"A":"...","B":"..."},"answer":"B","relevant_doc_ids":["d1"]}
```

`MedicalQuestion` contains the question and answer options. Its `candidate_claims` property derives one `CandidateClaim` per option; there is no single question-level claim or expected stance. See [schemas/medqa_question.schema.json](schemas/medqa_question.schema.json) and [schemas/candidate_claim.schema.json](schemas/candidate_claim.schema.json).

Retrieved evidence uses one JSON object per line:

```json
{"schema_version":"2.0","question_id":"q1","doc_id":"d1","rank":1,"text":"...","source":"...","evidence_type":"systematic_review","retrieval_score":0.91,"annotated_stances":{"A":"CONTRADICT","B":"SUPPORT"}}
```

`annotated_stances` is an optional fixture mapping from option label to stance. It only exercises the model-free classifier. A future Qwen or API classifier can consume unannotated evidence through the same `StanceClassifier` interface. See [schemas/retrieved_evidence.schema.json](schemas/retrieved_evidence.schema.json).

The final MCQ result contract is documented in [schemas/question_prediction.schema.json](schemas/question_prediction.schema.json). Producing this result remains the responsibility of a future interchangeable `AnswerGenerator`; the current command stops at claim-level aggregation.

## Adding model-backed components later

Implement the protocols in `src/quality_uncertainty_medrag/interfaces.py` and inject them into `BaselinePipeline`. Local Qwen and hosted API adapters can both implement `TextGenerationBackend`; model-specific parsing belongs in a separate stance classifier or answer generator. The baseline does not select a provider.
