# Medical RAG Research v2.2: architecture audit

Audit date: 2026-10-03. Research milestone: post-Q27 technical pilot.
This is an experimental research repository, not a production medical system.
The milestone label does not change the Python package version (`0.1.0`).
No algorithms, frozen inputs, checkpoints, or existing results were changed by this audit.

## Actual data flow

```text
Frozen MedQA question
  -> question-only query reformulation and cached relaxation cascade
  -> PubMed ESearch / EFetch, frozen question-level ranked records
  -> explicit publication-type mapping and configured quality weights
  -> question + one candidate option + title/abstract stance input
  -> frozen hard prediction OR seeded stochastic label observations
  -> deterministic Top-K question-option-document join
  -> matched hard / soft claim aggregation
  -> isolated Q27 five-option answer-selection adapter
  -> blind predictions, then evaluator-only gold join
```

This flow spans separate entrypoints. The baseline CLI does not execute the entire
real-data flow, and there is no completed full-pool MedQA QA evaluation.

## Directory responsibilities

```text
src/quality_uncertainty_medrag/  Core interfaces and frozen research modules
configs/baseline.yaml           Baseline settings and evidence-type weights
schemas/                       Input/output contracts
tests/                         Focused unit and research-interface tests
cloud/                         Research packaging, cloud jobs, isolated Q27 adapter
notebooks/                     Manual cloud execution notebooks
data/                          Committed synthetic fixtures; private real inputs
outputs/                       Mostly ignored frozen/derived research artifacts
research/aggregation_ablation/ Public byte-identical M2 code/test snapshots
docs/                          Method notes and this milestone documentation
```

## Stage-to-module map

All paths below are repository-relative. Private outputs are locators, not a
recommendation to distribute their contents.

| Stage | Actual modules / configuration | Responsibility and artifact/test links |
| --- | --- | --- |
| Question and option identity | `medqa_us.py`, `loaders.py`, `models.py` | Frozen development IDs and per-option `CandidateClaim`; real inputs in `data/processed/medqa_us_dev_50.jsonl`. Only synthetic datasets are tracked. |
| Query formulation | `clinical_query.py`, `clinical_query_relaxation.py`, `clinical_query_minimal.py`, `query_cache.py`, `query_inspection.py`; alternative `pubmed_query.py`, `query_relaxation.py` | Question-only LLM primary/core/minimal queries; the frozen cascade relaxes when fewer than five matches exist. Candidate answers and gold are not reformulation inputs. `tests/test_query_inspection.py` checks input isolation. |
| PubMed retrieval | `pubmed.py`, `pubmed_records.py`, `heldout_retrieval.py`, `cloud_runtime.py`, `cloud_bundle.py` | ESearch relevance order followed by EFetch metadata; no local dense or hybrid index. Title/abstract, PMID, publication types and ranks survive freezing. `heldout_retrieval.py` is a historical name for dev 11-30, not dev 31-50. Separate latter-partition tooling is `cloud/final_retrieval_validation.py` and `notebooks/kaggle_final_retrieval_dev_31_50.ipynb`. |
| Evidence quality | `pubmed_evidence_type.py`, `quality.py`, `configs/baseline.yaml` | Conservative exact publication-type rules and fixed weights. Generic Journal Article / Review and ambiguous Case Reports remain `OTHER`, weight zero. Unknown study design is not a finding of clinical inferiority or irrelevance. |
| Stance and annotations | `llm_stance.py`, `ollama_backend.py`, `stance_annotation.py`, `stance_directional_annotation.py`, `stance_comparison_run.py`, `stance_smoke.py` | Frozen prompt/parser, identity-aware caches, annotation/comparison tooling. `stance.py` supplies fixture annotations, not a live classifier. `docs/llm_stance.md` documents the classifier. |
| Self-consistency and uncertainty | `stance_self_consistency.py`, `stance_uncertainty_evaluation.py` | Seeded label observations, identity/configuration gates, checkpoint restoration, entropy and error diagnostics. M1 outputs are under `outputs/stance_uncertainty/evaluation/`. M1 notebooks are `colab_stance_self_consistency_10.ipynb` and `kaggle_stance_self_consistency_10.ipynb`. |
| Top-K deterministic join | `topk_aggregation_development.py` | Dev 1-30 only; first K ranked records, deterministic rank/document-ID ordering, identical ordered evidence across five options. Missing predictions stay missing. Outputs: `outputs/aggregation_development/topk_dev_1_30_k3_k5_v1/`; test: `tests/test_topk_aggregation_development.py`. |
| Claim aggregation | `aggregation.py`, `uncertainty_aggregation.py`; M2 `research/aggregation_ablation/matched_60_generation_free_v1/analysis.py` | Frozen majority, quality-hard and quality-entropy-soft modules. M2 adds the entropy-off soft comparison without rewriting those modules. Public analysis and adjacent tests are byte-identical copies of the originals in ignored outputs; results remain private. |
| Q27 selection / evaluation | `cloud/question27_adapter.py`, `cloud/question27_pilot.py`, `cloud/prepare_question27_pilot.py` | Matched modal hard labels from the same ten seeds used for soft frequencies; both quality conditions; five-option scoring and explicit incomplete/undefined/abstention states. Test: `tests/test_question27_pilot.py`; notebook: `notebooks/colab_medqa_question27_pilot.ipynb`; outputs: `outputs/aggregation_pilot/medqa_question27_k3_v1/`. |
| Scaffold evaluation | `pipeline.py`, `cli.py`, `evaluation.py`, `interfaces.py` | `BaselinePipeline` composes retrieval, quality and per-claim aggregation. `quality-medrag-run` defaults to synthetic/precomputed inputs and claim/retrieval metrics; it is not the Q27 MCQ experiment. An answer-generator protocol is not an implemented full-system generator. |

Module names without a prefix are inside `src/quality_uncertainty_medrag/`.
NLI, sentence-level NLI, truncation and logprob probes are optional experimental
branches, not dependencies of the completed Q27 pilot.

## Dependency overview

- Core package: Python >=3.9; `PyYAML>=6,<7`. Development tests use `pytest>=7,<10`.
- Retrieval and Ollama clients use standard-library HTTP. Network/model services
  are execution-time dependencies only for their explicitly invoked jobs.
- Q27 packaged runtime is standard-library-only and includes selected core source;
  it does not require the optional NLI stack.
- The optional, isolated `cloud/nli_requirements.txt` pins torch, transformers,
  huggingface_hub, tokenizers and safetensors; its documented Python minimum is 3.10.
- Real experiment reproduction requires separately provisioned frozen question,
  retrieval, annotation and checkpoint files. A source-only Git clone cannot
  reproduce these results without those inputs.
- The original MedQA source adapter refers to an external Med-RR reference tree.
  That tree is not vendored and is not a runtime import dependency of Q27.

## Frozen Q27 semantics

For each complete pair, empirical label frequencies define `s = pS - pC` and
`u3 = H(p)/log(3)`. They are not calibrated probabilities. A unique modal label
defines matched hard stance; unresolved modal ties remain valid ties, not missing
predictions. Historical temperature-zero labels must not replace matched labels.

The four conditions are unweighted matched hard (A), quality matched hard (B),
quality soft (C), and quality/entropy soft (D). D uses `q*(1-u3)`; C omits the
entropy factor. The unit-weight control disables quality analytically and never
rewrites stored quality metadata.

The Q27 adapter retains native claim-level eligibility. It selects a unique highest
positive eligible option, otherwise abstains, with numerical tie tolerance `1e-12`.
Missing required samples mean `INCOMPLETE`; a zero denominator is undefined, not
an imputed zero. Gold is joined only after blind scoring.

## Technical debt and risks

1. M2 originals remain in ignored `outputs/`. Byte-identical copies in `research/`
   preserve code identity without refactoring. Its private inputs/results still
   require separate restoration; source availability is not data availability.
2. The v2.2 staging review extends `.gitignore` to cover raw data, local caches,
   model weights, cloud archives/checkpoints, credentials and temporary/editor
   directories. Ignore rules do not remove already-tracked files or history and
   do not substitute for reviewing an exact staged diff.
   A minimal `.gitattributes` LF policy protects source/configuration byte hashes
   across Windows/Linux checkouts; no bulk renormalization was performed.
3. Source-byte hashes intentionally couple archives to specific module versions.
   Do not edit pinned code or update historical hashes merely to pass validation.
4. M2 and Q27 duplicate controlled soft-score adapter logic, and jobs repeat
   checkpoint/hash/backup utilities. This is research-job duplication, not a
   justification to refactor the completed experiments in this milestone.
5. Historical hard labels, matched modal labels and alternative directional
   normalizations must be named explicitly; they are not interchangeable.
6. Q27 preparation reports saying 150 samples are missing are initial inventory
   snapshots. The completed archive/evaluation, not those snapshots, establishes
   the current 150/150 status.
7. The Q27 preparer contains a personal interpreter path; the notebook contains
   a machine-specific Windows project path. Both remain unstaged and unchanged.
   The public source-only milestone therefore does not provide that Q27 notebook
   or runtime bundle; archived hashes/locators do not imply their inclusion.
8. No root license was found; raw MedQA/PubMed redistribution permissions were not
   verified. Small files and already-tracked files are not automatically safe.
9. README changes and older cloud jobs describe multiple research phases. Their
   claims/settings need individual review rather than inclusion by directory.

The audit did not inspect unrelated caches, model environments, or temporary test
trees recursively. Git reported a permission warning for
`.tmp_colab_adapter_tests_ascii/`; no access or deletion was attempted.
