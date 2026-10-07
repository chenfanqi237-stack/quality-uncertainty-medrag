# Research v2.4: post-holdout, pre-independent-review milestone

Status: completed internal-holdout analysis, with manuscript integration deferred
at the researcher's explicit request. No paper or Overleaf edit is part of this
milestone. This document contains aggregate results and reproducibility metadata
only, not medical questions, abstracts, gold records or reviewer annotations.

## Scope and scientific configuration

Development and internal holdout are separate cohorts of 30 and 20 questions.
They are not pooled into a primary estimate. The holdout uses the already frozen
question-level K<=3 retrieval snapshot, 55 documents, 275 option-document pairs,
and ten seeds (101-110), producing 2,750 validated observations. One question has
an empty evidence pool and remains in the denominator.

The matched comparison retains identical evidence and stochastic observations:
A is unweighted modal-hard aggregation; B is quality-weighted modal hard;
C retains empirical stance frequencies; D adds normalized entropy reweighting.
The primary comparison is B versus C with unit analysis weights. Native claim
semantics and the positive-score, unique-answer selective policy are unchanged.
Original publication-type weights remain unchanged, including OTHER=0.

Frozen inference identity: qwen3:8b, Ollama 0.34.4, thinking enabled,
temperature 0.7. Model digest:
`500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41`.
No inference, retrieval, sampling, tuning or annotation occurred during import.

## Independently verified internal-holdout results

| Method | Quality condition | Correct/20 | Answered/20 | Accuracy | Coverage | Answered accuracy | Selective risk |
|---|---|---:|---:|---:|---:|---:|---:|
| A matched hard | Quality invariant | 4/20 | 9/20 | 20.00% | 45.00% | 44.44% | 55.56% |
| B matched hard | Original | 1/20 | 1/20 | 5.00% | 5.00% | 100.00% | 0.00% |
| C frequency-based soft | Original | 1/20 | 1/20 | 5.00% | 5.00% | 100.00% | 0.00% |
| D entropy-weighted soft | Original | 1/20 | 1/20 | 5.00% | 5.00% | 100.00% | 0.00% |
| B matched hard | Unit | 4/20 | 9/20 | 20.00% | 45.00% | 44.44% | 55.56% |
| C frequency-based soft | Unit | 7/20 | 13/20 | 35.00% | 65.00% | 53.85% | 46.15% |
| D entropy-weighted soft | Unit | 8/20 | 14/20 | 40.00% | 70.00% | 57.14% | 42.86% |

Matched unit-weight hard-to-frequency transitions comprise eight identical
answers, zero different-answer replacements, five abstention-to-answer changes,
one answer-to-abstention change, and six joint abstentions. Three newly selected
answers are correct and two are wrong; the removed hard answer was wrong.

The primary coverage difference is +20 percentage points. Its frozen
conservative marginal-bound working-model interval is
[-20.661428711733376, +52.19944465456091] percentage points. It uses two 97.5%
exact Clopper-Pearson bounds on the marginal discordance rates with a Bonferroni
difference construction; it is not an exact paired confidence interval.

The observed accuracy difference is +15 percentage points. The supplementary
Newcombe method 10 approximate paired interval, pre-specified for this holdout,
is [-3.714213705333805, +33.130249931992706] percentage points. Individual
proportion intervals are exact 95% Clopper-Pearson intervals. Both difference
intervals span zero. Raw paired counts remain primary; no significance test or
statistical superiority claim is made.

## Separate-cohort comparison and sensitivity

| Cohort | Hard coverage | Soft coverage | Coverage difference | Hard accuracy | Soft accuracy | Accuracy difference | Soft-only | Hard-only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Development, N=30 | 13/30 | 19/30 | +20 pp | 7/30 | 11/30 | +13.33 pp | 6 | 0 |
| Internal holdout, N=20 | 9/20 | 13/20 | +20 pp | 4/20 | 7/20 | +15 pp | 5 | 1 |

The net coverage effect reproduced descriptively in direction and magnitude,
not as statistical replication proof. Retained frequencies added both correct
and incorrect answers in both cohorts. Clinical safety, calibrated
probabilities, statistical superiority and broader generalization are not
established.

Entropy weighting was answer-level inert on development. On holdout it changed
36/95 jointly defined unit-weight option scores, two rankings and one final
decision (an abstention became a correct answer). Native claim decisions did
not change. Under original weights, 0/10 jointly defined scores and no final
decisions changed. The small answer-level effect is not evidence of universal
entropy benefit.

Only two holdout records have positive original quality weights, across two of
20 questions. Original B/C/D each have 90/100 undefined option scores; unit
methods have 5/100, corresponding to the empty pool. Original-to-unit final
decisions change in 8/20, 12/20 and 13/20 questions for B/C/D. OTHER=0 denotes
unclassified study design under this mapper, not poor or irrelevant evidence.

Vanilla-RAG holdout v1 remains incomplete: four valid generated predictions,
one strict-format failure after three attempts, fourteen missing generated
predictions, and one structural empty-pool abstention. No partial Vanilla
accuracy, coverage or risk is reported. The completed development Vanilla
baseline remains valid and separate.

## Integrity, blinding and operational provenance

The transfer's SHA256, CRC and all 37,957 listed member hashes were verified.
The 59 packaged runtime files were hashed; 56 existing project scientific
references matched exactly. The 5,515-revision native linear journal was folded
in memory and matched the final snapshot; native final validation and complete
blind-score reproduction also passed. This is not a claim that the original
sampler was cold-restored and rescanned at every historical revision.

Hash-linked records confirm dual-freeze verification at
2026-10-07 09:48:52.217017 UTC, before first recorded gold access at
09:49:36.104024 UTC (20:48:52 and 20:49:36 Sydney). Local independent gold
reading occurred only after completed identity and blind-score reproof. Only
the twenty holdout records were decoded for local scoring.

The canonical source stores the gold as an A-E `answer` field, whereas the
frozen evaluator expected `answer_idx`. The preserved adapter changes the
evaluator reader only. It does not change prediction bytes, question IDs,
options, equations, thresholds or scoring. This is an evaluator/data-schema
compatibility fix, not a scientific-method modification.

Authorized runtime, recovery, checkpoint-cadence, archive-path, controller and
resident-worker amendments are retained in the private provenance. The
auxiliary receipt's closed-gold-gate error occurred after the core evaluation
completed; the failed receipt and STOP logs are preserved, not hidden or
presented as a failed core evaluation.

The independent 90-pair development disagreement review remains pending.
No review annotation or clinical adjudication was generated in this milestone.

## Reproducibility and private storage

Canonical analysis:
`outputs/holdout_evaluation/dev31_50_holdout_v1/canonical_analysis_v1/`.
Readable imported provenance:
`outputs/holdout_evaluation/dev31_50_holdout_v1/completed_gpu_run_20261007/`.
Private original archive:
`cloud/checkpoint/dev31_50_holdout/dev31_50_holdout_results_transfer_v1.zip`.
Private freeze ledger:
`outputs/research_freeze/post_holdout_pre_review_v1/freeze_manifest.json`.
Private backup:
`cloud/checkpoint/post_holdout_freeze/post_holdout_pre_review_backup_v1.zip`.

Transfer SHA256:
`eea8c0f8df00147f34da1344f9971e95d40c7e4dd422edee5e3cc4eb56660f75`.
Primary blind SHA256:
`25ef5e04db22d400c325f5ecf98376665709b7b21b7f5e7da5e1187416f506f0`.
Vanilla incomplete freeze SHA256:
`52eada0e590fa229978399ec836ab3c48207a24218a7954f2ec40df34cea9e95`.

`cloud/integrate_holdout_results_v1.py` is an isolated CPU-only integration
utility; `tests/test_holdout_results_integration.py` contains synthetic checks.
Existing scientific modules were not edited. Its executing analysis source is
preserved privately as `analysis_code_executed.py` and matches the recorded
analysis-code hash. A subsequently added private-backup helper is operational
only and has a separate current-file hash in the freeze ledger.

The same-disk backup is not an off-machine backup. Copy that ZIP and its SHA256
sidecar to a separate physical device or private cloud location; no upload was
performed. Checkpoints, gold records, medical source text, review materials,
local manuscript snapshots and imported operational logs remain private.
