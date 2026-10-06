# Research v2.3: Post-Dev30 / Pre-Independent-Review freeze

Snapshot date: 6 October 2026. This is an inspected DEVELOPMENT cohort, not holdout evaluation or clinical validation. No independent-review annotations are completed.

## Frozen findings

The stochastic experiment has 3,250/3,250 valid observations (325 exact option/evidence pairs, ten seeds 101–110), zero missing, failed or interrupted units. There are 30 development questions, 22 nonempty pools, eight empty pools and 65 selected evidence documents. Native final revision is 5,347; transport revision is 5,348. Recorded generation attempts (3,252) are audit records, not an exact HTTP-request count.

| Method | Condition | Correct/30 | Answered/30 | Wrong answered |
| --- | --- | --- | --- | --- |
| A matched hard | Quality-invariant | 7/30 | 13/30 | 6 |
| B matched hard | Original | 0/30 | 1/30 | 1 |
| C soft | Original | 0/30 | 1/30 | 1 |
| D entropy soft | Original | 0/30 | 1/30 | 1 |
| B matched hard | Unit | 7/30 | 13/30 | 6 |
| C soft | Unit | 11/30 | 19/30 | 8 |
| D entropy soft | Unit | 11/30 | 19/30 | 8 |
| Vanilla RAG | No quality weighting | 12/30 | 19/30 | 7 |

The primary unit-weight B→C contrast expands six abstentions to answers: four correct and two wrong. All 13 hard answers are unchanged; there are no answer replacements or answer→abstain transitions. Observed coverage difference is +6/30 (+20 percentage points); accuracy difference is +4/30 (+13.33 points). Raw paired counts remain primary; no significance or superiority is claimed.

The coverage interval [−6.98,+41.16] points is a conservative marginal-bound interval derived from two 97.5% marginal Clopper–Pearson discordance-rate bounds, not an exact paired CI. The secondary exploratory Newcombe method 10 approximate 95% paired accuracy interval is [−0.58,+26.98] points. Both span zero. Individual method proportions use separate 95% marginal Clopper–Pearson intervals.

Entropy changes 1/35 original-quality and 48/110 unit-weight jointly defined scores, but zero native claim decisions, rankings or final answers. The current publication-type gate gives OTHER=0 to 56/65 records and positive quality to only seven questions; this is not a general claim that quality weighting is harmful. Vanilla is an external full-answer baseline, not a matched representation arm. Its 11 abstentions comprise eight structural and three model-issued abstentions.

## Frozen identities and private archive locations

| Artifact | Project-relative location | SHA256 |
| --- | --- | --- |
| Primary protocol | `docs/multiqu_primary_protocol_v1.md` | `85f690a87e9022973dbfcd118ea1ebb404635e5e708f689ae33fd6a4e392fc82` |
| Full native recovery transport | `cloud/checkpoint/dev30_completed_20261006_v1/dev30_kaggle_export_r005348_a6d87e9ee75c42dba2094c68563859fd.zip` | `a219d649d6106aa07980e4b1916458debd142e76da7e13595570736b34349023` |
| Native final snapshot | Native archive nested in the recovery transport | `c45e4194c1292c471daeb71c6772d58a46ea780359a51957f5a7e11f36d3a96f` |
| Primary blind predictions | `outputs/dev30_cpu_20261006_v1/independent_blind/blind_predictions.json` | `9c9dacef47455f601e520d676136f1cdb8f2b0ed9f1e84cee7b75eb178776e85` |
| Vanilla return transport | `cloud/checkpoint/dev30_vanilla_rag/dev30_vanilla_rag_results_transfer_v1.zip` | `a993ab5e16517256e4f43a6906336cd6699c81c914a236a968e0f741c076054d` |
| Vanilla blind predictions | Completed-run blind artifact within its return transport | `0adffcca4cf0740cdca055ec12240f60f7cb2b03f7366055166257e7d41a22d1` |
| Vanilla recovery snapshot | Completed-run recovery artifact within its return transport | `a30d0b0912ac5aa7efa331946c5a994046f3b54bc419301c634104727a49f65c` |

The private machine-readable inventory is `outputs/research_freeze/post_dev30_pre_review_v1/freeze_manifest.json`. It records source paths, bytes, SHA256, roles, classification and public/private disposition. Canonical archives, observations, protocols and old result files are immutable; this milestone only creates new snapshots and documentation.

## Independent review and blinding

The post-hoc explanatory cohort includes all 90 option/evidence pairs from six changed questions. Eight exact historical labels exist but none has verified pre-outcome independent-blind provenance; independent eligible coverage is 0/90, so 90 new independent annotations remain necessary. The primary researcher has seen the outcomes and cannot serve as their independent blinded reviewer.

Deliver only the two-member reviewer ZIP under `reviewer_delivery/`; never send the private mapping, outcome reports, manuscript or source manifests to the reviewer. Review uses supplied question/candidate/title/abstract only. Reviewer stance remains SUPPORT/CONTRADICT/IRRELEVANT with HIGH/MEDIUM/LOW confidence. A reviewer without clinical qualifications provides independent stance annotation, not clinical adjudication. No annotation mechanism finding is established yet.

## Backup and reproduction boundary

Private recovery ZIP: `cloud/checkpoint/post_dev30_freeze/post_dev30_pre_review_backup_v1.zip`, with an external SHA256 sidecar and internal path/size/hash manifest. The archive must pass CRC, exact-member and temporary-directory restore checks. Copy this ZIP AND its sidecar to an independent device or private cloud destination and verify the copied SHA256. Local creation alone is not an off-machine backup; no external upload is performed here.

The backup includes the completed analysis, frozen gold-free Dev30 execution inputs, source dependencies and protocols, review assets/private mapping, and manuscript snapshots. It excludes model weights, Python environments, unrelated checkpoints, raw all-50 MedQA data and holdout content. Re-running evaluator-only gold joins requires separately obtaining the licensed source under its access terms; already saved evaluated outputs are preserved. Generation-free review merge uses `cloud/dev30_disagreement_review_v1.py merge`; it rejects blank or non-independent submissions and preserves disagreement rather than automatically adjudicating.

## Public milestone boundary

Only reviewed public-safe code, synthetic CPU tests and aggregate/hash-only documentation are candidates for staging. Raw question/abstract text, reviewer CSVs/mappings, annotations, checkpoint ZIPs, manuscript binary snapshots and machine-specific provenance remain private and ignored. Existing README edits remain unstaged. No inference, annotation, retrieval, holdout evaluation, commit or push is part of this milestone. Suggested future commit: `Freeze Dev30 natural QA comparison and review protocol`.
