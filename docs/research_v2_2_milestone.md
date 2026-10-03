# Medical RAG Research v2.2 - Post-Q27 Technical Pilot

Audit date: 2026-10-03. This documents a technical research milestone, not a claim
of better medical QA or a production release. See `research_v2_2_architecture.md`
for the actual module/data-flow map.

## Objective and verified milestones

Study hard versus soft stance aggregation for selective Medical RAG using frozen,
identical retrieved evidence and matched underlying samples where available.

| Milestone | Verified artifact scope | Confirmed result and interpretation |
| --- | --- | --- |
| M1 | 60 pairs x 10 seeds; 600 valid, no failed/missing samples; final uncertainty evaluation | Three-class entropy error AUROC 0.5814 / AP 0.3823; directional entropy AUROC 0.3222 / AP 0.2447; error prevalence 17/60. Reference annotations are AI-assisted adjudicated, not independent clinician labels. |
| M2 | Generation-free matched 60-pair ablation; 50 question-option groups, 40 single-evidence and 10 two-evidence groups | Quality-disabled B-to-C: 9 native decision changes, 18 score changes; C-to-D: 6 score changes, no native decision change. Original quality leaves 41/50 B/C/D scores undefined. These are claim-level diagnostics, not full-pool answer selection or QA improvement. |
| M3 | Q27, K=3; 15 pairs x 10 seeds; 150 valid, zero failed/missing/interrupted | All A/B/C/D methods abstain in both quality conditions. All 40 option scores are defined; reason is `NO_POSITIVE_ELIGIBLE_SCORE`. Observations: 0 SUPPORT, 14 CONTRADICT, 136 IRRELEVANT. Saved blind predictions exactly match generation-free recomputation. |

M2 contains two unresolved modal pair ties. Missing observations and valid modal
ties remain distinct. No verified natural SUPPORT/CONTRADICT conflict group, no
calibrated probabilities and no established answer-accuracy improvement are claimed.
The seven-question positive-quality subset is not a representative QA evaluation.
Dev 31-50 remains separate from development; this audit did not run it.

## Frozen identity and artifact locations

Paths are repository-relative. These locators do not authorize publishing raw data.

| Artifact | Path | SHA256 |
| --- | --- | --- |
| Immutable M1 archive | `cloud/checkpoint/checkpoint_valid600_20261002T152105Z_553a09b9.zip` | `356cdb4f30de17952e22176bc6127b41384bae3a795baf8c3724612162ee4f34` |
| M1 pair uncertainty | `outputs/stance_uncertainty/evaluation/checkpoint_valid600_20261002T152105Z_553a09b9_final_evaluation_v1/pair_uncertainty.csv` | `ec7e725a65d2a08118e188eef1ddf9302e5bc6d829fd3c9281b26f0696bb21ac` |
| M1 analysis manifest | Same M1 evaluation directory, `analysis_manifest.json` | `f102d04783ee0a5f2a3e160f4bccef28cc1826e30983e95ba5df1c65d887965f` |
| M2 results | `outputs/aggregation_ablation/matched_60_generation_free_v1/ablation_results.csv` | `8cdebd036506ed8eb0060d7d86f80ca31627d13c5191547eb743ce620c7924eb` |
| M2 frozen analysis source | Same M2 directory, `analysis.py` | `0080a0ebe23cefd5a8fe19816511f8cf1f197c372c5650cc33dad96e34146928` |
| Q27 final archive | `cloud/checkpoint/question27_download/q27_final_valid150_r000302_20261003T092959459406+0000_77e56b49.zip` | `1531a91c7ee2756cd2a92881300b16610841a0b515f0146b699ae55abc96dd79` |
| Q27 pilot input | `outputs/aggregation_pilot/medqa_question27_k3_v1/pilot_input.jsonl` | `8a0f3e91a82153909ae0939daf94b4beacc92e3086f94c5039984954a19ebc1f` |
| Q27 runtime bundle | `cloud/medqa_question27_k3_runtime_v1.zip` | `57f431c98e9a89998fced8f738e16d3c5b034926a13fbb4e26c062666f820a82` |
| Q27 notebook | `notebooks/colab_medqa_question27_pilot.ipynb` | `0aa69c34c330999c0762b69e878f41c4a84fec9e9806ab38f8fb918509716601` |
| Q27 blind predictions | `outputs/aggregation_pilot/medqa_question27_k3_v1/evaluation_82b4f527c792413a8e6b1d2430ab15d4/blind_predictions.json` | `731efcd0960a558ec2e2c5a5113e3dc8255d3cc7478c625ea026e6a2f746822f` |

Frozen sampling: Ollama `0.34.4`, `qwen3:8b`, thinking enabled, temperature `0.7`,
seeds `101..110`, unchanged three-class prompt/parser.

- Model digest: `500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41`.
- Prompt SHA256: `c9db305b4c2a7ec23700851c332f8109c9d2e3b62efd83e9eb1c50520670886a`.
- Q27 evidence identities: PMIDs `39709592`, `26780003`, `21663949`; identical order
  for all five options, two positive-quality articles. The original 600-sample
  archive contains no reusable Q27 pair observations.
- Final archive/configuration verification is offline provenance verification,
  not a new live verification or invocation of an Ollama installation.

Archive SHA/CRC, manifests, exact pair-seed identities, frozen input and source
hashes were checked. The historical M1 manifest records base commit `8ad6afb`;
its recorded source hashes still match current files. Historical provenance was
not rewritten to the current Git commit.

## Reproduction entrypoints

Use the repository root, an appropriately provisioned Python environment, and
authorized frozen assets at the recorded relative paths. No checkpoint is part
of the proposed Git staging list. Do not regenerate data to compensate for absent
private inputs. The following commands do not call a model.

Read-only M2 reproduction using the private original artifact directory:

```powershell
python -B outputs/aggregation_ablation/matched_60_generation_free_v1/analysis.py --check
```

The public source/test snapshots are in
`research/aggregation_ablation/matched_60_generation_free_v1/`. They are byte-for-byte
copies of the two frozen M2 scripts, not rewritten aggregation algorithms. The
matching directory depth preserves their repository-root discovery. Tests run with:

```powershell
python -B -m unittest discover -s research/aggregation_ablation/matched_60_generation_free_v1 -p test_analysis.py -v
```

To use the copied script's `--check`, authorized users must restore the original
`ablation_results.csv`, `ablation_summary.md` and `manifest.json` alongside the
copied scripts, in addition to all separately supplied frozen input dependencies.
Those three result files are ignored. Without `--check`, the original script
writes new results beside itself and refuses to overwrite existing results; do
not invoke that mode as a substitute for verifying the archived experiment.

Q27 preparation/runtime consistency check (requires the separately retained,
unstaged preparer and its existing private artifacts):

```powershell
python -B cloud/prepare_question27_pilot.py --check
```

Generation-free Q27 evaluation into a NEW output directory:

```powershell
python -B cloud/question27_pilot.py evaluate --checkpoint cloud/checkpoint/question27_download/q27_final_valid150_r000302_20261003T092959459406+0000_77e56b49.zip --sha256 1531a91c7ee2756cd2a92881300b16610841a0b515f0146b699ae55abc96dd79 --gold-source data/processed/medqa_us_dev_50.jsonl --output-dir outputs/aggregation_pilot/medqa_question27_k3_v1/reproduction_new
```

M1 evaluation entrypoint is
`python -m quality_uncertainty_medrag.stance_uncertainty_evaluation`, with required
`--project-root`, `--checkpoint`, `--output-dir`, `--mode FINAL_EVALUATION` and the
recorded `--expected-checkpoint-sha256`. It requires the frozen reference and hard
prediction files as well as the archive. Select a new output directory.

Top-K builder entrypoint is
`python -m quality_uncertainty_medrag.topk_aggregation_development --project-root . --output-dir <new-directory>`.
It is deterministic but writes derived outputs; focused tests reran it only in
disposable temporary directories, never over existing research output directories.
The baseline `quality-medrag-run` CLI is a scaffold/synthetic demonstration, not
the reproduction command for M1-M3.

## Git inspection and bounded safety audit

- Branch `main`; local tracking ref `origin/main`; HEAD `d4387ae`.
- Remote: `https://github.com/chenfanqi237-stack/quality-uncertainty-medrag.git`.
  No fetch/push or live visibility check was performed.
- Three existing commits: initialization `908ebb5`, baseline `8ad6afb`, freeze
  `d4387ae`. The index was empty before staging review; the existing tracked
  worktree change was `README.md` (+322/-3 lines), which remains unstaged and intact.
  The staging review changes only ignore policy, documentation and safe copies;
  original research code and assets remain unchanged.
- 110 files were already tracked; none was >=1 MiB. Real MedQA/retrieval files were
  not tracked; the two committed data files are synthetic fixtures.
- Three already-tracked M1 evaluation files were inspected despite the outputs
  ignore rule: manifest, summary and evaluation JSON. No copied abstracts were
  found in those files; their current contents have no changes to stage.
- A bounded 138-file relevant text scan found no high-confidence embedded API
  token/private key/credential URL. A literal secret-named test sentinel is not a
  credential. This is not a complete repository/history secret certification.
- The subsequent public-staging review examined all 126 unique blobs across the
  three locally reachable commits. No high-confidence credentials, binary archives
  or model weights were detected. Existing machine-local paths occur in historical
  README/reference/MedQA adapter and a synthetic path-validation test. Ignore rules
  do not remediate those historical paths; no history rewrite was authorized.
- Frozen-source checks against all 50 questions and 567 retrieval records found
  no copied twenty-word clinical-text windows in those historical blobs or the
  final allowlist. Decoded Python/JSON literals were also inspected. This bounded
  content check is not a general legal, privacy or secret-audit certification.
- All five inspected notebooks had no saved execution outputs/counts. Q27 had ten
  code cells. Some frozen preparer/notebook/README/reference examples contain
  local Windows paths; inclusion requires review, not silent rewriting.
- Known inspected raw retrieval file: 3,803,940 bytes. Original M1 archive:
  1,359,842 bytes. Q27 runtime: 62,479 bytes and contains clinical question and
  abstract text. Exclude it for content reasons despite its small size.

## Reviewed public staging allowlist

The original 12-file proposal was reviewed individually, including credentials,
decoded personal paths, embedded source text, binary content and size. The Q27
preparer and notebook were rejected for this public staging operation because of
their local-path examples; they were not edited. Safe M2 scripts and the generic
annotation guide were copied out of ignored outputs. The final explicit allowlist
contains these 12 files, not their parent directories. A small LF policy was added
after Git warned about future CRLF conversion of hash-pinned text. No README edits
are included, and no bulk renormalization is performed.

```text
.gitignore
.gitattributes
src/quality_uncertainty_medrag/topk_aggregation_development.py
tests/test_topk_aggregation_development.py
cloud/question27_adapter.py
cloud/question27_pilot.py
tests/test_question27_pilot.py
docs/research_v2_2_architecture.md
docs/research_v2_2_milestone.md
research/aggregation_ablation/matched_60_generation_free_v1/analysis.py
research/aggregation_ablation/matched_60_generation_free_v1/test_analysis.py
docs/q27_abstract_stance_annotation_guide_v1.md
```

No ignore override is required for these destinations. The annotation guide is a
byte-identical copy containing general instructions only, not reviewer judgments.
Existing tracked core algorithms/configs/tests already reside in HEAD and do not
need redundant staging. No candidate contains detected embedded MedQA question
or Q27 abstract text, binary payloads or credentials; all are small text files.

| Original candidate | Public-staging decision |
| --- | --- |
| Top-K builder and its test | Include unchanged source; real inputs remain private. |
| Q27 adapter, pilot orchestration and test | Include unchanged source; no embedded clinical input text. |
| Q27 preparer | Exclude: personal interpreter and local project paths; preserve frozen bytes. |
| Q27 notebook | Exclude: machine-specific local project path; preserve frozen bytes. |
| Architecture and milestone documents | Include reviewed, portable documentation; hashes/locators only. |
| M2 analysis and test in outputs | Exclude original paths; include identical copies under research. |
| Generic annotation guide in outputs | Exclude original path; include identical copy under docs. |

### Review before any additional inclusion

Keep out of the initial allowlist: modified `README.md`; historical cloud
instructions/scripts/notebooks and Kaggle account/config metadata; unrelated
method notes; derived reports and research annotations. Examples of small files
eligible for a separate review are M2 `manifest.json` / `ablation_summary.md`, Q27
`input_manifest.json` / `sampling_manifest.json` / `reproducibility_manifest.json`,
and Q27 blind prediction/evaluation summaries. Safe size does not establish safe
content, correct claims, or redistribution permission. Gold-containing evaluator
outputs are not automatically approved for public inclusion.

### Exclude from this milestone

- `cloud/checkpoint/`, `cloud/backups/`, runtime/upload/bundle ZIPs including the
  Q27 runtime; model weights and local dependency environments.
- Real `data/processed/`, `data/retrieved/`, `data/stance_smoke/`, and data caches;
  raw or licensed question/evidence text, full text, or answer records.
- Canonical evidence JSONL/CSV, reference/review rows, stochastic caches, mutable
  sampling state and input copies under outputs. In particular, Q27
  `pilot_input.jsonl`, its blinded review CSV, and `q27_failure_diagnosis_v1.md`
  contain clinical source text and must stay out of this commit.
- Credentials / `.env*`, editor state, temporary pytest directories, `.tmp_*`,
  root `t/` and numbered `t*` test directories, bytecode/build artifacts.

`.gitignore` now explicitly protects these local asset classes, while preserving
the tracked synthetic fixtures. It does not untrack files, remove data or rewrite
history. Restored/generated M2 results beside the copied source are also ignored.
Do not rely on `.gitignore` instead of inspecting the exact staged diff, and do
not use `git add .` or `git add -A`.

`.gitattributes` keeps Python, YAML, JSON, notebooks, Markdown and Git policy files
in LF form during checkout so source/configuration hashes do not depend on the
Windows line-ending setting. This changes checkout policy, not algorithms or
current frozen assets; no `--renormalize` operation is included.

## Validation, incomplete research and readiness

The prior architecture audit passed eight selected CPU-only Q27 unit checks,
M2 read-only `--check`, archive/manifests/hash/configuration and sample identity
gates, saved Q27 blind-prediction recomputation, and `git diff --check`. The staging
review additionally checks copied-source identity, focused CPU tests, ignore
coverage, the exact staged inventory and full staged contents. No full suite,
model call, retrieval, GPU job, commit, push, data deletion or algorithm refactoring
is part of this operation. Staging is explicitly authorized; committing is not.

Focused staging-review tests: 9 copied M2 unit tests, 12 selected Q27 unit tests,
and 4 Top-K tests passed (25 total, zero test failures). The bundled interpreter
lacks pytest/PyYAML, so the Top-K tests used the already-installed local environment
with those dependencies; no package installation was performed. M2's original
read-only reproduction check also passed. All three source/guide copies match
their original SHA256 exactly.

Research remains incomplete: independent abstract-only annotation; adequate
matched five-option question coverage; pre-specified missingness/tie/abstention
evaluation across natural retrieval; separation of soft-frequency and entropy
effects; unknown-quality sensitivity and error analysis. A single abstaining
question and the restricted 60-pair reference do not establish general QA effects.

Readiness: this is a narrow source/documentation milestone, with the two
portability-sensitive candidates omitted. A self-contained public
result-reproduction release is NOT ready: private frozen assets and their
redistribution permissions are not supplied/verified. Existing historical local
paths and the absence of a root license remain disclosed risks. README changes
remain a separate decision. A later explicit user instruction is required to
commit or push, regardless of staged-file safety.

Suggested commit message:

```text
Document Medical RAG Research v2.2 post-Q27 technical pilot
```
