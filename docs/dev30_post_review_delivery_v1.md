# Independent stance review delivery and return

The current mechanism review is pending and post-hoc explanatory. Ninety exact pairs are required; zero historical labels have verified pre-outcome independent-blind eligibility. No review results are reported.

Give an outcome-unaware independent reviewer only the private delivery ZIP and its SHA256 sidecar. The ZIP contains exactly a blank anonymous review CSV and its annotation guide. Do not provide the manuscript, GitHub research reports, model outputs, answer keys, private identity mapping, previous annotations or transition identities. A CSV containing supplied medical evidence is not public-safe merely because its IDs are anonymous.

The reviewer edits only stance, confidence and notes. Labels are SUPPORT/CONTRADICT/IRRELEVANT and confidence HIGH/MEDIUM/LOW. Use only the supplied question, candidate, evidence title and abstract; do not consult external clinical knowledge or full texts. Return `dev30_disagreement_review_reviewer_<ID>_v1.csv` as a separate file. The coordinator records its SHA256 and timezone-aware submission time plus attestations of independent human origin, outcomes unseen, not the primary researcher and supplied-text-only review. Do not claim clinical adjudication without clinical qualification.

Future generation-free merge is implemented in `cloud/dev30_disagreement_review_v1.py`. It requires a private submission JSON with `reviewer_id`, `review_csv`, `review_csv_sha256`, `reviewed_at`, `independent_human`, `outcomes_unseen`, `not_primary_researcher`, `supplied_text_only`, and `annotation_origin="independent_human"`. Paths are resolved relative to that submission file. This metadata is not included in the outgoing reviewer ZIP.

From the project root, after independent labels are actually returned:

```text
python -B cloud/dev30_disagreement_review_v1.py merge --submission PRIVATE_SUBMISSION.json --output NEW_PRIVATE_MERGE.json
```

Supply another `--submission` for a second independent reviewer. Blank labels, changed source text, extra identities, fourth stance labels, duplicate submissions or missing independence attestations block analysis. Disagreements are retained, not overwritten; Cohen's kappa requires two reviewers and is undefined in degenerate cases. Model-frequency agreement and minority-direction indicators are descriptive, not calibration or clinical truth. All returned annotations and merge outputs remain private. Do not run the merge while the review template is blank.
