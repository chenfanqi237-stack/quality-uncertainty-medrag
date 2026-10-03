# Abstract-only stance review guide

Review each of the 15 rows independently using only its question, candidate answer, evidence title and abstract. Judge the evidence–candidate relationship in the question's context, not which answer is medically correct from memory. Do not consult answer keys, full articles, external sources, prior annotations or research analyses. Work independently before discussing judgments with other reviewers.

## Three stance labels

Enter exactly one of these in `reviewer_stance`:

- **SUPPORT:** The evidence provides information that supports the candidate answer in the context of the medical question.
- **CONTRADICT:** The evidence provides information that conflicts with or argues against the candidate answer in the context of the medical question.
- **IRRELEVANT:** The evidence does not provide sufficient information to support or contradict the candidate answer. This includes clearly unrelated evidence and related evidence that cannot establish either direction.

Shared medical words alone do not establish SUPPORT. Missing support is not automatically CONTRADICT. Consider the reported population, intervention, comparator and outcome when deciding whether a finding applies to the contextual candidate. Differences do not automatically imply irrelevance, and matching every demographic detail is not required. Do not supply missing facts from outside knowledge or treat an unreported outcome as a negative finding. Distinguish an explicitly negative finding from an absence of adequate evidence about an effect.

## INSUFFICIENT INFORMATION: a separate review flag

`insufficient_evidence_flag` is **not a fourth stance label**. Enter `TRUE` when information necessary to establish contextual SUPPORT or CONTRADICT is unavailable or ambiguous in the supplied title/abstract; use IRRELEVANT and explain what is missing. Enter `FALSE` when the relationship can be judged without that information gap, including clear off-topic irrelevance. A blank flag means not yet reviewed, not FALSE.

Do not enter `INSUFFICIENT INFORMATION` in `reviewer_stance`. This flag distinguishes inadequate evidence from clear irrelevance without changing the frozen three-class schema.

## Completing the CSV

1. Save your own copy as `q27_review_reviewer_<ID>.csv`. Preserve the supplied master.
2. Edit only `reviewer_stance`, `reviewer_reasoning` and `insufficient_evidence_flag`. Keep pair identities and all source text unchanged.
3. Give a brief evidence-based rationale in `reviewer_reasoning`, citing the relevant phrase or explaining the missing link. Record material applicability ambiguity rather than guessing the intended answer.
4. Check that all 15 rows have your three review fields completed before returning your copy. Initial judgments must come from you; disagreements can be adjudicated separately afterward.

The CSV uses UTF-8, comma delimiters and quoted multiline fields. Import it as UTF-8/comma-separated text if your spreadsheet application does not display it correctly. Preserve PMID and pair identity as text, enable text wrapping to read the complete question/abstract, and save back to CSV without changing source contents. Blank review fields are intentional; no reference annotations have been supplied.
