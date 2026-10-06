# Dev30 statistical interval audit v1

Reporting addendum only. The original predictions, results, summaries and frozen protocol are unchanged. Exact inputs, checks and full-precision bounds are in `statistical_interval_audit_v1.json`; source/result hashes are in `result_freeze_review_v1.json`.

## Paired counts first

For unit-weight matched B versus C, N=30: both answered 13, neither answered 11, hard-only 0, soft-only 6. All six changes are ABSTAIN to ANSWER. Four yield a correct selected answer and two yield an incorrect selected answer. There are no answer-option swaps or ANSWER to ABSTAIN changes.

Observed coverage difference: +6/30 = +20 percentage points. Observed overall-accuracy difference: +4/30 = +13.3333 percentage points. Correct selected/N uses a binary correct-selection indicator: an abstention contributes no correct selection, but is not a wrong selected answer. The paired correctness table is both correct 7, soft-only correct 4, hard-only correct 0, neither correct 19.

## Individual proportions

All 24 published accuracy, coverage and answered-accuracy intervals in the eight-row method table were independently recomputed and match. They are 95% marginal Clopper-Pearson intervals (each tail .025), using denominators 30, 30 and answered count respectively. These are descriptive working-model intervals, not independent confirmation, clinical-population coverage guarantees or simultaneous intervals over all methods. Answered accuracy and its interval would be undefined at zero answers.

## Correct wording for the locked coverage interval

The existing [-6.9793, +41.1624] percentage-point interval is a **CONSERVATIVE MARGINAL-BOUND INTERVAL**. The frozen evaluator takes separate 97.5% exact CP intervals for the soft-only rate (6/30) and hard-only rate (0/30), then reports [Lplus-Uminus, Uplus-Lminus]. The Bonferroni argument provides at least nominal 95% simultaneous coverage under the stated iid/common-distribution question working model. It does not assume independence of these two indicators within a question, but does not exploit their paired covariance.

The original summary's phrase "conservative paired95% working-model coverage interval" must not be reused as "exact paired CP confidence interval". The construction is documented correctly in the frozen protocol/evaluator; the label in future manuscript reporting should be explicit. This addendum supersedes ambiguous wording only, without editing frozen files or altering the primary endpoint.

## Separate paired accuracy interval (post-hoc reporting)

A standard paired interval was computed separately: Newcombe's 1998 hybrid score **method 10**, using uncorrected Wilson marginal score bounds and the specified continuity correction to positive phi. On the paired correctness cells (7,4,0,19), the nominal approximate 95% interval for soft minus hard accuracy is **[-0.5788, +26.9770] percentage points**. It is not exact CP and does not replace raw paired counts or the locked primary coverage reporting. No significance test or superiority claim is made.

The implementation follows section 5, method 10, p.2639, and was checked against published Table III examples on p.2641. [Newcombe (1998), original paper, Statistics in Medicine 17:2635–2650](https://www.site.uottawa.ca/~nat/Courses/csi5388/Newcombe.1998.pdf), [publisher DOI](https://doi.org/10.1002/(SICI)1097-0258(19981130)17:22%3C2635::AID-SIM954%3E3.0.CO;2-C).

Both interval constructions are working-model descriptions for an already inspected, fixed DEVELOPMENT set. They do not correct pilot inspection, post-hoc selection or multiple comparisons and cannot establish safer or superior medical QA. The finite-cohort differences themselves are directly observed.
