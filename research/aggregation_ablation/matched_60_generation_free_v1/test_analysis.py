"""Focused offline boundaries and real frozen-input validation; no inference."""

import csv
import io
import math
import unittest

import analysis as a


def fixture(p, q=1.0, doc="1"):
    """Explicitly synthetic numerical unit-test input, never a clinical result."""
    return {"key": ("synthetic", "A", doc), "pair_id": doc, "quality": q,
            "p": {s: float(v) for s, v in zip(a.Stance, p)},
            "counts": {s.value: round(v * 10) for s, v in zip(a.Stance, p)},
            "raw": {"schema_version": "synthetic-test", "question_id": "synthetic", "doc_id": doc,
                    "rank": int(doc), "text": "synthetic test", "source": "synthetic-test",
                    "evidence_type": "other", "retrieval_score": 0,
                    "metadata": {"title": "synthetic", "abstract": "synthetic", "publication_types": ["Journal Article"]}},
            "input": {"question_stem": "synthetic numerical test", "candidate_option_text": "synthetic claim"}}


class FocusedTests(unittest.TestCase):
    def test_unique_argmax_tie_is_unresolved(self):
        result, labels = a.evaluate([fixture((.5, .5, 0))], "quality_disabled")
        self.assertEqual(labels, [None])
        self.assertTrue(all(x["decision"] == "ABSTAIN" for x in result.values()))

    def test_all_zero_quality_undefined_not_zero(self):
        result, _ = a.evaluate([fixture((1, 0, 0), 0)], "original_quality")
        self.assertEqual(result["A"]["decision"], "SUPPORT")
        for m in "BCD":
            self.assertIsNone(result[m]["score"])
            self.assertEqual(result[m]["decision"], "ABSTAIN")

    def test_single_contributor_entropy_cancels(self):
        for group in ([fixture((.6, .3, .1), .2)], [fixture((.6, .3, .1), .2), fixture((0, 1, 0), 0, "2")]):
            result, _ = a.evaluate(group, "original_quality")
            self.assertAlmostEqual(result["C"]["score"], .3, delta=a.TOL)
            self.assertAlmostEqual(result["D"]["score"], .3, delta=a.TOL)

    def test_uniform_frequency_zero_effective_denominator(self):
        result, _ = a.evaluate([fixture((1/3, 1/3, 1/3))], "quality_disabled")
        self.assertEqual(result["C"]["score"], 0)
        self.assertIsNone(result["D"]["score"])
        self.assertEqual(result["D"]["abstention_reason"], "ZERO_ENTROPY_EFFECTIVE_WEIGHT")

    def test_synthetic_entropy_can_change_sign(self):
        group = [fixture((.4, .2, .4)), fixture((0, .1, .9), doc="2")]
        result, _ = a.evaluate(group, "quality_disabled")
        self.assertEqual(result["C"]["decision"], "SUPPORT")
        self.assertEqual(result["D"]["decision"], "CONTRADICT")

    def test_numerical_comparison_does_not_rewrite_native_sign(self):
        self.assertEqual(a.signed_decision(1e-17), "SUPPORT")
        self.assertEqual(a.signed_decision(1e-17, a.TOL), "ABSTAIN")
        self.assertRaises(ValueError, a.probabilities, {label: 0 for label in a.LABELS})

    def test_source_conditional_direction_is_not_soft_stance(self):
        row = {"n_support": "9", "n_contradict": "0", "n_irrelevant": "1",
               "p_support": ".9", "p_contradict": "0", "p_irrelevant": ".1",
               "u_3": "0.29590327428938457", "directional_score": "1.0"}
        _, p = a.validate_frequency(row)
        self.assertEqual(p[a.Stance.SUPPORT] - p[a.Stance.CONTRADICT], .9)

    def test_real_frozen_join_and_all_zero_stratum(self):
        groups, samples, _, _ = a.load_inputs()
        self.assertEqual(sum(map(len, groups.values())), 60)
        self.assertEqual(sum(map(len, samples.values())), 600)
        positive = [sum(p["quality"] > 0 for p in g) for g in groups.values()]
        self.assertEqual(a.Counter(positive), {0: 41, 1: 9})
        for group in groups.values():
            disabled, _ = a.evaluate(group, "quality_disabled")
            self.assertEqual(disabled["A"]["score"], disabled["B"]["score"])
            self.assertEqual(disabled["A"]["decision"], disabled["B"]["decision"])
            original, _ = a.evaluate(group, "original_quality")
            if original["C"]["score"] is not None:
                self.assertAlmostEqual(original["C"]["score"], original["D"]["score"], delta=a.TOL)

    def test_csv_projection_excludes_labels_and_preserves_missingness(self):
        groups, samples, _, _ = a.load_inputs()
        rows, _, _ = a.analyze(groups, samples)
        exported = list(csv.DictReader(io.StringIO(a.csv_text(rows))))
        self.assertEqual(len(exported), 100)
        self.assertFalse(any("reference" in field or "gold" in field for field in exported[0]))
        zero = [r for r in exported if r["condition"] == "original_quality" and r["analysis_positive_weight_count"] == "0"]
        self.assertEqual(len(zero), 41)
        self.assertTrue(all(r["B_score"] == "" and r["D_score_defined"] == "False" for r in zero))


if __name__ == "__main__":
    unittest.main()
