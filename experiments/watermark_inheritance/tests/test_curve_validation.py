"""Check that numerical verification distinguishes roundoff from bad results."""
import math
import unittest

from experiments.watermark_inheritance.curve_validate import verify_log10_probability


class ProbabilityValidationTests(unittest.TestCase):
    def test_legacy_subnormal_tail_records_analytic_value(self):
        # Saved outputs from the frozen SciPy 1.16.3 log(sf) implementation.
        for count, stored in ((610, -318.95602549311644), (615, -321.5738215832928)):
            expected = count * math.log10(.3)
            record = verify_log10_probability(stored, expected)
            self.assertEqual(record["recomputed_log10_p"], expected)
            self.assertEqual(record["stored_log10_p"], stored)

    def test_two_sided_tail_roundoff_uses_one_sided_quantum(self):
        expected = math.log10(2) + 615 * math.log10(.3)
        stored = math.log10(2) - 321.5738215832928
        self.assertEqual(verify_log10_probability(stored, expected, factor=2)["factor"], 2)

    def test_discrepancies_outside_subnormal_roundoff_fail(self):
        for stored, expected in ((-2.1, -2), (-318.9, -319.2), (-400, -401)):
            with self.subTest(stored=stored), self.assertRaises(AssertionError):
                verify_log10_probability(stored, expected)

    def test_matching_normal_and_extreme_logs_need_no_exception(self):
        for value in (-2., -1000.):
            self.assertIsNone(verify_log10_probability(value, value))


if __name__ == "__main__":
    unittest.main()
