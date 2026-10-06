"""The paper bands must reflect sample SD across complete matched runs."""
import unittest

from experiments.watermark_inheritance.paper_curve import summarize


def rows():
    return [{"run_id": f"run_{run:02d}", "stage": stage, "steps": 512,
             "examples_seen": 16384, "unique_examples_seen": 16384,
             "watermark_advantage": value}
            for stage, values in (("target", (.2, .4, .6)), ("control", (.1, .1, .1)))
            for run, value in enumerate(values)]


class PaperCurveTests(unittest.TestCase):
    def test_band_uses_sample_sd_not_population_sd_or_standard_error(self):
        summary, run_ids, excluded = summarize(rows())
        self.assertEqual(len(run_ids), 3)
        self.assertEqual(excluded, [])
        self.assertAlmostEqual(summary[0]["mean_advantage"], .4)
        self.assertAlmostEqual(summary[0]["std_advantage"], .2)
        self.assertEqual(summary[1]["std_advantage"], 0)

    def test_partial_checkpoint_does_not_change_replicate_count(self):
        partial = dict(rows()[0], steps=1024, examples_seen=32768, unique_examples_seen=32768)
        summary, _, excluded = summarize(rows() + [partial])
        self.assertEqual([r["steps"] for r in summary], [512, 512])
        self.assertEqual(excluded, [1024])

    def test_duplicate_or_mismatched_exposure_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summarize(rows() + [rows()[0]])
        mismatched = rows()
        mismatched[0]["examples_seen"] += 1
        with self.assertRaisesRegex(ValueError, "Mismatched"):
            summarize(mismatched)


if __name__ == "__main__":
    unittest.main()
