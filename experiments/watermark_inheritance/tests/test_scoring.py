"""Scientific invariants of reading mode, calibration, and paired inference."""
import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from experiments.watermark_inheritance import score
from experiments.watermark_inheritance.report import CHECKPOINT_ORDER, make_report, table_rows


class ToyWatermark:
    ngram = 2
    gamma_exact = 0.25

    def __init__(self, tokenizer=None, config=None, key=None):
        self.key = key or 0

    def is_green(self, context, token):
        return int((sum(context)+token+self.key) % 4 == 0)


class EligibilityTests(unittest.TestCase):
    def test_first_completion_context_is_not_preemptively_excluded(self):
        self.assertEqual(list(score.eligible_predictions([1, 2], [3], [9])), [((1, 2), 9)])

    def test_prompt_and_completion_repeats_excluded(self):
        self.assertEqual(list(score.eligible_predictions([1, 2, 1, 2], [3, 1, 2, 4], [8, 9, 0, 1])),
                         [((2, 3), 9), ((3, 1), 0)])

    def test_prediction_scored_not_stored_token_and_state_resets(self):
        watermark = ToyWatermark()
        first, _ = score.example_watermark_score([1, 2], [3], [1], watermark)
        second, _ = score.example_watermark_score([1, 2], [3], [1], watermark)
        self.assertEqual(first, second)
        self.assertEqual(first["green"], 1)  # stored token 3 would be non-green
        self.assertAlmostEqual(first["z"], math.sqrt(3))

    def test_neutral_n0(self):
        result, records = score.example_watermark_score([1, 1, 1], [1, 1], [1, 1], ToyWatermark())
        self.assertEqual(result, {"z": 0.0, "green": 0, "eligible": 0})
        self.assertEqual(records, [])

    def test_global_tuple_dedup_is_separate_and_budget_is_exact(self):
        tape = score.AggregateRadioactivity(2, 0.25)
        tape.add([((1, 2), 1, 1), ((1, 2), 1, 1), ((1, 2), 2, 0), ((4, 5), 3, 1)])
        result = tape.result()
        self.assertEqual(result["eligible_tokens"], 2)
        self.assertEqual(result["green_tokens"], 1)
        other_side = score.AggregateRadioactivity(2, 0.25)
        other_side.add([((1, 2), 1, 1)])
        with self.assertRaisesRegex(ValueError, "fixed budget"):
            other_side.result()


class StatisticalTests(unittest.TestCase):
    def test_strict_threshold_and_tie_rule(self):
        result = score.calibrate_threshold([2, 3], [0, 1])
        self.assertEqual(result["threshold"], 1)
        self.assertEqual(result["balanced_accuracy"], 1)
        result = score.calibrate_threshold([1, 1], [1, 1])
        self.assertEqual(result["threshold"], 1)
        self.assertEqual(score.evaluate_shard([1, 1], [1, 1], 1)["a"], 0)

    def test_threshold_matches_exhaustive_search(self):
        for s in itertools.product(range(3), repeat=2):
            for sp in itertools.product(range(3), repeat=2):
                candidates = [math.nextafter(min(s+sp), -math.inf), *sorted(set(s+sp))]
                expected = max(candidates, key=lambda c: (sum(x > c for x in s)+sum(x <= c for x in sp), c))
                self.assertEqual(score.calibrate_threshold(s, sp)["threshold"], expected)

    def test_exact_paired_swap_matches_all_sign_permutations(self):
        for positive in range(6):
            for negative in range(6):
                discordant = positive+negative
                if not discordant:
                    continue
                observed = abs(positive-negative)
                p_exact = sum(abs(sum(signs)) >= observed for signs in itertools.product((-1, 1), repeat=discordant))/2**discordant
                result = score.paired_label_swap([1]*positive+[0]*negative+[1], [0]*positive+[1]*negative+[1])
                self.assertAlmostEqual(result["p_shard"], p_exact)
                self.assertEqual(result["decision_ties"], 1)

    def test_signed_gap_and_absolute_advantage_are_distinct(self):
        result = score.evaluate_shard([0, 0], [1, 1], 0)
        self.assertEqual(result["signed_gap"], -1)
        self.assertEqual(result["advantage"], 1)
        self.assertAlmostEqual(result["p_shard"], .5)

    def test_extreme_binomial_tail_has_no_floor(self):
        result = score.log10_binomial_survival(10000, 10000, .25)
        self.assertAlmostEqual(result, 10000*math.log10(.25), places=8)
        n, k = 1000, 950
        numerator = sum(math.comb(n, j)*3**(n-j) for j in range(k, n+1))
        expected = math.log10(numerator)-n*math.log10(4)
        self.assertAlmostEqual(score.log10_binomial_survival(k, n, .25), expected, places=9)
        self.assertEqual(score.log10_binomial_survival(0, 0, .25), 0)

    def test_subnormal_binomial_tail_keeps_log_precision(self):
        # P[X=n] = p**n is analytic. These tails are nonzero binary64 values,
        # but computing log(sf) has already lost significant digits.
        for count in (600, 610, 615):
            with self.subTest(count=count):
                expected = count * math.log10(.3)
                self.assertAlmostEqual(
                    score.log10_binomial_survival(count, count, .3), expected, places=10)

    def test_mink_uses_completion_bottom_fifth(self):
        self.assertEqual(score.min_k_score([-9, -8, -1, -1, -1, -1, -1, -1, -1, -1]), -8.5)
        self.assertEqual(score.min_k_score([-4, -1]), -4)
        self.assertEqual(score.min_k_score([]), 0)

    def test_mink_matches_repository_definition_at_every_supported_length(self):
        from src.shard_audit.scoring.mia_scores import min_k_logprob
        for length in range(1, 257):
            values = list(range(-length, 0))
            self.assertEqual(score.min_k_score(values), min_k_logprob(values, 20), length)


class InferenceTests(unittest.TestCase):
    def test_pilot_limit_keeps_exact_count_and_both_splits(self):
        pairs = [{"pair_id": i, "audit_split": "calibration" if i < 32 else "audit"} for i in range(128)]
        self.assertEqual(score.select_pilot_pairs(pairs, 128), pairs)
        selected = score.select_pilot_pairs(pairs, 40)
        self.assertEqual(len(selected), 40)
        self.assertEqual(sum(row["audit_split"] == "calibration" for row in selected), 10)
        for count in (1, 127):
            skewed = [{"pair_id": i, "audit_split": "calibration" if i < count else "audit"} for i in range(128)]
            selected = score.select_pilot_pairs(skewed, 120)
            self.assertEqual(len(selected), 120)
            self.assertEqual({row["audit_split"] for row in selected}, {"calibration", "audit"})

    def test_teacher_forcing_alignment_padding_and_valid_vocabulary(self):
        import torch

        class ToyModel:
            def __call__(self, input_ids, attention_mask, use_cache):
                self.assertions = (attention_mask.tolist(), use_cache)
                logits = torch.zeros((*input_ids.shape, 7))
                logits[:, :, 6] = 1000  # padded model output must never be predicted
                logits.scatter_(2, ((input_ids+1) % 5).unsqueeze(-1), 10)
                return SimpleNamespace(logits=logits)

        model = ToyModel()
        results = score.infer_batch(model, [([1, 2], [3, 4]), ([4], [0, 1])], "cpu", 5, 0)
        self.assertEqual(results[0]["predictions"], [3, 4])
        self.assertEqual(results[1]["predictions"], [0, 1])
        self.assertGreater(results[1]["min_k20"], -.001)
        self.assertEqual(model.assertions, ([[1, 1, 1, 1], [1, 1, 1, 0]], False))

    def test_record_alignment_rejects_wrong_pair_or_length(self):
        pair = {"pair_id": "p", "s_ids": [1], "s_prime_ids": [2]}
        with self.assertRaisesRegex(ValueError, "pair alignment"):
            score.validate_prediction_records([pair], [{"pair_id": "q"}])
        with self.assertRaisesRegex(ValueError, "token alignment"):
            score.validate_prediction_records([pair], [{"pair_id": "p", "s": {"predictions": []}}])

    def test_threshold_is_frozen_before_audit_decisions(self):
        pairs, predictions = [], []
        for index in range(4):
            pairs.append({"pair_id": index, "source_id": index, "audit_split": "calibration" if index < 2 else "audit",
                          "prompt_ids": [index+1, index+2], "s_ids": [3], "s_prime_ids": [4]})
            predictions.append({"pair_id": index, "s": {"predictions": [1], "min_k20": -1},
                                "s_prime": {"predictions": [2], "min_k20": -2}})
        config = {"run_id": "test", "pairs": 4, "calibration_pairs": 2,
                  "radio_token_budget": 1, "watermark_key": 0, "wrong_key": 1}
        original = score.evaluate_shard
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            def checked_evaluate(*args):
                self.assertTrue((directory/"thresholds.json").is_file())
                return original(*args)
            with patch.object(score, "Maryland", ToyWatermark), patch.object(score, "evaluate_shard", checked_evaluate):
                result = score.score_predictions(pairs, predictions, None, config, directory, {"x": 1}, "T")
                self.assertEqual(result["audit_pairs"], 2)
                self.assertEqual(len(result["radioactivity"]), 4)
                rows = table_rows([result])
                self.assertEqual(sum(row["primary"] for row in rows), 1)
                frozen = json.loads((directory/"thresholds.json").read_text())
                self.assertEqual(frozen["detectors"]["watermark"]["calibration_pairs"], 2)
                # Reversing final-audit predictions cannot change the calibration.
                predictions[-1]["s"], predictions[-1]["s_prime"] = predictions[-1]["s_prime"], predictions[-1]["s"]
                score.score_predictions(pairs, predictions, None, config, directory, {"x": 1}, "T")
                self.assertEqual(json.loads((directory/"thresholds.json").read_text()), frozen)
                # Exercise the actual table/plot writer on a full six-checkpoint
                # synthetic fixture; a completed-looking partial run must fail.
                run_dir = directory/"run"
                score.write_json(run_dir/"config.json", config)
                score_dir = run_dir/"scores"/"T"
                score.write_json(score_dir/"metrics.json", result)
                score.write_json(score_dir/"thresholds.json", frozen)
                with self.assertRaisesRegex(ValueError, "Missing final checkpoint"):
                    make_report([run_dir])
                for name in CHECKPOINT_ORDER:
                    score.write_json(run_dir/"scores"/name/"metrics.json",
                                     {**result, "checkpoint": name, "primary_comparison": name == "T"})
                    score.write_json(run_dir/"scores"/name/"thresholds.json", frozen)
                report = make_report([run_dir])
                self.assertTrue(report["complete"])
                self.assertEqual(len(report["table"]), 48)
                for name in ("main_results.csv", "results.csv", "results.json", "advantage.png",
                             "advantage.pdf", "radioactivity.png", "radioactivity.pdf"):
                    self.assertGreater((run_dir/"reports"/name).stat().st_size, 20)


if __name__ == "__main__":
    unittest.main()
