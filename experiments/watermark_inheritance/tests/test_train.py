"""Scientific data boundaries and accumulation invariants for watermark training."""
import unittest

from experiments.watermark_inheritance.train import (
    collate, dolly_examples, logical_batches, parent_examples, response_example, valid_vocab_loss,
)


class TrainingDataTests(unittest.TestCase):
    def test_parent_uses_all_s_and_never_s_prime(self):
        pairs = [
            {"prompt_ids": [1, 2], "s_ids": [3, 0], "s_prime_ids": [900], "audit_split": "calibration"},
            {"prompt_ids": [4, 5], "s_ids": [6, 0], "s_prime_ids": [901], "audit_split": "audit"},
        ]
        examples = parent_examples(pairs, 1024)
        self.assertEqual([r["input_ids"] for r in examples], [[1, 2, 3, 0], [4, 5, 6, 0]])
        self.assertEqual([r["labels"] for r in examples], [[-100, -100, 3, 0], [-100, -100, 6, 0]])
        # No second EOS is appended; no text or S-prime tokenization occurs.
        pairs[0]["s_prime_ids"] = None
        self.assertEqual(parent_examples(pairs, 1024), examples)

    def test_prompt_is_trimmed_before_response(self):
        row = response_example([1, 2, 3, 4, 5], [6, 7, 8], 5)
        self.assertEqual(row["input_ids"], [4, 5, 6, 7, 8])
        self.assertEqual(row["labels"], [-100, -100, 6, 7, 8])
        self.assertEqual(row["response_tokens"], 3)

    def test_long_response_retains_one_causal_context_token(self):
        row = response_example([1, 2], [3, 4, 5, 6, 7, 8], 4)
        self.assertEqual(row["input_ids"], [2, 3, 4, 5])
        self.assertEqual(row["labels"], [-100, 3, 4, 5])

    def test_dolly_preserves_response_and_discards_padding(self):
        rows = [{"input_ids": [1, 2, 3, 4, 5, 0, 0], "labels": [-100, -100, -100, 4, 5, -100, -100]}]
        self.assertEqual(dolly_examples(rows, 3), [{"input_ids": [3, 4, 5], "labels": [-100, 4, 5], "response_tokens": 2}])

    def test_invalid_or_empty_responses_fail_loudly(self):
        for ids, labels in [([1, 2], [-100, -100]), ([1, 2], [1, 2]),
                            ([1, 2, 3, 4], [-100, 2, -100, 4]), ([1, 2], [-100, 3])]:
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                dolly_examples([{"input_ids": ids, "labels": labels}], 1024)
        with self.assertRaises(ValueError):
            response_example([1], [], 1024)

    def test_accumulation_preserves_every_example_and_short_tail(self):
        result = list(logical_batches(list(range(67)), 32, 6))
        self.assertEqual([sum(map(len, batch)) for batch in result], [32, 32, 3])
        self.assertTrue(all(len(micro) <= 6 for batch in result for micro in batch))
        self.assertEqual([i for batch in result for micro in batch for i in micro], list(range(67)))

    def test_padding_masks_do_not_hide_real_eos_tokens(self):
        import torch
        rows = [response_example([1], [2, 0], 8), response_example([1], [3], 8)]
        inputs, labels = collate(rows, pad_token_id=0, device="cpu")
        self.assertEqual(inputs["attention_mask"].tolist(), [[1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0]])
        self.assertEqual(labels.tolist(), [[-100, 2, 0, -100, -100, -100, -100, -100], [-100, 3, -100, -100, -100, -100, -100, -100]])
        self.assertEqual(int((labels[:, 1:] != -100).sum()), 3)

    def test_ce_ignores_padded_vocabulary_prompt_and_padding(self):
        import torch
        logits = torch.zeros(1, 5, 7, requires_grad=True)
        logits.data[:, :, 5:] = 1000  # Impossible padded-vocabulary IDs.
        labels = torch.tensor([[-100, -100, 2, 3, -100]])
        loss = valid_vocab_loss(logits, labels, vocab_size=5)
        self.assertAlmostEqual(float(loss.detach()), 2 * __import__("math").log(5), places=5)
        loss.backward()
        self.assertEqual(float(logits.grad[:, :, 5:].abs().sum()), 0)
        self.assertEqual(float(logits.grad[:, [0, 3, 4], :].abs().sum()), 0)

    def test_microbatch_token_weighting_matches_combined_batch(self):
        import torch
        torch.manual_seed(3)
        logits = torch.randn(3, 4, 6, requires_grad=True)
        labels = torch.tensor([[-100, 1, 2, -100], [-100, 3, -100, -100], [-100, 2, 4, 1]])
        count = int((labels[:, 1:] != -100).sum())
        (valid_vocab_loss(logits, labels, 5) / count).backward()
        combined = logits.grad.clone()
        logits.grad.zero_()
        for start, stop in [(0, 2), (2, 3)]:
            (valid_vocab_loss(logits[start:stop], labels[start:stop], 5) / count).backward()
        torch.testing.assert_close(logits.grad, combined)


if __name__ == "__main__":
    unittest.main()
