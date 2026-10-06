"""Data isolation, loss masks, overlap matching, and generation resume tests."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.watermark_inheritance.common import write_jsonl
from experiments.watermark_inheritance.generate import pair_record, read_journal, sampling_seed, validate_pair
from experiments.watermark_inheritance.prepare import (
    CompletionIndex, encode_dolly, normalize_text, prepare_dolly, select_prompts,
    validate_tokenizer, verify_upstream,
)


class CharacterTokenizer:
    eos_token_id = 1000
    vocab_size = 1001

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def __len__(self):
        return self.vocab_size

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids if not skip_special_tokens or i != self.eos_token_id)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()

    def test_article_groups_and_pilot_are_disjoint(self):
        rows = []
        for article in range(20):
            rows += [{"text": f" = Article {article} = "}, {"text": " = = Section = = "},
                     {"text": f"Article {article:03d} " + "abc " * 30},
                     {"text": f"Another long paragraph for article {article:03d} " + "def " * 30}]
        main, pilot, meta = select_prompts(rows, self.tokenizer, count=10, pilot_count=4,
                                            calibration_count=2, prompt_tokens=64, seed=10)
        self.assertEqual(len(main), 10)
        self.assertEqual(sum(row["audit_split"] == "calibration" for row in main), 2)
        self.assertEqual(len({row["article_id"] for row in main + pilot}), 14)
        self.assertTrue(all(len(row["prompt_ids"]) == 64 for row in main + pilot))
        self.assertEqual(len({tuple(row["prompt_ids"]) for row in main + pilot}), 14)
        self.assertEqual(meta["eligible_articles"], 20)
        self.assertEqual(select_prompts(rows, self.tokenizer, 10, 4, 2, 64, 10)[:2], (main, pilot))

    def test_response_preserved_before_context(self):
        row = {"instruction": "Explain", "context": "context " * 100, "response": "Original answer"}
        result = encode_dolly(row, self.tokenizer, max_length=100)
        supervised = [label for label in result["labels"] if label != -100]
        self.assertEqual(supervised, self.tokenizer.encode(row["response"]) + [1000])
        self.assertTrue(result["context_truncated"])
        self.assertFalse(result["response_truncated"])
        self.assertFalse(result["instruction_truncated"])
        self.assertLessEqual(len(result["input_ids"]), 100)
        self.assertEqual(result["labels"][0], -100)

    def test_long_response_has_reported_last_resort_truncation(self):
        result = encode_dolly({"instruction": "Explain", "response": "x" * 2000}, self.tokenizer, 100)
        self.assertTrue(result["response_truncated"])
        self.assertEqual(len(result["input_ids"]), 100)
        self.assertEqual(result["input_ids"][-1], self.tokenizer.eos_token_id)
        self.assertGreater(result["response_tokens"], 50)

    def test_index_finds_full_inclusion_and_normalized_inclusion(self):
        phrase = "This is a generated full completion with several words."
        index = CompletionIndex([phrase, "tiny", "", "\n\n"])
        self.assertTrue(index.contains_completion("prefix " + phrase + " suffix"))
        self.assertFalse(index.contains_completion("a tiny result"))
        self.assertTrue(index.contains_completion("tiny"))
        self.assertFalse(CompletionIndex(["."]).contains_completion("A normal answer."))
        self.assertFalse(index.contains_completion(phrase[:32] + "other ending"))
        self.assertFalse(index.contains_completion("unrelated\n\n"))
        normalized = CompletionIndex([normalize_text("This  is\nMY completion")])
        self.assertTrue(normalized.contains_completion(normalize_text("THIS IS MY COMPLETION")))

    def test_dolly_excludes_both_shards_and_records_exact_input_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pairs = [{"s_text": "original private completion with enough unique text", "s_prime_text": "Audit PRIVATE Completion"}]
            write_jsonl(root / "prepared/pairs.jsonl", pairs)
            rows = [{"instruction": "First", "response": "before original private completion with enough unique text after"},
                    {"instruction": "Second", "response": "audit private   completion"},
                    {"instruction": "Third", "response": "Remaining original answer"}]
            manifest = prepare_dolly(root, {"pairs": 1, "max_length": 100}, self.tokenizer, rows)
            self.assertTrue(manifest["overlap_checked"])
            self.assertEqual(manifest["eligible_rows"], 1)
            self.assertEqual(manifest["exact_overlap_excluded"], 1)
            self.assertEqual(manifest["normalized_overlap_excluded"], 1)
            self.assertEqual(len(manifest["pair_file_sha256"]), 64)

    def test_paired_seeds_assignment_ties_and_resume_validation(self):
        prompt = {"pair_id": "pair_00000", "source_id": "source", "audit_split": "audit",
                  "prompt_ids": [65, 66], "prompt_text": "AB"}
        config = {"seed": 1, "watermark_key": 55}
        record = pair_record(prompt, [[67, 1000], [67, 1000]], self.tokenizer, config, "full")
        self.assertTrue(record["tied"])
        self.assertNotEqual(record["sampling_seed_s"], record["sampling_seed_s_prime"])
        self.assertNotEqual(sampling_seed(1, "full", "pair_00000", 0),
                            sampling_seed(1, "pilot", "pair_00000", 0))
        validate_pair(record, prompt, config, "full", 1001)
        with self.assertRaises(ValueError):
            validate_pair(record, prompt, dict(config, seed=2), "full", 1001)

    def test_only_partial_tail_is_recovered(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "journal.jsonl"
            path.write_bytes(b'{"pair_id":"ok"}\n{"pair_id":')
            self.assertEqual(read_journal(path), [{"pair_id": "ok"}])
            self.assertEqual(path.read_bytes(), b'{"pair_id":"ok"}\n')
            path.write_bytes(b'{"pair_id":"ok"}')
            self.assertEqual(read_journal(path), [{"pair_id": "ok"}])
            self.assertTrue(path.read_bytes().endswith(b"\n"))
            path.write_bytes(b'{broken}\n')
            with self.assertRaises(ValueError):
                read_journal(path)

    def test_changed_upstream_revision_or_dirty_source_is_rejected(self):
        config = {"upstream_path": "/tmp/upstream", "upstream_commit": "pinned"}
        with patch("subprocess.check_output", side_effect=["pinned\n", ""]):
            self.assertEqual(verify_upstream(config, "pinned")["revision"], "pinned")
        with patch("subprocess.check_output", side_effect=["changed\n", ""]):
            with self.assertRaisesRegex(RuntimeError, "revision changed"):
                verify_upstream(config, "pinned")
        with patch("subprocess.check_output", side_effect=["pinned\n", " M wm/detector.py"]):
            with self.assertRaisesRegex(RuntimeError, "modified"):
                verify_upstream(config, "pinned")

    def test_real_added_tokenizer_vocabulary_is_allowed_but_specials_must_fit(self):
        validate_tokenizer(self.tokenizer)
        class AddedTokenizer(CharacterTokenizer):
            def __len__(self):
                return self.vocab_size + 1
        validate_tokenizer(AddedTokenizer())
        invalid = AddedTokenizer()
        invalid.eos_token_id = 2000
        with self.assertRaisesRegex(ValueError, "outside the watermark vocabulary"):
            validate_tokenizer(invalid)


if __name__ == "__main__":
    unittest.main()
