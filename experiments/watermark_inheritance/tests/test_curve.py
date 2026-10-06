"""Continuous trajectories must preserve data order, exposure and provenance."""
import json
from pathlib import Path
import tempfile
import unittest

from experiments.watermark_inheritance.curve_train import batch_indices, checkpoint_schedule, validate_parent, train
from experiments.watermark_inheritance.train import _identity, _digest


class CurveTests(unittest.TestCase):
    def test_schedule_uses_optimizer_boundaries_and_includes_endpoint(self):
        self.assertEqual(checkpoint_schedule([2, 4], 4), [2, 4])
        for steps in ([0, 4], [2, 2, 4], [4, 2], [2, 5], [2]):
            with self.assertRaises(ValueError):
                checkpoint_schedule(steps, 4)

    def test_resumed_order_has_no_skipped_or_repeated_examples(self):
        whole = list(batch_indices(13, 4, 7, 0, 12))
        resumed = list(batch_indices(13, 4, 7, 5, 12))
        self.assertEqual(whole[5:], resumed)
        for start in (0, 4, 8):
            self.assertEqual(sorted(i for _, indices in whole[start:start + 4] for i in indices), list(range(13)))
        self.assertEqual([len(indices) for _, indices in whole], [4, 4, 4, 1] * 3)

    def test_parent_validates_original_configuration_and_pair_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old, new = root / "old", root / "new"
            (old / "prepared").mkdir(parents=True)
            (new / "prepared").mkdir(parents=True)
            parent = old / "checkpoints/parent"
            parent.mkdir(parents=True)
            config = {"seed": 3, "watermark_key": 4, "model_path": "base", "tokenizer_path": "base",
                      "epochs": 3, "ngram": 2, "gamma": .25, "salt_key": 35317}
            (old / "config.json").write_text(json.dumps(config))
            for run in (old, new):
                (run / "prepared/pairs.jsonl").write_text('{"id":1}\n')
            state = {"identity": _identity(config, old / "prepared/pairs.jsonl", "parent"),
                     "complete": True, "epoch_complete": True, "epoch": 3}
            (parent / "stage_state.json").write_text(json.dumps(state))
            new_config = dict(config, parent_run_dir=str(old), parent_config_sha256=_digest(old / "config.json"))
            self.assertEqual(validate_parent(new, new_config), parent)
            (new / "prepared/pairs.jsonl").write_text('{"id":2}\n')
            with self.assertRaises(ValueError):
                validate_parent(new, new_config)
            (new / "prepared/pairs.jsonl").write_text('{"id":1}\n')
            new_config["watermark_key"] = 5
            with self.assertRaises(ValueError):
                validate_parent(new, new_config)

    def test_mid_epoch_resume_matches_uninterrupted_training_exactly(self):
        import torch
        from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from experiments.watermark_inheritance.common import write_json, write_jsonl
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, original = root / "base", root / "original"
            parent = original / "checkpoints/parent"
            base.mkdir()
            parent.mkdir(parents=True)
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
                {"<eos>": 0, "<unk>": 1, **{f"v{i}": i for i in range(2, 32)}}, unk_token="<unk>")),
                eos_token="<eos>", unk_token="<unk>", pad_token="<eos>")
            tokenizer.save_pretrained(base)
            torch.manual_seed(91)
            model = GPTNeoXForCausalLM(GPTNeoXConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=32,
                rotary_pct=.5, hidden_dropout=.1, attention_dropout=.1))
            model.save_pretrained(base)
            model.save_pretrained(parent)
            tokenizer.save_pretrained(parent)
            pairs = original / "prepared/pairs.jsonl"
            write_jsonl(pairs, [{"pair_id": 1}])
            config = {"model_path": str(base), "tokenizer_path": str(base), "seed": 9, "epochs": 2,
                      "max_length": 16, "effective_batch_size": 2, "train_micro_batch": 1,
                      "learning_rate": .001, "weight_decay": .01, "warmup_ratio": .03, "pilot": True}
            write_json(original / "config.json", config)
            write_json(parent / "stage_state.json", {"identity": _identity(config, pairs, "parent"),
                       "complete": True, "epoch_complete": True, "epoch": 2})
            clean = root / "clean.jsonl"
            write_jsonl(clean, [{"input_ids": [2, 3 + i, 10 + i, 0], "labels": [-100, -100, 10 + i, 0]}
                               for i in range(5)])
            manifest = root / "manifest.json"
            write_json(manifest, {"overlap_checked": True, "output_file_sha256": _digest(clean),
                       "pair_sha256": {str(pairs): _digest(pairs)}, "eligible_rows": 5})
            config.update(parent_run_dir=str(original), parent_config_sha256=_digest(original / "config.json"),
                          clean_data_path=str(clean), clean_manifest_path=str(manifest), clean_rows=5,
                          checkpoint_steps=[2, 3, 6], max_steps=6)
            runs = [root / name for name in ("uninterrupted", "resumed")]
            for run in runs:
                write_json(run / "config.json", config)
                (run / "prepared").mkdir()
                (run / "prepared/pairs.jsonl").symlink_to(pairs)
            uninterrupted = train(runs[0], "target", "cpu")
            partial = train(runs[1], "target", "cpu", stop_after_step=2)
            self.assertFalse(partial["epoch_complete"])
            self.assertTrue(partial["checkpoint_complete"])
            resumed = train(runs[1], "target", "cpu")
            for key in ("examples_seen", "response_tokens_seen", "input_tokens_seen", "loss_sum", "learning_rate"):
                self.assertEqual(uninterrupted[key], resumed[key])
            self.assertEqual(resumed["examples_seen"], 10)
            self.assertEqual(resumed["response_tokens_seen"], 20)
            left, right = [GPTNeoXForCausalLM.from_pretrained(run / "checkpoints/target_step6").state_dict() for run in runs]
            for key in left:
                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
            self.assertFalse((runs[1] / "checkpoints/target_step2/training_state.pt").exists())


if __name__ == "__main__":
    unittest.main()
