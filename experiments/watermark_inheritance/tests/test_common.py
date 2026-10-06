import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from experiments.watermark_inheritance.common import Maryland


class UpstreamCompatibilityTests(unittest.TestCase):
    def test_added_token_ids_belong_to_greenlist_domain(self):
        class AddedTokenizer:
            vocab_size = 78
            def __len__(self):
                return 101
        upstream = Path(__file__).resolve().parents[3] / ".cache/watermark/radioactive-watermark"
        config = dict(upstream_path=str(upstream), watermark_key=101)
        adapter = Maryland(AddedTokenizer(), config)
        self.assertEqual(adapter.vocab_size, 101)
        self.assertEqual(adapter.gamma_exact, 25 / 101)
        self.assertIn(adapter.is_green((99, 100), 100), (0, 1))

    def test_greenlist_and_scores_equal_authors_code(self):
        import torch
        upstream = Path(__file__).resolve().parents[3] / ".cache/watermark/radioactive-watermark"
        sys.path.insert(0, str(upstream))
        from wm.detector import MarylandDetector
        from wm.generator import MarylandGenerator
        tokenizer = SimpleNamespace(vocab_size=101)
        model = SimpleNamespace(config=SimpleNamespace(to_dict=lambda: {}, pad_token_id=None, eos_token_id=0))
        config = dict(upstream_path=str(upstream), watermark_key=101, gamma=.25,
                      delta=3., ngram=2, salt_key=35317, watermark_cache_size=2)
        for key in [101, 900101, 203]:
            adapter = Maryland(tokenizer, config, key=key)
            reference = MarylandDetector(tokenizer, ngram=2, seed=key, gamma=.25, delta=3.)
            generator = MarylandGenerator(model, tokenizer, ngram=2, seed=key, gamma=.25, delta=3.)
            for context in [(0, 1), (50, 100), (2, 2), (0, 1)]:
                ref_seed = reference.get_seed_rng(context)
                reference.rng.manual_seed(ref_seed)
                expected = torch.randperm(101, generator=reference.rng)[:25]
                self.assertTrue(torch.equal(adapter.greenlist(context), expected))
                logits = generator.logits_processor(torch.zeros(1, 101), torch.tensor([context]))
                actual = {token for token in range(101) if adapter.is_green(context, token)}
                self.assertEqual(actual, set(expected.tolist()))
                self.assertEqual(actual, set(torch.where(logits[0] == 3.)[0].tolist()))
            self.assertEqual(adapter.gamma_exact, 25 / 101)
            self.assertLessEqual(len(adapter._cache), 2)

    def test_adapter_does_not_change_sampling_rng(self):
        import torch
        upstream = Path(__file__).resolve().parents[3] / ".cache/watermark/radioactive-watermark"
        config = dict(upstream_path=str(upstream), watermark_key=101)
        state = torch.get_rng_state().clone()
        adapter = Maryland(SimpleNamespace(vocab_size=101), config)
        adapter.greenlist((2, 3))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
