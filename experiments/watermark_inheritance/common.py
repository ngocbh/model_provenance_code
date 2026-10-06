"""Durable artifacts and a thin adapter to the authors' Maryland watermark."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from collections import OrderedDict
from types import SimpleNamespace


def load_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("w") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def load_config(run_dir):
    return load_json(Path(run_dir) / "config.json")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class Maryland:
    """Same hash and CPU randperm as radioactive-watermark, with bounded caching.

    The tokenizer's real vocabulary is the common generation/scoring domain.
    Pythia's extra padded model-output entries must be masked by callers.
    CPU RNG is deliberate: CUDA randperm does not produce the same greenlists.
    """

    def __init__(self, tokenizer, config, key=None):
        import torch

        upstream = str(Path(config["upstream_path"]).resolve())
        if not (Path(upstream) / "wm/detector.py").is_file():
            raise FileNotFoundError(f"Missing pinned upstream source: {upstream}")
        if upstream not in sys.path:
            sys.path.insert(0, upstream)
        detector_type = importlib.import_module("wm.detector").MarylandDetector
        # Pythia has 23 real added whitespace tokens beyond vocab_size. Include
        # them; only the further model-padding entries are invalid token IDs.
        vocab_size = len(tokenizer) if hasattr(tokenizer, "__len__") else tokenizer.vocab_size
        upstream_tokenizer = SimpleNamespace(vocab_size=vocab_size)
        # The unused upstream hashtable initializer must not change sampling RNG.
        with torch.random.fork_rng(devices=[]):
            self.detector = detector_type(
                upstream_tokenizer, ngram=config.get("ngram", 2),
                seed=config["watermark_key"] if key is None else key,
                seeding="hash", salt_key=config.get("salt_key", 35317),
                gamma=config.get("gamma", 0.25), delta=config.get("delta", 3.0),
            )
        self.vocab_size = int(vocab_size)
        self.green_size = int(config.get("gamma", 0.25) * self.vocab_size)
        self.gamma_exact = self.green_size / self.vocab_size
        self.ngram = int(config.get("ngram", 2))
        self.cache_size = int(config.get("watermark_cache_size", 2048))
        self._cache = OrderedDict()

    def _entry(self, context):
        import torch

        context = tuple(int(x) for x in context)
        if len(context) != self.ngram:
            raise ValueError(f"Expected {self.ngram} context tokens, got {context}")
        if context in self._cache:
            self._cache.move_to_end(context)
            return self._cache[context]
        self.detector.rng.manual_seed(self.detector.get_seed_rng(context))
        indices = torch.randperm(self.vocab_size, generator=self.detector.rng)[:self.green_size].clone()
        mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        mask[indices] = True
        entry = (indices, mask)
        self._cache[context] = entry
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return entry

    def greenlist(self, context):
        return self._entry(context)[0]

    def is_green(self, context, token):
        token = int(token)
        if not 0 <= token < self.vocab_size:
            raise ValueError(f"Token {token} is outside the watermark vocabulary")
        return int(self._entry(context)[1][token])
