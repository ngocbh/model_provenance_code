"""Prepare a shared, pinned clean corpus for continuous fine-tuning curves."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import random

from .common import load_config, read_jsonl, write_json
from .prepare import CompletionIndex, encode_dolly, file_sha256, normalize_text

DATASET = "HuggingFaceH4/ultrachat_200k"
REVISION = "8049631c405ae6576f93f445c6b8166f76f5505a"
STEPS = [512, 1024, 2048, 3125, 6250, 12500, 18750]


def initialize_worker(tokenizer_path, pair_paths):
    from transformers import AutoTokenizer
    global TOKENIZER, EXACT, NORMALIZED
    TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    texts = [row[field] for path in pair_paths for row in read_jsonl(path)
             for field in ("s_text", "s_prime_text")]
    EXACT = CompletionIndex(texts, width=32)
    NORMALIZED = CompletionIndex(map(normalize_text, texts), width=32)


def encode_row(item):
    index, row = item
    messages = row["messages"]
    if len(messages) < 2 or [m["role"] for m in messages[:2]] != ["user", "assistant"]:
        return "invalid_roles", None
    fields = {"instruction": messages[0]["content"], "response": messages[1]["content"], "context": ""}
    encoded = encode_dolly(fields, TOKENIZER, 1024)
    if encoded is None:
        return "empty", None
    values = [fields["instruction"], fields["response"]]
    values += ["\n\n".join(values), TOKENIZER.decode(encoded["input_ids"], skip_special_tokens=True)]
    if any(EXACT.contains_completion(value) for value in values):
        return "exact_overlap", None
    if any(NORMALIZED.contains_completion(normalize_text(value)) for value in values):
        return "normalized_overlap", None
    encoded.update(source_id=f"ultrachat:train_sft:{row['prompt_id']}", source_row=index,
                   instruction_sha256=hashlib.sha256(normalize_text(fields["instruction"]).encode()).hexdigest())
    return "eligible", encoded


def prepare(root: Path, original: Path, workers=4):
    from datasets import load_dataset, load_from_disk
    root, original = root.resolve(), original.resolve()
    setup = root / "setup"
    setup.mkdir(parents=True, exist_ok=True)
    if (setup / "data_manifest.json").exists():
        raise FileExistsError("Prepared corpus already exists; preserve immutable experiment inputs")
    cache = Path(".cache/watermark/data/ultrachat_200k_train_sft").resolve()
    if cache.exists():
        dataset = load_from_disk(cache)
        from .common import load_json
        if load_json(cache / "pinned_revision.json") != {"dataset": DATASET, "revision": REVISION}:
            raise ValueError("Cached dataset revision mismatch")
    else:
        dataset = load_dataset(DATASET, revision=REVISION, split="train_sft")
        dataset.save_to_disk(cache)
        write_json(cache / "pinned_revision.json", {"dataset": DATASET, "revision": REVISION})
    originals = [original / f"run_{i:02d}" for i in range(3)]
    pair_paths = [run / "prepared/pairs.jsonl" for run in originals]
    if any(len(read_jsonl(path)) != 10000 for path in pair_paths):
        raise ValueError("All original complete paired shards are required")
    counts, sources, instructions = Counter(), set(), set()
    order = list(range(len(dataset)))
    random.Random(8675309).shuffle(order)
    output = setup / "clean.jsonl"
    temporary = output.with_suffix(".jsonl.partial")
    total_input, total_response = 0, 0
    with mp.get_context("spawn").Pool(workers, initializer=initialize_worker,
            initargs=(load_config(originals[0])["tokenizer_path"], pair_paths)) as pool, temporary.open("w") as stream:
        results = pool.imap(encode_row, ((i, dataset[i]) for i in order), chunksize=32)
        for status, row in results:
            counts["examined_rows"] += 1
            if row is None:
                counts[status] += 1
                continue
            if row["source_id"] in sources or row["instruction_sha256"] in instructions:
                counts["duplicate_instruction"] += 1
                continue
            sources.add(row["source_id"])
            instructions.add(row["instruction_sha256"])
            stream.write(json.dumps(row) + "\n")
            total_input += len(row["input_ids"])
            total_response += row["response_tokens"]
            counts["eligible_rows"] += 1
            for field in ("response_truncated", "instruction_truncated", "context_truncated"):
                counts[field] += int(row[field])
            if counts["eligible_rows"] % 10000 == 0:
                print(json.dumps(dict(counts)), flush=True)
            if counts["eligible_rows"] == 200000:
                break
        stream.flush()
        os.fsync(stream.fileno())
    if counts["eligible_rows"] != 200000:
        raise ValueError(f"Need 200000 distinct clean examples, got {dict(counts)}")
    temporary.rename(output)
    manifest = {"dataset": DATASET, "revision": REVISION, "split": "train_sft", "source_rows": len(dataset),
                "selection_seed": 8675309, "selection": "first user/assistant exchange; unique normalized instructions",
                "format": "same response-only instruction format as original Dolly experiment",
                "max_length": 1024, **dict(counts), "input_tokens_per_epoch": total_input,
                "response_tokens_per_epoch": total_response, "overlap_checked": True,
                "overlap_policy": "exact and NFKC/casefold/whitespace normalized full-completion substrings; >=32 characters, otherwise original-field equality",
                "pair_sha256": {str(path.resolve()): file_sha256(path) for path in pair_paths},
                "output_file_sha256": file_sha256(output)}
    write_json(setup / "data_manifest.json", manifest)
    for parent_run in originals:
        run = root / parent_run.name
        (run / "prepared").mkdir(parents=True, exist_ok=True)
        (run / "prepared/pairs.jsonl").symlink_to(parent_run / "prepared/pairs.jsonl")
        config = load_config(parent_run)
        config.update(experiment="watermark_finetuning_curve", parent_run_dir=str(parent_run),
                      parent_config_sha256=file_sha256(parent_run / "config.json"),
                      clean_data_path=str(output), clean_manifest_path=str(setup / "data_manifest.json"),
                      clean_rows=200000, checkpoint_steps=STEPS, max_steps=18750,
                      primary_checkpoint_name="target_step18750", dataset_revision=REVISION,
                      log_every=50)
        write_json(run / "config.json", config)
        write_json(run / "config.lock.yaml", config)
    print(json.dumps({"event": "data_ready", **manifest}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/watermark_curve"))
    parser.add_argument("--original", type=Path, default=Path("artifacts/watermark_inheritance"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    prepare(args.root, args.original, args.workers)
