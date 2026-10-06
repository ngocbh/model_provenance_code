"""Resumable, independently seeded Maryland completion-pair generation."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import random
import time

from .common import Maryland, fingerprint, load_config, load_json, read_jsonl, write_json, write_jsonl
from .prepare import file_sha256, validate_tokenizer, verify_upstream


def sampling_seed(seed, phase, pair_id, side):
    return int(fingerprint([int(seed), phase, str(pair_id), str(side)])[:15], 16)


def watermark_config_id(config):
    return fingerprint({key: config.get(key, default) for key, default in [
        ("watermark_key", None), ("gamma", .25), ("delta", 3.), ("ngram", 2), ("salt_key", 35317)]})


def generation_config_id(config):
    return fingerprint({key: config.get(key, default) for key, default in [
        ("seed", None), ("watermark_key", None), ("gamma", .25), ("delta", 3.),
        ("ngram", 2), ("salt_key", 35317), ("temperature", .8), ("top_p", .95),
        ("max_new_tokens", 256), ("model_path", None), ("tokenizer_path", None)]})


def pair_record(prompt, completions, tokenizer, config, phase):
    pair_id = prompt["pair_id"]
    seeds = [sampling_seed(config["seed"], phase, pair_id, side) for side in range(2)]
    assignment_seed = sampling_seed(config["seed"], phase, pair_id, "assignment")
    s_side = random.Random(assignment_seed).randrange(2)
    s_ids, sp_ids = completions[s_side], completions[1 - s_side]
    record = dict(prompt, s_ids=s_ids, s_prime_ids=sp_ids,
                  s_text=tokenizer.decode(s_ids, skip_special_tokens=True),
                  s_prime_text=tokenizer.decode(sp_ids, skip_special_tokens=True),
                  s_length=len(s_ids), s_prime_length=len(sp_ids),
                  sampling_seed_s=seeds[s_side], sampling_seed_s_prime=seeds[1 - s_side],
                  assignment_seed=assignment_seed, s_draw_index=s_side, tied=s_ids == sp_ids,
                  s_stop_reason="eos" if s_ids and s_ids[-1] == tokenizer.eos_token_id else "length",
                  s_prime_stop_reason="eos" if sp_ids and sp_ids[-1] == tokenizer.eos_token_id else "length",
                  watermark_config_id=watermark_config_id(config),
                  generation_config_id=generation_config_id(config))
    return record


def read_journal(path):
    """Recover only an interrupted final JSON line, rejecting all other damage."""
    path = Path(path)
    if not path.exists():
        return []
    result = []
    with path.open("r+b") as stream:
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line:
                break
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if not line.endswith(b"\n") and not stream.read(1):
                    stream.truncate(start)
                    stream.flush()
                    os.fsync(stream.fileno())
                    break
                raise ValueError(f"Corrupt generation journal {path} at byte {start}")
            if not line.endswith(b"\n"):
                # A complete JSON record whose final newline was interrupted.
                stream.seek(0, os.SEEK_END)
                stream.write(b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            result.append(row)
    return result


def validate_pair(row, prompt, config, phase, vocab_size=None):
    for key in ("pair_id", "source_id", "audit_split", "prompt_ids", "prompt_text"):
        if row.get(key) != prompt[key]:
            raise ValueError(f"Pair {prompt['pair_id']} mismatched {key}")
    if row.get("watermark_config_id") != watermark_config_id(config):
        raise ValueError("Watermark configuration changed since generation")
    if row.get("generation_config_id") != generation_config_id(config):
        raise ValueError("Generation configuration changed since generation")
    assignment_seed = sampling_seed(config["seed"], phase, row["pair_id"], "assignment")
    s_side = random.Random(assignment_seed).randrange(2)
    if row.get("assignment_seed") != assignment_seed or row.get("s_draw_index") != s_side:
        raise ValueError("Invalid paired random side assignment")
    for side, index in [("s", s_side), ("s_prime", 1 - s_side)]:
        ids = row[side + "_ids"]
        if not 1 <= len(ids) <= int(config.get("max_new_tokens", 256)):
            raise ValueError("Invalid generated sequence length")
        if row.get(side + "_length") != len(ids):
            raise ValueError("Stored generation length disagrees with token IDs")
        if row.get("sampling_seed_" + side) != sampling_seed(config["seed"], phase, row["pair_id"], index):
            raise ValueError("Mismatched independent sampling seed")
        if vocab_size is not None and any(not 0 <= token < vocab_size for token in ids):
            raise ValueError("Generation contains padded/out-of-vocabulary token")
    if row.get("tied") != (row["s_ids"] == row["s_prime_ids"]):
        raise ValueError("Invalid tie indicator")


def sample_batch(model, tokenizer, watermark, prompts, seeds, config, device):
    """Cached decoding with independent CUDA RNG streams, invariant to sharding.

    Each sample pre-draws uniforms from its own seeded generator. Inverse-CDF
    sampling is exactly categorical after the upstream temperature/top-p rule;
    this avoids launching one multinomial kernel per row at every token.
    """
    import torch

    max_tokens = int(config.get("max_new_tokens", 256))
    temperature, top_p = float(config.get("temperature", .8)), float(config.get("top_p", .95))
    if temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("Independent completion sampling requires temperature > 0 and 0 < top_p <= 1")
    if len({len(prompt) for prompt in prompts}) != 1:
        raise ValueError("Generation expects fixed-length, unpadded prompts")
    if any(len(prompt) < watermark.ngram for prompt in prompts):
        raise ValueError("Prompt shorter than watermark context")
    batch = len(prompts)
    inputs = torch.tensor(prompts, device=device, dtype=torch.long)
    uniforms = torch.stack([torch.rand(max_tokens, device=device,
                                      generator=torch.Generator(device=device).manual_seed(seed)) for seed in seeds])
    histories, outputs = [list(prompt) for prompt in prompts], [[] for _ in prompts]
    finished = [False] * batch
    past = None
    vocab = watermark.vocab_size
    with torch.inference_mode():
        for step in range(max_tokens):
            prediction = model(input_ids=inputs, past_key_values=past, use_cache=True)
            past = prediction.past_key_values
            # The model embedding table has padded entries with no tokenizer ID.
            logits = prediction.logits[:, -1, :vocab].float().contiguous()
            green = torch.stack([watermark.greenlist(tokens[-watermark.ngram:]) for tokens in histories]).to(device)
            logits.scatter_add_(1, green, torch.full(green.shape, float(config.get("delta", 3.)), device=device))
            probs = torch.softmax(logits / temperature, dim=-1)
            probs, indices = torch.sort(probs, dim=-1, descending=True)
            cumulative = probs.cumsum(dim=-1)
            probs.masked_fill_(cumulative - probs > top_p, 0.)
            probs.div_(probs.sum(dim=-1, keepdim=True))
            cumulative = probs.cumsum(dim=-1).contiguous()
            # Sample relative to the actual endpoint: forcing only the last
            # entry to 1 could assign rounding-residual mass to a top-p-masked
            # trailing token. searchsorted(left) also handles rounded endpoints.
            draws = uniforms[:, step:step + 1] * cumulative[:, -1:]
            selected = torch.searchsorted(cumulative, draws.contiguous()).clamp_max(vocab - 1)
            next_tokens = indices.gather(1, selected).squeeze(1)
            next_ids = next_tokens.tolist()
            for i, token in enumerate(next_ids):
                if not finished[i]:
                    outputs[i].append(token)
                    histories[i].append(token)
                    finished[i] = token == tokenizer.eos_token_id
            if all(finished):
                break
            inputs = next_tokens[:, None]
    return outputs


def _paths(run_dir, phase, rank, world_size):
    base = Path(run_dir)
    prompts = base / "prepared" / ("pilot_prompts.jsonl" if phase == "pilot" else "prompts.jsonl")
    journal = base / "generated" / f"{phase}.rank{rank:03d}-of-{world_size:03d}.jsonl"
    return prompts, journal


def worker_identity(args, config, rank, prompt_path):
    manifest = load_json(Path(args.run_dir) / "prepared/manifest.json")
    current_upstream = verify_upstream(config, manifest["upstream"]["revision"])
    if current_upstream != manifest["upstream"]:
        raise RuntimeError("Current upstream checkout does not match prepared manifest")
    return {"generation_config_id": generation_config_id(config), "phase": args.phase,
            "rank": rank, "world_size": args.world_size,
            "prompt_file_sha256": file_sha256(prompt_path), "model": manifest["model"],
            "tokenizer": manifest["tokenizer"], "upstream": manifest["upstream"]}


def run_worker(args, config):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not 0 <= args.rank < args.world_size:
        raise ValueError("rank must be in [0, world_size)")
    prompt_path, journal = _paths(args.run_dir, args.phase, args.rank, args.world_size)
    prompts = read_jsonl(prompt_path)[args.rank::args.world_size]
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = worker_identity(args, config, args.rank, prompt_path)
        metadata_path = journal.with_suffix(".metadata.json")
        if metadata_path.exists() and load_json(metadata_path) != identity:
            raise RuntimeError("Worker resume metadata does not match current pinned inputs")
        write_json(metadata_path, identity)
        completed = read_journal(journal)
        by_id = {row["pair_id"]: row for row in prompts}
        seen = set()
        for row in completed:
            if row["pair_id"] in seen or row["pair_id"] not in by_id:
                raise ValueError("Duplicate or foreign pair in generation shard")
            validate_pair(row, by_id[row["pair_id"]], config, args.phase)
            seen.add(row["pair_id"])
        pending = [row for row in prompts if row["pair_id"] not in seen]
        if not pending:
            print(json.dumps({"phase": args.phase, "rank": args.rank, "complete_pairs": len(completed), "resumed": True}), flush=True)
            return
        torch.set_num_threads(int(config.get("cpu_threads", 4)))
        device = args.device
        if device.startswith("cuda"):
            torch.cuda.set_device(device)
        tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"], local_files_only=True)
        validate_tokenizer(tokenizer)
        model = AutoModelForCausalLM.from_pretrained(
            config["model_path"], local_files_only=True, torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
            attn_implementation="sdpa").to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        watermark = Maryland(tokenizer, config)
        start = time.monotonic()
        tokens, generated_pairs = 0, 0
        # --batch-size counts completions, with both independent draws together.
        pair_batch = max(1, args.batch_size // 2)
        with journal.open("a") as stream:
            for offset in range(0, len(pending), pair_batch):
                batch = pending[offset:offset + pair_batch]
                inputs = [row["prompt_ids"] for row in batch for _ in range(2)]
                seeds = [sampling_seed(config["seed"], args.phase, row["pair_id"], side) for row in batch for side in range(2)]
                outputs = sample_batch(model, tokenizer, watermark, inputs, seeds, config, device)
                for i, prompt in enumerate(batch):
                    record = pair_record(prompt, outputs[2 * i:2 * i + 2], tokenizer, config, args.phase)
                    validate_pair(record, prompt, config, args.phase, watermark.vocab_size)
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                tokens += sum(map(len, outputs))
                generated_pairs += len(batch)
                elapsed = time.monotonic() - start
                progress = {"phase": args.phase, "rank": args.rank, "complete_pairs": len(completed) + generated_pairs,
                            "assigned_pairs": len(prompts), "new_tokens": tokens, "elapsed_seconds": elapsed,
                            "tokens_per_second": tokens / elapsed, "pairs_per_second": generated_pairs / elapsed,
                            "remaining_seconds": (len(pending) - generated_pairs) * elapsed / generated_pairs,
                            "batch_size": args.batch_size, "device": device,
                            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.startswith("cuda") else 0}
                write_json(journal.with_suffix(".progress.json"), progress)
                print(json.dumps(progress), flush=True)


def _log10_binomial_tail(green, count, gamma):
    import numpy as np
    from scipy.special import logsumexp
    from scipy.stats import binom

    if count == 0 or green == 0:
        return 0.
    result = float(binom.logsf(green - 1, count, gamma))
    if not math.isfinite(result):
        result = float(logsumexp(binom.logpmf(np.arange(green, count + 1), count, gamma)))
    return result / math.log(10.)


def pilot_metrics(pairs, tokenizer, config):
    keys = {"correct_key": Maryland(tokenizer, config), "wrong_key": Maryland(tokenizer, config, key=config["wrong_key"])}
    counts = {name: [0, 0] for name in keys}
    global_seen = set()
    lengths, repetition = [], []
    eos_count = 0
    for row in pairs:
        for side in ("s", "s_prime"):
            ids = row[side + "_ids"]
            lengths.append(len(ids))
            eos_count += int(ids[-1] == tokenizer.eos_token_id)
            grams = [tuple(ids[i:i + 4]) for i in range(max(0, len(ids) - 3))]
            repetition.append(1 - len(set(grams)) / len(grams) if grams else 0.)
            prompt, ngram = row["prompt_ids"], int(config.get("ngram", 2))
            seen = {tuple(prompt[i:i + ngram]) for i in range(max(0, len(prompt) - ngram))}
            sequence = prompt + ids
            for position in range(len(prompt), len(sequence)):
                context = tuple(sequence[position - ngram:position])
                token = sequence[position]
                entry = context + (token,)
                eligible = context not in seen and entry not in global_seen
                seen.add(context)
                if not eligible:
                    continue
                global_seen.add(entry)
                for name, detector in keys.items():
                    counts[name][0] += detector.is_green(context, token)
                    counts[name][1] += 1
    stats = {"pairs": len(pairs), "completions": len(lengths), "mean_length": sum(lengths) / len(lengths),
             "min_length": min(lengths), "max_length": max(lengths), "eos_fraction": eos_count / len(lengths),
             "mean_repeated_4gram_fraction": sum(repetition) / len(repetition),
             "ties": sum(row["tied"] for row in pairs),
             "detector_note": "Stored generated tokens, unique contexts per sample and context/token tuples globally; pilot only."}
    for name, (green, count) in counts.items():
        gamma = keys[name].gamma_exact
        stats[name] = {"green": green, "eligible_tokens": count, "gamma_exact": gamma,
                       "green_rate": green / count if count else 0.,
                       "log10_p": _log10_binomial_tail(green, count, gamma)}
    return stats


def merge(args, config):
    from transformers import AutoTokenizer

    prompt_path, _ = _paths(args.run_dir, args.phase, 0, args.world_size)
    prompts = read_jsonl(prompt_path)
    expected = int(config.get("pilot_pairs", 128) if args.phase == "pilot" else config["pairs"])
    if len(prompts) != expected:
        raise ValueError(f"Expected {expected} prompts, found {len(prompts)}")
    by_id = {row["pair_id"]: row for row in prompts}
    if len(by_id) != len(prompts):
        raise ValueError("Duplicate prompt pair IDs")
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"], local_files_only=True)
    validate_tokenizer(tokenizer)
    records, worker_progress = {}, []
    for rank in range(args.world_size):
        _, journal = _paths(args.run_dir, args.phase, rank, args.world_size)
        with journal.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if load_json(journal.with_suffix(".metadata.json")) != worker_identity(args, config, rank, prompt_path):
                raise RuntimeError("Merge worker metadata does not match pinned inputs")
            assigned = {row["pair_id"] for row in prompts[rank::args.world_size]}
            rows = read_journal(journal)
            if {row["pair_id"] for row in rows} != assigned or len(rows) != len(assigned):
                raise ValueError(f"Generation rank {rank} incomplete or contains duplicate/foreign IDs")
            for row in rows:
                validate_pair(row, by_id[row["pair_id"]], config, args.phase, len(tokenizer))
                if row["pair_id"] in records:
                    raise ValueError("Overlapping worker pair IDs")
                records[row["pair_id"]] = row
            progress = journal.with_suffix(".progress.json")
            if progress.exists():
                worker_progress.append(load_json(progress))
    if set(records) != set(by_id):
        raise ValueError("Merged dataset does not contain exactly the prepared prompts")
    ordered = [records[row["pair_id"]] for row in prompts]
    output = Path(args.run_dir) / "prepared" / ("pilot_pairs.jsonl" if args.phase == "pilot" else "pairs.jsonl")
    if output.exists() and read_jsonl(output) != ordered:
        raise RuntimeError("Refusing to overwrite a different merged generation dataset")
    write_jsonl(output, ordered)
    summary = {"phase": args.phase, "pairs": len(ordered), "ties": sum(row["tied"] for row in ordered),
               "tokens": sum(row[side + "_length"] for row in ordered for side in ("s", "s_prime")),
               "pair_file_sha256": file_sha256(output), "generation_config_id": generation_config_id(config),
               "watermark_config_id": watermark_config_id(config), "workers": worker_progress}
    if args.phase == "pilot":
        summary["pilot"] = pilot_metrics(ordered, tokenizer, config)
    else:
        from .prepare import build_inputs
        build_inputs(args.run_dir, config, config.get("shared_dir", ".cache/watermark"), dolly_only=True)
    write_json(output.with_suffix(".manifest.json"), summary)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--phase", choices=["pilot", "full"], default="full")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    config = load_config(args.run_dir)
    args.batch_size = args.batch_size or int(config.get("generation_batch_size", 32))
    if args.world_size < 1 or args.batch_size < 2:
        raise ValueError("Need world_size >= 1 and batch_size >= 2")
    if args.merge:
        merge(args, config)
    else:
        run_worker(args, config)


if __name__ == "__main__":
    main()
