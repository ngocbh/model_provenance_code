"""Continuous clean fine-tuning, atomically checkpointed at fixed optimizer steps."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import time
import json

from .common import load_config, load_json, write_json
from .train import (_digest, _identity, _seed_everything, _restore_rng, _write_checkpoint,
                    collate, dolly_examples, valid_vocab_loss)


def checkpoint_schedule(steps, max_steps):
    if not steps or steps != sorted(set(steps)) or steps[0] < 1 or steps[-1] != max_steps:
        raise ValueError("Checkpoint steps must increase strictly and end at max_steps")
    return steps


def batch_indices(rows, batch_size, seed, start_step, max_steps):
    """Yield the same shuffled logical batches before and after a resume."""
    import torch
    steps_per_epoch = math.ceil(rows / batch_size)
    cached_epoch, order = None, None
    for step in range(start_step, max_steps):
        epoch, position = divmod(step, steps_per_epoch)
        if epoch != cached_epoch:
            generator = torch.Generator().manual_seed(seed + epoch + 1)
            order = torch.randperm(rows, generator=generator).tolist()
            cached_epoch = epoch
        yield epoch + 1, order[position * batch_size:(position + 1) * batch_size]


def validate_parent(run_dir, config):
    original = Path(config["parent_run_dir"])
    if _digest(original / "config.json") != config["parent_config_sha256"]:
        raise ValueError("Original parent configuration changed")
    parent_config = load_config(original)
    for key in ("seed", "watermark_key", "wrong_key", "ngram", "gamma", "delta", "salt_key",
                "model_path", "tokenizer_path", "pairs", "calibration_pairs"):
        if config.get(key) != parent_config.get(key):
            raise ValueError(f"Parent/continuation mismatch: {key}")
    pair_path = original / "prepared/pairs.jsonl"
    if _digest(pair_path) != _digest(Path(run_dir) / "prepared/pairs.jsonl"):
        raise ValueError("Continuation audit must use the original paired shard")
    parent = original / "checkpoints/parent"
    state = load_json(parent / "stage_state.json")
    if (state.get("identity") != _identity(parent_config, pair_path, "parent")
            or not state.get("complete") or not state.get("epoch_complete")
            or state.get("epoch") != parent_config["epochs"]):
        raise ValueError("Continuation requires the original complete, matching parent")
    return parent


def train(run_dir, stage, device="cuda:0", stop_after_step=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    run_dir = Path(run_dir)
    config = load_config(run_dir)
    if stage not in ("target", "control") or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("One GPU per target or control trajectory is required")
    torch.set_num_threads(int(config.get("cpu_threads", 1)))
    parent = validate_parent(run_dir, config)
    source = parent if stage == "target" else Path(config["model_path"])
    data = Path(config["clean_data_path"])
    manifest = load_json(config["clean_manifest_path"])
    identity = _identity(config, data, f"curve_{stage}")
    identity["parent_state_sha256"] = _digest(parent / "stage_state.json")
    identity["manifest_sha256"] = _digest(Path(config["clean_manifest_path"]))
    pair_digest = _digest(run_dir / "prepared/pairs.jsonl")
    if (not manifest.get("overlap_checked") or manifest["output_file_sha256"] != identity["data_sha256"]
            or pair_digest not in manifest["pair_sha256"].values()):
        raise ValueError("Clean corpus or union-of-shards overlap manifest mismatch")
    max_steps = int(config["max_steps"])
    milestones = checkpoint_schedule(config["checkpoint_steps"], max_steps)
    if stop_after_step is not None and stop_after_step not in milestones:
        raise ValueError("Graceful stops must coincide with a durable checkpoint")
    checkpoint_root = run_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    latest, previous = None, None
    for step in milestones:
        candidate = checkpoint_root / f"{stage}_step{step}"
        if not candidate.exists():
            continue
        state = load_json(candidate / "stage_state.json")
        if state["identity"] != identity or not state.get("checkpoint_complete") or state["global_step"] != step:
            raise ValueError(f"Invalid completed interval checkpoint: {candidate}")
        latest, previous = candidate, state
    if previous and previous["complete"]:
        write_json(run_dir / f"training_{stage}.json", previous)
        return previous
    if previous and stop_after_step is not None and previous["global_step"] >= stop_after_step:
        return previous
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"], local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Read one row at a time to avoid retaining two copies of the encoded corpus.
    examples = []
    with data.open() as stream:
        for line in stream:
            examples.extend(dolly_examples([json.loads(line)], int(config["max_length"])))
    if len(examples) != config["clean_rows"] or len(examples) != manifest["eligible_rows"]:
        raise ValueError("Clean training row count mismatch")
    vocab_size = len(tokenizer)
    if any(min(row["input_ids"]) < 0 or max(row["input_ids"]) >= vocab_size for row in examples):
        raise ValueError("Invalid training token ID")
    batch_size, micro = int(config["effective_batch_size"]), int(config["train_micro_batch"])
    steps_per_epoch = math.ceil(len(examples) / batch_size)
    if max_steps != int(config["epochs"]) * steps_per_epoch:
        raise ValueError("max_steps must cover the declared complete epochs")
    stage_seed = int(config["seed"]) + 202  # Matched target/control data order and RNG.
    _seed_everything(stage_seed)
    model = AutoModelForCausalLM.from_pretrained(str(latest or source), torch_dtype=torch.float32,
                                               attn_implementation="sdpa", local_files_only=True)
    model.config.use_cache = False
    model.to(device)
    if config.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
    is_cuda = str(device).startswith("cuda")
    if is_cuda and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA is required")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]),
                                 weight_decay=float(config["weight_decay"]), fused=is_cuda)
    warmup = math.ceil(max_steps * float(config["warmup_ratio"]))

    def multiplier(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, max_steps - warmup)
        return max(0.0, .5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    start_step, seen, tokens, input_tokens, loss_sum_total = 0, 0, 0, 0, 0.0
    if latest:
        saved = torch.load(latest / "training_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        _restore_rng(saved["rng"])
        del saved
        start_step = previous["global_step"]
        seen, tokens = previous["examples_seen"], previous["response_tokens_seen"]
        input_tokens, loss_sum_total = previous["input_tokens_seen"], previous["loss_sum"]
    model.train()
    started, initial_seen = time.monotonic(), seen
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    print(json.dumps({"event": "train_start", "stage": stage, "source": str(source),
                      "resume": str(latest) if latest else None, "rows": len(examples),
                      "max_steps": max_steps, "checkpoint_steps": milestones,
                      "response_tokens_per_epoch": sum(r["response_tokens"] for r in examples)}), flush=True)
    for global_step, (epoch, indices) in enumerate(batch_indices(len(examples), batch_size, stage_seed,
                                                               start_step, max_steps), start=start_step + 1):
        optimizer.zero_grad(set_to_none=True)
        batch_tokens = sum(examples[i]["response_tokens"] for i in indices)
        for start in range(0, len(indices), micro):
            chunk = [examples[i] for i in indices[start:start + micro]]
            inputs, labels = collate(chunk, tokenizer.pad_token_id, device)
            with torch.autocast(device_type="cuda" if is_cuda else "cpu", dtype=torch.bfloat16, enabled=is_cuda):
                logits = model(**inputs, use_cache=False).logits
                loss_sum = valid_vocab_loss(logits, labels, vocab_size)
                loss = loss_sum / batch_tokens
            loss.backward()
            loss_sum_total += float(loss_sum.detach())
            input_tokens += sum(len(row["input_ids"]) for row in chunk)
            del logits, loss, loss_sum, inputs, labels
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("max_grad_norm", 1.0)))
        if not math.isfinite(float(grad_norm)):
            raise FloatingPointError(f"Nonfinite gradient at step {global_step}")
        optimizer.step()
        scheduler.step()
        seen += len(indices)
        tokens += batch_tokens
        if global_step == 1 or global_step % int(config.get("log_every", 50)) == 0:
            print(json.dumps({"event": "train_step", "stage": stage, "step": global_step, "epoch": epoch,
                              "examples_seen": seen, "response_tokens_seen": tokens,
                              "train_loss": loss_sum_total / tokens, "lr": scheduler.get_last_lr()[0],
                              "examples_per_second": (seen - initial_seen) / (time.monotonic() - started)}), flush=True)
        if global_step not in milestones:
            continue
        if is_cuda:
            torch.cuda.synchronize(device)
        complete = global_step == max_steps
        metadata = {"identity": identity, "stage": stage, "epoch": epoch, "global_step": global_step,
                    "checkpoint_complete": True, "epoch_complete": global_step % steps_per_epoch == 0,
                    "complete": complete, "pilot": bool(config.get("pilot")),
                    "dataset_rows": len(examples), "examples_seen": seen, "unique_examples_seen": min(seen, len(examples)),
                    "response_tokens_seen": tokens, "input_tokens_seen": input_tokens, "loss_sum": loss_sum_total,
                    "train_loss": loss_sum_total / tokens, "learning_rate": scheduler.get_last_lr()[0],
                    "effective_batch_size": batch_size, "micro_batch_size": micro, "source": str(source),
                    "steps_per_epoch": steps_per_epoch, "checkpoint_kind": "continuous_clean_interval",
                    "loss_description": "cumulative training response-token loss, not held-out performance",
                    "max_memory_allocated": torch.cuda.max_memory_allocated(device) if is_cuda else None}
        destination = checkpoint_root / f"{stage}_step{global_step}"
        save_start = time.monotonic()
        _write_checkpoint(destination, model, tokenizer, metadata,
                          optimizer=None if complete else optimizer, scheduler=scheduler)
        # Never mutate published metadata: prediction identity includes its mtime.
        if latest and (latest / "training_state.pt").exists():
            (latest / "training_state.pt").unlink()
        latest = destination
        write_json(run_dir / f"progress_{stage}.json", metadata)
        print(json.dumps({"event": "checkpoint_ready", "checkpoint": str(destination),
                          "save_seconds": time.monotonic() - save_start, **metadata}), flush=True)
        if stop_after_step == global_step:
            return metadata
    write_json(run_dir / f"training_{stage}.json", metadata)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=["target", "control"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stop-after-step", type=int, help="Yield after this durable checkpoint; resume preserves optimizer and RNG")
    args = parser.parse_args()
    train(args.run_dir, args.stage, args.device, args.stop_after_step)
