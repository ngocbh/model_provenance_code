"""Full-parameter, response-only Pythia training for watermark inheritance.

Run as ``python -m experiments.watermark_inheritance.train --run-dir ...``.
An epoch is the durable resume boundary. Pilot checkpoints are deliberately
ineligible as parent models or completed experimental stages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Iterable


def response_example(prompt_ids: list[int], response_ids: list[int], max_length: int) -> dict:
    """Keep response tokens preferentially, retaining a causal prompt token.

    Generated completion IDs are used verbatim, including an existing EOS. A
    length-limited generation is not changed into an EOS-terminated generation.
    """
    if max_length < 2 or not prompt_ids or not response_ids:
        raise ValueError("Need a prompt, a response, and max_length >= 2")
    response = list(response_ids[: max_length - 1])
    prompt = list(prompt_ids[-(max_length - len(response)) :])
    return {"input_ids": prompt + response, "labels": [-100] * len(prompt) + response,
            "response_tokens": len(response)}


def parent_examples(pairs: Iterable[dict], max_length: int) -> list[dict]:
    """Consume S from every split. S-prime is never an input to this function."""
    return [response_example(row["prompt_ids"], row["s_ids"], max_length) for row in pairs]


def dolly_examples(rows: Iterable[dict], max_length: int) -> list[dict]:
    result = []
    for row in rows:
        ids, labels = list(row["input_ids"]), list(row["labels"])
        if len(ids) != len(labels) or not ids:
            raise ValueError("Dolly input_ids and labels must have equal positive lengths")
        supervised = [i for i, value in enumerate(labels) if value != -100]
        if not supervised:
            raise ValueError("Dolly row has no response tokens")
        first, last = supervised[0], supervised[-1]
        if first == 0 or supervised != list(range(first, last + 1)):
            raise ValueError("Dolly needs a masked prompt followed by a contiguous response")
        if ids[first:last + 1] != labels[first:last + 1]:
            raise ValueError("Dolly response labels must be its original token IDs")
        result.append(response_example(ids[:first], ids[first:last + 1], max_length))
    return result


def logical_batches(order: list[int], effective_batch_size: int, micro_batch_size: int):
    """Yield exact logical batches, without dropping or duplicating the tail."""
    if effective_batch_size <= 0 or micro_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")
    for start in range(0, len(order), effective_batch_size):
        logical = order[start:start + effective_batch_size]
        yield [logical[i:i + micro_batch_size] for i in range(0, len(logical), micro_batch_size)]


def collate(examples: list[dict], pad_token_id: int, device: str):
    import torch

    length = max(len(row["input_ids"]) for row in examples)
    # Tensor-core-friendly dynamic padding; attention and labels mask all pads.
    length = math.ceil(length / 8) * 8
    ids = torch.full((len(examples), length), pad_token_id, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    attention = torch.zeros_like(ids)
    for i, row in enumerate(examples):
        size = len(row["input_ids"])
        ids[i, :size] = torch.tensor(row["input_ids"], dtype=torch.long)
        labels[i, :size] = torch.tensor(row["labels"], dtype=torch.long)
        attention[i, :size] = 1
    return {"input_ids": ids.to(device), "attention_mask": attention.to(device)}, labels.to(device)


def valid_vocab_loss(logits, labels, vocab_size: int):
    """Summed causal CE, excluding Pythia's padded non-tokenizer vocabulary."""
    import torch.nn.functional as F

    # FP32 CE is essential even when the forward pass uses BF16 autocast.
    return F.cross_entropy(logits[:, :-1, :vocab_size].float().reshape(-1, vocab_size),
                           labels[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _identity(config: dict, data_path: Path, stage: str) -> dict:
    return {"schema": 1, "stage": stage, "data_sha256": _digest(data_path),
            "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True,
                separators=(",", ":")).encode()).hexdigest(),
            "loss": "completion_token_mean_valid_tokenizer_vocab", "precision": "fp32_master_bf16_autocast"}


def _checkpoint_name(stage: str, epoch: int, epochs: int) -> str:
    return "parent" if stage == "parent" and epoch == epochs else f"{stage}_epoch{epoch}"


def _seed_everything(seed: int):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_state():
    import numpy as np
    import torch

    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(state: dict):
    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _write_checkpoint(destination: Path, model, tokenizer, metadata: dict,
                      optimizer=None, scheduler=None):
    import torch
    from .common import write_json

    if destination.exists():
        raise FileExistsError(f"Refusing to replace checkpoint: {destination}")
    temporary = destination.with_name(f".{destination.name}.saving-{os.getpid()}")
    if temporary.exists():
        # A same-PID stale temporary directory is never a valid checkpoint.
        raise FileExistsError(f"Stale checkpoint temporary directory: {temporary}")
    temporary.mkdir(parents=True)
    try:
        model.save_pretrained(temporary, safe_serialization=True, max_shard_size="2GB")
        tokenizer.save_pretrained(temporary)
        if optimizer is not None:
            torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "rng": _rng_state()}, temporary / "training_state.pt")
        write_json(temporary / "stage_state.json", metadata)
        temporary.rename(destination)
    except BaseException:
        # Preserve partial output for inspection; it cannot be resumed as complete.
        raise


def train_stage(run_dir: Path, stage: str, device: str = "cuda", max_steps: int | None = None) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .common import load_config, load_json, read_jsonl, write_json

    run_dir = Path(run_dir)
    config = load_config(run_dir)
    torch.set_num_threads(int(config.get("cpu_threads", 1)))
    if stage not in ("parent", "target", "control"):
        raise ValueError(f"Unknown stage: {stage}")
    if max_steps is not None and (not config.get("pilot", False) or max_steps < 1):
        raise ValueError("--max-steps requires config pilot=true and a positive step count")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This worker uses one GPU; launch independent runs on other GPUs")
    data_path = run_dir / "prepared" / ("pairs.jsonl" if stage == "parent" else "dolly.jsonl")
    identity = _identity(config, data_path, stage)
    if stage != "parent":
        manifest = load_json(run_dir / "prepared" / "dolly_manifest.json")
        if not manifest.get("overlap_checked", False):
            raise ValueError("Dolly training requires an overlap check against the complete generated shard")
        if (manifest.get("pair_file_sha256") != _digest(run_dir / "prepared" / "pairs.jsonl") or
                manifest.get("output_file_sha256") != identity["data_sha256"]):
            raise ValueError("Dolly overlap manifest does not match current pairs/Dolly files")
    epochs = int(config.get("epochs", 3))
    if epochs < 1:
        raise ValueError("epochs must be positive")
    checkpoint_root = run_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    latest, previous_metadata = None, None
    for epoch in range(1, epochs + 1):
        candidate = checkpoint_root / _checkpoint_name(stage, epoch, epochs)
        if not candidate.exists():
            continue
        metadata = load_json(candidate / "stage_state.json")
        if metadata.get("identity") != identity or not metadata.get("epoch_complete", False):
            raise ValueError(f"Checkpoint identity/completion mismatch: {candidate}")
        if metadata["epoch"] != epoch or metadata["global_step"] != metadata["expected_steps_per_epoch"] * epoch:
            raise ValueError(f"Checkpoint epoch/step mismatch: {candidate}")
        latest, previous_metadata = candidate, metadata
    if previous_metadata and previous_metadata["epoch"] == epochs:
        if not previous_metadata.get("complete", False):
            raise ValueError(f"Final checkpoint is not complete: {latest}")
        write_json(run_dir / f"training_{stage}.json", previous_metadata)
        print(json.dumps({"event": "already_complete", "checkpoint": str(latest)}), flush=True)
        return previous_metadata
    source = str(latest) if latest else config["model_path"]
    if stage == "target" and latest is None:
        parent_path = checkpoint_root / "parent"
        parent_state = load_json(parent_path / "stage_state.json")
        parent_identity = _identity(config, run_dir / "prepared" / "pairs.jsonl", "parent")
        if parent_state.get("identity") != parent_identity or not parent_state.get("complete", False):
            raise ValueError("Target requires a completed parent with the same immutable configuration")
        source = str(parent_path)
    tokenizer = AutoTokenizer.from_pretrained(config.get("tokenizer_path", config["model_path"]), local_files_only=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs an EOS or PAD token")
        tokenizer.pad_token = tokenizer.eos_token
    max_length = int(config.get("max_length", 1024))
    rows = list(read_jsonl(data_path))
    if stage == "parent" and len(rows) != int(config["pairs"]):
        raise ValueError(f"Expected {config['pairs']} complete training pairs, found {len(rows)}")
    examples = parent_examples(rows, max_length) if stage == "parent" else dolly_examples(rows, max_length)
    del rows
    if not examples:
        raise ValueError("Training dataset is empty")
    # Pythia includes real added whitespace tokens beyond tokenizer.vocab_size.
    # Only the model's entries at or above len(tokenizer) are padding.
    vocab_size = len(tokenizer)
    if any(min(row["input_ids"]) < 0 or max(row["input_ids"]) >= vocab_size for row in examples):
        raise ValueError("Training token IDs exceed tokenizer vocabulary")
    effective_batch = int(config.get("effective_batch_size", 32))
    micro_batch = int(config.get("train_micro_batch", 2))
    if effective_batch < 1 or micro_batch < 1:
        raise ValueError("Batch sizes must be positive")
    steps_per_epoch = math.ceil(len(examples) / effective_batch)
    total_steps = epochs * steps_per_epoch
    stage_seed = int(config["seed"]) + (101 if stage == "parent" else 202)
    _seed_everything(stage_seed)
    model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=torch.float32,
                                               attn_implementation="sdpa", local_files_only=True)
    model.config.use_cache = False
    model.to(device)
    if config.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
    # All parameters, including layer norms and embeddings, follow the declared
    # full-parameter AdamW recipe; this experiment does not use adapters.
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 1e-5)),
                                 weight_decay=float(config.get("weight_decay", 0.01)),
                                 fused=str(device).startswith("cuda"))
    warmup = math.ceil(total_steps * float(config.get("warmup_ratio", 0.03)))

    def lr_multiplier(step):
        if step < warmup:
            return float(step) / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    first_epoch = 1
    global_step = 0
    history = []
    if latest:
        state_path = latest / "training_state.pt"
        if not state_path.exists():
            raise ValueError(f"Incomplete stage lacks resumable optimizer state: {latest}")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        _restore_rng(state["rng"])
        del state
        first_epoch = previous_metadata["epoch"] + 1
        global_step = previous_metadata["global_step"]
        history = previous_metadata["history"]
    is_cuda = str(device).startswith("cuda")
    if is_cuda:
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This training recipe requires CUDA BF16 support")
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    stage_start = time.monotonic()
    print(json.dumps({"event": "train_start", "stage": stage, "source": source,
                      "rows": len(examples), "response_tokens_per_epoch": sum(row["response_tokens"] for row in examples),
                      "steps_per_epoch": steps_per_epoch, "micro_batch": micro_batch,
                      "effective_batch": effective_batch, "resume_epoch": first_epoch}), flush=True)
    for epoch in range(first_epoch, epochs + 1):
        epoch_start = time.monotonic()
        epoch_tokens, epoch_examples, epoch_loss, epoch_input_tokens = 0, 0, 0.0, 0
        generator = torch.Generator().manual_seed(stage_seed + epoch)
        order = torch.randperm(len(examples), generator=generator).tolist()
        for micro_batches in logical_batches(order, effective_batch, micro_batch):
            optimizer.zero_grad(set_to_none=True)
            batch_response_tokens = sum(examples[i]["response_tokens"] for micro in micro_batches for i in micro)
            for indices in micro_batches:
                chunk = [examples[i] for i in indices]
                inputs, labels = collate(chunk, tokenizer.pad_token_id, device)
                with torch.autocast(device_type="cuda" if is_cuda else "cpu", dtype=torch.bfloat16, enabled=is_cuda):
                    logits = model(**inputs, use_cache=False).logits
                    loss_sum = valid_vocab_loss(logits, labels, vocab_size)
                    loss = loss_sum / batch_response_tokens
                loss.backward()
                epoch_loss += float(loss_sum.detach())
                epoch_tokens += sum(row["response_tokens"] for row in chunk)
                epoch_input_tokens += sum(len(row["input_ids"]) for row in chunk)
                epoch_examples += len(chunk)
                del logits, loss, loss_sum, inputs, labels
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("max_grad_norm", 1.0)))
            if not math.isfinite(float(grad_norm)):
                raise FloatingPointError(f"Nonfinite gradient at {stage} epoch {epoch} step {global_step}")
            optimizer.step()
            scheduler.step()
            global_step += 1
            if global_step % int(config.get("log_every", 10)) == 0 or global_step == 1:
                elapsed = time.monotonic() - epoch_start
                print(json.dumps({"event": "train_step", "stage": stage, "epoch": epoch, "step": global_step,
                                  "train_loss": epoch_loss / epoch_tokens, "lr": scheduler.get_last_lr()[0],
                                  "response_tokens_per_second": epoch_tokens / elapsed,
                                  "input_tokens_per_second": epoch_input_tokens / elapsed,
                                  "examples_per_second": epoch_examples / elapsed,
                                  "elapsed_seconds": elapsed}), flush=True)
            if max_steps is not None and global_step >= max_steps:
                if is_cuda:
                    torch.cuda.synchronize(device)
                metadata = {"identity": identity, "stage": stage, "epoch": epoch, "global_step": global_step,
                            "complete": False, "epoch_complete": False, "pilot": True,
                            "train_loss": epoch_loss / epoch_tokens, "examples_seen": epoch_examples,
                            "response_tokens_seen": epoch_tokens, "input_tokens_seen": epoch_input_tokens,
                            "train_seconds": time.monotonic() - stage_start,
                            "max_memory_allocated": torch.cuda.max_memory_allocated(device) if is_cuda else None}
                destination = checkpoint_root / f"pilot_{stage}_step{global_step}"
                save_start = time.monotonic()
                _write_checkpoint(destination, model, tokenizer, metadata)
                metadata["save_seconds"] = time.monotonic() - save_start
                write_json(destination / "stage_state.json", metadata)
                print(json.dumps({"event": "pilot_complete", "checkpoint": str(destination), **metadata}), flush=True)
                return metadata
        if is_cuda:
            torch.cuda.synchronize(device)
        epoch_metrics = {"epoch": epoch, "examples": epoch_examples, "response_tokens": epoch_tokens,
                         "input_tokens": epoch_input_tokens, "train_loss": epoch_loss / epoch_tokens,
                         "train_seconds": time.monotonic() - epoch_start}
        history.append(epoch_metrics)
        complete = epoch == epochs
        metadata = {"identity": identity, "stage": stage, "epoch": epoch, "global_step": global_step,
                    "expected_steps_per_epoch": steps_per_epoch, "complete": complete, "epoch_complete": True,
                    "dataset_rows": len(examples), "effective_batch_size": effective_batch,
                    "micro_batch_size": micro_batch, "history": history,
                    "loss_description": "training-set completion/response token mean; not held-out performance",
                    "max_memory_allocated": torch.cuda.max_memory_allocated(device) if is_cuda else None}
        destination = checkpoint_root / _checkpoint_name(stage, epoch, epochs)
        save_start = time.monotonic()
        _write_checkpoint(destination, model, tokenizer, metadata,
                          optimizer=None if complete else optimizer, scheduler=scheduler)
        epoch_metrics["save_seconds"] = time.monotonic() - save_start
        write_json(destination / "stage_state.json", metadata)
        # Only the latest incomplete epoch needs AdamW state. Keep every model.
        if latest and (latest / "training_state.pt").exists():
            (latest / "training_state.pt").unlink()
        latest = destination
        print(json.dumps({"event": "epoch_complete", "checkpoint": str(destination), **epoch_metrics}), flush=True)
    write_json(run_dir / f"training_{stage}.json", metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--stage", choices=("parent", "target", "control", "all"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    if args.max_steps is not None and args.stage == "all":
        parser.error("A partial pilot must choose one stage; its parent is ineligible for target training")
    for stage in (("parent", "target", "control") if args.stage == "all" else (args.stage,)):
        train_stage(args.run_dir, stage, args.device, args.max_steps)


if __name__ == "__main__":
    main()
