"""Execute pilot and GPU stages with at most the allocated 2 or 4 devices."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time

from .common import load_config, load_json, read_jsonl, write_json, write_jsonl

BASE = Path("artifacts/watermark_inheritance")
MODULE = "experiments.watermark_inheritance."
CHECKPOINTS = {"B": None, "P": "parent", "T1": "target_epoch1", "T2": "target_epoch2",
               "T": "target_epoch3", "T0": "control_epoch3"}


def command(module, *args):
    return [sys.executable, "-u", "-m", MODULE + module, *map(str, args)]


def run_command(cmd, log):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    print(json.dumps({"event": "start", "command": cmd, "log": str(log),
                      "utc": datetime.now(timezone.utc).isoformat()}), flush=True)
    with log.open("a") as stream:
        stream.write("\nCOMMAND " + json.dumps(cmd) + "\n")
        stream.flush()
        subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, check=True)
    elapsed = time.monotonic() - start
    print(json.dumps({"event": "finished", "log": str(log), "seconds": elapsed}), flush=True)
    return elapsed


def parallel(tasks, gpus):
    devices = queue.Queue()
    for device in range(gpus):
        devices.put(device)

    def execute(task):
        device = devices.get()
        try:
            cmd, log = task
            return run_command(cmd + ["--device", f"cuda:{device}"], log)
        finally:
            devices.put(device)
    with ThreadPoolExecutor(max_workers=gpus) as pool:
        return list(pool.map(execute, tasks))


def generation(run_dir, phase, gpus):
    parallel([(command("generate", "--run-dir", run_dir, "--phase", phase,
                       "--rank", rank, "--world-size", gpus),
               run_dir / "logs" / f"generate_{phase}_{rank}.log") for rank in range(gpus)], gpus)
    run_command(command("generate", "--run-dir", run_dir, "--phase", phase,
                        "--world-size", gpus, "--merge"), run_dir / "logs" / f"merge_{phase}.log")


def snapshot_sources(destination):
    import shutil
    destination.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for source in Path(__file__).parent.glob("*.py"):
        hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
        shutil.copy2(source, destination / source.name)
    write_json(destination / "sha256.json", hashes)


def pilot_training_tasks(run, config, max_steps=3):
    """Resume the pilot by reusing only fully written, matching pilot artifacts."""
    from .train import _identity

    tasks = []
    for stage in ("parent", "control"):
        checkpoint = run / "checkpoints" / f"pilot_{stage}_step{max_steps}"
        if checkpoint.exists():
            metadata = load_json(checkpoint / "stage_state.json")
            data = run / "prepared" / ("pairs.jsonl" if stage == "parent" else "dolly.jsonl")
            if (metadata.get("identity") != _identity(config, data, stage)
                    or not metadata.get("pilot") or metadata.get("complete")
                    or metadata.get("epoch_complete") or metadata.get("global_step") != max_steps
                    or not list(checkpoint.glob("*.safetensors"))):
                raise ValueError(f"Existing pilot checkpoint does not match this pilot: {checkpoint}")
            print(json.dumps({"event": "reuse_pilot_training", "checkpoint": str(checkpoint)}), flush=True)
            continue
        tasks.append((command("train", "--run-dir", run, "--stage", stage,
                              "--max-steps", max_steps), run / "logs" / f"train_{stage}.log"))
    return tasks


def pilot(gpus):
    full = BASE / "run_00"
    generation(full, "pilot", gpus)
    run = BASE / "pilot"
    pairs = read_jsonl(full / "prepared/pilot_pairs.jsonl")
    calibration = len(pairs) // 4
    for index, row in enumerate(pairs):
        row["audit_split"] = "calibration" if index < calibration else "audit"
    config = dict(load_config(full), run_id="pilot", pilot=True, pairs=len(pairs),
                  calibration_pairs=calibration, radio_token_budget=1000, log_every=1)
    if (run / "config.json").exists() and load_config(run) != config:
        raise ValueError("Existing pilot configuration mismatch")
    write_json(run / "config.json", config)
    write_json(run / "config.lock.yaml", config)
    write_jsonl(run / "prepared/pairs.jsonl", pairs)
    run_command(command("prepare", "--run-dir", run, "--stage", "dolly"), run / "logs/prepare_dolly.log")
    parallel(pilot_training_tasks(run, config), gpus)
    parallel([(command("score", "--run-dir", run, "--checkpoint-name", name,
                       "--checkpoint-path", path, "--limit", len(pairs)), run / "logs" / f"score_{name}.log")
              for name, path in [("B", config["model_path"]),
                                 ("P", run / "checkpoints/pilot_parent_step3")]], gpus)
    # Explicitly verify that training changed real weights after warmup and that
    # the saved checkpoint was reloaded successfully by the reading-mode scorer.
    from safetensors import safe_open
    import torch
    tensor_name = "gpt_neox.layers.0.attention.dense.weight"

    def sample_weight(path):
        path = Path(path)
        index = path / "model.safetensors.index.json"
        filename = load_json(index)["weight_map"][tensor_name] if index.exists() else "model.safetensors"
        with safe_open(path / filename, framework="pt", device="cpu") as stream:
            return stream.get_slice(tensor_name)[:32, :32].clone()

    difference = (sample_weight(config["model_path"]) - sample_weight(run / "checkpoints/pilot_parent_step3")).abs().max().item()
    if difference == 0 or not torch.isfinite(torch.tensor(difference)):
        raise ValueError("Pilot did not demonstrate a finite parameter update")
    generated = load_json(full / "prepared/pilot_pairs.manifest.json")
    training = {stage: load_json(run / "checkpoints" / f"pilot_{stage}_step3/stage_state.json")
                for stage in ["parent", "control"]}
    # The candidate 100k aggregate budget was prespecified before the pilot.
    # Confirm a fourfold safety margin from pilot eligible-token density.
    baseline = load_json(run / "scores" / f"pilot_B_{len(pairs)}/metrics.json")
    result = {"complete": True, "gpus": gpus, "generation": generated,
              "training": training, "max_parameter_change": difference,
              "baseline_metrics": baseline, "main_radio_token_budget": 100000,
              "completed_utc": datetime.now(timezone.utc).isoformat()}
    write_json(run / "pilot_summary.json", result)
    print(json.dumps({"event": "pilot_complete", "summary": str(run / "pilot_summary.json")}), flush=True)


def main_gpu(gpus):
    summary = load_json(BASE / "pilot/pilot_summary.json")
    pilot_stats = summary["generation"]["pilot"]
    if (not summary.get("complete") or summary["max_parameter_change"] <= 0
            or pilot_stats["correct_key"]["green_rate"] <= 0.4
            or pilot_stats["correct_key"]["log10_p"] >= -6
            or abs(pilot_stats["wrong_key"]["green_rate"] - pilot_stats["wrong_key"]["gamma_exact"]) > 0.05
            or pilot_stats["mean_length"] < 32
            or pilot_stats["mean_repeated_4gram_fraction"] > 0.5):
        raise ValueError("Pilot validation failed; inspect pilot_summary.json before the full run")
    pilot_config = load_config(BASE / "pilot")
    examples = read_jsonl(BASE / "pilot/scores" / f"pilot_B_{pilot_config['pairs']}/examples.jsonl")
    audit = [row for row in examples if row["audit_split"] == "audit"]
    min_mean_eligible = min(sum(row[side]["eligible"] for row in audit) / len(audit)
                            for side in ["s", "s_prime"])
    if min_mean_eligible * 8000 < 4 * 100000:
        raise ValueError("Pilot does not support the candidate aggregate token budget with safety margin")
    write_json(BASE / "pilot/validation.json", {"pilot_verified": True,
        "main_radio_token_budget": 100000, "pilot_min_mean_eligible": min_mean_eligible,
        "budget_rule": "100k candidate confirmed by >=4x projected eligible-token margin",
        "utc": datetime.now(timezone.utc).isoformat()})
    runs = [BASE / f"run_{index:02d}" for index in range(3)]
    for run in runs:
        generation(run, "full", gpus)
    parallel([(command("train", "--run-dir", run, "--stage", stage),
               run / "logs" / f"train_{stage}.log")
              for run in runs for stage in ["parent", "control"]], gpus)
    parallel([(command("train", "--run-dir", run, "--stage", "target"),
               run / "logs/train_target.log") for run in runs], gpus)
    tasks = []
    for run in runs:
        for name, folder in CHECKPOINTS.items():
            path = run / "checkpoints" / folder if folder else load_config(run)["model_path"]
            tasks.append((command("score", "--run-dir", run, "--checkpoint-name", name,
                                  "--checkpoint-path", path, "--predictions-only"),
                          run / "logs" / f"predict_{name}.log"))
    parallel(tasks, gpus)
    write_json(BASE / "gpu_complete.json", {"complete": True, "gpus": gpus,
               "job_id": os.environ.get("SLURM_JOB_ID"), "runs": list(map(str, runs)),
               "utc": datetime.now(timezone.utc).isoformat()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["pilot", "main"], required=True)
    parser.add_argument("--gpus", type=int, choices=[2, 4], required=True)
    args = parser.parse_args()
    import torch
    observed_gpus = torch.cuda.device_count()
    if observed_gpus != args.gpus:
        if args.phase != "main" or observed_gpus not in (2, 4):
            raise ValueError(f"Expected {args.gpus} allocated GPUs; see {observed_gpus}")
        # A pending Slurm request can be resized without losing its job ID and
        # dependencies. Never repartition a partially generated full dataset.
        old_parts = list(BASE.glob(f"run_*/generated/full.rank*-of-{args.gpus:03d}.jsonl"))
        if old_parts:
            raise ValueError("Cannot change the GPU count after full generation has begun")
        print(json.dumps({"event": "allocation_resized_before_start", "original_gpus": args.gpus,
                          "allocated_gpus": observed_gpus}), flush=True)
        args.gpus = observed_gpus
    source_snapshot = BASE / "setup" / f"source_{args.phase}_{os.environ.get('SLURM_JOB_ID', 'local')}"
    snapshot_sources(source_snapshot)
    if args.phase == "main":
        for index in range(3):
            run = BASE / f"run_{index:02d}"
            write_json(run / "runtime.json", {"source_snapshot": str(source_snapshot.resolve()),
                "package_versions": str((BASE / "setup/packages.txt").resolve()),
                "code_commit": load_config(run)["code_commit"],
                "job_id": os.environ.get("SLURM_JOB_ID"), "gpus": args.gpus,
                "gpu_names": [torch.cuda.get_device_name(i) for i in range(args.gpus)],
                "utc": datetime.now(timezone.utc).isoformat()})
    (pilot if args.phase == "pilot" else main_gpu)(args.gpus)


if __name__ == "__main__":
    main()
