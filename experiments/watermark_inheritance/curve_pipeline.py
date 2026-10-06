"""Train independent trajectories while a separate GPU drains checkpoint audits."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

from .common import load_config, load_json, write_json
from .pipeline import command, run_command


def submit_score(root, run, name, checkpoint):
    record = root / "setup/score_jobs" / f"{run.name}_{name}.json"
    if record.exists():
        return load_json(record)["job_id"]
    script = Path(__file__).with_name("curve_score.sbatch").resolve()
    cmd = ["sbatch", "--parsable", f"--job-name=wmc_{run.name}_{name}",
           f"--output={root}/setup/logs/score_%j.out", f"--error={root}/setup/logs/score_%j.err",
           str(script), sys.executable, str(root), str(run), name, str(checkpoint)]
    result = subprocess.run(cmd, text=True, capture_output=True, check=True)
    job_id = result.stdout.strip().split(";")[0]
    if not job_id.isdigit():
        raise ValueError(f"Unexpected sbatch response: {result.stdout}")
    write_json(record, {"job_id": job_id, "checkpoint": name, "run_dir": str(run), "command": cmd,
                        "submitted_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"event": "cpu_audit_submitted", "job_id": job_id, "run": run.name, "checkpoint": name}), flush=True)
    return job_id


def pipeline(root, gpus):
    import torch
    root = Path(root).resolve()
    actual = torch.cuda.device_count()
    if actual != gpus or gpus < 2:
        raise ValueError(f"Requested {gpus} GPUs, visible {actual}; training and evaluation need separate devices")
    runs = [root / f"run_{i:02d}" for i in range(3)]
    configs = [load_config(run) for run in runs]
    source_manifest = load_json(root / "setup/source_manifest.json")
    from .prepare import file_sha256
    for relative, expected in source_manifest["files"].items():
        if file_sha256(Path.cwd() / relative) != expected:
            raise ValueError(f"Frozen source file changed: {relative}")
    jobs = [(run, stage) for stage in ("target", "control") for run in runs]
    pending = [(run, stage, step) for run, config in zip(runs, configs)
               for stage in ("target", "control") for step in config["checkpoint_steps"]]
    stopped = threading.Event()
    evaluation_error = []
    submitted = []
    started = time.monotonic()
    write_json(root / "setup/gpu_runtime.json", {"job_id": __import__("os").environ.get("SLURM_JOB_ID"),
               "started_utc": datetime.now(timezone.utc).isoformat(), "gpus": gpus,
               "training_workers": gpus - 1, "inference_device": gpus - 1,
               "devices": [torch.cuda.get_device_name(i) for i in range(gpus)],
               "source_root": str(Path.cwd()), "source_manifest": source_manifest})

    def evaluate():
        try:
            while pending:
                ready = []
                for run, stage, step in pending:
                    checkpoint = run / "checkpoints" / f"{stage}_step{step}"
                    if (checkpoint / "stage_state.json").exists():
                        ready.append((run, stage, step))
                if not ready:
                    if stopped.is_set():
                        raise RuntimeError(f"Training ended with {len(pending)} checkpoints missing")
                    time.sleep(10)
                    continue
                # Earlier exposures first; all checkpoint directories are published atomically.
                for run, stage, step in sorted(ready, key=lambda item: item[2]):
                    name = f"{stage}_step{step}"
                    checkpoint = run / "checkpoints" / name
                    run_command(command("score", "--run-dir", run, "--checkpoint-name", name,
                                        "--checkpoint-path", checkpoint, "--predictions-only", "--device", f"cuda:{gpus - 1}"),
                                run / "logs" / f"predict_{name}.log")
                    submitted.append(submit_score(root, run, name, checkpoint))
                    pending.remove((run, stage, step))
        except BaseException as error:
            evaluation_error.append(error)
            print(json.dumps({"event": "evaluation_error", "error": repr(error)}), flush=True)

    def worker(device):
        # Thread-safe iterator; tasks are submitted before workers start.
        while True:
            with job_lock:
                if not jobs:
                    return
                run, stage = jobs.pop(0)
            run_command(command("curve_train", "--run-dir", run, "--stage", stage, "--device", f"cuda:{device}"),
                        run / "logs" / f"train_{stage}.log")

    job_lock = threading.Lock()
    evaluator = threading.Thread(target=evaluate, name="checkpoint-evaluation")
    evaluator.start()
    training_errors = []
    with ThreadPoolExecutor(max_workers=gpus - 1) as executor:
        futures = [executor.submit(worker, device) for device in range(gpus - 1)]
        for future in futures:
            try:
                future.result()
            except BaseException as error:
                training_errors.append(repr(error))
    stopped.set()
    evaluator.join()
    write_json(root / "setup/gpu_completion.json", {"training_errors": training_errors,
               "evaluation_errors": list(map(repr, evaluation_error)), "cpu_score_jobs": submitted,
               "seconds": time.monotonic() - started})
    if training_errors or evaluation_error:
        raise RuntimeError(f"Trajectory pipeline failed: {training_errors}, {evaluation_error}")
    result = subprocess.run(["sbatch", "--parsable", "--job-name=wm_curve_final",
                "--dependency=afterany:" + ":".join(submitted),
                f"--output={root}/setup/logs/final_%j.out", f"--error={root}/setup/logs/final_%j.err",
                str(Path(__file__).with_name("curve_final.sbatch").resolve()), sys.executable, str(root)],
                check=True, text=True, capture_output=True)
    write_json(root / "setup/final_job.json", {"job_id": result.stdout.strip().split(";")[0], "score_jobs": submitted})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--gpus", type=int, default=4)
    args = parser.parse_args()
    pipeline(args.root, args.gpus)
