"""Refresh the live fine-tuning curve from complete, immutable audit artifacts."""
from __future__ import annotations

import argparse
import csv
import fcntl
import os
from pathlib import Path

from .common import load_json, write_json


def point(metrics, state, run_id, stage):
    row = {"run_id": run_id, "stage": stage, "checkpoint": metrics["checkpoint"],
           "steps": state.get("global_step", 0), "examples_seen": state.get("examples_seen", 0),
           "unique_examples_seen": state.get("unique_examples_seen", 0),
           "response_tokens_seen": state.get("response_tokens_seen", 0),
           "input_tokens_seen": state.get("input_tokens_seen", 0),
           "audit_pairs": metrics["audit_pairs"]}
    for detector in ("watermark", "min_k20"):
        row.update({f"{detector}_{key}": value for key, value in metrics["shard"][detector].items()})
    for radio in metrics["radioactivity"]:
        if radio["probe"] == "s_prime":
            row.update({f"radio_{radio['key']}_{key}": value for key, value in radio.items()
                        if key not in ("probe", "key")})
    return row


def report(root):
    root = Path(root).resolve()
    output = root / "reports"
    output.mkdir(parents=True, exist_ok=True)
    # Multiple CPU score jobs can finish together; serialize publication.
    with (output / ".report.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = []
        for run in sorted(root.glob("run_*")):
            config = load_json(run / "config.json")
            original = Path(config["parent_run_dir"])
            for stage, baseline in (("target", "P"), ("control", "B")):
                rows.append(point(load_json(original / f"scores/{baseline}/metrics.json"), {}, run.name, stage))
            for path in sorted(run.glob("scores/*/metrics.json")):
                metrics = load_json(path)
                state = load_json(run / "checkpoints" / metrics["checkpoint"] / "stage_state.json")
                if metrics["pilot"] or not state.get("checkpoint_complete"):
                    raise ValueError(f"An incomplete/pilot result cannot enter the curve: {path}")
                rows.append(point(metrics, state, run.name, state["stage"]))
        rows.sort(key=lambda r: (r["run_id"], r["stage"], r["steps"]))
        if not rows:
            return []
        temporary = output / f"curve.csv.tmp.{os.getpid()}"
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(output / "curve.csv")
        write_json(output / "curve.json", {"rows": rows, "completed_nonbaseline_points": sum(r["steps"] > 0 for r in rows),
                   "planned_nonbaseline_points": 42,
                   "interpretation": "Repeated measurements on a fixed audit; intermediate points are descriptive. All prespecified points are retained."})
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(2, 2, figsize=(12, 8))
        specifications = [
            ("examples_seen", "watermark_advantage", "Fine-tuning examples seen (includes repeats)", "Watermark shard advantage"),
            ("response_tokens_seen", "watermark_advantage", "Supervised response tokens seen", "Watermark shard advantage"),
            ("examples_seen", "min_k20_advantage", "Fine-tuning examples seen (includes repeats)", "MIN-K 20% shard advantage"),
            ("steps", "radio_correct_log10_p_radio", "Optimizer steps", "Correct-key S′ log10(p)")]
        colors = ["#2878b5", "#d47622", "#3c9563"]
        for axis, (x, y, xlabel, ylabel) in zip(axes.flat, specifications):
            for index, run_id in enumerate(sorted({r["run_id"] for r in rows})):
                for stage, style in (("target", "-"), ("control", "--")):
                    selected = [r for r in rows if r["run_id"] == run_id and r["stage"] == stage]
                    axis.plot([r[x] for r in selected], [r[y] for r in selected], style, marker="o", markersize=3,
                              color=colors[index], label=f"{run_id} {stage}", alpha=.85)
            axis.set(xlabel=xlabel, ylabel=ylabel)
            axis.grid(alpha=.2)
            if "advantage" in y:
                axis.set_ylim(0, 1)
        axes[0, 0].legend(fontsize=8, ncol=2)
        count = sum(r["steps"] > 0 for r in rows)
        figure.suptitle(f"Continuous clean fine-tuning on 200k UltraChat examples — {count}/42 audits complete")
        figure.tight_layout()
        for extension in ("png", "pdf"):
            temporary = output / f"curve.tmp.{os.getpid()}.{extension}"
            figure.savefig(temporary, dpi=180)
            temporary.replace(output / f"curve.{extension}")
        plt.close(figure)
        return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/watermark_curve"))
    args = parser.parse_args()
    report(args.root)
