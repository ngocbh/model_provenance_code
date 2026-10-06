"""Render a compact paper figure with means and sample standard deviations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev


def summarize(rows):
    run_ids = sorted({r["run_id"] for r in rows})
    if len(run_ids) < 2:
        raise ValueError("At least two runs are required for sample standard deviation")
    groups = {}
    for row in rows:
        stage, step, run = row["stage"], row["steps"], row["run_id"]
        if stage not in ("target", "control"):
            raise ValueError(f"Unknown stage: {stage}")
        value = row["watermark_advantage"]
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid advantage: {value}")
        group = groups.setdefault((stage, step), {})
        if run in group:
            raise ValueError(f"Duplicate observation: {run}, {stage}, {step}")
        group[run] = row
    summary, excluded = [], []
    for step in sorted({r["steps"] for r in rows}):
        if any(set(groups.get((stage, step), {})) != set(run_ids)
               for stage in ("target", "control")):
            excluded.append(step)
            continue
        matched = [groups[(stage, step)][run] for stage in ("target", "control") for run in run_ids]
        for field in ("examples_seen", "unique_examples_seen"):
            if len({r[field] for r in matched}) != 1:
                raise ValueError(f"Mismatched {field} at step {step}")
        for stage in ("target", "control"):
            values = [groups[(stage, step)][run]["watermark_advantage"] for run in run_ids]
            summary.append({"stage": stage, "steps": step,
                            "examples_seen": matched[0]["examples_seen"],
                            "unique_examples_seen": matched[0]["unique_examples_seen"],
                            "n": len(values), "mean_advantage": mean(values),
                            "std_advantage": stdev(values)})
    if not summary:
        raise ValueError("No complete matched checkpoints are available")
    return summary, run_ids, excluded


def render(input_path, output_prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, MultipleLocator, StrMethodFormatter
    import numpy as np

    input_path, output_prefix = Path(input_path), Path(output_prefix)
    raw = input_path.read_bytes()
    summary, run_ids, excluded = summarize(json.loads(raw)["rows"])
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    style = {"font.family": "serif", "font.serif": ["STIXGeneral"],
             "font.size": 10, "axes.labelsize": 11.5, "xtick.labelsize": 10,
             "ytick.labelsize": 10, "legend.fontsize": 10,
             "axes.spines.top": False, "axes.spines.right": False,
             "axes.linewidth": .65, "axes.edgecolor": "#4B5563",
             "text.color": "#20252B", "axes.labelcolor": "#20252B",
             "xtick.color": "#20252B", "ytick.color": "#20252B",
             "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
             "savefig.facecolor": "white", "figure.facecolor": "white"}
    with plt.rc_context(style):
        figure, axis = plt.subplots(figsize=(3.8, 2.8), layout="constrained")
        lower, upper = [], []
        for stage, label, color, line, marker in (
                ("target", "Target", "#286B9E", "-", "o"),
                ("control", "Control", "#747C85", "--", "s")):
            selected = [r for r in summary if r["stage"] == stage]
            x = np.array([r["steps"] for r in selected])
            y = np.array([r["mean_advantage"] * 100 for r in selected])
            sd = np.array([r["std_advantage"] * 100 for r in selected])
            lower.extend(y - sd)
            upper.extend(y + sd)
            axis.fill_between(x, y - sd, y + sd, color=color, alpha=.18, linewidth=0, zorder=2)
            axis.plot(x, y, color=color, linestyle=line, linewidth=1.7,
                      marker=marker, markersize=3.8, markeredgecolor="white",
                      markeredgewidth=.4, label=label, zorder=3)
        maximum_step = max(1, max(r["steps"] for r in summary))
        axis.set(xlabel="Fine-tuning steps", ylabel="Watermark advantage (%)",
                 xlim=(-.025 * maximum_step, 1.025 * maximum_step),
                 ylim=(min(-2, min(lower) - .8), max(80, math.ceil(max(upper) / 10) * 10)))
        axis.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        axis.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        axis.yaxis.set_major_locator(MultipleLocator(20))
        axis.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#E5E7EB", linewidth=.5)
        axis.tick_params(direction="out", length=3, width=.6, pad=3)
        axis.legend(loc="upper right", frameon=False, handlelength=2.2,
                    borderaxespad=.25, labelspacing=.3)
        files = []
        for extension in ("pdf", "svg", "png"):
            path = output_prefix.with_suffix(f".{extension}")
            figure.savefig(path, dpi=600, bbox_inches="tight", pad_inches=.03)
            files.append(path)
        plt.close(figure)
    table = output_prefix.with_suffix(".csv")
    with table.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    files.append(table)
    metadata = {"input_file": input_path.name, "input_sha256": hashlib.sha256(raw).hexdigest(),
                "run_ids": run_ids, "standard_deviation_ddof": 1,
                "band": "Mean +/- one sample standard deviation across runs; not a confidence interval",
                "included_steps": sorted({r["steps"] for r in summary}), "excluded_incomplete_steps": excluded,
                "files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    output_prefix.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("experiments/watermark_inheritance/results/2026-10-06-distinct/curve.json"))
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()
    prefix = args.output_prefix or args.input.with_name("advantage_vs_steps_paper")
    print(json.dumps(render(args.input, prefix), indent=2))
