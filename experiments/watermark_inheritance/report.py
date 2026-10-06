"""Tables and plots for watermark survival and the distinct shard advantage."""
from __future__ import annotations

import argparse
import csv
import io
import os
from pathlib import Path

from .common import load_config, load_json, write_json

CHECKPOINT_ORDER = ["B", "P", "T1", "T2", "T", "T0"]


def table_rows(metrics):
    rows = []
    for result in metrics:
        for score_name, shard in result["shard"].items():
            for radio in result["radioactivity"]:
                rows.append({"run": result["run_id"], "checkpoint": result["checkpoint"],
                    "score": score_name,
                    "primary": result["primary_comparison"] and score_name == "watermark"
                               and radio["probe"] == "s_prime" and radio["key"] == "correct",
                    "a": shard["a"], "a_prime": shard["a_prime"],
                    "signed_gap": shard["signed_gap"], "advantage": shard["advantage"],
                    "p_shard": shard["p_shard"], "log10_p_shard": shard["log10_p_shard"],
                    "audit_pairs": shard["audit_pairs"], "threshold": shard["threshold"],
                    "radioactivity_probe": radio["probe"], "radioactivity_key": radio["key"],
                    "eligible_tokens": radio["eligible_tokens"], "green_rate": radio["green_rate"],
                    "gamma_exact": radio["gamma_exact"], "log10_p_radio": radio["log10_p_radio"],
                    "s_n0_rate": result["diagnostics"]["s"]["audit_n0_rate"],
                    "s_prime_n0_rate": result["diagnostics"]["s_prime"]["audit_n0_rate"],
                    "decision_ties": shard["decision_ties"], "raw_score_ties": shard["raw_score_ties"]})
    return rows


def atomic_text(path, text):
    temporary = path.with_name(path.name+f".tmp.{os.getpid()}")
    with temporary.open("w") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def plot_results(metrics, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_ids = list(dict.fromkeys(result["run_id"] for result in metrics))
    colors = {run_id: f"C{index}" for index, run_id in enumerate(run_ids)}
    labels = ["B\npretrained", "P\nprivate S", "T1\nDolly 1", "T2\nDolly 2", "T\nDolly 3", "T0\ncontrol"]
    for plot_name in ("advantage", "radioactivity"):
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
        for run_id in run_ids:
            by_checkpoint = {result["checkpoint"]: result for result in metrics if result["run_id"] == run_id}
            present = [(i, by_checkpoint[name]) for i, name in enumerate(CHECKPOINT_ORDER) if name in by_checkpoint]
            for axis_index, axis in enumerate(axes):
                if plot_name == "advantage":
                    score = ("watermark", "min_k20")[axis_index]
                    values = [result["shard"][score]["advantage"] for _, result in present]
                    axis.set_title(("Watermark shard distinguisher", "MIN-K 20% (diagnostic)")[axis_index])
                    axis.set_ylabel("Absolute final-audit shard advantage")
                    # Conservative reference only, not the exact paired p-value.
                    if len(run_ids) == 1 and present:
                        import math
                        m = present[0][1]["audit_pairs"]
                        axis.axhline(math.sqrt(2*math.log(40)/m), color="0.65", linestyle=":",
                                     label="Paired Hoeffding α=0.05")
                else:
                    probe = ("s_prime", "s")[axis_index]
                    values = [next(-r["log10_p_radio"] for r in result["radioactivity"]
                                   if r["probe"] == probe and r["key"] == "correct") for _, result in present]
                    wrong = [next(-r["log10_p_radio"] for r in result["radioactivity"]
                                  if r["probe"] == probe and r["key"] == "wrong") for _, result in present]
                    wrong_branch = [(i, v) for (i, _), v in zip(present, wrong) if i < 5]
                    axis.plot([i for i, _ in wrong_branch], [v for _, v in wrong_branch], "x--",
                              color=colors[run_id], alpha=.55, label=f"{run_id}: wrong key")
                    for (i, _), value in zip(present, wrong):
                        if i == 5:
                            axis.scatter([i], [value], marker="x", color=colors[run_id], alpha=.55)
                    axis.set_title(("Unseen S′ probe (primary)", "Supervised S probe")[axis_index])
                    axis.set_ylabel("−log₁₀(p radio); higher is stronger evidence")
                # T0 is an independent branch, so do not connect it as an epoch.
                inherited = [(i, v) for (i, _), v in zip(present, values) if i < 5]
                if inherited:
                    axis.plot([i for i, _ in inherited], [v for _, v in inherited], "o-",
                              color=colors[run_id], label=run_id)
                control = [(i, v) for (i, _), v in zip(present, values) if i == 5]
                for i, value in control:
                    axis.scatter([i], [value], marker="s", color=colors[run_id])
                axis.set_xticks(range(6), labels)
                axis.grid(alpha=.2)
        for axis in axes:
            # Setting limits before plotting disables autoscaling and can clip
            # later series (especially the much stronger correct-key signal).
            axis.set_ylim(bottom=0, top=1 if plot_name == "advantage" else None)
            axis.legend(fontsize=8)
        figure.savefig(output_dir/f"{plot_name}.png", dpi=180)
        figure.savefig(output_dir/f"{plot_name}.pdf")
        plt.close(figure)


def make_report(run_dirs, output_dir=None, allow_partial=False):
    run_dirs = [Path(path) for path in run_dirs]
    if output_dir is None:
        if len(run_dirs) != 1:
            raise ValueError("An explicit output directory is needed for a combined report")
        output_dir = run_dirs[0]/"reports"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, manifest, missing = [], [], []
    for run_dir in run_dirs:
        config = load_config(run_dir)
        run_metrics = []
        for checkpoint in CHECKPOINT_ORDER:
            path = run_dir/"scores"/checkpoint/"metrics.json"
            if not path.exists():
                missing.append(f"{config['run_id']}/{checkpoint}")
                continue
            result = load_json(path)
            if result["pilot"] or result["checkpoint"] != checkpoint or result["run_id"] != config["run_id"]:
                raise ValueError(f"Invalid final metrics identity: {path}")
            if (result["pairs"] != config["pairs"] or result["calibration_pairs"] != config["calibration_pairs"]
                    or result["audit_pairs"] != config["pairs"]-config["calibration_pairs"]):
                raise ValueError(f"Incomplete calibration/audit evidence in {path}")
            thresholds = load_json(run_dir/"scores"/checkpoint/"thresholds.json")
            if thresholds["prediction_identity"] != result["prediction_identity"]:
                raise ValueError(f"Metrics and frozen threshold identities differ: {path}")
            if any(row["eligible_tokens"] != config["radio_token_budget"] for row in result["radioactivity"]):
                raise ValueError(f"Aggregate comparisons are not matched to the fixed budget: {path}")
            run_metrics.append(result)
        metrics.extend(run_metrics)
        manifest.append({"run_id": config["run_id"], "run_dir": str(run_dir.resolve()),
                         "config_path": str((run_dir/"config.json").resolve()),
                         "completed_checkpoints": [row["checkpoint"] for row in run_metrics]})
    if missing and not allow_partial:
        raise ValueError("Missing final checkpoint results: "+", ".join(missing))
    if not metrics:
        raise ValueError("No completed checkpoint metrics available")
    rows = table_rows(metrics)
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(output_dir/"results.csv", text.getvalue())
    primary_table = [row for row in rows if row["score"] == "watermark"
                     and row["radioactivity_probe"] == "s_prime" and row["radioactivity_key"] == "correct"]
    primary_text = io.StringIO(newline="")
    primary_writer = csv.DictWriter(primary_text, fieldnames=list(rows[0]))
    primary_writer.writeheader()
    primary_writer.writerows(primary_table)
    atomic_text(output_dir/"main_results.csv", primary_text.getvalue())
    report = {"complete": not missing, "missing": missing, "runs": manifest,
              "primary_comparison": "T (Dolly epoch 3), watermark shard score; correct-key S-prime radioactivity",
              "notes": ["Other checkpoint/score comparisons are diagnostic; no multiplicity adjustment is claimed.",
                        "Radioactivity is model-level evidence of a common watermark, not shard advantage.",
                        "Exact p_shard conditions on discordant pairs and swaps labels within each pair.",
                        "Finite log10 p-values are retained when ordinary p-values underflow.",
                        "No confidence in false-positive rates is inferred from only three controls."],
              "table": rows, "checkpoint_metrics": metrics}
    write_json(output_dir/"results.json", report)
    plot_results(metrics, output_dir)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir", type=Path)
    group.add_argument("--run-dirs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    result = make_report(args.run_dirs or [args.run_dir], args.output_dir, args.allow_partial)
    print(f"Report: {len(result['checkpoint_metrics'])} checkpoints; complete={result['complete']}")


if __name__ == "__main__":
    main()
