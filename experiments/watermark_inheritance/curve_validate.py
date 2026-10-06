"""Independently verify the 600k single-pass curve audits and matched exposures."""
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.special import gammaln, logsumexp


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9), (actual, expected)


def binomial_log_sum(n, lo, hi, probability):
    k = np.arange(lo, hi + 1, dtype=np.float64)
    terms = (gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
             + k * math.log(probability) + (n - k) * math.log1p(-probability))
    return float(logsumexp(terms))


def validate(root, require_matched_through=0):
    BASE = Path(root).resolve()
    exposures = {}
    records = []
    for run in sorted(BASE.glob("run_*")):
        config = read(run / "config.json")
        assert config == read(run / "config.lock.yaml")
        assert config["epochs"] == 1 and config["clean_rows"] == 600000
        source = []
        ties = 0
        with (run / "prepared/pairs.jsonl").open() as stream:
            for line in stream:
                pair = json.loads(line)
                ties += pair["s_ids"] == pair["s_prime_ids"]
                source.append({key: pair[key] for key in
                               ("pair_id", "source_id", "audit_split", "s_length", "s_prime_length")})
        assert len(source) == 10000
        assert len({p["source_id"] for p in source}) == 10000
        calibration_ids = [p["pair_id"] for p in source if p["audit_split"] == "calibration"]
        assert len(calibration_ids) == 2000
        metrics_paths = sorted((run / "scores").glob("*/metrics.json"))
        allowed = {f"{stage}_step{step}" for stage in ("target", "control") for step in config["checkpoint_steps"]}
        assert {p.parent.name for p in metrics_paths} <= allowed
        for path in metrics_paths:
            folder = path.parent
            metrics = read(path)
            thresholds = read(folder / "thresholds.json")
            identity = read(folder / "predictions.meta.json")
            examples = rows(folder / "examples.jsonl")
            decisions = rows(folder / "audit_decisions.jsonl")
            assert metrics["schema"] == identity["schema"] == thresholds["schema"] == 2
            assert not metrics["pilot"]
            assert metrics["run_id"] == run.name
            assert metrics["primary_comparison"] == (folder.name == config["primary_checkpoint_name"])
            state = read(run / "checkpoints" / folder.name / "stage_state.json")
            stage, step = folder.name.split("_step")
            step = int(step)
            assert state["stage"] == stage and state["global_step"] == step
            assert state["checkpoint_complete"] and not state["pilot"]
            assert state["examples_seen"] == state["unique_examples_seen"] == step * 32
            assert state["dataset_rows"] == 600000 and state["epoch"] == 1
            assert state["complete"] == (step == config["max_steps"])
            assert state["identity"]["data_sha256"] == read(BASE / "setup/data_manifest.json")["output_file_sha256"]
            assert identity["config_sha256"] == fingerprint(config)
            assert identity["pairs_sha256"] == digest(run / "prepared/pairs.jsonl")
            checkpoint = (run / "checkpoints" / folder.name).resolve()
            assert identity["checkpoint_path"] == str(checkpoint)
            for filename, size, modified in identity["checkpoint_files"]:
                stat = (checkpoint / filename).stat()
                assert (stat.st_size, stat.st_mtime_ns) == (size, modified)
            exposures[(run.name, stage, step)] = state
            assert metrics["pairs"] == 10000 and metrics["calibration_pairs"] == 2000
            assert metrics["audit_pairs"] == 8000
            assert metrics["prediction_identity"] == thresholds["prediction_identity"] == fingerprint(identity)
            assert identity["scorer_sha256"] == digest(folder / "scoring_source.py")
            assert thresholds["calibration_pair_ids_sha256"] == fingerprint(calibration_ids)
            assert thresholds["radio_token_budget"] == 100000
            assert thresholds["neutral_n0_score"] == 0
            assert len(examples) == 10000 and len(decisions) == 8000
            for name, expected in metrics["artifact_sha256"].items():
                assert digest(folder / name) == expected
            assert (folder / "thresholds.json").stat().st_mtime_ns <= (folder / "audit_decisions.jsonl").stat().st_mtime_ns
            gamma = thresholds["gamma_exact"]
            close(gamma, 12569 / 50277)
            for datum, example in zip(source, examples):
                for key in ("pair_id", "source_id", "audit_split"):
                    assert datum[key] == example[key]
                for side in ("s", "s_prime"):
                    score = example[side]
                    n, g = score["eligible"], score["green"]
                    assert 0 <= g <= n <= datum[side + "_length"]
                    assert score["completion_tokens"] == datum[side + "_length"]
                    z = (g - gamma * n) / math.sqrt(n * gamma * (1 - gamma)) if n else 0.0
                    close(score["z"], z)
                    assert math.isfinite(score["min_k20"])
            calibration = [e for e in examples if e["audit_split"] == "calibration"]
            audit = [e for e in examples if e["audit_split"] == "audit"]
            assert len(audit) == 8000
            assert metrics["diagnostics"]["audit_source_groups"] == 8000
            assert metrics["diagnostics"]["identical_completion_pairs"] == ties
            for side in ("s", "s_prime"):
                assert metrics["diagnostics"][side]["audit_n0_count"] == sum(e[side]["eligible"] == 0 for e in audit)
            for detector, field in (("watermark", "z"), ("min_k20", "min_k20")):
                frozen = thresholds["detectors"][detector]
                assert frozen["comparison"] == ">" and frozen["calibration_pairs"] == 2000
                c = frozen["threshold"]
                close(frozen["balanced_accuracy"],
                      (sum(e["s"][field] > c for e in calibration)
                       + sum(e["s_prime"][field] <= c for e in calibration)) / 4000)
                positives = negatives = s_yes = sp_yes = raw_ties = 0
                for example, decision in zip(audit, decisions):
                    assert decision["pair_id"] == example["pair_id"]
                    ds, dp = decision[detector]["s"], decision[detector]["s_prime"]
                    assert type(ds) is bool and type(dp) is bool
                    assert ds == (example["s"][field] > c)
                    assert dp == (example["s_prime"][field] > c)
                    s_yes += ds
                    sp_yes += dp
                    positives += ds and not dp
                    negatives += dp and not ds
                    raw_ties += example["s"][field] == example["s_prime"][field]
                result = metrics["shard"][detector]
                close(result["threshold"], c)
                close(result["a"], s_yes / 8000)
                close(result["a_prime"], sp_yes / 8000)
                close(result["signed_gap"], (s_yes - sp_yes) / 8000)
                close(result["advantage"], abs(s_yes - sp_yes) / 8000)
                discordant = positives + negatives
                assert result["discordant_pairs"] == discordant
                assert result["positive_pairs"] == positives and result["negative_pairs"] == negatives
                assert result["decision_ties"] == 8000 - discordant and result["raw_score_ties"] == raw_ties
                logp = min(0.0, math.log(2) + binomial_log_sum(discordant, 0, min(positives, negatives), .5)) if discordant else 0.0
                close(result["log10_p_shard"], logp / math.log(10))
                close(result["p_shard"], math.exp(logp))
            assert {(r["probe"], r["key"]) for r in metrics["radioactivity"]} == {
                (p, k) for p in ("s", "s_prime") for k in ("correct", "wrong")}
            for radio in metrics["radioactivity"]:
                n, g = radio["eligible_tokens"], radio["green_tokens"]
                assert n == 100000 and 0 <= g <= n
                close(radio["green_rate"], g / n)
                close(radio["gamma_exact"], gamma)
                close(radio["log10_p_radio"], min(0.0, binomial_log_sum(n, g, n, gamma)) / math.log(10))
            records.append({"run": run.name, "checkpoint": folder.name, "metrics_sha256": digest(path)})

    run_names = [run.name for run in sorted(BASE.glob("run_*"))]
    assert run_names == ["run_00", "run_01", "run_02"]
    steps = read(BASE / "run_00/config.json")["checkpoint_steps"]
    matched_steps = []
    for step in steps:
        present = all((run, stage, step) in exposures for run in run_names for stage in ("target", "control"))
        if present:
            matched_steps.append(step)
            for run in run_names:
                target, control = exposures[(run, "target", step)], exposures[(run, "control", step)]
                for field in ("examples_seen", "unique_examples_seen", "response_tokens_seen", "input_tokens_seen", "learning_rate"):
                    assert target[field] == control[field]
                assert target["source"] != control["source"]
        if step <= require_matched_through:
            assert present, f"Matched audits through step {require_matched_through} are incomplete at {step}"
    expected = 2 * len(run_names) * len(steps)
    result = {"complete": len(records) == expected, "matched_steps": matched_steps, "expected_checkpoints": expected, "verified_utc": datetime.now(timezone.utc).isoformat(),
              "checkpoints": len(records), "pair_scores": len(records) * 10000, "final_pair_decisions_per_detector": len(records) * 8000,
              "aggregate_tests": len(records) * 4,
              "checks": ["Single-pass exposure, checkpoint/model/config identity, and matched target/control prefixes", "Source/split/length alignment", "Frozen calibration membership and accuracy",
                         "Per-example z reconstruction and neutral scores", "All strict-threshold final decisions",
                         "Counts, signed gaps, advantages and within-pair outcomes",
                         "Exact paired and aggregate p-values by independent finite binomial sums",
                         "Equal 100k aggregate budgets with both probes and keys", "Artifact and scorer hashes"],
              "records": records}
    (BASE / "setup/curve_scoring_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/watermark_curve_distinct"))
    parser.add_argument("--require-matched-through", type=int, default=0)
    args = parser.parse_args()
    validate(args.root, args.require_matched_through)
