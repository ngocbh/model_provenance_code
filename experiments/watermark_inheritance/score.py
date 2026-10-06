"""Unbiased reading-mode predictions, frozen calibration, and paired audits.

Run with ``python -m experiments.watermark_inheritance.score --run-dir RUN
--checkpoint-name T --checkpoint-path RUN/checkpoints/target_epoch3``.  Prediction
extraction and CPU watermark scoring can be split using --predictions-only and
--cpu-only.  Neither stage changes a model's logits with a watermark processor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

from .common import Maryland, fingerprint, load_config, load_json, read_jsonl, write_json, write_jsonl
from src.shard_audit.scoring.mia_scores import min_k_logprob

SCHEMA = 2
CHECKPOINT_PATHS = {"P": "parent", "T1": "target_epoch1", "T2": "target_epoch2",
                    "T": "target_epoch3", "T0": "control_epoch3"}


def eligible_predictions(prompt_ids, completion_ids, predictions, ngram=2):
    """Yield (context, prediction) at nonrepeated completion contexts.

    The prediction at completion offset i comes from logits[prompt_length+i-1].
    Prompt contexts whose predicted position is *inside* the prompt initialize
    the exclusion set.  The final prompt context predicts completion offset 0,
    so it is checked when processing that position, not prematurely excluded.
    The whole stored sequence is within the attention window (validated at load).
    """
    if ngram < 1 or not prompt_ids or len(completion_ids) != len(predictions):
        raise ValueError("Need a prompt, positive ngram, and one prediction per completion token")
    tokens = list(prompt_ids) + list(completion_ids)
    start = len(prompt_ids)
    seen = {tuple(tokens[position-ngram:position]) for position in range(ngram, start)}
    for offset, prediction in enumerate(predictions):
        position = start + offset
        if position < ngram:
            continue
        context = tuple(tokens[position-ngram:position])
        if context in seen:
            continue
        seen.add(context)
        yield context, int(prediction)


def example_watermark_score(prompt_ids, completion_ids, predictions, watermarker):
    records = [(context, prediction, watermarker.is_green(context, prediction))
               for context, prediction in eligible_predictions(
                   prompt_ids, completion_ids, predictions, watermarker.ngram)]
    count = len(records)
    green = sum(row[2] for row in records)
    gamma = watermarker.gamma_exact
    if not 0 < gamma < 1:
        raise ValueError("The exact greenlist fraction must be strictly between 0 and 1")
    z = (green-gamma*count) / math.sqrt(count*gamma*(1-gamma)) if count else 0.0
    return {"z": z, "green": green, "eligible": count}, records


def min_k_score(log_probabilities, fraction=0.2):
    """Use the repository's MIN-K definition, including its ceiling rule."""
    if not log_probabilities:
        return 0.0
    return min_k_logprob(log_probabilities, 100 * fraction)


def log10_binomial_survival(green, count, probability):
    """log10 P[Binomial(count, probability) >= green], without a p-value floor.

    scipy's survival function is accurate in its representable range.  For an
    underflowing tail, sum exact log-PMF terms in bounded chunks using logsumexp.
    This preserves e.g. log10(p)=-10000 instead of substituting a finite floor.
    """
    from scipy.special import gammaln, logsumexp
    from scipy.stats import binom
    import numpy as np

    if not 0 <= green <= count or not 0 < probability < 1:
        raise ValueError("Invalid binomial count or probability")
    if green == 0:
        return 0.0
    log_tail = float(binom.logsf(green-1, count, probability))
    if math.isfinite(log_tail):
        return log_tail / math.log(10)
    log_total = -math.inf
    constant = gammaln(count+1)
    for start in range(green, count+1, 8192):
        values = np.arange(start, min(start+8192, count+1), dtype=np.float64)
        log_pmf = (constant-gammaln(values+1)-gammaln(count-values+1)
                   + values*math.log(probability)+(count-values)*math.log1p(-probability))
        log_total = float(np.logaddexp(log_total, logsumexp(log_pmf)))
    return log_total / math.log(10)


def paired_label_swap(decisions_s, decisions_s_prime):
    """Exact two-sided within-pair sign swap, conditioning on discordant pairs."""
    if not decisions_s or len(decisions_s) != len(decisions_s_prime):
        raise ValueError("Need equally sized, nonempty paired decisions")
    positive = sum(bool(s) and not bool(sp) for s, sp in zip(decisions_s, decisions_s_prime))
    negative = sum(not bool(s) and bool(sp) for s, sp in zip(decisions_s, decisions_s_prime))
    discordant = positive + negative
    log10_p = min(0.0, math.log10(2)+log10_binomial_survival(
        max(positive, negative), discordant, 0.5)) if discordant else 0.0
    return {"p_shard": 10.0**log10_p, "log10_p_shard": log10_p,
            "discordant_pairs": discordant, "positive_pairs": positive,
            "negative_pairs": negative, "decision_ties": len(decisions_s)-discordant}


def calibrate_threshold(scores_s, scores_s_prime):
    """Maximize balanced accuracy for strict score > c using calibration only.

    All equally good thresholds are resolved toward the largest threshold. This
    deterministic rule includes both constant classifiers and handles score ties
    as a group; it never breaks a tie using shard identity or evaluation order.
    """
    if not scores_s or len(scores_s) != len(scores_s_prime):
        raise ValueError("Calibration requires nonempty matched pairs")
    if not all(math.isfinite(x) for x in [*scores_s, *scores_s_prime]):
        raise ValueError("Calibration scores must be finite")
    groups = {}
    for label, values in ((1, scores_s), (0, scores_s_prime)):
        for value in values:
            groups.setdefault(float(value), [0, 0])[label] += 1
    count = len(scores_s)
    true_positive, true_negative = count, 0
    best_correct = count
    threshold = math.nextafter(min(groups), -math.inf)
    for value in sorted(groups):
        negatives, positives = groups[value]
        true_positive -= positives
        true_negative += negatives
        correct = true_positive + true_negative
        if correct >= best_correct:
            best_correct, threshold = correct, value
    return {"threshold": threshold, "comparison": ">", "calibration_pairs": count,
            "balanced_accuracy": best_correct/(2*count),
            "tie_rule": "largest_threshold", "direction": "larger_predicts_s"}


def evaluate_shard(scores_s, scores_s_prime, threshold):
    decisions_s = [score > threshold for score in scores_s]
    decisions_sp = [score > threshold for score in scores_s_prime]
    count = len(decisions_s)
    if not count or len(decisions_sp) != count:
        raise ValueError("Final audit requires nonempty matched pairs")
    a, a_prime = sum(decisions_s)/count, sum(decisions_sp)/count
    return {"a": a, "a_prime": a_prime, "signed_gap": a-a_prime,
            "advantage": abs(a-a_prime), "audit_pairs": count,
            "raw_score_ties": sum(s == sp for s, sp in zip(scores_s, scores_s_prime)),
            **paired_label_swap(decisions_s, decisions_sp)}


class AggregateRadioactivity:
    """A bounded, per-checkpoint/per-probe tuple tape with a prespecified budget."""

    def __init__(self, budget, gamma_exact):
        if budget < 1:
            raise ValueError("radio_token_budget must be fixed to a positive value before audit")
        self.budget, self.gamma_exact = budget, gamma_exact
        self.seen = set()
        self.records = []

    def add(self, records):
        for context, prediction, green in records:
            if len(self.records) == self.budget:
                break
            token_tuple = (*context, prediction)
            if token_tuple not in self.seen:
                self.seen.add(token_tuple)
                self.records.append((context, prediction, green))

    def result(self, watermarker=None):
        if len(self.records) != self.budget:
            raise ValueError(f"Only {len(self.records)} distinct eligible tuples for fixed budget {self.budget}")
        green = sum(row[2] if watermarker is None else watermarker.is_green(row[0], row[1])
                    for row in self.records)
        return {"eligible_tokens": self.budget, "green_tokens": green,
                "green_rate": green/self.budget, "gamma_exact": self.gamma_exact,
                "log10_p_radio": log10_binomial_survival(green, self.budget, self.gamma_exact)}


def infer_batch(model, examples, device, vocab_size, pad_token_id):
    """Predict stored completion positions and their unbiased token likelihoods."""
    import torch

    if not examples:
        return []
    lengths = [len(prompt)+len(completion) for prompt, completion in examples]
    if any(not prompt for prompt, _ in examples):
        raise ValueError("A causal prompt token is required")
    inputs = torch.full((len(examples), max(lengths)), pad_token_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(inputs)
    for row, ((prompt, completion), length) in enumerate(zip(examples, lengths)):
        inputs[row, :length] = torch.tensor(prompt+completion, dtype=torch.long, device=device)
        attention[row, :length] = 1
    with torch.inference_mode():
        logits = model(input_ids=inputs, attention_mask=attention, use_cache=False).logits
        if logits.shape[-1] < vocab_size:
            raise ValueError("Model has fewer output entries than the tokenizer vocabulary")
        outputs = []
        for row, (prompt, completion) in enumerate(examples):
            start, stop = len(prompt)-1, len(prompt)+len(completion)-1
            # The padded Pythia output entries cannot win argmax or normalization.
            completion_logits = logits[row, start:stop, :vocab_size].float()
            predictions = completion_logits.argmax(-1)
            targets = inputs[row, start+1:stop+1]
            log_probs = completion_logits.gather(1, targets[:, None]).squeeze(1)
            log_probs = log_probs-torch.logsumexp(completion_logits, dim=-1)
            outputs.append({"predictions": predictions.cpu().tolist(),
                            "min_k20": min_k_score(log_probs.cpu().tolist()),
                            "completion_tokens": len(completion)})
    return outputs


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prediction_identity(config, pairs_path, checkpoint_path, checkpoint_name, limit):
    path = Path(checkpoint_path).resolve()
    files = []
    for pattern in ("*.safetensors", "pytorch_model*.bin", "config.json", "stage_state.json"):
        for item in sorted(path.glob(pattern)):
            stat = item.stat()
            files.append([item.name, stat.st_size, stat.st_mtime_ns])
    if not files:
        raise FileNotFoundError(f"No local model files at {path}")
    return {"schema": SCHEMA, "config_sha256": fingerprint(config),
            "pairs_sha256": file_digest(pairs_path), "checkpoint_path": str(path),
            "checkpoint_name": checkpoint_name, "checkpoint_files": files,
            "limit": limit, "topk": 1, "score_vocab": "len(tokenizer)",
            "min_k_fraction": 0.2, "min_k_count_rule": "max(1,ceil(length*20/100))",
            "scorer_sha256": file_digest(Path(__file__)), "watermark_bias": False}


def extract_predictions(pairs, tokenizer, checkpoint_path, config, output_dir, identity, device):
    """Atomically resume at complete paired records under an immutable identity."""
    import torch
    from transformers import AutoModelForCausalLM

    final = output_dir / "predictions.jsonl"
    meta = output_dir / "predictions.meta.json"
    partial = output_dir / "predictions.inprogress.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    if meta.exists():
        if load_json(meta) != identity:
            raise ValueError(f"Scoring identity changed: {meta}")
    else:
        if final.exists() or partial.exists():
            raise ValueError("Unidentified existing predictions; refuse unsafe resume")
        write_json(meta, identity)
        (output_dir / "scoring_source.py").write_bytes(Path(__file__).read_bytes())
    if final.exists():
        predictions = read_jsonl(final)
        validate_prediction_records(pairs, predictions)
        return predictions
    completed = []
    if partial.exists():
        # Only a crash-interrupted final line may be discarded. All prior lines
        # must parse and correspond exactly to the prefix of the fixed pair list.
        data = partial.read_bytes()
        if data and not data.endswith(b"\n"):
            data = data[:data.rfind(b"\n")+1]
            partial.write_bytes(data)
        completed = [json.loads(line) for line in data.splitlines()]
        validate_prediction_records(pairs[:len(completed)], completed)
    if len(completed) > len(pairs):
        raise ValueError("Prediction cache contains too many pairs")
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, torch_dtype=dtype,
        local_files_only=True, attn_implementation="sdpa").to(device).eval()
    model.requires_grad_(False)
    model_limit = getattr(model.config, "max_position_embeddings", config["max_length"])
    if any(len(row["prompt_ids"])+len(row[side+"_ids"]) > model_limit
           for row in pairs for side in ("s", "s_prime")):
        raise ValueError("A stored probe exceeds the model attention window; do not truncate silently")
    batch_pairs = max(1, int(config.get("score_batch_size", 8))//2)
    started, token_count = time.monotonic(), 0
    initial_count = len(completed)
    with partial.open("a") as stream:
        for offset in range(initial_count, len(pairs), batch_pairs):
            batch = pairs[offset:offset+batch_pairs]
            examples = [(row["prompt_ids"], row[side+"_ids"])
                        for row in batch for side in ("s", "s_prime")]
            outputs = infer_batch(model, examples, device, len(tokenizer),
                                  tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
            for index, row in enumerate(batch):
                record = {"pair_id": row["pair_id"], "s": outputs[index*2],
                          "s_prime": outputs[index*2+1]}
                stream.write(json.dumps(record, allow_nan=False)+"\n")
                completed.append(record)
            stream.flush()
            token_count += sum(len(completion) for _, completion in examples)
            if offset == initial_count or (offset-initial_count) % (batch_pairs*20) == 0:
                print(json.dumps({"event": "prediction_progress", "pairs": len(completed),
                    "total": len(pairs), "completion_tokens_per_second": token_count/max(1e-9, time.monotonic()-started)}), flush=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, final)
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    validate_prediction_records(pairs, completed)
    return completed


def validate_prediction_records(pairs, predictions):
    if len(pairs) != len(predictions):
        raise ValueError("Prediction and data pair counts differ")
    for pair, record in zip(pairs, predictions):
        if pair["pair_id"] != record["pair_id"]:
            raise ValueError("Prediction/data pair alignment mismatch")
        for side in ("s", "s_prime"):
            if len(pair[side+"_ids"]) != len(record[side]["predictions"]):
                raise ValueError("Prediction/completion token alignment mismatch")


def select_pilot_pairs(pairs, limit):
    """Limit pairs without dropping one split or changing a full-pilot request."""
    if limit < 2:
        raise ValueError("A scoring pilot needs at least two pairs")
    calibration = [row for row in pairs if row["audit_split"] == "calibration"]
    audit = [row for row in pairs if row["audit_split"] == "audit"]
    if not calibration or not audit:
        raise ValueError("Both calibration and audit records are required in a scoring pilot")
    if limit >= len(pairs):
        return pairs
    calibration_count = min(len(calibration), max(1, limit*len(calibration)//len(pairs)))
    audit_count = min(len(audit), limit-calibration_count)
    calibration_count = limit-audit_count
    selected = {row["pair_id"] for row in calibration[:calibration_count]+audit[:audit_count]}
    return [row for row in pairs if row["pair_id"] in selected]


def score_predictions(pairs, predictions, tokenizer, config, output_dir, identity, checkpoint_name, pilot=False):
    """Freeze calibration thresholds on disk before final audit decisions."""
    import torch

    # Thousands of short CPU randperm calls run much faster without thread fanout.
    torch.set_num_threads(1)
    correct = Maryland(tokenizer, config)
    wrong = Maryland(tokenizer, config, key=config["wrong_key"])
    if config["wrong_key"] == config["watermark_key"]:
        raise ValueError("The prespecified wrong key must differ from the correct key")
    budget = int(config["radio_token_budget"])
    tapes = {side: AggregateRadioactivity(budget, correct.gamma_exact) for side in ("s", "s_prime")}
    scored = []
    started, eligible_count = time.monotonic(), 0
    for index, (pair, prediction) in enumerate(zip(pairs, predictions)):
        row = {"pair_id": pair["pair_id"], "source_id": pair["source_id"],
               "audit_split": pair["audit_split"]}
        for side in ("s", "s_prime"):
            result, records = example_watermark_score(pair["prompt_ids"], pair[side+"_ids"],
                                                     prediction[side]["predictions"], correct)
            row[side] = {**result, "min_k20": prediction[side]["min_k20"],
                         "completion_tokens": len(pair[side+"_ids"])}
            eligible_count += result["eligible"]
            if pair["audit_split"] == "audit":
                tapes[side].add(records)
        scored.append(row)
        if index == 0 or (index+1) % 100 == 0:
            print(json.dumps({"event": "watermark_scoring_progress", "pairs": index+1,
                "total": len(pairs), "eligible_tokens_per_second": eligible_count/max(1e-9, time.monotonic()-started)}), flush=True)
    write_jsonl(output_dir/"examples.jsonl", scored)
    calibration = [row for row in scored if row["audit_split"] == "calibration"]
    audit = [row for row in scored if row["audit_split"] == "audit"]
    if not calibration or not audit:
        raise ValueError("Both calibration and audit records are required (including for pilots)")
    if not pilot and (len(calibration) != config["calibration_pairs"] or len(scored) != config["pairs"]):
        raise ValueError("Incomplete full-run calibration or audit split")
    thresholds = {"schema": SCHEMA, "prediction_identity": fingerprint(identity),
                  "calibration_pair_ids_sha256": fingerprint([row["pair_id"] for row in calibration]),
                  "neutral_n0_score": 0.0, "gamma_exact": correct.gamma_exact,
                  "radio_token_budget": budget, "detectors": {}}
    for score, field in (("watermark", "z"), ("min_k20", "min_k20")):
        thresholds["detectors"][score] = calibrate_threshold(
            [row["s"][field] for row in calibration], [row["s_prime"][field] for row in calibration])
    threshold_path = output_dir/"thresholds.json"
    if threshold_path.exists():
        if load_json(threshold_path) != thresholds:
            raise ValueError("Frozen detector thresholds differ; refuse to rewrite after final audit")
    else:
        write_json(threshold_path, thresholds)
    # All final classifier decisions occur strictly after the freeze above.
    shard = {}
    for score, field in (("watermark", "z"), ("min_k20", "min_k20")):
        threshold = thresholds["detectors"][score]["threshold"]
        shard[score] = {"threshold": threshold, **evaluate_shard(
            [row["s"][field] for row in audit], [row["s_prime"][field] for row in audit], threshold)}
    radioactivity = []
    for side in ("s_prime", "s"):
        for name, watermarker in (("correct", None), ("wrong", wrong)):
            radioactivity.append({"probe": side, "key": name, **tapes[side].result(watermarker)})
    diagnostics = {"identical_completion_pairs": sum(row["s_ids"] == row["s_prime_ids"] for row in pairs),
                   "neutral_n0_score": 0.0, "independence_assumption":
                   "Independent paired completion draws conditional on the shared prompts and sources",
                   "audit_source_groups": len({row["source_id"] for row in audit})}
    for side in ("s", "s_prime"):
        diagnostics[side] = {"audit_n0_count": sum(row[side]["eligible"] == 0 for row in audit),
                            "audit_n0_rate": sum(row[side]["eligible"] == 0 for row in audit)/len(audit),
                            "audit_empty_completion_count": sum(row[side]["completion_tokens"] == 0 for row in audit)}
    decisions = [{"pair_id": row["pair_id"], **{
        score: {side: row[side][field] > thresholds["detectors"][score]["threshold"] for side in ("s", "s_prime")}
        for score, field in (("watermark", "z"), ("min_k20", "min_k20"))}} for row in audit]
    write_jsonl(output_dir/"audit_decisions.jsonl", decisions)
    result = {"schema": SCHEMA, "run_id": config["run_id"], "checkpoint": checkpoint_name,
              "pilot": pilot, "primary_comparison": checkpoint_name == "T" and not pilot,
              "prediction_identity": fingerprint(identity), "pairs": len(scored),
              "calibration_pairs": len(calibration), "audit_pairs": len(audit),
              "shard": shard, "radioactivity": radioactivity, "diagnostics": diagnostics,
              "thresholds_path": str(threshold_path.resolve()),
              "scoring_seconds": time.monotonic()-started}
    result["artifact_sha256"] = {name: file_digest(output_dir/name) for name in
        ("examples.jsonl", "thresholds.json", "audit_decisions.jsonl")}
    write_json(output_dir/"metrics.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-name", required=True, choices=["B", *CHECKPOINT_PATHS])
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, help="Pilot only; selected pairs are split-aware")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--predictions-only", action="store_true")
    mode.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.run_dir)
    pilot = args.limit is not None or bool(config.get("pilot", False))
    checkpoint_path = args.checkpoint_path or (Path(config["model_path"]) if args.checkpoint_name == "B"
                       else args.run_dir/"checkpoints"/CHECKPOINT_PATHS[args.checkpoint_name])
    if args.checkpoint_name != "B" and args.limit is None:
        state = load_json(checkpoint_path/"stage_state.json")
        if not state.get("epoch_complete"):
            raise ValueError("An incomplete/pilot checkpoint cannot be used for a full audit")
    pairs_path = args.run_dir/"prepared"/"pairs.jsonl"
    pairs = read_jsonl(pairs_path)
    if len({row["pair_id"] for row in pairs}) != len(pairs):
        raise ValueError("Duplicate pair identifiers")
    if any(row["audit_split"] not in {"calibration", "audit"} for row in pairs):
        raise ValueError("Unrecognized audit split")
    sources = {}
    for row in pairs:
        source = row["source_id"]
        if source in sources and sources[source] != row["audit_split"]:
            raise ValueError("A source group crosses calibration and final-audit splits")
        sources[source] = row["audit_split"]
    if args.limit is not None:
        pairs = select_pilot_pairs(pairs, args.limit)
    elif len(pairs) != config["pairs"]:
        raise ValueError("Full scoring requires the complete generated pair corpus")
    output_dir = args.run_dir/"scores"/(args.checkpoint_name if args.limit is None else f"pilot_{args.checkpoint_name}_{args.limit}")
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = prediction_identity(config, pairs_path, checkpoint_path, args.checkpoint_name, args.limit)
    completed_path = output_dir/"metrics.json"
    if completed_path.exists() and not args.predictions_only:
        completed = load_json(completed_path)
        if completed["prediction_identity"] != fingerprint(identity):
            raise ValueError("Completed score identity differs from this data/model/configuration")
        for name, digest in completed["artifact_sha256"].items():
            if file_digest(output_dir/name) != digest:
                raise ValueError(f"Completed scoring artifact changed: {output_dir/name}")
        print(json.dumps({"event": "already_scored", "metrics": str(completed_path),
                          "checkpoint": args.checkpoint_name}), flush=True)
        return
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"], local_files_only=True)
    if args.cpu_only:
        if load_json(output_dir/"predictions.meta.json") != identity:
            raise ValueError("CPU scoring predictions do not match this run/checkpoint")
        predictions = read_jsonl(output_dir/"predictions.jsonl")
        validate_prediction_records(pairs, predictions)
    else:
        predictions = extract_predictions(pairs, tokenizer, checkpoint_path, config, output_dir, identity, args.device)
    if not args.predictions_only:
        result = score_predictions(pairs, predictions, tokenizer, config, output_dir, identity,
                                   args.checkpoint_name, pilot=pilot)
        print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
