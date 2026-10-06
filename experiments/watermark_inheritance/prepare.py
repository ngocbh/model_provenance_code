"""Pinned downloads, article-grouped prompts, and response-only Dolly examples."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import re
import subprocess
import unicodedata

from .common import fingerprint, load_config, load_json, read_jsonl, write_json, write_jsonl


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_versions():
    names = ["torch", "transformers", "datasets", "huggingface_hub", "tokenizers", "scipy", "numpy"]
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def verify_upstream(config, expected_revision=None):
    """Reject source drift before creating or consuming watermark artifacts."""
    upstream = Path(config["upstream_path"]).resolve()
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    if dirty:
        raise RuntimeError("Watermark upstream checkout is modified")
    for expected in (config.get("upstream_commit"), expected_revision):
        if expected is not None and revision != expected:
            raise RuntimeError(f"Watermark upstream revision changed: expected {expected}, found {revision}")
    return {"path": str(upstream), "revision": revision,
            "url": "https://github.com/facebookresearch/radioactive-watermark"}


def validate_tokenizer(tokenizer):
    # Pythia adds real whitespace tokens beyond its base BPE vocabulary.
    # The common Maryland adapter exposes this complete domain to upstream.
    if tokenizer.vocab_size > len(tokenizer):
        raise ValueError("Tokenizer length cannot be smaller than its base vocabulary")
    if hasattr(tokenizer, "get_vocab"):
        if set(tokenizer.get_vocab().values()) != set(range(len(tokenizer))):
            raise ValueError("Tokenizer IDs must cover a contiguous effective vocabulary")
    for name in ("eos_token_id", "bos_token_id", "pad_token_id"):
        value = getattr(tokenizer, name, None)
        if value is not None and not 0 <= value < len(tokenizer):
            raise ValueError(f"Tokenizer {name} lies outside the watermark vocabulary")


def download_inputs(config, shared_dir):
    """Resolve immutable revisions once, then resume their downloads safely."""
    from datasets import load_dataset, load_from_disk
    from huggingface_hub import HfApi, snapshot_download

    shared_dir = Path(shared_dir)
    shared_dir.mkdir(parents=True, exist_ok=True)
    plan_path = shared_dir / "download_plan.json"
    api = HfApi()
    if plan_path.exists():
        plan = load_json(plan_path)
    else:
        model_id = "EleutherAI/pythia-1b"
        plan = {"model": {"id": model_id, "revision": api.model_info(model_id).sha},
                "datasets": {}}
        for name, repo_id, subset in [
            ("wikitext", "Salesforce/wikitext", "wikitext-103-raw-v1"),
            ("dolly", "databricks/databricks-dolly-15k", None),
        ]:
            plan["datasets"][name] = {"id": repo_id, "revision": api.dataset_info(repo_id).sha,
                                      "config": subset, "split": "train"}
        write_json(plan_path, plan)
    model_path = Path(config["model_path"]).resolve()
    snapshot_download(plan["model"]["id"], revision=plan["model"]["revision"],
                      local_dir=str(model_path), allow_patterns=[
                          "*.safetensors", "*.safetensors.index.json", "config.json",
                          "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                          "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
                      ])
    manifest = dict(plan)
    manifest["model"] = dict(plan["model"], path=str(model_path))
    manifest["tokenizer"] = dict(manifest["model"])
    manifest["datasets"] = {}
    for name, spec in plan["datasets"].items():
        destination = shared_dir / "data" / name
        marker = shared_dir / "data" / (name + ".revision.json")
        if destination.exists():
            if not marker.exists() or load_json(marker) != spec:
                raise RuntimeError(f"Existing dataset lacks matching pinned metadata: {destination}")
            dataset = load_from_disk(str(destination))
        else:
            dataset = load_dataset(spec["id"], name=spec["config"], revision=spec["revision"], split="train")
            dataset.save_to_disk(str(destination))
            write_json(marker, spec)
        manifest["datasets"][name] = dict(spec, rows=len(dataset), path=str(destination.resolve()),
                                            fingerprint=dataset._fingerprint)
    manifest["upstream"] = verify_upstream(config)
    manifest["packages"] = package_versions()
    write_json(shared_dir / "download_manifest.json", manifest)
    return manifest


ARTICLE_HEADING = re.compile(r"^=\s+([^=\n]+?)\s+=$")
ANY_HEADING = re.compile(r"^=+\s+.*?\s+=+$")


def select_prompts(rows, tokenizer, count=10000, pilot_count=128, calibration_count=2000,
                   prompt_tokens=64, seed=1729):
    """Choose one unique paragraph prefix per top-level WikiText article."""
    if not 0 <= calibration_count <= count or prompt_tokens < 2:
        raise ValueError("Invalid prompt/split configuration")
    candidates, prefixes = [], set()
    article, title, accepted_article = None, None, False
    for row_index, row in enumerate(rows):
        text = row["text"].strip()
        heading = ARTICLE_HEADING.fullmatch(text)
        if heading:
            article, title, accepted_article = row_index, heading.group(1), False
            continue
        if article is None or accepted_article or not text or ANY_HEADING.fullmatch(text):
            continue
        # Restrict to a single paragraph, preserving its original row identifier.
        for paragraph in text.split("\n\n"):
            ids = tokenizer.encode(paragraph.strip(), add_special_tokens=False)
            if len(ids) < prompt_tokens:
                continue
            prefix = tuple(ids[:prompt_tokens])
            if prefix in prefixes:
                continue
            prefixes.add(prefix)
            candidates.append({"source_id": f"wikitext:train:article:{article}:row:{row_index}",
                               "article_id": f"wikitext:train:article:{article}",
                               "source_row": row_index, "article_heading_row": article,
                               "article_title": title, "prompt_ids": list(prefix),
                               "prompt_text": tokenizer.decode(prefix, skip_special_tokens=False)})
            accepted_article = True
            break
    if len(candidates) < count + pilot_count:
        raise ValueError(f"Only {len(candidates)} unique article prefixes, need {count + pilot_count}")
    rng = random.Random(seed)
    rng.shuffle(candidates)
    main, pilot = candidates[:count], candidates[count:count + pilot_count]
    calibration = set(rng.sample(range(count), calibration_count))
    for i, row in enumerate(main):
        row.update(pair_id=f"pair_{i:05d}", audit_split="calibration" if i in calibration else "audit")
    for i, row in enumerate(pilot):
        row.update(pair_id=f"pilot_{i:05d}", audit_split="pilot")
    return main, pilot, {"eligible_articles": len(candidates), "prompt_tokens": prompt_tokens,
                         "prompt_seed": seed, "main_pairs": count, "pilot_pairs": pilot_count,
                         "calibration_pairs": calibration_count, "audit_pairs": count - calibration_count}


def normalize_text(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


class CompletionIndex:
    """Match substantial substrings; short completions require whole-field equality.

    A pilot completion consisting of '.' is not evidence that every punctuated
    Dolly response copied a shard example. Short text still matches if it is an
    entire original field. The 32-character rule is fixed before full generation.
    """
    def __init__(self, texts, width=32):
        self.width = width
        self.anchors = defaultdict(list)
        self.short = defaultdict(set)
        for text in set(texts):
            if not text.strip():
                continue
            if len(text) < width:
                self.short[len(text)].add(text)
            else:
                self.anchors[text[:width]].append(text)

    def contains_completion(self, text):
        for size, patterns in self.short.items():
            if len(text) == size and text in patterns:
                return True
        seen = set()
        for i in range(len(text) - self.width + 1):
            anchor = text[i:i + self.width]
            if anchor in seen:
                continue
            patterns = self.anchors.get(anchor)
            if patterns:
                seen.add(anchor)
                if any(pattern in text for pattern in patterns):
                    return True
        return False


def encode_dolly(row, tokenizer, max_length=1024):
    """Discard context before instruction, and response only as a last resort."""
    instruction = str(row.get("instruction", "")).strip()
    response = str(row.get("response", ""))
    context = str(row.get("context", "")).strip()
    if not instruction or not response.strip():
        return None
    encode = lambda text: list(tokenizer.encode(text, add_special_tokens=False))
    heading, ending = encode("### Instruction:\n"), encode("\n\n### Response:\n")
    instruction_ids, context_ids = encode(instruction), encode(context)
    context_heading = encode("\n\n### Context:\n")
    response_ids = encode(response)
    eos = tokenizer.eos_token_id
    if eos is not None and (not response_ids or response_ids[-1] != eos):
        response_ids.append(eos)
    if not response_ids:
        return None
    response_original = len(response_ids)
    available = max_length - len(heading) - len(ending)
    if available < 2:
        raise ValueError("max_length is too short for Dolly formatting")
    # Keep at least one instruction token so every example remains conditioned.
    response_limit = available - 1
    if len(response_ids) > response_limit:
        response_ids = response_ids[:response_limit]
        if eos is not None:
            response_ids[-1] = eos
    prompt_budget = available - len(response_ids)
    kept_instruction = instruction_ids[:prompt_budget]
    prompt = heading + kept_instruction
    context_budget = prompt_budget - len(kept_instruction) - len(context_heading)
    kept_context = context_ids[:max(context_budget, 0)] if context_ids else []
    if kept_context:
        prompt += context_heading + kept_context
    prompt += ending
    return {"input_ids": prompt + response_ids, "labels": [-100] * len(prompt) + response_ids,
            "response_tokens": len(response_ids), "original_response_tokens": response_original,
            "response_truncated": len(response_ids) < response_original,
            "instruction_truncated": len(kept_instruction) < len(instruction_ids),
            "context_truncated": len(kept_context) < len(context_ids)}


def prepare_dolly(run_dir, config, tokenizer, dataset):
    run_dir = Path(run_dir)
    pair_path = run_dir / "prepared/pairs.jsonl"
    pairs = read_jsonl(pair_path) if pair_path.exists() else None
    if pairs is not None and len(pairs) != int(config["pairs"]):
        raise ValueError("Dolly overlap check requires the complete paired dataset")
    texts = [row[field] for row in pairs or [] for field in ("s_text", "s_prime_text")]
    overlap_width = int(config.get("overlap_substring_min_chars", 32))
    exact = CompletionIndex(texts, width=overlap_width)
    normalized = CompletionIndex(map(normalize_text, texts), width=overlap_width)
    result, counts = [], defaultdict(int)
    for i, row in enumerate(dataset):
        counts["source_rows"] += 1
        encoded = encode_dolly(row, tokenizer, int(config.get("max_length", 1024)))
        if encoded is None:
            counts["ineligible_rows"] += 1
            continue
        # Check original fields conservatively, plus actual formatted training
        # text so a completion that contains a formatting marker cannot leak.
        text = "\n\n".join(str(row.get(k, "")) for k in ("instruction", "context", "response"))
        text += "\n\n" + tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)
        fields = [str(row.get(k, "")) for k in ("instruction", "context", "response")]
        if pairs is not None and any(exact.contains_completion(value) for value in [text, *fields]):
            counts["exact_overlap_excluded"] += 1
            continue
        if pairs is not None and any(normalized.contains_completion(normalize_text(value)) for value in [text, *fields]):
            counts["normalized_overlap_excluded"] += 1
            continue
        encoded["source_id"] = f"dolly:train:{i}"
        encoded["source_row"] = i
        for name in ("response_truncated", "context_truncated", "instruction_truncated"):
            counts[name] += int(encoded[name])
        result.append(encoded)
    output = run_dir / "prepared/dolly.jsonl"
    write_jsonl(output, result)
    manifest = dict(counts, eligible_rows=len(result), overlap_checked=pairs is not None,
                    overlap_excluded=counts["exact_overlap_excluded"] + counts["normalized_overlap_excluded"],
                    ignored_blank_completions=sum(not text.strip() for text in texts),
                    overlap_policy="exact/normalized completion substring for >=32 characters; shorter completions require whole original field equality",
                    overlap_substring_min_chars=overlap_width,
                    pair_file_sha256=file_sha256(pair_path) if pairs is not None else None,
                    output_file_sha256=file_sha256(output), config_id=fingerprint(config))
    write_json(run_dir / "prepared/dolly_manifest.json", manifest)
    return manifest


def build_inputs(run_dir, config, shared_dir, dolly_only=False):
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    run_dir, shared_dir = Path(run_dir), Path(shared_dir)
    manifest = load_json(shared_dir / "download_manifest.json")
    verify_upstream(config, manifest["upstream"]["revision"])
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"], local_files_only=True)
    validate_tokenizer(tokenizer)
    if not dolly_only:
        wikitext = load_from_disk(manifest["datasets"]["wikitext"]["path"])
        prompts, pilot, selection = select_prompts(
            wikitext, tokenizer, count=int(config["pairs"]), pilot_count=int(config.get("pilot_pairs", 128)),
            calibration_count=int(config["calibration_pairs"]), prompt_tokens=int(config.get("prompt_tokens", 64)),
            seed=int(config.get("prompt_seed", 1729)))
        for name, rows in [("prompts", prompts), ("pilot_prompts", pilot)]:
            path = run_dir / "prepared" / (name + ".jsonl")
            if path.exists() and read_jsonl(path) != rows:
                raise RuntimeError(f"Refusing to change existing prepared prompts: {path}")
            write_jsonl(path, rows)
        manifest = dict(manifest, selection=selection, config_id=fingerprint(config),
                        tokenizer_vocab_size=tokenizer.vocab_size, tokenizer_length=len(tokenizer),
                        effective_vocab_size=len(tokenizer),
                        eos_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id,
                        special_tokens_map=tokenizer.special_tokens_map,
                        prompt_file_sha256=file_sha256(run_dir / "prepared/prompts.jsonl"),
                        pilot_prompt_file_sha256=file_sha256(run_dir / "prepared/pilot_prompts.jsonl"))
        code_root = Path(__file__).resolve().parents[2]
        manifest["code_revision"] = subprocess.check_output(
            ["git", "-C", str(code_root), "rev-parse", "HEAD"], text=True).strip()
        manifest["packages"] = package_versions()
        manifest["source_file_sha256"] = {
            str(path.relative_to(code_root)): file_sha256(path)
            for path in sorted(Path(__file__).parent.glob("*.py"))}
        write_json(run_dir / "prepared/manifest.json", manifest)
    dolly = load_from_disk(manifest["datasets"]["dolly"]["path"])
    result = prepare_dolly(run_dir, config, tokenizer, dolly)
    print(json.dumps({"stage": "prepare", "run_dir": str(run_dir), "dolly": result}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--shared-dir")
    parser.add_argument("--stage", choices=["download", "build", "dolly", "all"], default="all")
    args = parser.parse_args()
    config = load_config(args.run_dir)
    shared = Path(args.shared_dir or config.get("shared_dir", ".cache/watermark")).resolve()
    if args.stage in ("download", "all"):
        download_inputs(config, shared)
    if args.stage in ("build", "dolly", "all"):
        build_inputs(args.run_dir, config, shared, dolly_only=args.stage == "dolly")


if __name__ == "__main__":
    main()
