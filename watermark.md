# Pythia watermark inheritance experiment

**Goal:** Test whether a watermark learned from a private synthetic shard survives subsequent Dolly fine-tuning, and whether a watermark-based distinguisher can detect the target's dependence on that specific shard.

```text
Frozen generator + watermarker → 10,000 matched (s_i, s'_i) pairs

B = pretrained Pythia-1B
B ── fine-tune on S only ──→ P(S) ── fine-tune on Dolly ──→ T
B ──────────────────────────────── fine-tune on Dolly ──→ T0

S' is reserved for auditing; neither training stage consumes it.
```

Here, `s_i` and `s'_i` are two independently sampled completions of the same prompt; `S` and `S'` are the corresponding collections. We generate 20,000 completions, but train the parent on only 10,000.

This is a **Pythia adaptation**, not an exact reproduction. The reference paper trains on watermarked synthetic data and includes subsequent clean fine-tuning in its “purification” experiment. We change the models, dataset size, prompt source, and downstream dataset. [Reference paper, Sections 5 and 6.3](https://arxiv.org/html/2402.14904v2)

**1. Models and data**

| Component | First-run choice |
|---|---|
| Base `B` | [`EleutherAI/pythia-1b`](https://huggingface.co/EleutherAI/pythia-1b), final pretrained checkpoint |
| Synthetic generator | A frozen copy of `B`, before either fine-tuning stage |
| Prompt source | [`Salesforce/wikitext`](https://huggingface.co/datasets/Salesforce/wikitext), `wikitext-103-raw-v1`, training split |
| Downstream data | [`databricks/databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k), original human responses |
| Watermark implementation | Authors' [`radioactive-watermark`](https://github.com/facebookresearch/radioactive-watermark) repository |

Use natural text continuations for the first run: base Pythia is a completion model rather than an instruction-tuned assistant. This makes generation straightforward. Instruction prompts with a stronger instruction-tuned teacher can be a later variant. Using the base as generator is another departure from the reference experiment; the controls below are essential. [Pythia model card](https://huggingface.co/EleutherAI/pythia-1b)

**2. Construct the paired shards**

1. Select 10,000 distinct WikiText paragraph prefixes, each about 64 Pythia tokens. Remove empty rows, headings, and duplicate prefixes. Keep the original source identifiers. Prefer one prefix per article; keep all related prefixes together when splitting.
2. For each fixed prompt `q_i`, sample two continuations independently from the frozen generator, with the **same watermark configuration and key**, but different text-sampling randomness.
3. Randomly assign one completion to `s_i` and the other to `s'_i`. Keep generation and filtering identical on both sides. Do not select examples by which side has a stronger watermark. Retain identical-completion pairs as ties and report their frequency.
4. Use the Kirchenbauer/“Maryland” watermark with window `k=2`, greenlist fraction `gamma=0.25`, logit bias `delta=3`, temperature `0.8`, and top-p `0.95`. These are the reference paper's generation settings. [Section 5.1](https://arxiv.org/html/2402.14904v2#S5.SS1)
5. Proposed length: up to 256 new tokens per completion, with ordinary EOS stopping. Save actual lengths and original token IDs. Use a separate small pilot to inspect quality, repetition, and watermark detection before committing to the full dataset.

Save one JSONL record per pair: `pair_id`, source ID, prompt text/token IDs, both completion texts/token IDs, sampling seeds, watermark configuration ID, and audit split. Record the generator/tokenizer revisions. Keep the watermark key fixed within a run and distinct from the sampling seeds; some reference-code arguments called “seed” belong to the watermark itself.

Use one consistent tokenizer, vocabulary, hash/greenlist implementation, and special-token policy throughout. Generate fresh Pythia-tokenized data; the repository's existing Llama examples are only reference fixtures.

**3. Separate parent training from detector calibration**

Before evaluating models, assign 2,000 pairs to **detector calibration** and 8,000 pairs to **final audit**. Split by prompt/source group, keeping both members of each pair together.

The parent trains on **all 10,000 members of `S`**, including the final-audit members. “Held out” here means held out from constructing the detector, not from parent training. No part of `S'` is used for model training. Freeze detector choices using only calibration data, then evaluate once on the final-audit pairs.

Treat the fixed prompts, frozen generator, and shared watermark configuration as background information `U` in our provenance framework. The fresh completion draws provide the shard randomness. Keep `S` out of downstream data so its training-time influence reaches `T` through `P(S)`.

**4. Train the parent, target, and control**

| Checkpoint | Construction | Purpose |
|---|---|---|
| `B` | Original pretrained model | Check for pre-existing signal or scoring artifacts |
| `P(S)` | Fine-tune `B` on prompt + `s_i` | Check whether the first stage learns the signal |
| `T` | Start from `P(S)` and fine-tune on Dolly | Main inheritance experiment |
| `T0` | Start from `B` and fine-tune on the same Dolly data | Control without the private-shard training stage |

For the first run, use full-parameter causal-language-model fine-tuning. Compute loss only on the completion in stage one, and only on the response in stage two; mask prompts and padding. Format Dolly consistently as instruction, optional context, then response. Use the original responses, not watermarked replacements.

**Proposed starting recipe, not claimed as an optimized Pythia recipe:** AdamW, learning rate `1e-5`, effective batch size 32, cosine schedule with 3% warmup, maximum sequence length 1,024, and 3 epochs per stage. Use gradient accumulation and mixed precision as appropriate. Verify truncation preserves response tokens. Save `P(S)` and the target after each Dolly epoch; the prespecified main result is after epoch 3. Reset optimizer/scheduler state when beginning Dolly training.

Use all eligible Dolly rows for this inheritance run, after checking overlap with generated shard completions. Record the actual row count. If measuring downstream task quality, reserve a separate Dolly validation split before training and report the reduced training count; do not call training-set loss held-out performance.

**5. Evaluate watermark survival and shard advantage separately**

Both shards contain the same watermark. Running a text detector directly on `S` versus `S'` therefore does not test whether `T` inherited `S`. Likewise, `1 - watermark p-value` is not an advantage.

**A. Watermark survival: model-level radioactivity.** Use the paper's open-model “reading mode”: feed a stored sequence into the checkpoint and score its next-token predictions against the greenlists defined by the input prefixes. Disable watermarking during evaluation. Exclude repeated watermark contexts in the attention span and deduplicate scored context/prediction tuples across the aggregate test. [Paper, Section 4](https://arxiv.org/html/2402.14904v2#S4)

Use the same final-audit `S'` sequences as the primary probe corpus for `B`, `P(S)`, `T`, and `T0`. These completions were never training examples, so this tests transfer of the watermark beyond the exact members. Separately report the supervised result on final-audit `S`. Match the number of eligible scored tokens across checkpoint comparisons, with a budget fixed using the pilot/calibration data. Report green-token rate, eligible-token count, and `log10(p_radio)`; a smaller p-value is stronger evidence against the watermark null, not a larger effect size.

**B. Shard advantage: our proposed adapter to the provenance framework.** This is an additional measurement, not the original paper's reported metric.

For each sample `x = (prompt, completion)`:

- Feed the true prefix into `T` and obtain its most likely next token at each completion position. Score the prediction, not the stored completion token. Use `topk=1` as in the reference reading-mode default. [Reference implementation](https://github.com/facebookresearch/radioactive-watermark/blob/main/main_reed_wm.py)
- Form a per-example normalized green-token score, `z_T(x) = (G - gamma*N) / sqrt(N*gamma*(1-gamma))`, using eligible positions. Here `G` is the green-prediction count and `N` is the eligible count. Use the implementation's exact greenlist fraction when rounding its size.
- Apply repetition exclusions within each example, resetting state between examples. Reserve corpus-wide deduplication for measurement A; a global tape would make an individual sample's score depend on audit order. Define a fixed neutral result for `N=0` and report how often it occurs.
- On the 2,000 calibration pairs only, choose a threshold `c` that maximizes balanced classification accuracy for `A_T(x) = 1[z_T(x) > c]`, where 1 predicts membership in `S`. Freeze it before the final audit. Do not feed shard labels, IDs, or generation seeds into the distinguisher.

On the 8,000 final-audit pairs, report:

```text
a       = fraction of S samples with A_T(x) = 1
a_prime = fraction of S' samples with A_T(x) = 1
signed_gap = a - a_prime
advantage  = abs(signed_gap)
```

This follows Algorithm 1 and the advantage definition in our supplied `Model_Provenance.pdf`, Sections 4–5. Repeat the same calibration/audit procedure separately for `B`, `P(S)`, and `T0`.

Because examples are paired by prompt, use a **within-pair label-swap test** on the final signed differences, holding the calibrated detector fixed; report its two-sided p-value as `p_shard`. Its justification is exchangeability under the no-shard-dependence null, conditional on the shared background. Do not shuffle individual rows or automatically copy the draft's independent-shard threshold. For a simple conservative alternative under independent pairs and equal conditional marginals, the paired Hoeffding threshold is `sqrt(2*log(2/alpha)/m)`, approximately `0.0304` for `m=8000`, `alpha=0.05`. If multiple pairs share a source and their randomness cannot be treated independently, use source-level inference instead.

A strong aggregate watermark can coexist with near-zero shard advantage: `T` may learn the common watermark equally well on unseen same-key completions. That is a meaningful result. As a secondary comparison, run the draft's MIN-K 20% distinguisher on exactly the same splits to see whether instance membership is detectable by another score.

**6. Run order and deliverables**

Start with a small generation/detector pilot, then one full run. Check `P(S)` before interpreting a weak result on `T`: absent signal in the parent cannot establish that Dolly removed it. Evaluate the correct key and a prespecified wrong key for the aggregate detector, and compare `B`/`T0` controls. A useful follow-up is a matched branch trained on unwatermarked synthetic data before Dolly, to separate watermark effects from generic extra fine-tuning.

After the pipeline works, repeat with three independent runs, resampling completion pairs and watermark keys and varying training randomness. Report each run, rather than claiming a reliable false-positive rate from only three controls. Keep the final `T` watermark-based shard test as the primary comparison; other scores/checkpoints are diagnostic unless multiple-testing control is added.

Return the paired-data manifest, training configurations, model/tokenizer revisions, checkpoints, and a table with:

```text
run | checkpoint | a | a_prime | signed_gap | advantage | p_shard
    | radioactivity_probe | eligible_tokens | green_rate | log10(p_radio)
```

Also plot advantage and aggregate watermark strength before and after Dolly. A positive result supports this controlled provenance pipeline when the shard/watermark cannot reach the target through another training path. Watermark detection alone does not uniquely identify a parent if other models can train on data made with the same key. Failure to detect does not establish absence of derivation.

**7. Download/setup starting point**

Run these in the team's GPU environment, with a compatible PyTorch installation already available. These commands obtain inputs; they do not implement or launch the experiment.

```bash
python -m pip install transformers datasets accelerate huggingface_hub scipy
git clone https://github.com/facebookresearch/radioactive-watermark.git

python - <<'PY'
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download

Path("data").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)
api = HfApi()
model_id = "EleutherAI/pythia-1b"
model_sha = api.model_info(model_id).sha
snapshot_download(model_id, revision=model_sha, local_dir="models/pythia-1b")

sources = [
    ("Salesforce/wikitext", "wikitext-103-raw-v1", "data/wikitext"),
    ("databricks/databricks-dolly-15k", None, "data/dolly"),
]
manifest = {"model": {"id": model_id, "revision": model_sha}}
for repo_id, config, destination in sources:
    sha = api.dataset_info(repo_id).sha
    dataset = load_dataset(repo_id, name=config, revision=sha, split="train")
    dataset.save_to_disk(destination)
    manifest[repo_id] = {"revision": sha, "config": config, "rows": len(dataset)}

import json
Path("data/download_manifest.json").write_text(json.dumps(manifest, indent=2))
PY
```

The teammate or coding agent should implement four stages from this specification: prepare paired data, train checkpoints, score checkpoints, and produce the result table. Reuse the authors' watermark logic, adapting the data loader and tokenizer handling to Pythia. The repository's example reading-mode command and data format need checking against its actual script; this handoff does not assume a ready-made Pythia training command. Save the code commit and installed package versions with each run. No training or GPU validation has been performed for this handoff.
