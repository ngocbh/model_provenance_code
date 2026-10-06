# Pythia watermark inheritance

Implements [watermark.md](../../watermark.md): three independent 10,000-pair
WikiText runs, full-parameter Pythia-1B parent/Dolly training, reading-mode
watermark survival, calibrated paired shard advantage, and MIN-K diagnostics.
The separate 128-pair pilot never enters the full experiment.

## Completed experiment: 2026-10-06

All three runs completed. After three Dolly epochs, the watermark-based shard
advantages were 59.06%, 60.20%, and 61.60%; baseline/clean-control advantages
were at most 0.83%. Run 02's wrong-key radioactivity diagnostics had nominal
p-values of 0.0035 for T and 0.035 for T0; these are included in the full table.

[Main table](results/2026-10-06/main_results.csv) ·
[All diagnostics](results/2026-10-06/results.csv) ·
[Advantage plot](results/2026-10-06/advantage.png) ·
[Radioactivity plot](results/2026-10-06/radioactivity.png).
The result directory also preserves exact run configs, input revisions,
package versions, dataset checks, and validation records. Large generated
datasets and model checkpoints remain in the local artifact directory below.

Run state lives in `artifacts/watermark_inheritance/`. Each `run_00` through
`run_02` has a locked config, paired-data manifest, checkpoints, scores, and
reports. Shared model/data revisions are in `.cache/watermark/download_manifest.json`.
Installed packages and snapshots of the actual uncommitted experiment code
are recorded under `artifacts/watermark_inheritance/setup/`.

The Python executable is `.venv/bin/python`, a project environment using the
existing `dream` PyTorch installation plus the upstream `numba` dependency.
Generation, training, and evaluation all use the full 50,277-token tokenizer
vocabulary, including 23 added whitespace tokens; model-only padded output
entries are excluded. The upstream CPU hash/RNG and permutation are preserved.

## Commands

Submit from the repository root after creating the run configurations:

```bash
sbatch experiments/watermark_inheritance/setup.sbatch
sbatch experiments/watermark_inheritance/gpu.sbatch pilot 2
sbatch --job-name=wm_main --gres=gpu:h200:4 --cpus-per-task=16 \
  --mem=192G --time=08:00:00 --dependency=afterok:PILOT_JOB \
  experiments/watermark_inheritance/gpu.sbatch main 4
sbatch --dependency=afterok:MAIN_JOB experiments/watermark_inheritance/cpu_score.sbatch
sbatch --dependency=afterok:SCORE_ARRAY_JOB experiments/watermark_inheritance/report.sbatch
```

Use confirmed numeric job IDs in dependency arguments. Main execution checks
the pilot's detection, wrong-key control, generation quality, parameter update,
checkpoint reload, and aggregate token-budget evidence. Its 100,000-token
aggregate budget is fixed before the full audit. Generation is resumable by
paired records; training resumes at completed epochs; scoring reuses validated
prediction files. Keep the generation worker count fixed within a phase when
resuming. CPU scoring is separate so GPUs can be released after inference.

The pilot exposed an overlap-filter edge case: a completion consisting of one
period matched ordinary punctuation. Before full generation, the rule was fixed
to require whole-field equality for completions shorter than 32 characters;
longer completions are checked as substrings. Exact and normalized comparisons
cover both shards. The pilot recheck retained all 15,011 Dolly rows. Full-run
counts are recorded separately after checking each complete paired dataset.

Per-stage logs are in each run's `logs/`; scheduler logs and job history are in
`setup/`. Monitor exact IDs using `squeue`, `scontrol show job`, and `sacct`.
Slurm displayed timestamps use America/New_York; JSON timestamps use UTC.

Final combined `reports/main_results.csv` contains the correct-key S-prime
radioactivity plus primary watermark-based shard results. `results.csv` also
includes supervised S, wrong-key controls and MIN-K diagnostics. PNG/PDF plots
show changes from B to P through Dolly epochs, with T0 displayed separately.
All three runs and six scored checkpoints per run must exist before the final
report compiler accepts the experiment as complete.

## Verification

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest discover \
  -s experiments/watermark_inheritance/tests -v
```

Tests cover upstream compatibility, added-token handling, independent paired
sampling metadata, source/pilot separation, overlap removal, response masks,
batch accumulation, teacher-forcing alignment, deduplication, calibrated
thresholds and exact paired inference. Actual speed, memory use, training update
and reload are measured by the GPU pilot. Training losses are not held-out
task-quality measurements.

## Continuous fine-tuning stress test

The active single-pass extension uses `artifacts/watermark_curve_distinct`:
600,000 distinct first-turn examples from the full `openbmb/UltraChat` train split,
one epoch, and a 48-hour GPU allocation request. Its pinned source revision is
`f220fe796ce3ed62fbe1681b45ce6cbc9c6cabe0`. Both normalized instructions and encoded
training examples are deduplicated, and the same overlap exclusions apply.
The full UltraChat source differs from the smaller H4-curated source below.
The checkpoint schedule remains 512, 1024, 2048, 3125, 6250, 12500, and 18750 steps;
all examples on this trajectory are seen once. Each seed stays on one GPU and
alternates target/control at durable checkpoints, restoring the full optimizer,
scheduler, RNG and data position. This yields matched control curves early.

`reports/advantage_vs_steps.png` and `.pdf` are the dedicated requested plots,
with optimizer steps on the bottom axis and example counts on the top axis.
The CSV also records response-token exposure. Partial plots show their completed
and planned audit counts; completed audits are published while training proceeds.
The archived [advantage plot](results/2026-10-06-distinct/advantage_vs_steps.png),
[measurements](results/2026-10-06-distinct/curve.csv), and
[snapshot coverage](results/2026-10-06-distinct/snapshot_manifest.json) contain the
latest verified publication; the live artifacts may have additional measurements.
The earlier 200k run was stopped after all three targets reached 100,000 distinct
examples and is retained as a preliminary trajectory. Its planned three-epoch
endpoint was not reached, and its plotted controls are step-zero baselines only.

Verify the available single-pass audits independently from their saved pair
scores and decisions:

```bash
.venv/bin/python -m experiments.watermark_inheritance.curve_validate \
  --root artifacts/watermark_curve_distinct
```

The resulting `setup/curve_scoring_validation.json` records artifact hashes and
the steps with all three target/control pairs complete. Add
`--require-matched-through 1024`, for example, to require every scheduled point
through that step. This checks matching example/token exposure as well as
calibration membership, strict-threshold decisions and independently calculated
paired and aggregate p-values. Full completion requires all 42 checkpoint audits.

The initial follow-up in `artifacts/watermark_curve` keeps the original three private
parents and audit pairs, and trains each parent plus a matched base-model control
on the same 200,000 distinct UltraChat first-turn instruction/response examples.
The corpus is pinned to revision `8049631c405ae6576f93f445c6b8166f76f5505a`, deduplicated
by normalized instruction, and filtered against S and S-prime from all three runs.
This is a different-corpus stress test; it does not isolate the effect of increasing
the number of Dolly examples.

The initial plan called for three epochs, with one optimizer and cosine
schedule. Checkpoints at steps 512, 1024, 2048, 3125, 6250, 12500, and 18750 correspond
to 16,384, 32,768, 65,536, 100,000, 200,000, 400,000, and 600,000 examples seen.
Exposure above 200,000 includes repeats. Checkpoints record unique examples,
optimizer steps, input tokens, and supervised response tokens. Interval resumes
restore data position, optimizer, scheduler and RNG; a tiny-model integration test
checks exact equality with uninterrupted training.

Three GPU workers train the six trajectories, while a fourth GPU extracts audit
predictions from atomically published checkpoints. Independent Slurm CPU jobs score
those predictions and refresh `artifacts/watermark_curve/reports/curve.csv`,
`curve.json`, `curve.png`, and `curve.pdf` while training continues. The report keeps
every scheduled point, including controls, MIN-K, signed gaps and wrong-key results.
Calibration stays separate from the fixed final audit; intermediate measurements
are descriptive and are not independent confirmation tests. The final scheduled
target checkpoint is the primary endpoint.

Submission uses `curve_setup.sbatch`, followed by
`python -m experiments.watermark_inheritance.curve_submit --data-job JOB_ID`.
The submission tool freezes executable source under `setup/source` and records
source hashes, the data-job dependency and the GPU job ID. GPU completion and CPU
audit completion are recorded separately. The final CPU job verifies all 42 audits
are present. See [the design](../../docs/plans/2026-10-06-finetuning-curve.md).
