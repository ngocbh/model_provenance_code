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
