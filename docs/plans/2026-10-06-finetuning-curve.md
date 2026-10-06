# Continuous clean fine-tuning curve

Goal: measure how inherited watermark and shard-membership advantage change as clean fine-tuning exposure increases, while evaluation runs concurrently with training.

The previous experiment used about 15,000 Dolly rows for three epochs. Repeating those rows further would increase steps without increasing unique data. Instead, use a deterministic 200,000-example subset of UltraChat train_sft, one first user/assistant exchange per conversation, with exact and normalized overlap filtering against S and S-prime from all three runs. Use the same subset for every target and control. This changes corpus as well as scale and is a separate stress test, not a controlled Dolly-only sample-size ablation. UltraChat is synthetic instruction data.

Reuse each original completed P(S) checkpoint after verifying its original immutable config and pair hashes. Initialize matched controls from B. Keep all training hyperparameters and response-only Dolly-style formatting; reset optimizer once at the start of each continuous trajectory. Run three epochs over 200,000 rows (18,750 optimizer steps at effective batch 32). Freeze the schedule before observing results: steps 512, 1024, 2048, 3125, 6250, 12500, 18750 (16,384 through 600,000 examples seen). The cosine schedule spans the full trajectory and is never restarted at evaluation checkpoints. Counts above 200,000 include repeats.

Keep the same 2,000 calibration / 8,000 audit pairs, watermark keys, 100,000-token radioactivity budget, MIN-K baseline, and wrong-key diagnostics. Recalibrate each checkpoint's threshold on calibration pairs only, then report all scheduled audit points without selecting favorable steps. Intermediate results are descriptive repeated measurements, not independent confirmation tests. Primary endpoint is the final scheduled checkpoint. Report all three seeds, signed gap as well as absolute advantage, and exact examples/steps/response tokens.

Architecture: isolated curve preparation/training/orchestration/report modules reuse existing scientific primitives. Four H200s support three training workers and one prediction worker. CPU score jobs launch after prediction artifacts are complete. Atomic checkpoint directories prevent readers from observing partial saves. Every checkpoint retains weights; only the newest resumable checkpoint keeps optimizer state. Original experiment artifacts stay immutable.

Implementation sequence (small independently checked changes):

1. Add `curve_prepare.py`: pinned download, shared filtered corpus and manifests, original-parent references. Verify row/source uniqueness, overlap manifests, lengths and coverage on a Slurm CPU setup job.
2. Add parent validation, schedule and resume helpers in `curve_train.py` with tests in `tests/test_curve.py`. Run the new tests first, implement, rerun. Reuse existing loss/accumulation/checkpoint primitives.
3. Implement continuous training with atomic interval checkpoints and recorded cumulative exposure. Verify a tiny local model uninterrupted versus resumed training has identical weights, scheduler and exposure.
4. Extend score CLI to accept explicit named complete interval checkpoints, preserving pilot rejection; run scoring tests.
5. Add `curve_pipeline.py` and Slurm wrappers. Freeze source snapshot/config manifests before launch, verify shell syntax and failure propagation. Submit GPU job after data setup, then start prediction worker and CPU scoring asynchronously.
6. Add `curve_report.py`: refresh CSV and plots under a lock after each score; include baseline P/B at zero exposure. Verify report points against metric and checkpoint artifacts.
7. End-to-end verification: observe the first real checkpoint being scored while training advances beyond it; verify accounting, artifacts and recorded exposure. Monitor throughput and completion, report available curve honestly if the external deadline interrupts the planned trajectory.

Data preparation can run while training/orchestration implementation proceeds. Code changes and launch are authorized by the user's request; no optional review or approval loop is needed.

## Amendment: larger single pass

The user prefers distinct examples and one epoch, permits a two-day GPU request,
and requires a verified plot while training continues. Prepare a separate 600,000
example corpus from the full `openbmb/UltraChat` training data, pinned to
`f220fe796ce3ed62fbe1681b45ce6cbc9c6cabe0`. Keep one first user/assistant exchange
per normalized instruction, also reject identical encoded training examples, and
apply the same union-of-three-shards overlap filter. Use one epoch, preserving
the original 18,750-step optimizer schedule and all seven checkpoint steps.
The full source differs from the smaller H4-curated source; record this explicitly.

Preserve the existing 200k run and let its audits supply an early plot. Once the
new data and runnable source are ready, retain durable old checkpoints, stop only
the superseded GPU allocation, and launch the fresh single-pass trajectories
from the original parents/base controls using four H200s and a 48-hour limit.
Already-submitted CPU audits keep running. Archive the supersession reason and
exact last durable checkpoint for each old target; do not present the abandoned
three-epoch trajectory as complete. Use separate artifacts under
`artifacts/watermark_curve_distinct` and parameterize report titles/counts from
the actual configs. Continue until a plot with multiple measured fine-tuning
points is present and verified; label partial coverage while jobs continue.

For the single-pass run, assign one GPU to each seed and alternate its target and
matched control at checkpoint boundaries. Resume both trajectories with the same
model, optimizer, scheduler, RNG and data position. Keep each seed on its original
GPU to preserve CUDA RNG state. This supplies matched control curves early instead
of waiting for the full target trajectory. The fourth GPU continues predictions
asynchronously. The existing exact-resume integration test covers the yielding
boundary, including an already-reached checkpoint being a no-op.
