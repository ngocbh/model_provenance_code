# Plan: watermark inheritance experiment

**Goal**: Run the experiment specified in watermark.md on 2 or 4 GPUs and deliver auditable outputs by the deadline.
**Architecture**: Independent preparation/generation, training, prediction/scoring, and reporting stages with immutable run manifests and resumable outputs.
**Tech Stack**: Python, PyTorch, Transformers, datasets, scipy, upstream radioactive-watermark, Slurm Bouchet.

| Group | Steps | Files touched | Parallel |
|---|---|---|---|
| 1 | input preparation, common adapter; training; scoring/statistics | common.py/prepare.py/generate.py; train.py; score.py/report.py | Yes, independent files and agreed interfaces |
| 2 | module specification reviews, fixes, quality reviews | corresponding modules | Pipelined after each module |
| 3 | integration pilot and runtime estimate | pipeline.py/jobs/pilot.sbatch | After interfaces exist |
| 4 | full runs, monitoring and reports | jobs/run.sbatch/artifacts | After pilot verification |

All paths below are under `experiments/watermark_inheritance/` unless stated otherwise. Implementation is authorized by the request to run the experiment. Do not commit shared-worktree changes.

1. Pin model/data/upstream revisions and record installed versions. Validate downloaded metadata before generating (prepare.py).
2. Select one distinct 64-token paragraph prefix per WikiText article, split by article, reserve separate pilot (prepare.py). Check exact row counts and source uniqueness.
3. Implement shared JSON helpers and upstream Maryland adapter (common.py). Verify greenlist equality for multiple contexts and keys against upstream.
4. Generate seeded independent completion draws, randomly assign paired sides, retain ties and original tokens (generate.py). Verify identical schema/resume and text-detection pilot.
5. Build parent S-only and Dolly response-only datasets (train.py). Unit-test masks, truncation and absence of S-prime.
6. Train full parameters, checkpoint stages/epochs and training state (train.py). Verify optimizer reset, effective batches and resume metadata; GPU pilot demonstrates parameter change and reload.
7. Extract greedy predictions and completion MIN-K scores (score.py). Test index alignment against manually constructed sequences.
8. Apply within-example context exclusion and global tuple dedup only for aggregate tests (score.py). Verify order invariance for example scores and wrong-key controls.
9. Calibrate strict-greater thresholds only on calibration rows; compute held-out advantage and exact within-pair label-swap p-values (report.py). Test ties, null and perfectly separated samples.
10. Freeze aggregate budgets from pilot/calibration, generate complete table and before/after plots (report.py). Reject missing rows/checkpoints or insufficient matched token counts.
11. Review specification then code quality for each module using reviewer agents, fix material findings, and run `python -m unittest discover -s experiments/watermark_inheritance/tests`.
12. Compare Slurm test-only 2/4 GPU estimates immediately before request. Submit bounded pilot and main workload after measured validation; record IDs, timestamps, resource requests and deadlines.
13. Monitor actual scheduler handles and progress logs. Estimate remaining work from observed throughput; diagnose and resume bounded transient failures without duplicate writers.
14. End-to-end verification: all three runs have 10,000 complete pairs, correct 2,000/8,000 split, expected trained checkpoints, correct/wrong-key aggregate controls and both score families; verify accounting, CSV/JSON, plots, package/source revisions and final summary against watermark.md.
