#!/usr/bin/env bash
#SBATCH --job-name=wm_report
#SBATCH --account=pi_zy286
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/watermark_inheritance/setup/logs/%x-%j.out
#SBATCH --error=artifacts/watermark_inheritance/setup/logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
for run in run_00 run_01 run_02; do
    "$PWD/.venv/bin/python" -m experiments.watermark_inheritance.report \
        --run-dir "artifacts/watermark_inheritance/$run"
done
"$PWD/.venv/bin/python" -m experiments.watermark_inheritance.report \
    --run-dirs artifacts/watermark_inheritance/run_00 artifacts/watermark_inheritance/run_01 artifacts/watermark_inheritance/run_02 \
    --output-dir artifacts/watermark_inheritance/reports
