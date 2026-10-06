#!/usr/bin/env bash
#SBATCH --job-name=wm_curve_data
#SBATCH --account=pi_zy286
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/watermark_curve/setup/logs/%x-%j.out
#SBATCH --error=artifacts/watermark_curve/setup/logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HOME="$PWD/.cache/watermark/hf"
"$PWD/.venv/bin/python" -m experiments.watermark_inheritance.curve_prepare --workers "${SLURM_CPUS_PER_TASK:-4}" "$@"
