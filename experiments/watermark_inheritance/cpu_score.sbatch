#!/usr/bin/env bash
#SBATCH --job-name=wm_score
#SBATCH --account=pi_zy286
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=06:00:00
#SBATCH --array=0-17%6
#SBATCH --output=artifacts/watermark_inheritance/setup/logs/%x-%A_%a.out
#SBATCH --error=artifacts/watermark_inheritance/setup/logs/%x-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
checkpoints=(B P T1 T2 T T0)
task=${SLURM_ARRAY_TASK_ID:?}
run_index=$(( task / 6 ))
checkpoint_index=$(( task % 6 ))
run_name=$(printf 'run_%02d' "$run_index")
"$PWD/.venv/bin/python" -m experiments.watermark_inheritance.score \
    --run-dir "artifacts/watermark_inheritance/$run_name" \
    --checkpoint-name "${checkpoints[$checkpoint_index]}" --cpu-only
