#!/usr/bin/env bash
#SBATCH --job-name=wm_pilot
#SBATCH --account=prio_zy286
#SBATCH --partition=priority_gpu
#SBATCH --qos=prio_zy286
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:2
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/watermark_inheritance/setup/logs/%x-%j.out
#SBATCH --error=artifacts/watermark_inheritance/setup/logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HOME="$PWD/.cache/watermark/hf"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
PYTHON_BIN="$PWD/.venv/bin/python"
nvidia-smi --query-gpu=name,memory.total,uuid --format=csv
"$PYTHON_BIN" -m experiments.watermark_inheritance.pipeline --phase "${1:?pilot or main}" --gpus "${2:?2 or 4}"
