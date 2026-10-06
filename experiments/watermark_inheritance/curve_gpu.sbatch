#!/usr/bin/env bash
#SBATCH --job-name=wm_curve
#SBATCH --account=prio_zy286
#SBATCH --partition=priority_gpu
#SBATCH --qos=prio_zy286
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=160G
#SBATCH --time=26:00:00
#SBATCH --output=artifacts/watermark_curve/setup/logs/%x-%j.out
#SBATCH --error=artifacts/watermark_curve/setup/logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
curve_root="$PWD/artifacts/watermark_curve"
python_bin="$PWD/.venv/bin/python"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd "$curve_root/setup/source"
nvidia-smi --query-gpu=name,memory.total,uuid --format=csv
"$python_bin" -m experiments.watermark_inheritance.curve_pipeline --root "$curve_root" --gpus 4
