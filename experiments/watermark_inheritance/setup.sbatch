#!/usr/bin/env bash
#SBATCH --job-name=wm_setup
#SBATCH --account=pi_zy286
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/watermark_inheritance/setup/logs/%x-%j.out
#SBATCH --error=artifacts/watermark_inheritance/setup/logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HOME="$PWD/.cache/watermark/hf"
PYTHON_BIN=/home/nhb25/.conda/envs/dream/bin/python
"$PYTHON_BIN" -m experiments.watermark_inheritance.prepare --run-dir artifacts/watermark_inheritance/run_00 --stage download --shared-dir .cache/watermark
for run in run_00 run_01 run_02; do
    "$PYTHON_BIN" -m experiments.watermark_inheritance.prepare --run-dir "artifacts/watermark_inheritance/$run" --stage build --shared-dir .cache/watermark
done
"$PYTHON_BIN" -m pip freeze > artifacts/watermark_inheritance/setup/packages.txt
