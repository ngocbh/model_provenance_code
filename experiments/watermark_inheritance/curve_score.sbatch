#!/usr/bin/env bash
#SBATCH --account=pi_zy286
#SBATCH --partition=day
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=01:00:00
#SBATCH --constraint=cpugen:emeraldrapids|cpugen:turin
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
"${1:?python}" -m experiments.watermark_inheritance.score --run-dir "${3:?run}" \
    --checkpoint-name "${4:?name}" --checkpoint-path "${5:?checkpoint}" --cpu-only
"$1" -m experiments.watermark_inheritance.curve_report --root "${2:?root}"
