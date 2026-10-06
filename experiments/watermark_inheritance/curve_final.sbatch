#!/usr/bin/env bash
#SBATCH --account=pi_zy286
#SBATCH --partition=day
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"${1:?python}" -m experiments.watermark_inheritance.curve_report --root "${2:?root}"
"$1" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); r=json.loads((p/"reports/curve.json").read_text()); assert r["completed_nonbaseline_points"] == r["planned_nonbaseline_points"] == 42; print("All 42 interval audits complete")' "$2"
