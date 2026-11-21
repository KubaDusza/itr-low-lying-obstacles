#!/bin/bash -l
#SBATCH --job-name=itr-gt-quality-audit
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --output=/scratch/work/duszaj1/itr_instance_20261121/gt-quality-%j.log
set -euo pipefail
module load scicomp-pytorch-env/2026.1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
BASE=/scratch/work/duszaj1/itr_instance_20261121
TASK_TMP=$(mktemp -d /tmp/itr-gt-audit-XXXXXX)
trap 'rm -rf "$TASK_TMP"' EXIT
tar xf "$BASE/gt-quality-inputs.tar" -C "$TASK_TMP"
python -u "$TASK_TMP/code/audit_gt_assistance_quality.py" --root "$TASK_TMP" --out "$BASE/gt_quality_audit"
