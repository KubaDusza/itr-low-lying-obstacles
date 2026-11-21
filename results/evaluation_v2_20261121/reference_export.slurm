#!/bin/bash -l
#SBATCH --job-name=itr-reference-instances
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --output=/scratch/work/duszaj1/itr_instance_20261121/reference-pool-%j.log
set -euo pipefail
module load scicomp-pytorch-env/2026.1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
BASE=/scratch/work/duszaj1/itr_instance_20261121
TASK_TMP=$(mktemp -d /tmp/itr-reference-XXXXXX)
trap 'rm -rf "$TASK_TMP"' EXIT
tar xzf "$BASE/inputs.tar.gz" -C "$TASK_TMP"
tar xf "$BASE/reference-inputs.tar" -C "$TASK_TMP"
python -u "$TASK_TMP/code/export_reference_instance_pool.py" --root "$TASK_TMP" --out "$BASE/reference_pool"
python -u "$TASK_TMP/code/audit_reference_instance_pool.py" --root "$TASK_TMP" --proposals "$BASE/reference_pool"
