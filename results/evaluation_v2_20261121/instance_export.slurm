#!/bin/bash -l
#SBATCH --job-name=itr-instance-export-v2
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/scratch/work/duszaj1/itr_instance_20261121/v2/export-%j.log
set -euo pipefail
module load scicomp-pytorch-env/2026.1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
BASE=/scratch/work/duszaj1/itr_instance_20261121
TASK_TMP=$(mktemp -d /tmp/itr-instances-XXXXXX)
trap 'rm -rf "$TASK_TMP"' EXIT
mkdir "$TASK_TMP/code"
tar xzf "$BASE/inputs.tar.gz" -C "$TASK_TMP"
cp "$BASE/v2/refine_sam_instances.py" "$BASE/v2/export_instance_proposals.py" "$TASK_TMP/code/"
python -u "$TASK_TMP/code/export_instance_proposals.py" --root "$TASK_TMP" --out "$BASE/v2/proposals"
