#!/bin/bash -l
#SBATCH --job-name=itr-cost
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:25:00
#SBATCH --output=/scratch/work/duszaj1/itr_extended_20261122/cost-%j.log
set -euo pipefail
module load scicomp-pytorch-env/2026.1
BASE=/scratch/work/duszaj1/itr_extended_20261122
export PYTHONPATH="/scratch/work/duszaj1/itr_instances_20261122/samdeps:$BASE/deps:/scratch/work/duszaj1/itr_instances_20261122/deps${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export SLURM_TMPDIR
SLURM_TMPDIR=$(mktemp -d /tmp/itr-cost-${SLURM_JOB_ID}.XXXXXX)
trap 'rm -rf "$SLURM_TMPDIR"' EXIT
export YOLO_CONFIG_DIR="$SLURM_TMPDIR/yolo"
python -u "$BASE/triton_inference_cost.py" --base "$BASE" --device "$1"
