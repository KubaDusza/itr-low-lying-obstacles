#!/bin/bash -l
#SBATCH --job-name=itr-SAM-instances
#SBATCH --partition=gpu-a100-80g,gpu-h100-80g
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/work/duszaj1/itr_instances_20261122/teacher-%j.log
set -euo pipefail
module load scicomp-pytorch-env/2026.1
BASE=/scratch/work/duszaj1/itr_instances_20261122
export PYTHONPATH="$BASE/samdeps:$BASE${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export SLURM_TMPDIR
SLURM_TMPDIR=$(mktemp -d /tmp/itr-sam-box-${SLURM_JOB_ID}.XXXXXX)
trap 'rm -rf "$SLURM_TMPDIR"' EXIT
python -u "$BASE/triton_native_teacher.py" --base "$BASE"
