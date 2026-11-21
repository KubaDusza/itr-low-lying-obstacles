#!/bin/bash -l
#SBATCH --job-name=itr-rgbd210
#SBATCH --partition=gpu-a100-80g,gpu-h100-80g
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/work/duszaj1/itr_20261121/production/slurm-%A_%a.log
set -euo pipefail
module load scicomp-pytorch-env/2026.1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
BASE=/scratch/work/duszaj1/itr_20261121/production
export PYTHONPATH="$BASE/deps${PYTHONPATH:+:$PYTHONPATH}"
export SLURM_TMPDIR
SLURM_TMPDIR=$(mktemp -d "/tmp/itr-${SLURM_JOB_ID}.XXXXXX")
trap 'rm -rf "$SLURM_TMPDIR"' EXIT
if [[ "${1:-}" == gate ]]; then
    python -u "$BASE/code/triton_rgbd210_job.py" --base "$BASE" --gate
else
    python -u "$BASE/code/triton_rgbd210_job.py" --base "$BASE" --index "$SLURM_ARRAY_TASK_ID"
fi
