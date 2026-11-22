#!/bin/bash -l
#SBATCH --job-name=itr-box-detect
#SBATCH --partition=gpu-a100-80g,gpu-h100-80g
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/work/duszaj1/itr_instances_20261122/detect-%A_%a.log
set -euo pipefail
module load scicomp-pytorch-env/2026.1
BASE=/scratch/work/duszaj1/itr_instances_20261122
export PYTHONPATH="$BASE/deps${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export YOLO_CONFIG_DIR="$BASE/yolo-config"
export MPLCONFIGDIR="$BASE/matplotlib"
export SLURM_TMPDIR
SLURM_TMPDIR=$(mktemp -d /tmp/itr-box-${SLURM_JOB_ID}.XXXXXX)
trap 'rm -rf "$SLURM_TMPDIR"' EXIT
if [[ "${1:-}" == gate ]]; then
 python -u "$BASE/triton_instance_detection.py" --base "$BASE" --gate
else
 python -u "$BASE/triton_instance_detection.py" --base "$BASE" --source "${1}" --seed "${SLURM_ARRAY_TASK_ID:-0}"
fi
