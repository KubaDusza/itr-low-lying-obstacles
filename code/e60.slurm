#!/bin/bash -l
#SBATCH --job-name=itr-e60
#SBATCH --partition=gpu-a100-80g,gpu-h100-80g,gpu-h100-80g-short,gpu-h200-141g-short,gpu-h200-71g-ia
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=00:55:00
#SBATCH --output=/scratch/work/duszaj1/itr_extended_20261122/e60-%x-%A_%a.log
set -euo pipefail
module load scicomp-pytorch-env/2026.1
BASE=/scratch/work/duszaj1/itr_extended_20261122
export PYTHONPATH="$BASE/deps:$BASE/helpers:/scratch/work/duszaj1/itr_instances_20261122/deps${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export YOLO_CONFIG_DIR="$BASE/yolo-config" MPLCONFIGDIR="$BASE/matplotlib"
export SLURM_TMPDIR; SLURM_TMPDIR=$(mktemp -d /tmp/itr-e60-${SLURM_JOB_ID}.XXXXXX)
export TORCH_HOME="$BASE/torch-cache" HF_HOME="$BASE/hf-cache"
trap 'rm -rf "$SLURM_TMPDIR"' EXIT
case "$1" in
 box-gate) python -u "$BASE/triton_boxes_e60.py" --base "$BASE" --gate ;;
 seg-gate) python -u "$BASE/triton_seg_e60.py" --base "$BASE" --gate ;;
 reference|automatic) python -u "$BASE/triton_boxes_e60.py" --base "$BASE" --source "$1" --seed "$SLURM_ARRAY_TASK_ID" ;;
 seg) python -u "$BASE/triton_seg_e60.py" --base "$BASE" --seed "$SLURM_ARRAY_TASK_ID" ;;
esac
