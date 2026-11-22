#!/bin/bash -l
#SBATCH --job-name=itr-extended
#SBATCH --partition=gpu-a100-80g,gpu-h100-80g,gpu-h100-80g-short,gpu-h200-141g-short,gpu-h200-71g-ia
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/work/duszaj1/itr_extended_20261122/train-%A_%a.log
set -euo pipefail
module load scicomp-pytorch-env/2026.1
BASE=/scratch/work/duszaj1/itr_extended_20261122
export PYTHONPATH="$BASE/deps:$BASE/helpers:/scratch/work/duszaj1/itr_instances_20261122/deps${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export YOLO_CONFIG_DIR="$BASE/yolo-config" MPLCONFIGDIR="$BASE/matplotlib"
export SLURM_TMPDIR
SLURM_TMPDIR=$(mktemp -d /tmp/itr-extended-${SLURM_JOB_ID}.XXXXXX)
export TORCH_HOME="$BASE/torch-cache" HF_HOME="$BASE/hf-cache"
trap 'rm -rf "$SLURM_TMPDIR"' EXIT
if [[ "$1" == unet-gate ]]; then
 python -u "$BASE/triton_unet_logged.py" --base "$BASE" --gate
elif [[ "$1" == unet ]]; then
 python -u "$BASE/triton_unet_logged.py" --base "$BASE" --seed "$SLURM_ARRAY_TASK_ID"
else
 python -u "$BASE/triton_extended_boxes.py" --base "$BASE" --source "$1" --seed "$SLURM_ARRAY_TASK_ID" --epochs 200 --patience 20
fi
