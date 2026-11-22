#!/bin/bash -l
#SBATCH --job-name=itr-pred300
#SBATCH --partition=gpu-a100-80g,gpu-h100-80g-short,gpu-h200-141g-short
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/work/duszaj1/itr_test_refs_20261122/pred-%j.log
set -euo pipefail
module load scicomp-pytorch-env/2026.1
B=/scratch/work/duszaj1/itr_test_refs_20261122
export PYTHONPATH="/scratch/work/duszaj1/itr_20261121/production/deps:/scratch/work/duszaj1/itr_extended_20261122/deps:/scratch/work/duszaj1/itr_instances_20261122/deps${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export SLURM_TMPDIR; SLURM_TMPDIR=$(mktemp -d /tmp/itr-pred-${SLURM_JOB_ID}.XXXXXX); trap 'rm -rf "$SLURM_TMPDIR"' EXIT
export YOLO_CONFIG_DIR=$SLURM_TMPDIR/yolo YOLO_OFFLINE=true
python -u $B/deploy/predict_test300.py --out $B/pred
