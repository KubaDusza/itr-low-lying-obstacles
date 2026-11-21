#!/bin/bash -l
#SBATCH --job-name=itr-speed-probe
#SBATCH --partition=gpu-debug
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/work/duszaj1/itr_20261121/benchmark-%j.log
set -euo pipefail
module load scicomp-pytorch-env/2026.1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
cd /scratch/work/duszaj1/itr_20261121
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -u benchmark_esanet_compute.py --source esanet \
    --output "benchmark-${SLURM_JOB_ID}.json"
