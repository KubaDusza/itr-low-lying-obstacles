#!/bin/bash -l
#SBATCH --job-name=itr-box-analysis
#SBATCH --partition=batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/work/duszaj1/itr_instances_20261122/analysis-%j.log
set -euo pipefail
module load scicomp-python-env
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
BASE=/scratch/work/duszaj1/itr_instances_20261122
python "$BASE/analyse_instance_detection.py" --base "$BASE" --out "$BASE/analysis"
