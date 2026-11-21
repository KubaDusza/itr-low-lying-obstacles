#!/bin/bash -l
#SBATCH --job-name=itr-final-analysis
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --output=/scratch/work/duszaj1/itr_20261121/production/analysis-%j.log
set -euo pipefail
module load scicomp-pytorch-env/2026.1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR
MPLCONFIGDIR=$(mktemp -d /tmp/itr-mpl-XXXXXX)
export SLURM_TMPDIR
SLURM_TMPDIR=$(mktemp -d /tmp/itr-analysis-XXXXXX)
trap 'rm -rf "$SLURM_TMPDIR" "$MPLCONFIGDIR"' EXIT
BASE=/scratch/work/duszaj1/itr_20261121/production
python -u "$BASE/code/triton_finish_analysis.py" --base "$BASE" "$@"
