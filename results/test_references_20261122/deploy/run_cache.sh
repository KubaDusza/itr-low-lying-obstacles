#!/bin/bash -l
#SBATCH --job-name=itr-testsam
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/work/duszaj1/itr_test_refs_20261122/cache-%j.log
set -euo pipefail
module load scicomp-pytorch-env/2026.1
BASE=/scratch/work/duszaj1/itr_test_refs_20261122
export PYTHONPATH="/scratch/work/duszaj1/itr_instances_20261122/samdeps:/scratch/work/duszaj1/itr_instances_20261122/deps${PYTHONPATH:+:$PYTHONPATH}"
T=$(mktemp -d /tmp/itr-testsam-${SLURM_JOB_ID}.XXXXXX); trap 'rm -rf "$T"' EXIT
mkdir -p $T/data $T/code
tar xzf /scratch/work/duszaj1/itr_20261121/production/data.tar.gz -C $T/data ISOD/data/HDCL_HW_A ISOD/data/MEB_CM ISOD/data/MEB_HW_B
cp $BASE/deploy/*.py $T/code/
export ITR_DATA=$T/data HF_HOME=$T/hf HF_HUB_CACHE=$T/hf/hub HF_HUB_DISABLE_XET=1
python - <<'PY'
import os
from huggingface_hub import HfApi,snapshot_download
m='bodhicitta/sam3';r='cba430d22f6fdc3f06ad3841274ec7bb55885f2f'
files=HfApi().list_repo_files(m,revision=r)
snapshot_download(m,revision=r,allow_patterns=[f for f in files if f.endswith(('.safetensors','.json','.txt','.model','.tiktoken'))],cache_dir=os.environ['HF_HUB_CACHE'],max_workers=4)
PY
cd $T/code
python -u isod_sam_review.py cache --frames $BASE/deploy/test_frames.json --cache $T/cache \
  --model bodhicitta/sam3 --revision cba430d22f6fdc3f06ad3841274ec7bb55885f2f --source-context 1 --batch 2
mkdir -p $BASE/cache && cp -r $T/cache/. $BASE/cache/
ls $BASE/cache/*/ | wc -l > $BASE/cache/DONE
