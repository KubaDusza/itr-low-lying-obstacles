#!/usr/bin/env bash
# Run the RGB / Depth / RGB-D x seed 0,1,2 experiments one after another on ONE GPU.
# A run is skipped if runs/<name>.pt already exists; the first failure stops everything.
#
#   tmux new -s isod
#   cd ~/intro_to_research/itr-low-lying-obstacles/code
#   bash run_remaining.sh 1          # 1 = GPU id
#   (detach: Ctrl-b d   reattach: tmux attach -t isod)
set -euo pipefail

GPU=${1:?usage: bash run_remaining.sh <gpu id>}
PY=${PY:-/opt/venv/jdusza_venv/bin/python}
ROOT=${ROOT:-$HOME/datasets/isod_work}

cd "$(dirname "$0")"
mkdir -p "$ROOT/runs/logs"

for seed in 0 1 2; do
  for m in rgb depth rgbd; do
    name="unet_mobilenet_v2_${m}_s${seed}"
    if [ -e "$ROOT/runs/$name.pt" ]; then
      echo "[skip] $name (already finished)"
      continue
    fi
    log="$ROOT/runs/logs/$name.log"
    echo "[$(date '+%F %T')] start $name   log: $log"
    if ! CUDA_VISIBLE_DEVICES=$GPU "$PY" train_modal.py --root "$ROOT" --modality "$m" --seed "$seed" \
         > "$log" 2>&1; then
      echo "[$(date '+%F %T')] FAILED $name -- last lines of $log:"
      tail -n 20 "$log"
      exit 1
    fi
    echo "[$(date '+%F %T')] done  $name"
  done
done
echo "all runs finished"
