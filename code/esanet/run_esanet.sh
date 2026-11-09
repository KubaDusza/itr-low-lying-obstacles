#!/usr/bin/env bash
# The published RGB-D architecture on our data: ESANet, its own code, its own three arms.
#
#   bash run_esanet.sh [gpu] [root] [epochs] [seeds]
#
# Why this exists: every other arm in this project is our own U-Net with the input changed,
# which a reader can dismiss in one sentence -- "you only tested your own fusion". ESANet
# (Seichter et al., ICRA 2021) is the RGB-D segmentation model built for mobile robots: two
# ResNet-34 encoders with squeeze-and-excite fusion, and a --modality flag that natively
# accepts rgb, depth and rgbd. So the same three-way comparison runs inside a published
# model and the two answers can be checked against each other.
#
# The recipe is the authors': SGD, median-frequency class weighting, their augmentation. We
# do not transplant ours onto it, because the point is the model as published. Comparing
# across the two architectures therefore carries a recipe difference, which the write-up
# states; the comparison *within* ESANet's three arms is clean.
#
# Predictions are written through predict_esanet.py and scored by the project's own
# eval.py, so every arm in the final table is measured the same way.
set -eu
GPU=${1:-0}
ROOT=${2:-/data/cs_courses/jdusza_itr}
EPOCHS=${3:-60}
SEEDS=${4:-0}
PY=/opt/venv/jdusza_venv/bin/python
ESA=$ROOT/esanet
HERE=$(cd "$(dirname "$0")" && pwd)
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU

COMMON="--dataset isod --dataset_dir $ROOT/esanet_isod --raw_depth
  --encoder resnet34 --encoder_block NonBottleneck1D
  --nr_decoder_blocks 3 --context_module ppm --decoder_channels_mode decreasing
  --encoder_decoder_fusion add --fuse_depth_in_rgb_encoder SE-add
  --upsampling learned-3x3-zeropad --height 448 --width 640 --batch_size_valid 8"

cd "$ESA"
for seed in $SEEDS; do
  for m in rgb depth rgbd; do
    name=esanet_${m}_s${seed}
    res=$ESA/results/$name
    if [ -d "$res/isod" ]; then
      echo "skip training $name"
    else
      echo "=== train $name ==="
      ESANET_SEED=$seed $PY train.py $COMMON --modality "$m" \
        --pretrained_dir ./trained_models/imagenet \
        --batch_size 8 --epochs "$EPOCHS" --lr 0.01 --optimizer SGD \
        --class_weighting median_frequency --results_dir "$res" || exit 1
    fi

    ckpt=$($PY - "$res" <<'EOF'
import glob, sys, torch, os
dirs = sorted(glob.glob(f"{sys.argv[1]}/isod/checkpoints_*"))
d = dirs[-1]
latest = f"{d}/ckpt_latest.pth"
best = None
if os.path.exists(latest):
    ep = torch.load(latest, map_location="cpu", weights_only=False).get("best_miou_epoch")
    if ep and os.path.exists(f"{d}/ckpt_epoch_{ep}.pth"):
        best = f"{d}/ckpt_epoch_{ep}.pth"
if best is None:
    cands = sorted(glob.glob(f"{d}/ckpt_epoch_*.pth"),
                   key=lambda p: int(p.rsplit("_", 1)[1][:-4]))
    best = cands[-1] if cands else latest
print(best)
EOF
)
    echo "=== predict $name from $ckpt ==="
    $PY "$HERE/predict_esanet.py" $COMMON --modality "$m" \
      --ckpt_path "$ckpt" --out "$ROOT/pred/$name" || exit 1
  done
done
echo ALL_ESANET_RUNS_DONE
