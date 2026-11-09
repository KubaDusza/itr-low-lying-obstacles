#!/usr/bin/env bash
# The whole modality experiment on one GPU, start to finish.
#
#   bash run_modal.sh [gpu] [root]
#
# Safe to re-run: any run whose checkpoint exists is skipped, so an interrupted session
# continues where it stopped. Predictions are always rewritten, which is cheap.
#
# Blocks, in the order they are run:
#   1  the comparison          rgb / depth / rgbd / fuse, 3 seeds
#   2  is depth given a fair chance?  no conv1 surgery, geometric encoding, no pretraining
#   3  can the depth arm learn at all?  capacity and sufficiency checks
#   4  did we train it long enough?     schedule and learning-rate checks
#   5  where would a fusion gain come from?  blind controls and the geometric fusions
#   6  predictions, including the test-time depth ablations
set -u
GPU=${1:-0}
ROOT=${2:-/data/cs_courses/jdusza_itr}
PY=/opt/venv/jdusza_venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU ITR_DATA=$ROOT
cd "$(dirname "$0")"
mkdir -p "$ROOT/runs/modal"

# train <name> <args...> ; skips if the checkpoint is already there
train() {
  local name=$1; shift
  if [ -f "$ROOT/runs/modal/$name.pt" ]; then echo "skip $name"; return 0; fi
  echo "=== train $name ==="
  $PY train_modal.py --root "$ROOT" "$@" || exit 1
}

echo "######## 1. the comparison ########"
for s in 0 1 2; do
  for m in rgb depth rgbd fuse; do
    train "${m}_s${s}" --modality "$m" --seed "$s"
  done
done

echo "######## 2. is depth given a fair chance? ########"
for s in 0 1 2; do
  # the pretrained first convolution left untouched, depth tripled instead
  train "depth3_s${s}" --modality depth3 --seed "$s"
  # depth repackaged geometrically, HHA's idea: disparity, height above floor, tilt
  train "geo_s${s}" --modality geo --seed "$s"
  # no ImageNet weights anywhere: the fair comparison when the arms differ in function space
  train "rgb_scratch_s${s}" --modality rgb --seed "$s" --scratch
  train "depth_scratch_s${s}" --modality depth --seed "$s" --scratch
done

echo "######## 3. can the depth arm learn at all? ########"
# capacity: 50 training frames, 60 epochs, scored on those same frames. A model that can
# fit them has the capacity to represent these obstacles from this input.
train "rgb_overfit_s0"   --modality rgb   --seed 0 --limit 50 --epochs 60 --tag overfit
train "depth_overfit_s0" --modality depth --seed 0 --limit 50 --epochs 60 --tag overfit
train "geo_overfit_s0"   --modality geo   --seed 0 --limit 50 --epochs 60 --tag overfit
# sufficiency: the same arm, same schedule, on the drivable-floor task that depth suits
train "rgb_floor_s0"   --modality rgb   --seed 0 --target floor --tag floor
train "depth_floor_s0" --modality depth --seed 0 --target floor --tag floor

echo "######## 4. did we train it long enough? ########"
train "depth_long_s0" --modality depth --seed 0 --epochs 45 --tag long
train "depth_lr1e3_s0" --modality depth --seed 0 --lr 1e-3 --tag lr1e3
train "depth_lr3e3_s0" --modality depth --seed 0 --lr 3e-3 --tag lr3e3
train "rgb_long_s0"   --modality rgb   --seed 0 --epochs 45 --tag long

echo "######## 5. where would a fusion gain come from? ########"
for s in 0 1 2; do
  # same architecture and parameter count, depth replaced by a constant during training
  train "rgbd_blind_s${s}" --modality rgbd --seed "$s" --blind
  train "fuse_blind_s${s}" --modality fuse --seed "$s" --blind
  # fusion of the geometric encoding, early and mid-level
  train "rgbgeo_s${s}"  --modality rgbgeo  --seed "$s"
  train "fusegeo_s${s}" --modality fusegeo --seed "$s"
done

echo "######## 6. predictions on the test split ########"
for s in 0 1 2; do
  for m in rgb depth depth3 geo rgbd fuse rgbgeo fusegeo; do
    $PY predict_modal.py --root "$ROOT" --modality "$m" --seed "$s" || exit 1
  done
  for m in rgb depth; do
    $PY predict_modal.py --root "$ROOT" --modality "$m" --seed "$s" --scratch || exit 1
  done
  for m in rgbd fuse; do
    $PY predict_modal.py --root "$ROOT" --modality "$m" --seed "$s" --blind || exit 1
    for ab in const shuffle; do
      $PY predict_modal.py --root "$ROOT" --modality "$m" --seed "$s" --depth "$ab" || exit 1
    done
  done
done
for r in depth_long depth_lr1e3 depth_lr3e3 rgb_long; do
  m=${r%%_*}; tag=${r#*_}
  $PY predict_modal.py --root "$ROOT" --modality "$m" --seed 0 --tag "$tag" || exit 1
done

echo "######## did the depth path move off zero? ########"
$PY diagnose.py depth --root "$ROOT"
echo ALL_MODAL_RUNS_DONE
