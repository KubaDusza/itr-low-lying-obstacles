#!/bin/zsh
# One-shot export of every GTE-only ESANet checkpoint (selected epoch, weights only).
set -u
OUT=${0:A:h}/weights; mkdir -p $OUT
typeset -A RUNS
RUNS=(
 esanet_full_rgb_reference_s0 "glaurung /data/dream/cs_courses/jdusza_itr/sam_review_20261114/paired_v1/runs/esanet_full_rgb_reference_s0"
 esanet_full_rgb_reference_s1 "glaurung /data/dream/cs_courses/jdusza_itr/sam_review_20261114/paired_v1/runs/esanet_full_rgb_reference_s1"
 esanet_full_rgb_reference_s2 "glaurung /data/dream/cs_courses/jdusza_itr/sam_review_20261114/paired_v1/runs/esanet_full_rgb_reference_s2"
 esanet_full_depth_reference_s0 "rivendell /mnt/gpu_storage/cs_courses/jdusza_itr/depth_parallel_v2/runs/esanet_full_depth_reference_s0"
 esanet_full_depth_reference_s1 "rivendell /mnt/gpu_storage/cs_courses/jdusza_itr/depth_parallel_v2/runs/esanet_full_depth_reference_s1"
 esanet_full_depth_reference_s2 "rivendell /mnt/gpu_storage/cs_courses/jdusza_itr/depth_parallel_v2/runs/esanet_full_depth_reference_s2"
 rgb_sam_v2_s0_recovery_glaurung "glaurung /data/dream/cs_courses/jdusza_itr/sam_v2_20261121/runs/rgb_sam_v2_s0_recovery_glaurung"
 rgb_sam_v2_s1 "rivendell /mnt/gpu_storage/cs_courses/jdusza_itr/sam_v2_20261121/runs/rgb_sam_v2_s1"
 rgb_sam_v2_s2 "rivendell /mnt/gpu_storage/cs_courses/jdusza_itr/sam_v2_20261121/runs/rgb_sam_v2_s2"
 depth_sam_v2_s0 "rivendell /mnt/gpu_storage/cs_courses/jdusza_itr/sam_v2_20261121/runs/depth_sam_v2_s0"
 depth_sam_v2_s1 "glaurung /data/dream/cs_courses/jdusza_itr/sam_v2_20261121/runs/depth_sam_v2_s1"
 depth_sam_v2_s2 "glaurung /data/dream/cs_courses/jdusza_itr/sam_v2_20261121/runs/depth_sam_v2_s2"
)
for run in ${(k)RUNS}; do
  host=${${=RUNS[$run]}[1]}; dir=${${=RUNS[$run]}[2]}
  [[ -s $OUT/$run.pth ]] && { echo "have $run"; continue; }
  echo "== $run on $host"
  PY=/opt/venv/jdusza_venv/bin/python; [[ $host == glaurung ]] && PY=/data/dream/cs_courses/jdusza_itr/sam_review_20261114/venv/bin/python; ssh -o ConnectTimeout=20 $host "cd /tmp && $PY - '$dir' '/tmp/jdusza_export_$run.pth'" <<'PY' || { echo "FAILED $run"; continue; }
import sys,pathlib,torch,json
d=pathlib.Path(sys.argv[1]);out=sys.argv[2]
lat=sorted(d.rglob('ckpt_latest.pth'));assert lat,'no ckpt_latest'
st=max(((p,torch.load(p,map_location='cpu',weights_only=False)) for p in lat),key=lambda a:a[1]['epoch'])
p,s=st;best=s['best_miou_epoch'];cands=sorted(p.parent.glob(f'ckpt_epoch_{best}.pth'));assert cands,best
ck=torch.load(cands[-1],map_location='cpu',weights_only=False)
torch.save(dict(state_dict=ck['state_dict'],epoch=ck.get('epoch',best),source=str(cands[-1]),last_epoch=s['epoch']),out)
print(json.dumps(dict(source=str(cands[-1]),best=best,last_epoch=s['epoch'])))
PY
  scp -q $host:/tmp/jdusza_export_$run.pth $OUT/$run.pth && ssh $host "rm -f /tmp/jdusza_export_$run.pth" && echo "ok $run $(du -h $OUT/$run.pth | cut -f1)"
done
ls $OUT | wc -l
