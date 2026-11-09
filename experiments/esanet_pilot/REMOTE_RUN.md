# REMOTE_RUN — esanet_pilot (v2, 2026-11-09)

You are a Claude Code session on the university Linux host **shire** (reached from `gtlpc103`
with `ssh shire`). **You have no chat history.** This file is the complete task. Work through the
phases in order and **stop at every STOP**: report, wait for the human, do not improvise.

## 1. Purpose
The project asks whether stereo depth improves segmentation of small, low-lying floor obstacles.
The agreed method compares **ESANet R34-NBt1D** (Seichter et al., ICRA 2021, official code
TUI-NICR/ESANet) with RGB, depth and RGB-D input on ISOD, using our existing site split and
derived obstacle masks. This run **does not train the models for the paper**. It checks that the
data are what we think they are, prepares a human review of the obstacle masks, and (later, only
with separate approval) checks that the pipeline runs on the GPU and measures time, memory and disk.

**Annotation status (agreed with Jakub, 2026-11-09).** Train only on ISOD; never on GTE-Floor.
The ISOD obstacle masks in `ISOD_gt/` come from an automatic wall-removal rule and have **not
been verified**. They are annotation version **v0: frozen, provisional**. They may be used for the
technical pilot only; definitive experiments wait for the review in
`experiments/isod_mask_review/PROTOCOL.md`. Never modify, move or overwrite `ISOD_gt/`.

**Phases authorised now: A, B, C, C2 (no GPU).** Phases D and E use the GPU and need a separate,
explicit "go" from the human, even if A–C2 pass.

## 2. Hard rules
- Never ask for, type or log passwords, tokens or keys. If something needs a login, the human does it.
- Do **not** modify, move or overwrite anything in the dataset folder (`ISOD/`, `ISOD_gt/`, `splits.json`).
- Do **not** install anything into `/opt/venv/jdusza_venv` (owned by collaborator `jdusza`). Only *use* it.
- Do **not** write runs, checkpoints or downloads into the home directory `/home/GTL/jgomezel`
  (network disk, ~5.7 GiB free on 2026-11-09). Use `$SCRATCH` (Phase A). Do not assume `/tmp` persists.
- Do not delete files, kill processes other than your own Phase E run, edit `.bashrc`, use more than one GPU, or run anything
  beyond the pilot in Phase E. No `git push`, no force operations.
- If something differs from this file, STOP and report it; never guess a path.

## 3. Known facts (re-verify; user-checked on 2026-11-09)
| Item | Value |
|---|---|
| Repo | `/home/GTL/jgomezel/intro_to_research/itr-low-lying-obstacles` |
| Code for this task | branch **`esanet-pilot`** (child of `62c3036` on `juan-rgbd`; check with `git log --oneline -8`): `code/check_isod_esanet.py`, `code/review_isod_masks.py`, `code/train_esanet.py` |
| Dataset root (`--root`) | `/home/GTL/jgomezel/datasets/isod_work` → `ISOD/data/<site>/{rgb,depth,label,mask}/<i>.png`, `ISOD_gt/<site>/<i>.png`, `splits.json` (14/3/3 sites), `depth_stats.json` (U-Net runs) |
| Python | `/opt/venv/jdusza_venv/bin/python` (PyTorch 2.14.0+cu130, CUDA available) |
| GPUs | 2 × RTX 3090 24 GiB |

Expected data conventions (taken from `code/isod.py`, `prep_gt.py`, `train_modal.py`; **verified in Phase C, not assumed**):
RGB 8-bit 3-channel; depth 16-bit **millimetres**, 0 = no reading; `label/` 255 = drivable floor,
0 = not drivable; `ISOD_gt/` 255 = obstacle, **derived automatically** by `isod.obstacles(label)`
(not hand-corrected); frames 640×422. ESANet labels built by `make_esanet_label`: 0 = void/ignored
(not drivable and not obstacle = walls/furniture), 1 = background, 2 = obstacle.

## Phase A — read-only preflight
```bash
hostname; date
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
free -h; df -h ~ /tmp; quota -s 2>/dev/null || echo "no quota command"
ls -ld /data /data/cs_courses /data/cs_courses/* 2>&1 | head    # course storage lead from the README
cd /home/GTL/jgomezel/intro_to_research/itr-low-lying-obstacles
git status --short; git branch --show-current; git rev-parse --short HEAD
ls /home/GTL/jgomezel/datasets/isod_work; du -sh /home/GTL/jgomezel/datasets/isod_work
/opt/venv/jdusza_venv/bin/python - <<'EOF'
import importlib
for m in ["torch", "torchvision", "cv2", "numpy", "pandas", "matplotlib", "PIL"]:
    try:
        x = importlib.import_module(m); print(m, getattr(x, "__version__", "ok"))
    except Exception as e:
        print(m, "MISSING", e)
import torch; print("cuda", torch.cuda.is_available(), torch.cuda.device_count())
EOF
curl -sS -o /dev/null -w "drive.google.com %{http_code}\n" --max-time 10 https://drive.google.com
curl -sS -o /dev/null -w "github.com %{http_code}\n" --max-time 10 https://github.com
```
**STOP.** Show the output, then ask the human:
1. Which directory is the **approved persistent storage** for ~5 GB now (up to ~10 GB later)? → `SCRATCH`
2. May this session *use* (read/execute only) `/opt/venv/jdusza_venv`?
Do not continue without both answers, or if a package above is MISSING, or if the repo is not clean.

## Phase B — get the code and ESANet (small downloads only)
```bash
export SCRATCH=<answer 1>
cd /home/GTL/jgomezel/intro_to_research/itr-low-lying-obstacles
git fetch origin esanet-pilot && git checkout esanet-pilot && git log --oneline -8   # must show 62c3036 below the new commits
mkdir -p $SCRATCH/esanet_pilot && cd $SCRATCH/esanet_pilot
git clone https://github.com/TUI-NICR/ESANet.git && git -C ESANet rev-parse --short HEAD     # 820c5bb was tested
mkdir -p ESANet/trained_models/imagenet && cd ESANet/trained_models/imagenet
/opt/venv/jdusza_venv/bin/python -m gdown 1neUb6SJ87dIY1VvrSGxurVBQlH8Pd_Bi \
  || echo "no gdown/drive access: STOP and ask the human to copy imagenet_r34_NBt1D.tar.gz here"
tar -xvzf imagenet_r34_NBt1D.tar.gz
find . -name "r34_NBt1D.pth" -o -name "logs.csv"; sha256sum $(find . -name "*.pth"); du -sh .
```
`--pretrained_dir` must be the folder that directly contains `r34_NBt1D.pth` (ESANet looks for
exactly that name). If `git fetch` finds no `esanet-pilot` branch, STOP: the human has not pushed it yet.

## Phase C — data audit (read-only on the dataset, ~2–5 min CPU)
```bash
cd /home/GTL/jgomezel/intro_to_research/itr-low-lying-obstacles/code
/opt/venv/jdusza_venv/bin/python check_isod_esanet.py \
  --root /home/GTL/jgomezel/datasets/isod_work --out $SCRATCH/esanet_pilot/audit \
  --esanet $SCRATCH/esanet_pilot/ESANet
```
Report from `audit/isod_audit.json`: `split_sizes`, `frame_sizes`, `floor_label_values`,
`obstacle_mask_values`, `frames_where_mask_differs_from_automatic_rule`, `per_split`,
`esanet_preprocessing`, `problems`. Open `audit/isod_audit.png` and describe it. Columns: RGB, depth, ESANet label
(black = ignored, grey = background, red = obstacle). The yellow obstacle outlines should sit on
the objects in **both** RGB and depth.

How to read it:
- `frames_where_mask_differs_from_automatic_rule` = 0 → the obstacle masks are exactly the automatic rule:
  **no manual correction**. If > 0, list `examples_of_differing_frames`. Do not "fix" anything.
- `floor_label_values` other than {0, 255} → STOP (make_esanet_label would treat them as floor).
- Any entry in `problems`, or outlines visibly off the objects in depth → **STOP**.

## Phase C2 — mask review package (read-only on the dataset, CPU, a few minutes)
```bash
cd /home/GTL/jgomezel/intro_to_research/itr-low-lying-obstacles/code
/opt/venv/jdusza_venv/bin/python review_isod_masks.py \
  --root /home/GTL/jgomezel/datasets/isod_work --out $SCRATCH/esanet_pilot/mask_review_v0
du -sh $SCRATCH/esanet_pilot/mask_review_v0; ls $SCRATCH/esanet_pilot/mask_review_v0/review_images | wc -l
```
Report `summary.json` (fingerprint, hint counts per split, `frames_masks_differ_from_rule`) and
the number of review images (expected: 100 systematic + up to 150 flagged). **Do not judge the
images yourself and do not fill in `review_log.csv`**: the human reviewers do that, following
`experiments/isod_mask_review/PROTOCOL.md`. Then pack the folder for download (one small file,
expected ~20–40 MB; these review/pilot files are the only files allowed in the home directory):
```bash
tar czf ~/isod_mask_review_v0.tgz -C $SCRATCH/esanet_pilot mask_review_v0 && ls -lh ~/isod_mask_review_v0.tgz
```
Also pack the 20-frame annotation-pilot package (exact v0 files + checksums, see
`experiments/isod_mask_review/ANNOTATION_PILOT.md` §9; expected ~15–40 MB):
```bash
cd /home/GTL/jgomezel/datasets/isod_work
tar czf ~/isod_v0_pilot.tgz -T $SCRATCH/esanet_pilot/mask_review_v0/pilot_files.txt
cp $SCRATCH/esanet_pilot/mask_review_v0/pilot_files.sha256 $SCRATCH/esanet_pilot/mask_review_v0/pilot_frames.csv ~/
ls -lh ~/isod_v0_pilot.tgz ~/pilot_files.sha256 ~/pilot_frames.csv
```
Report `pilot_frames.csv` (how many objective and difficult frames were found).
The human downloads these files in VS Code (Explorer → right-click → Download) and deletes them
from the home directory afterwards.
**STOP.** Phases D and E start only after an explicit "go" from the human.

## Phase D — GPU smoke test (2 epochs × 3 batches, real batch size 8, ~minutes) — needs explicit go
Uses the provisional v0 masks; nothing from this phase is a result. Pick a GPU that Phase A showed as idle (`G=0` or `G=1`), then:
```bash
cd /home/GTL/jgomezel/intro_to_research/itr-low-lying-obstacles/code
P=$SCRATCH/esanet_pilot/ESANet/trained_models/imagenet   # folder containing r34_NBt1D.pth
for m in rgb depth rgbd; do
  CUDA_VISIBLE_DEVICES=$G /opt/venv/jdusza_venv/bin/python train_esanet.py \
    --esanet $SCRATCH/esanet_pilot/ESANet --pretrained_dir $P \
    --root /home/GTL/jgomezel/datasets/isod_work --out $SCRATCH/esanet_pilot/runs \
    --modality $m --seed 0 --smoke 2>&1 | grep -v Warning | tee $SCRATCH/esanet_pilot/smoke_$m.log
done
```
The first call computes `runs/isod_train_stats_ISOD_gt.json` (depth mean/std of valid train
pixels in mm, class weights, mask fingerprint) once and reuses it in later runs. Each log starts with
`Masks: ISOD_gt, fingerprint ...`, which must equal the fingerprint in the Phase C2 `summary.json`. Expected: exit 0, finite losses, 10 test PNGs at
640×422 (the 10-frame test score is not a result), `best.pt` ≈ 125 MB (RGB, depth) and ≈ 180 MB (RGB-D). Params 32.1 M / 32.0 M / 46.9 M.
Validation numbers after 3 batches are meaningless. Report `peak_gpu_memory_MiB`, the train statistics
(sanity: depth mean should be a few thousand mm; obstacle class weight ≫ background weight) and errors.
STOP on: out-of-memory, NaN, `Missing key`/pretrained-load error, or depth mean outside ~500–6000 mm.

## Phase E — time-bounded pilot (one run, RGB-D, 10 epochs) — needs explicit go
Only if A–D passed and the human said go. Provisional v0 masks: the numbers are pipeline checks only. In `tmux` (detach Ctrl-b d, reattach `tmux attach -t esanet`):
```bash
tmux new -s esanet
cd /home/GTL/jgomezel/intro_to_research/itr-low-lying-obstacles/code
CUDA_VISIBLE_DEVICES=$G /opt/venv/jdusza_venv/bin/python train_esanet.py \
  --esanet $SCRATCH/esanet_pilot/ESANet --pretrained_dir $P \
  --root /home/GTL/jgomezel/datasets/isod_work --out $SCRATCH/esanet_pilot/runs \
  --modality rgbd --seed 0 --epochs 10 2>&1 | grep -v Warning | tee $SCRATCH/esanet_pilot/pilot_rgbd.log
```
While it runs, check `nvidia-smi` once in a while. **Hard stop: kill only this run** (Ctrl-c in its
tmux) if an epoch takes over 5 minutes or another user's job appears on the same GPU.
The 10-epoch OneCycle schedule is not a shortened 500-epoch run (the learning rate peaks at
epoch 1–2 instead of 50), so its IoU is not a result. It measures cost and checks that training works:
- seconds/epoch = last `seconds` in `history.csv` ÷ 10 → **projected 500-epoch hours** = s/epoch × 500 / 3600
- train_loss should fall; val_IoU should rise well above 0. Rough reference only: the U-Net RGB runs
  reached val IoU ≈ 0.76 in 15 epochs. A value near 0 after 10 epochs is a red flag.
- `test_eval.json` has test IoU/precision/recall (same scoring functions as `eval.py`). Report it as "pilot, not a result".
- `du -sh $SCRATCH/esanet_pilot/runs/*` for the disk footprint.
**STOP.** Do not start other modalities, seeds or longer runs.

## 4. Return: `$SCRATCH/esanet_pilot/REMOTE_RESULTS.md` (also paste it to the human)
1. What ran (phases, commands), host, GPU id, date. Repo commit and ESANet commit (in `config.json`),
   mask folder and fingerprint.
2. Phase A output (trimmed), the `SCRATCH` and venv answers.
3. Audit summary + description of `isod_audit.png`; mask-review `summary.json` and image count.
4. Smoke: per modality exit code, params, peak memory, best.pt size, train statistics.
5. Pilot: `history.csv` content, s/epoch, projected 500-epoch hours, peak memory, disk, test metrics (labelled *pilot*).
6. Everything that differed from this file, every error (last 20 lines), and what did not run and why.
Copy `isod_audit.json`, `isod_audit.png`, the review `summary.json` and (if D/E ran) `history.csv`,
`config.json`, `test_eval.json` next to it (they are small); leave `best.pt` and predictions in `$SCRATCH`.

## 5. What each script adds to ESANet (for reviewers)
Unchanged from ESANet: model (`build_model`, paper defaults), augmentation (rescale 1.0–1.4, crop
448×640, HSV jitter, flip), normalisation with `depth_mode='raw'` (invalid depth stays 0), weighted
cross-entropy with deep supervision at 1/8, 1/16, 1/32, SGD (lr 0.01, momentum 0.9, Nesterov, wd 1e-4),
OneCycle stepped once per epoch, batch 8, fp32. Ours: ISOD reading and the 3-value label; train-only
statistics; validation loss and obstacle IoU/precision/recall each epoch; selection by validation
obstacle IoU, one checkpoint (weights only); test masks at 640×422 scored with `isod.ignore_region`,
`isod.score`, `isod.totals` — the functions `eval.py` uses for every other method (void excluded,
pixel counts summed over frames; checked identical to `eval.py` on 300 frames).

## 6. Text to paste into a fresh Claude Code session on shire
> I am on shire (via gtlpc103) in VS Code, in `/home/GTL/jgomezel/intro_to_research/itr-low-lying-obstacles`.
> You do not have my previous chats. Run `git fetch origin esanet-pilot` and read the task with
> `git show origin/esanet-pilot:experiments/esanet_pilot/REMOTE_RUN.md`, then follow it
> exactly, phase by phase, stopping at every STOP to wait for me. Today only Phases A, B, C and C2
> (no GPU); never start D or E without my explicit go. Never ask for or print passwords. Do not
> modify the dataset or ISOD_gt, install into `/opt/venv/jdusza_venv`, put runs or downloads in my
> home directory (except the review and pilot packages), delete files, push, or train. Finish by writing
> REMOTE_RESULTS.md as specified.
