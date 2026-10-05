# Roadmap

Status of the project and what is left. Ordered by dependency, not by date.
Background and decisions: [`NOTES.md`](NOTES.md). Results so far: [`README.md`](README.md).

**Question:** does adding stereo depth to RGB improve detection of low-lying floor
obstacles (cables, gloves, screws), compared with RGB or depth alone?

**Secondary:** can SAM pseudo-labels replace hand labels for training a lightweight model?

---

## Done

- [x] **Dataset.** ISOD: 2,000 RGB-D frames, 20 indoor sites, hand-drawn floor labels.
- [x] **Obstacle ground truth.** Derived from the hand labels by removing the walls-and-furniture
      component; about 8.8 objects per frame. `code/prep_gt.py`
- [x] **Splits by site** (14 train / 3 val / 3 test), so test rooms are never trained on.
- [x] **Metrics:** IoU, precision, recall on the obstacle class, walls and furniture ignored.
      `code/eval.py`
- [x] **Depth baseline** (ISOD's own ground-plane method): IoU 0.031, recall 0.039.
- [x] **Three RGB models:** YOLOv8n-seg 0.744, U-Net 0.776, DeepLabV3+ 0.757 IoU.
- [x] **Agreement analysis:** the RGB models agree with each other (0.86) more than with
      the ground truth, so they fail on similar pixels.
- [x] **Epoch check:** 80 epochs is no better than 20; validation plateaus around epoch 12–15.
- [x] **Infrastructure:** GPU server (`shire`, `/opt/venv/jdusza_venv`), repo, figures,
      training curves.

## 1. Fair depth-only baseline

The original depth number comes from a classic geometric method, which is a weak opponent.
A learned model on the same data gives a fairer three-way comparison.

- [x] Train the same U-Net + MobileNetV2 with depth-only input.
- [x] Use 16-bit depth in millimetres with explicit handling of invalid depth (`0`).
- [x] Compute depth normalisation statistics from the training sites only.
- [x] Evaluate with the same site split and metrics.
- [x] Repeat with seeds 0, 1 and 2.

**Result:** learned Depth reaches `0.390 ± 0.003` IoU, far above the geometric
ground-plane baseline (`0.031`), but still well below RGB (`0.772 ± 0.001`).

## 2. RGB-D fusion — the main question

### Early fusion

- [x] Train U-Net + MobileNetV2 with 4-channel RGB-D input.
- [x] Preserve the pretrained RGB filters.
- [x] Initialise the added depth filter to zero.
- [x] Verify that paired RGB and RGB-D runs start from identical comparable weights.
- [x] Train RGB, Depth and RGB-D with the same split, augmentation, loss, batch size,
      learning rate and 15-epoch schedule.
- [x] Run 3 seeds for every modality.
- [x] Compare RGB / Depth / RGB-D in one table.
- [x] Report mean ± sample standard deviation.

Main pixel-level result:

- RGB: `0.772 ± 0.001` IoU
- Depth: `0.390 ± 0.003` IoU
- RGB-D: `0.772 ± 0.003` IoU

**Conclusion:** early RGB-D fusion does not improve overall pixel-level segmentation
over RGB alone on ISOD.

### Where does depth help?

- [x] Compute per-site metrics.
- [x] Analyse object recall by size.
- [x] Analyse object recall by distance.
- [x] Analyse RGB / Depth detection overlap.
- [x] Analyse RGB / RGB-D detection overlap.
- [x] Repeat the object analysis with 50% and 25% hit thresholds.
- [x] Generate summary figures.

At the 50% object-hit threshold:

- small objects (<144 px): RGB-D improves recall by about `+0.024`
- medium objects: about `+0.005`
- large objects: about `+0.001`
- far objects (>=2900 mm): about `+0.024`
- near objects (<1442 mm): about `-0.005`
- all objects: about `+0.010`

The small- and far-object gains are positive in all three seeds.

Depth provides little direct complementarity: only 15 of 3,471 test objects are
detected by Depth while missed by RGB, and RGB-D recovers only 2 of those 15.

Current hypothesis: depth may help mainly as scene / floor-geometry context rather than
as a direct obstacle cue. This is not yet demonstrated mechanistically.

Other fusion designs (late fusion, two-branch models such as ESANet) are not needed for
the ISOD result: depth finds almost no objects that RGB misses, so there is little for a
better fusion to recover. They are listed under section 6, conditional on the robot data.

## 3. Robot-camera evaluation — next priority

The next experiment tests whether the ISOD conclusion transfers to a different camera,
sensor geometry and indoor environment.

### Pilot

Before collecting the full dataset:

- [ ] Record one scene with about 10 RGB-D frames.
- [ ] Verify RGB / depth registration.
- [ ] Verify camera resolution and aspect ratio.
- [ ] Verify depth format is uint16 millimetres.
- [ ] Confirm `0 = invalid depth`.
- [ ] Measure invalid-depth rate.
- [ ] Record camera mounting height.
- [ ] Record camera downward tilt.
- [ ] Look up ISOD's camera height and tilt (MOSTS paper) and compare with the robot camera.
- [ ] Crop to the ISOD aspect ratio (640:422) and resize to 640×422; store in the ISOD
      folder layout (`<root>/ISOD/data/<scene>/{rgb,depth,label}/`).

### Full recording

If the pilot is correct:

- [ ] Record 4–6 indoor scenes.
- [ ] Collect about 20–30 labelled frames per scene.
- [ ] Target approximately 120–150 frames total.
- [ ] Include 8–10 obstacle types.
- [ ] Include high- and low-contrast objects.
- [ ] Include several floor textures.
- [ ] Include distances around 0.5–1 m, 1–2 m and >2 m.
- [ ] Include approximately 10% empty-floor frames.
- [ ] Optionally include a dim-light condition.

### Annotation and evaluation

- [ ] Label drivable floor in the same style as ISOD.
- [ ] While labelling, note each obstacle's contrast with the floor (high / low); the
      contrast analysis needs it.
- [ ] Derive obstacle masks using the same connected-component procedure.
- [ ] Write `predict.py`: run the 9 frozen checkpoints on a new data folder.
- [ ] Point the existing scoring (`eval.py` / `analyze_objects.py`) at the robot data.
- [ ] Keep the 9 ISOD-trained models frozen.
- [ ] Use the same `depth_stats.json`.
- [ ] Do not fine-tune on robot data.
- [ ] Report RGB, Depth and RGB-D IoU / precision / recall.
- [ ] Report RGB-D − RGB paired by seed.
- [ ] Measure performance drop from ISOD to robot data.
- [ ] Analyse object recall by size, distance and visual contrast.
- [ ] Check whether Depth detects objects that RGB misses under domain shift.

The key question is whether RGB-D is more robust than RGB when the camera and environment
change.

## 4. SAM pseudo-labels — secondary question

This is no longer the immediate priority.

- [ ] Run SAM on the training split, restricted to the floor region or using prompts.
- [ ] **Label quality:** compare SAM masks with hand-derived obstacle masks.
- [ ] Check specifically whether SAM misses thin cables and other small objects.
- [ ] **Downstream:** train the lightweight model on SAM pseudo-labels and test on the
      hand-labelled test set.
- [ ] Optional data-size curve: 100 / 250 / 500 / 1,000 / 1,400 training images.

Only proceed if time remains after the robot-camera evaluation.

## 5. Per-class results

- [ ] Tag objects as cable / flat-soft / small-solid. Objects are not tracked across
      frames, so tag a sample (e.g. 30 test frames) rather than all 3,471 instances.
- [ ] Report recall and IoU per class.
- [ ] Compare whether depth behaves differently on flat vs raised objects.

This is useful, but lower priority than the robot-camera evaluation.

## 6. Possible controls and extensions

### Depth-context placebo

- [ ] Pair RGB with depth from another frame of the same scene (3 extra training runs).
- [ ] Compare with correctly aligned RGB-D.
- [ ] Compare with RGB-only.

If mismatched scene depth still helps, that would support the hypothesis that the depth
channel contributes scene / floor context rather than direct obstacle evidence.

### Alternative RGB-D fusion

Only investigate if the robot-camera experiment shows stronger complementarity between
RGB and Depth.

Possible approaches:

- [ ] separate RGB and Depth encoders
- [ ] mid-level feature fusion
- [ ] late fusion
- [ ] depth-gated RGB features

## 7. Write-up and reproducibility

- [x] Main RGB / Depth / RGB-D results table.
- [x] Per-site analysis.
- [x] Per-object size and distance analysis.
- [x] Object-overlap analysis.
- [x] Summary figures.
- [x] Store run configs and training histories.
- [x] Store result CSVs.
- [x] Store checkpoint SHA-256 hashes.
- [x] Final README cleanup.
- [x] Final ROADMAP cleanup.
- [ ] Final `git status` review.
- [ ] Commit scripts, results and figures.
- [ ] Push the `juan-rgbd` branch.
- [ ] Add robot-camera results when available.
- [ ] Final report discussion and limitations.

Known limitations to discuss:

- coarse cable labels in ISOD;
- objects touching wall regions can disappear from derived ground truth;
- only one public RGB-D dataset;
- early fusion is only one possible fusion architecture;
- ISOD train and test use the same camera;
- the scene-context explanation is currently a hypothesis;
- pixel metrics are dominated by large objects (hence the object-level analysis);
- only 3 test sites; the seed std measures training noise, not variation between rooms;
- ISOD depth has very few invalid pixels (< 1%), suggesting it was already hole-filled,
  which may smooth away thin objects.

---

## How to run anything

See [`README.md`](README.md).

Short version:

- `prep_gt.py` builds the obstacle ground truth;
- `to_yolo.py` exports the YOLO dataset;
- `train_seg.py` runs the original RGB semantic baselines;
- `train_modal.py` trains RGB, Depth or RGB-D U-Net models;
- `summarize.py` aggregates the 9 modality runs;
- `analyze_objects.py` performs per-site and per-object analysis;
- `make_figures.py` regenerates the main modality figures;
- `eval.py` scores prediction masks against the ground truth.