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
- [x] **Metrics**: IoU, precision, recall on the obstacle class, walls and furniture ignored. `code/eval.py`
- [x] **Depth baseline** (ISOD's own ground-plane method): IoU 0.031, recall 0.039.
- [x] **Three RGB models**: YOLOv8n-seg 0.744, U-Net 0.776, DeepLabV3+ 0.757 IoU.
- [x] **Agreement analysis**: the RGB models agree with each other (0.86) more than with
      the ground truth, so they fail on the same pixels.
- [x] **Epoch check**: 80 epochs is no better than 20; validation plateaus around epoch 12–15.
- [x] **Infrastructure**: GPU server (shire, `/opt/venv/jdusza_venv`), repo, figures, training curves.

## 1. Fair depth-only baseline

The current depth number comes from a classic geometric method, which is a weak opponent.
A learned model on the same data makes the three-way comparison fair.

- [ ] Feed depth into the same U-Net (1-channel input, or depth replicated to 3 channels).
- [ ] Normalise depth sensibly (metres, clipped; zero = no reading needs its own handling).
- [ ] Score with `eval.py`, add the row.

*Small change to `train_seg.py`; about 10 minutes of GPU. Expected: better than 0.031 but
still well behind RGB, which is the interesting part.*

## 2. RGB-D fusion — the main question

- [ ] **Early fusion:** U-Net with a 4-channel input (RGB + depth). Cheapest real fusion.
- [ ] **Late fusion:** take the RGB model's mask and accept or reject each object using depth.
- [ ] **ESANet** (two encoder branches, the published RGB-D model) if time allows.
- [ ] Compare all of it in one table: RGB / depth / RGB-D, same splits, same metrics.
- [ ] Analyse *where* depth helps: near vs. far, flat vs. raised objects, shiny floors.

*The headline result of the project. If fusion does not beat RGB alone, that is still a
result, and the per-condition analysis is what makes it publishable rather than a null.*

## 3. SAM pseudo-labels — the secondary question

- [ ] Run SAM (SAM 3 is already installed in `/opt/venv/sam3_venv`) on the training split,
      restricted to the floor region, or with text prompts.
- [ ] **Label quality:** SAM masks vs. hand masks on the same images. Does it miss cables?
- [ ] **Downstream:** retrain the best model on SAM labels, test on the hand-labelled test
      set, report the gap against hand labels.
- [ ] Optional: data-size curve (100 / 250 / 500 / 1,000 / 1,400 images) for both label
      sources. Nearly free — each run is minutes.

## 4. Per-class results

- [ ] Tag each ground-truth object as cable / flat-soft / small-solid. Roughly a click per
      object, or automatically with SAM 3 text prompts on each object crop.
- [ ] Report IoU and recall per class for every method.

*This is where the project's premise gets tested: the classes should fail differently, with
depth worst on flat objects and better on raised ones.*

## 5. Our own recordings (robot + depth camera)

- [ ] Record 50–100 scenes from the robot's viewpoint: cables, latex gloves, screws, plus
      warehouse-style items that no public dataset has (stretch wrap, flattened cardboard).
- [ ] Vary the conditions deliberately: distance, floor texture, lighting, contrast.
- [ ] Label them, using SAM with hand correction (this also uses the pipeline from step 3).
- [ ] **Cross-dataset test:** train on ISOD, test on our data, with no fine-tuning. Does the
      conclusion transfer to a different sensor and different rooms?

*The strongest part of the project for a report: it moves beyond re-running a public
benchmark. Worth starting the recording early, since it needs lab access, not GPU time.*

## 6. Write-up

- [ ] Results table and figures (several already exist in `figures/`).
- [ ] Per-condition and per-class analysis.
- [ ] Limitations: coarse cable labels in ISOD, objects touching walls dropped from the
      ground truth, a single depth sensor, one building.

---

## How to run anything

See [`README.md`](README.md). Short version: `prep_gt.py` builds the ground truth,
`to_yolo.py` exports the YOLO dataset, `train_*.py` trains on the server, `eval.py` scores
any method's masks against the ground truth.
