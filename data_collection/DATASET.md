# GTE-Floor: RGB-D recordings of low-lying floor obstacles

64 runs, 320 annotated frames, one indoor lab room, an Intel RealSense D435 mounted on a
TurtleBot. Recorded 2026-10-12 to answer two questions that
[ISOD](https://www.kaggle.com/datasets/yuchen66/indoor-small-object-dataset) cannot:

1. **Does anything learned on ISOD transfer** to a different camera, room and mounting height?
2. **What happens when an obstacle is hard to see in RGB?** ISOD contains no controlled
   variation of obstacle visibility, so it cannot separate "depth is useless" from "RGB
   happened to be sufficient in every ISOD frame".

Capture protocol: [`PLAN.md`](PLAN.md) · run sheet: [`runs.xlsx`](runs.xlsx) · per-run index:
[`dataset_index.csv`](dataset_index.csv) · results: [`../README.md`](../README.md).

---

## 1. The rig

The camera is the robot's camera: a RealSense D435 on a small tripod bolted to a TurtleBot
(Kobuki base), so the viewpoint is the one a real robot would have rather than a convenient
one.

| camera at **52 cm** (the robot's own deck) | camera at **25 cm** (lower deck) |
|---|---|
| ![52 cm, side](../figures/dataset/52h_side_camera.jpg) | ![25 cm, side](../figures/dataset/25h_side_camera.jpg) |
| tilt 30°, bottom-centre of the depth frame reads 0.56–0.61 m | tilt 21°, bottom-centre reads 0.33 m |
| floor visible from 0.67 m to 3.3 m | floor visible from 0.37 m to the horizon |

Every run was shot at **both** heights with everything else identical, because the geometry
predicts height is the dominant variable. For an obstacle of height `Δh` at distance `d` seen
from height `h`, the grazing view turns height into a range difference of `Δh·d/h`, against
stereo noise `σ = k·d²`:

```
SNR = Δh / (h · k · d)          k ≈ 0.0045 m⁻¹ for this D435
```

So halving the mount height should double depth's signal-to-noise at every distance. The two
blocks exist to test that, and the same comparison turns out to matter for RGB as well.

### Distances

Three tape marks on the floor at 0.75 / 1.5 / 2.5 m from the point below the lens, which the
camera reads as slant ranges of **0.87 / 1.56 / 2.60 m**. They were chosen so a 4 mm cable sits
*above* the stereo noise at the near mark and *below* it at the far one — predicted SNR 2.0 /
1.1 / 0.7 at 52 cm, double that at 25 cm. A ladder where the answer is obvious at both ends
would measure nothing.

![The setup](../figures/dataset/process_chess_52h_poster_1_5m.jpg)

*Chess pieces on the white poster at the 1.5 m line, 52 cm block. The yellow measuring tape
runs along the floor with the three marks on it; the robot with the camera is bottom right.*

Cardboard tubes were stood at the marks so the distances would be visible in depth as well as
in colour — tape is invisible to a depth sensor. Two fell outside the colour camera's narrower
70° view, but the one at the 1.5 m mark is in frame in both blocks and reads **1.47 m** at
52 cm and **1.52 m** at 25 cm. A vertical tube spans from the floor's slant range at its base
to the horizontal distance at camera height, so those values sit exactly where they should,
and they confirm the distance marks and the depth alignment against an object you can point
at rather than against the fitted ground plane. The tubes play no part in the ground truth:
being static and present in every run of a group, the median background absorbs them like any
other fixture.

### Backgrounds

Two: the bare speckled lab floor, and a **white poster** laid over the same spot. The poster is
not there for variety — it is there to flip which objects are hard to see while leaving their
geometry untouched. A white rag on a grey floor is easy for RGB; the same rag on white paper is
nearly invisible, and depth doesn't care either way. That dissociation is the whole point.

---

## 2. The object sets, and why each one exists

Seven sets, one per run, laid across the tape line about 20 cm apart (closer and the scoring
merges them into one connected component). Two properties drive every result: **height above
the floor**, which decides whether depth can resolve an object against its noise, and
**contrast against the background**, which decides whether RGB can.

### 1. Cables — thin-linear (C1)

![cables](../figures/dataset/1_cables.jpg)

Four cables crossed by **colour × thickness**: black thin, black thick, white thin, white
thick. 2–6 mm tall, so depth is at the edge of its noise floor at the near mark and hopeless at
the far one. Laid in open S-curves rather than coils, so they stay genuinely thin — coiled
cables become compact blobs, which is exactly the flaw ISOD's own annotations have.

The thickness pairing separates "too thin to see" from "too low to measure": a thick white
cable and a thin white cable have the same contrast but different height and width.

![cables, predictions](../figures/dataset/pred_cables.jpg)

*Every run of this set: both camera heights, both backgrounds where recorded, all three
distances. Green = found, red = missed, orange = false positive. Where the derived mask did
not pass the reliability check the ground-truth column says so and the predictions are shown
in plain blue, since there is nothing trustworthy to compare them against.*

### 2. Tape rolls — raised, graded (C4)

![tapes](../figures/dataset/2_tapes.jpg)

Five rolls spanning roughly 20–50 mm in height. **This is the positive control**: depth *must*
find a 50 mm object at 0.87 m, where its predicted SNR is above 20. If it cannot, the depth
pipeline is broken and nothing else in the dataset is interpretable. The graded diameters also
give a height ladder inside a single object type, with colour and material held constant.

![tapes, predictions](../figures/dataset/pred_tapes.jpg)

### 3. Hex keys — thin, metallic, specular

![hexkeys](../figures/dataset/3_hexkeys.jpg)

Five sizes of the same shape. Thin enough that depth is blind, but dark and shiny against a
pale floor, so RGB has strong local contrast on a tiny area. They also test the specular case:
stereo matching tends to fail on shiny surfaces and return *no* depth there, which is a
different failure mode from measuring the floor correctly.

![hexkeys, predictions](../figures/dataset/pred_hexkeys.jpg)

### 4. Random clutter — mixed heights

![random](../figures/dataset/4_random.jpg)

Screwdriver, Rubik's cube, pen, keycap remover: everyday objects spanning about 8 mm to 57 mm
in a single frame. This is the realism check — one frame containing an object depth should
find, one it should miss, and two in between, with no curation by colour.

![random, predictions](../figures/dataset/pred_random.jpg)

### 5. Screws — the small limit, plus a pile

![screws](../figures/dataset/5_screws.jpg)

Five single screws of different sizes **and one pile**. The singles probe the smallest object
either modality can resolve; at 2.60 m a screw is a handful of pixels and below even the 20-px
minimum the ground truth enforces. The pile tests whether clustering rescues them: the same
material at the same height, but an order of magnitude more pixels.

![screws, predictions](../figures/dataset/pred_screws.jpg)

### 6. Chess pieces — colour × height, fully crossed

![chess](../figures/dataset/6_chess.jpg)

White queen, white pawn, black queen, black pawn: the cleanest dissociation in the dataset.
Queen and pawn differ in height by roughly a factor of two; white and black differ only in
albedo. **Depth should be indifferent to colour and sensitive to height; RGB should be the
opposite.** Put the same four pieces on the white poster and the white ones nearly vanish for
RGB while their geometry is unchanged — if depth's score holds and RGB's collapses, the two
modalities have been cleanly separated by a single variable.

![chess, predictions](../figures/dataset/pred_chess.jpg)

### 7. Rags — colour × flatness, and the hardest item in the set

![rags](../figures/dataset/7_rags.jpg)

Four cloths crossed by **colour × how they were laid**: white flat, white crumpled, pink folded
flat, purple bundled. Crumpling changes height (a few mm flat, 30–40 mm bundled) without
changing colour or material; colour changes contrast without changing height. The predictions
are explicit:

| | depth | RGB on bare floor | RGB on white poster |
|---|---|---|---|
| purple bundled | should find it (raised) | easy (coloured) | easy |
| pink flat | should miss it (flat) | easy (coloured) | easy |
| white crumpled | should find it (raised) | moderate | hard |
| **white flat** | **should miss it (flat)** | moderate | **neither modality has a cue** |

The white flat rag on the white poster is the single hardest object in the dataset: near-zero
height and near-zero contrast. It is the one case where a fusion model has nothing to fuse, and
all three RGB models do indeed miss it.

![rags, predictions](../figures/dataset/pred_rags.jpg)

### 8. Empty floor

One run per block with nothing on the ground. No labelling is needed and every predicted pixel
is wrong by construction, which makes it the only **false-positive** measurement in the
project — ISOD has no empty-floor frames at all. It turned out to separate the models more
sharply than any other condition.

---

## 3. The grid

| | bare floor | white poster |
|---|---|---|
| **camera 52 cm, tilt 30°** | 7 sets × 3 distances + empty | cables, chess, rags × 3 distances + empty |
| **camera 25 cm, tilt 21°** | 7 sets × 3 distances + empty | cables, chess, rags × 3 distances + empty |

The poster was shot for **cables, chess and rags** specifically, because those are the three
sets containing both pale and dark versions of the same object. Flipping the background flips
which half is hard to see while the geometry stays fixed — that is what makes the contrast
comparison a controlled one rather than a correlation.

**Five frames per run**, taken from the middle of a 3-second recording with the camera and
scene static. They do not add scenes; they measure how much a prediction moves under sensor
noise alone, and they stop one unlucky frame from driving a cell of the grid.

Run names are `<height>_<background>_<set>_<distance>`, e.g. `h25_poster_rags_150`.

---

## 4. Frames

```
data/OWN/<run>/
  rgb/<i>.png      640x362 colour, 8-bit BGR
  depth/<i>.png    640x362 uint16, millimetres, 0 = no reading, aligned to the colour frame
  gt/<i>.png       640x362 uint8, 255 = obstacle
  floor.png        640x362 uint8, 255 = scored region (the fitted ground plane)
  i = 0..4
```

Same layout as ISOD, so `eval.py` and `analyze_objects.py` run unchanged.

Frames are the full 848×480 sensor output scaled uniformly — **no crop**. An earlier version
centre-cropped to ISOD's 1.517 aspect and threw away **8.2% of the obstacle pixels**, because
the D435's colour lens (70°) is already much narrower than its depth lens (90°) and the
aligned frames live on the colour grid. The uniform scale also lands closer to ISOD's angular
resolution: 9.1 px per degree against ISOD's 7.1, where the crop gave 10.3.

**Depth alignment** (`code/db3_to_frames.py`) is done in numpy, because `pyrealsense2` has no
macOS-arm64 wheel. It is the same forward projection `rs2::align` performs — deproject, rigid
transform, reproject, keep the nearest hit — using this camera's own calibration (depth
`fx = fy = 423.55`, colour `fx = 608.15`, `fy = 606.93`, baseline 15.06 mm), with a 2×2 splat
because the colour stream has 1.44× the focal length and a bare forward map leaves a grid of
holes. Verified: after alignment, depth discontinuities land on 1.38× the mean colour gradient
versus 1.14× before, spurious depth edges fall from 876 to 283, and valid coverage rises from
98.2% to 99.4%.

The stereo shadow down the left edge remains — the first ~20 columns have no depth at all,
since depth comes from the left IR imager — and is most of the 6.8% invalid-depth rate against
ISOD's 1.3%. Cropping it away would also have cut real obstacles, so it is kept and reported.

---

## 5. Ground truth, without hand labelling

`code/own_gt.py`. Every run in a group was shot from the same pose at the same tape line with a
**different set of objects**, so the per-pixel median across the group is the empty scene: an
obstacle occupies a pixel in one set out of seven, never in the majority. Differencing each
frame against that median gives the obstacle mask.

This is also the only thing that works on the poster runs. The poster moves with the objects,
so a fixed empty reference is useless — but the poster is present in *every* run of a distance
group, so the median puts it in the background where it belongs.

Three corrections:

- **Registration.** The tripod was nudged during the session, so every frame is aligned to its
  group anchor by phase correlation on the top third of the image: wall and furniture, never
  floor.
- **Gain matching.** Auto-exposure was on, so each frame is scaled to the background's median
  brightness over the floor before differencing.
- **Hysteresis.** A component survives only if it contains a pixel more than 45 levels from the
  background, grown down to 18. Glossy floor reflections shift by a few levels across broad
  areas and never reach the high threshold; a real object has a hard core.

The scored region (`floor.png`) is the ground plane fitted to the group's median depth, with a
tolerance growing as `z²` to match the stereo noise. Walls, furniture and anyone walking past
fall outside it, which is ISOD's ignore convention.

**Validation.** On the empty-scene runs the method returns **0 objects and 0 px** wherever
registration succeeded — that is the noise floor of the whole procedure. On `rags` it finds
exactly the four rags including the white one; on `chess`, exactly the four pieces.

### Reliability

Registration fails on some runs of the 52 cm block, where people moving at the desk break the
phase correlation. Each run therefore carries a flag: the share of the scored floor marked as
obstacle, a few per cent for a sane scene. Reliability is measured where nothing should have changed: the mean difference between a
registered, gain-matched frame and its group background over the **static top band** — wall and
furniture, never floor. That does not punish a run for containing a physically large object,
which an earlier "share of floor flagged" criterion did. A run is **unreliable** above a
residual of 14 levels (the distribution runs 2.6 at the 10th percentile to 30 at the 95th), or
if it flags more than 35% of the floor, or 2% for an empty run.

**51 of 64 runs pass.** The unreliable ones stay in the dataset — the RGB and depth frames are
fine, only the automatic mask is suspect — but they are excluded from every metric. The flag is
in `dataset_index.csv` and `data/OWN/gt_summary.json`.

---

## 6. What the models do on it

The three ISOD-trained RGB models, frozen: no fine-tuning, same threshold.

| Method | IoU | Precision | Recall | ISOD IoU |
|---|---|---|---|---|
| **YOLOv8n-seg** | **0.267** | 0.535 | 0.347 | 0.744 |
| DeepLabV3+ (MobileNetV2) | 0.208 | 0.344 | 0.345 | 0.757 |
| U-Net (MobileNetV2) | 0.176 | 0.250 | 0.374 | 0.776 |

![results](../figures/own_results.png)

Three findings, each tied to a configuration above:

- **Transfer is poor and the ranking inverts.** Every model loses two thirds to four fifths of
  its ISOD IoU, and YOLOv8n-seg — the weakest of the three on ISOD — is the strongest here.
- **Mounting height beats architecture.** On bare floor at 52 cm, YOLO scores
  0.487 / 0.180 / 0.042 across the three distances; at 25 cm, 0.715 / 0.464 / 0.419. Overall
  IoU rises by half or more for every model. That is why both heights were shot, and the depth geometry predicted the direction
  before any data existed.
- **The semantic models hallucinate on bare floor.** On the empty runs U-Net predicts 8,999
  obstacle px per frame and DeepLabV3+ 4,657, while YOLO predicts **zero**. That is invisible
  on ISOD, which has no empty-floor frames.

Depth and RGB-D are not in this table yet: those checkpoints live in a collaborator's home
directory on the GPU server and are not readable from this account. The aligned depth is in the
dataset and the training script is on the `juan-rgbd` branch, so it is a retrain away.

---

## 7. Limitations

- **One room, one floor type** plus a poster. Site-to-site variation is not measurable here.
- Our D435 is not ISOD's D455: colour FoV 70°×43° against 90°×65°, so the same object covers
  about 1.45× more pixels at a given distance; stereo baseline 50 mm against 95 mm, so depth is
  roughly half as precise at range.
- Ground truth is derived, not drawn. It inherits whatever the median background missed and it
  cannot separate an object from its own shadow.
- ISOD's frames come from handheld walking video and carry motion blur; these are static shots,
  so they are *cleaner* than the training distribution.
- Objects are deliberately placed and spaced, so density and occlusion are far more benign than
  a real warehouse floor.

---

## 8. Regenerating

The `.db3` recordings (8.5 GB) are not in the repository. From them:

```bash
python code/db3_to_frames.py      # .db3 -> data/OWN/<run>/{rgb,depth}
python code/own_gt.py             # -> gt/, floor.png, gt_summary.json
```

Then on the GPU server:

```bash
rsync -az data/OWN shire:/data/cs_courses/jdusza_itr/
CUDA_VISIBLE_DEVICES=1 python predict_own.py --root /data/cs_courses/jdusza_itr
rsync -az shire:/data/cs_courses/jdusza_itr/OWN_pred data/
python code/eval_own.py && python code/plot_own.py
```
