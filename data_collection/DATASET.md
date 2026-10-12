# GTE-Floor: our own RGB-D recordings of low-lying floor obstacles

64 recordings, 320 annotated frames, one indoor room, an Intel RealSense D435. Recorded
2026-10-12 to test whether models trained on [ISOD](https://www.kaggle.com/datasets/yuchen66/indoor-small-object-dataset)
transfer to a different camera, room and mounting height, and to create the one condition ISOD
lacks: an obstacle that is hard to see in RGB.

Frames live in `data/OWN/<run>/`, the per-run index is
[`dataset_index.csv`](dataset_index.csv), the capture protocol is [`PLAN.md`](PLAN.md).

## Layout

```
data/OWN/<run>/
  rgb/<i>.png      640x362 colour, 8-bit BGR
  depth/<i>.png    640x362 uint16, millimetres, 0 = no reading, aligned to the colour frame
  gt/<i>.png       640x362 uint8, 255 = obstacle
  floor.png        640x362 uint8, 255 = scored region (the fitted ground plane)
  i = 0..4
```

Same layout as ISOD. The frames are 640x362 rather than ISOD's 640x422: the full 848x480
sensor output scaled uniformly, with no crop, so nothing is cut off the sides. An earlier
version centre-cropped to ISOD's 1.517 aspect and lost **8.2% of the obstacle pixels**,
because the D435's colour lens (70 degrees) is already much narrower than its depth lens
(90 degrees). The uniform scale also lands closer to ISOD's angular resolution: 9.1 px per
degree against ISOD's 7.1, where the crop gave 10.3.

## The grid

Every run is `<height>_<background>_<set>_<distance>`. Four blocks, each shot from one fixed
pose, with an empty-scene run at the start.

| | bare floor | white poster |
|---|---|---|
| **camera 52 cm, tilt 30°** | 7 sets × 3 distances + empty | 3 sets × 3 distances + empty |
| **camera 25 cm, tilt 21°** | 7 sets × 3 distances + empty | 3 sets × 3 distances + empty |

**Distances** are tape marks on the floor at 0.75 / 1.5 / 2.5 m from the point below the lens,
which the camera reads as slant ranges of **0.87 / 1.56 / 2.60 m**. The marks were chosen so
that a 4 mm cable sits above the stereo noise at the near mark and below it at the far one
(`SNR = Δh / (h·k·d)`, `k ≈ 0.0045 m⁻¹`): predicted SNR 2.0 / 1.1 / 0.7 at 52 cm, double that
at 25 cm.

**Object sets** — each run contains one set, laid across the tape line about 20 cm apart:

| set | items | classes |
|---|---|---|
| `cables` | 4 cables: black thin, black thick, white thin, white thick, laid in open S-curves | C1 thin-linear |
| `tapes` | 5 tape rolls of different diameter | C4 raised, 20–50 mm |
| `hexkeys` | 5 hex keys of different size | C1/C3 thin metal, specular |
| `random` | screwdriver, Rubik's cube, pen, keycap remover | C3/C4 mixed |
| `screws` | 5 single screws of different sizes + 1 pile | C3 small-compact |
| `chess` | white queen, white pawn, black queen, black pawn | C4 raised, matched for colour and size |
| `rags` | white flat, white crumpled, pink folded flat, purple bundled | C2 flat-soft |
| `empty` | nothing | false-positive reference |

The `cables`, `chess` and `rags` sets each contain **both pale and dark items**, so the poster
runs carry a low- and a high-contrast version of the same object class inside the same frame:
same exposure, same distance, same forward pass. The white flat rag on the white poster is the
hardest case in the set — near-zero height and near-zero contrast.

**Camera pose**, measured rather than assumed:

| | tilt | bottom-centre of the depth frame | floor visible |
|---|---|---|---|
| 52 cm | 30° | 0.56–0.61 m | 0.67 – 3.3 m |
| 25 cm | 21° | 0.33 m | 0.37 m → horizon |

ISOD's own framing, measured over 40 frames from its 20 sites, is 0.61 m at the bottom-centre
and 0.89 m median — so the 52 cm block reproduces ISOD's viewpoint closely.

## How the frames were made

`code/db3_to_frames.py`. The viewer records ROS2 `.db3`; five frames are taken from the middle
of each clip, depth is aligned to the colour frame, and the result is centre-cropped to ISOD's
1.517 aspect and resized to 640×422.

Alignment is done in numpy because `pyrealsense2` has no macOS-arm64 wheel. It is the same
forward projection `rs2::align` performs — deproject, rigid transform, reproject, keep the
nearest hit — using this camera's own calibration (depth `fx=fy=423.55`, colour `fx=608.15`,
`fy=606.93`, baseline 15.06 mm), with a 2×2 splat because the colour stream has 1.44× the
focal length and a bare forward map would leave a grid of holes. Verified: depth
discontinuities land on 1.38× the mean colour gradient after alignment versus 1.14× before,
spurious depth edges fall from 876 to 283, and valid coverage rises from 98.2% to 99.4%.

The stereo shadow down the left edge remains (the first ~20 columns have no depth at all,
since depth comes from the left IR imager), which is most of the 6.8% invalid-depth rate
against ISOD's 1.3%. Cropping it away would also have cut real obstacles, so it is kept and
reported instead.

## How the ground truth was made

`code/own_gt.py`. No hand labelling.

Every run in a group was shot from the same pose at the same tape line with a different set of
objects, so the **per-pixel median across the group is the empty scene** — an obstacle occupies
a pixel in one set out of seven, never in the majority. Differencing each frame against that
median gives the obstacle mask. On the poster runs this is the only thing that works: the
poster moves with the objects, but it is present in every run of a distance group, so the
median puts it in the background where it belongs.

Three corrections:

- **Registration.** The tripod was nudged during the session, so every frame is aligned to its
  group anchor by phase correlation on the top third of the image — wall and furniture, never
  floor.
- **Gain matching.** Auto-exposure was on, so each frame is scaled to the background's median
  brightness over the floor before differencing.
- **Hysteresis.** A component is kept only if it contains a pixel more than 45 levels from the
  background, grown down to 18. Glossy floor reflections shift by a few levels over broad
  areas and never reach the high threshold; a real object has a hard core.

The scored region (`floor.png`) is the ground plane fitted to the group's median depth, with a
tolerance that grows as `z²` to match the stereo noise. Walls, furniture and anyone walking
past fall outside it, which is ISOD's ignore convention.

**Validation.** On the empty-scene runs the method finds **0 objects and 0 px** where
registration succeeded — that is the noise floor of the whole procedure. On `rags` it finds
exactly the 4 rags including the white one, on `chess` exactly the 4 pieces.

## Reliability

Registration fails on some runs of the 52 cm block, where people moving at the desk break the
phase correlation. Each run therefore carries a flag: the share of the scored floor marked as
obstacle, which is a few per cent for a sane scene. A run is **unreliable** when more than 12%
of the floor is flagged (2% for an empty run).

**51 of 64 runs pass.** The unreliable runs stay in the dataset — the RGB and depth frames are
fine, only the automatic mask is suspect — but they are excluded from every metric. The flag
is in `dataset_index.csv` and `data/OWN/gt_summary.json`.

## Known limitations

- **One room, one floor type** plus a poster. Site-to-site variation is not measurable here.
- Our D435 is not ISOD's D455: colour FoV 69°×42° against 90°×65°, so the same object covers
  about 1.45× more pixels at a given distance; stereo baseline 50 mm against 95 mm, so depth is
  roughly half as precise at range.
- Ground truth is derived, not hand-drawn. It inherits whatever the median background missed,
  and it cannot separate an object from its own shadow.
- ISOD's frames come from handheld walking video and carry motion blur; these are tripod shots,
  so they are *cleaner* than the training distribution.
- Objects are deliberately placed and spaced, so density and occlusion are more benign than a
  real warehouse floor.

## Regenerating

The `.db3` recordings (8.5 GB) are not in the repository. From them:

```bash
python code/db3_to_frames.py      # .db3 -> data/OWN/<run>/{rgb,depth}
python code/own_gt.py             # -> gt/, floor.png, gt_summary.json
```

Then, on the GPU server:

```bash
rsync -az data/OWN shire:/data/cs_courses/jdusza_itr/
CUDA_VISIBLE_DEVICES=1 python predict_own.py --root /data/cs_courses/jdusza_itr
rsync -az shire:/data/cs_courses/jdusza_itr/OWN_pred data/
python code/eval_own.py && python code/plot_own.py
```
