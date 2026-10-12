# Own-data collection: one room, one afternoon

What to lay on the floor, how to record it, and why each run exists. The ISOD-trained models
stay frozen — this is *evaluation* data, so coverage of conditions matters and frame count does
not. Statistics and background: [`../METHODOLOGY.md`](../METHODOLOGY.md) §6.

## 1. What we are testing

Our ISOD result: **RGB 0.772, depth 0.390, RGB-D 0.772** IoU. Depth adds nothing. ISOD cannot
tell us why, because it never varies how visible an object is in RGB, and it is one camera in
one building. Five questions follow:

| # | Question | Comes from | Prediction |
|---|---|---|---|
| 1 | Does the ISOD ranking survive a new camera and room? | any run with objects | all drop; RGB stays ahead |
| 2 | **Does depth rescue what RGB loses in low contrast?** | the poster runs | the only place depth can win |
| 3 | Does the far/small RGB-D gain reproduce? | the distance columns | ISOD gave +0.024 recall on far objects, CI [+0.013, +0.035] |
| 4 | Do the classes fail differently? | object labels, no extra recording | depth worst on C1/C2, fine on C4 |
| 5 | Is mounting height the real lever? | the h = 0.52 vs 0.25 halves of the grid | depth SNR ∝ 1/h at fixed look-ahead |

## 2. Equipment

**Camera: Intel RealSense D435** (not the D455 that recorded ISOD). As of 2026-10-12 it
enumerates at **5 Gb/s (USB 3)** with the current cable, so the full 848×480 modes are
available. If a recording session ever reports 480 Mb/s, the cable is USB 2 — swap it, since
on USB 2 the SDK drops to reduced resolutions and may fall back to 4:3 modes that crop the
colour field of view further from ISOD's.

Three differences from ISOD's D455, all documentable rather than fatal:

| | D435 (ours) | D455 (ISOD) | Consequence |
|---|---|---|---|
| RGB FoV | 69° × 42° | 90° × 65° | we align depth to colour, so our frames are narrower; the same object covers ~1.45× more pixels at the same distance |
| Depth FoV | 87° × 58° | 87° × 58° | wider than our colour, which is the harmless direction: every colour pixel has a depth candidate |
| Stereo baseline | 50 mm | 95 mm | our depth is ~half as precise at a given range |
| IMU | none | yes | irrelevant, no model of ours uses it |

**Software.** `brew install librealsense` (prebuilt `arm64_tahoe` bottle) gives
`realsense-viewer` on the MacBook, which is all that is needed to record. The **conversion**
cannot run on macOS: depth must be aligned to colour and `pyrealsense2` has no macOS-arm64
wheel, so run [`../code/record_rs.py`](../code/record_rs.py) `--bag` on a GTE Linux PC or the
GPU server. `rs-convert` from the bottle does not align, so it is not a substitute.

The viewer needs **root on macOS** (`sudo realsense-viewer`) because Apple's UVC driver claims
the depth interface first. Recordings are therefore owned by root; `sudo chown` them
afterwards if that gets in the way. Version 2.59 writes **ROS2 `.db3`**, not legacy `.bag`.

**To bring:** tripod or box, masking tape, a white poster (≥1 × 1 m), a desk lamp, a ruler, a
printed A4 photo of a cable on a floor, and a phone to photograph the setup for the methods
figure.

## 3. Camera pose — fixed for everything

Measured from 40 ISOD frames across all 20 sites, and reproduced in the pilot:

| | ISOD | pilot |
|---|---|---|
| depth at bottom-centre of frame | 611 mm | **607 mm** |
| median depth | 894 mm | 1086 mm |
| 1st–99th percentile | ~520–3130 mm | 574–3585 mm |

**Measured pose, 2026-10-12: height 52 cm, tilt 30°**, bottom-centre of the depth image reads
0.61 m. The colour frame covers floor from 0.67 m to 3.3 m. The second half of the grid repeats
everything at height 25 cm, tilt 21°, bottom-centre 0.33 m.

The tilt is measured from the depth image, not with a protractor. Hover the viewer's cursor at
the **bottom-centre of the depth frame**, read the distance `d`, then

    tilt θ = asin(h / d) − 29°        (29° = half the depth stream's 58° vertical FoV)

The pilot gives `asin(0.50 / 0.607) − 29 = 26.5°`. At that pose the **colour** frame (42°
vertical) covers floor from 0.68 m to about 5.2 m, so all three distance marks fit.

Aim by the marks rather than by the number: the 0.7 m line should sit just above the bottom
edge and the 3.0 m line comfortably inside the top. If both tape lines are visible in the
colour image the angle is right. Measure θ afterwards with the formula and write it in the log.

Record **848×480 depth and 848×480 colour, 30 fps**, 16:9 (the 4:3 modes crop the colour FoV),
post-processing filters **off** — the invalid-depth rate is a measurement, not a defect to
hide. Lock exposure and white balance before the object runs so "low contrast" means the
object and not the camera compensating.

Recordings go to `data/raw/` (gitignored), one file per run, **named by run id**. Keep "Ask me
every time" selected in the viewer so the names match the capture log.

## 4. Object classes

Two properties drive the results: **height above the floor**, which decides whether depth can
resolve an object against its noise, and **footprint**, which decides whether RGB can.

| class | property | height | examples |
|---|---|---|---|
| **C1 thin-linear** | long, a few mm wide | 2–6 mm | cables, zip tie, wire, string |
| **C2 flat-soft** | large area, almost no height | 1–5 mm | glove, rag, paper, tape strip, flattened cardboard |
| **C3 small-compact** | a few cm², low | 3–10 mm | screws, nuts, coins, bottle caps, LEGO, USB stick |
| **C4 raised** *(positive control)* | compact but tall | 40–120 mm | tape roll, can, cup, small box |
| **C5 flush** *(negative control)* | exactly zero | 0 mm | printed photo of a cable, sheet of paper, tape stripe |

C1–C3 are the classes the project is about. C4 **must** be detected by depth — if it is not,
the depth pipeline is broken and nothing else is interpretable. C5 **cannot** be, by
construction, while RGB should still find it: the cleanest demonstration of each modality's
failure mode, and a good figure.

Specular and transparent items (foil, clear cup, CD) are a **per-object flag**, not a class —
making them a class would double the grid for two items.

Every run contains all five classes at once: same frame means same exposure, same lighting and
the same model forward pass, and scoring is per connected component so the classes separate at
labelling time. Record each physical object **once** in
[`objects.csv`](objects.csv): id, class, height in mm (use the ruler), footprint, bright/dark,
specular flag. Every analysis axis then follows from the object id.

### Contrast: matched pairs, not a dark background

Contrast is varied by the **object**, on a fixed white poster: the same type of object in two
brightnesses, side by side at the same tape mark.

| low contrast (on white) | high contrast (on white) |
|---|---|
| white cable | black cable |
| white paper strip | dark card or tape |
| pale glove, tissue | dark rag |
| light cap, chalk | dark cap, black screw |

Both conditions then sit in the **same image**, so the comparison is within-frame and every
nuisance variable cancels. It also avoids a confound that swapping backgrounds would introduce:
a bare floor is speckled and a poster is uniform, so a floor↔poster difference would mix
contrast with texture.

**The one rule:** pair members lie side by side on the same tape line. A white cable at 0.7 m
and a black cable at 1.0 m confounds contrast with distance.

## 5. The grid — 12 runs

Distances are tape marks on the floor at **0.75 / 1.5 / 2.5 m** from the point below the lens,
which the camera reads as slant ranges of **0.87 / 1.56 / 2.60 m** (`d = sqrt(x² + h²)`;
measured 2026-10-12). The marks land at rows 395 / 133 / 37 of 480. Everything below uses the
slant range, because that is what the sensor noise scales with and what the analysis reads out
of the depth map per object — the tape is for repeatability between runs, not for the numbers. An object of
height `Δh` on the floor, seen at distance `d` from a camera at height `h`, shifts the measured
range by `Δd = Δh·d/h` (the grazing view amplifies height into range), against stereo noise
`σ = k·d²` with `k = δ/(B·f) ≈ 0.0045 m⁻¹` for this D435. So

    SNR = Δh / (h · k · d)

| object | 0.87 m | 1.56 m | 2.60 m |
|---|---|---|---|
| cable, 4 mm | **2.0** | 1.1 | **0.7** |
| glove, 15 mm | 7.4 | 4.1 | 2.5 |
| tape roll, 45 mm | 22 | 12 | 7.4 |

The cable crosses the noise floor inside the ladder — above it at the near mark, below it at the
far one — so the near mark can produce a positive result and the far mark should produce a null.
The glove stays above 2.5 everywhere, so if depth misses a glove at 2.5 m the explanation is
not sensor noise. The tape roll must be found at all three or the pipeline is broken.

The near mark cannot go below the camera height — the nearest floor point is the one directly
underneath — and at this tilt the frame starts at 0.67 m, so 0.75 m is as close as objects can
sit without being clipped. At the far end the 2.5 m line is only ~37 px from the top edge, so
**lay objects across the line**, not extending away from the camera, and check nothing is
clipped before recording. Tilting up 2° (bottom-centre reading 0.61 → 0.63 m) buys ~35 px of
headroom if needed. Note `SNR ∝ 1/(h·d)`: halving the mount height doubles every number above, which is what the
lower half of the grid measures.

All five classes present, 20 cm apart, 30 cm clear of the walls, 5 frames per run.

The whole grid runs at **two camera heights**, 0.52 m and 0.25 m, with the same distance marks,
so only `h` changes between the halves:

| | 0.75 m | 1.5 m | 2.5 m |
|---|---|---|---|
| **bare floor**, h = 0.52 | run 1 | run 2 | run 3 |
| **white poster**, h = 0.52 | run 4 | run 5 | run 6 |
| **bare floor**, h = 0.25 | run 7 | run 8 | run 9 |
| **white poster**, h = 0.25 | run 10 | run 11 | run 12 |

The poster runs are the experiment: contrast varies inside every frame. The bare-floor runs
give realistic floor texture. The columns sweep apparent size and depth noise together; the
rows test the `1/h` scaling, which predicts every SNR above should double in the lower half.

One tilt per height, set so both the near and far tape lines are visible:

| | tilt θ | bottom-centre of depth reads | colour frame covers |
|---|---|---|---|
| h = 0.52 m | 30° | 0.61 m | 0.67 – 3.3 m |
| h = 0.25 m | 21° | 0.33 m | 0.37 m – horizon |

There is no tilt sweep: tilt only frames the scene, it does not change `sin α = h/d`. Steeper
angles were considered and rejected — at h = 0.52 m a 45° tilt shows floor from only 0.57 m to
1.28 m, losing two of the three marks, for about +40% SNR at the near mark alone. The low half
of the grid gets more than double the SNR *and* keeps the full ladder.

### The 5 frames of a run are 5 poses

Shift the tripod ±5–10 cm laterally and change the yaw slightly between frames, keeping the
height fixed — height is a factor now, so it must not vary inside a run. Five copies of one
frame carry one frame of information.

## 6. Probes — one run each, not crossed

| # | Run | Isolates |
|---|---|---|
| 13 | **dim light** (lamp only), poster, 1.5 m, h = 0.52 | RGB's failure mode; depth uses its own projector and should be unaffected |
| 14 | **handheld sweep**, 1–2 min walking at ≤1 m/s, camera drifting 0.2→1 m with tilt | motion blur and viewpoint spread — this is how ISOD itself was recorded, so it tests whether the fixed-mount result generalises to a moving robot |
| 15 | **empty floor**, h = 0.52 | false positives: any prediction is wrong. No labelling |
| 16 | **empty poster**, h = 0.52 | same |

There is deliberately **no camera-tilt probe**. For an object lying on the floor the ray angle
is fixed by geometry, `sin α = h/d`; tilting the camera only moves the object within the frame
and does not change the measurement. Height and distance are the only geometric variables,
and both are already in the grid.

16 runs, about 80 frames, roughly an hour.

### How far does the camera actually need to see?

Stopping distance is `v·t_react + v²/2a`: about 0.13 m for a TurtleBot at 0.25 m/s, 0.23 m at
0.5 m/s, 0.70 m at 1 m/s. The near mark at 0.75 m is already several times what the robot
needs, so a low camera that only sees 0.4–1 m is a valid robot configuration — and it is the
configuration where depth works best, which is worth stating in the report. The far mark is
not there for safety: it exists because ISOD's far bin is where the RGB-D gain appeared, and
because 2.5 m is where depth demonstrably dies (SNR 0.7 for a cable).

## 7. The capture log

[`capture_log.csv`](capture_log.csv), one row per run, filled in **at capture time**: run id,
background, distance, lighting, camera height, tilt. The contrast label lives per object in
`objects.csv`, assigned by eye **before** anyone sees a prediction. Reconstructing either
afterwards does not work.

## 8. After the lab

1. `python code/record_rs.py --bag data/raw/<run>.db3 --scene <run> --out data/OWN` on Linux —
   aligns depth to colour, keeps uint16 millimetres with 0 = invalid, centre-crops to ISOD's
   1.517 aspect and resizes to 640×422, into the ISOD folder layout so `eval.py` and
   `analyze_objects.py` run unchanged. The centre crop also removes the stereo shadow on the
   left edge, which took the pilot's invalid-depth rate from 6.78% to 2.35% against ISOD's 1.3%.
2. Label obstacles **directly** (SAM click prompts plus hand correction), not the floor. ISOD
   forced us to derive objects from floor labels; our own data has no such constraint, and
   direct labels remove the "objects touching a wall get dropped" limitation.
3. Keep the ignore convention: walls, furniture and anything not floor-level are ignore, not
   negative.
4. Report: IoU / precision / recall per input; the drop from ISOD per input; object recall by
   **contrast**, class, height and distance; paired RGB-D − RGB with the McNemar and bootstrap
   tests from `METHODOLOGY.md` §4 D; false positives on runs 11–12; and detection rate against
   **height ÷ depth-noise-at-that-distance**, which collapses both axes into one curve and
   predicts where depth works on any sensor, not just this one.

## 9. Limitations to write down now

- **One room**, one floor type plus a poster. Site-to-site variation is not measurable here.
- A different camera from ISOD's, with a narrower colour FoV and a shorter stereo baseline.
- ISOD frames came from walking video and carry motion blur; our tripod frames are cleaner, so
  the mismatch runs in the easy direction. Probe 10 is the only run that matches ISOD's capture
  style.
- Contrast is a hand-assigned binary label. Cheap upgrade if wanted: compute mean
  |object − surrounding floor| in Lab space from the labels afterwards and report it as a number.
- Objects are deliberately placed and spaced, so density and occlusion are more benign than a
  real warehouse floor.
