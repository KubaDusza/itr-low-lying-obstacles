# Methodology: why the input-channel surgery is valid, and what still needs support

Written after the feedback on the bibliography report and after Juan's RGB / depth / RGB-D runs
(branch `juan-rgbd`). Companion to [`NOTES.md`](NOTES.md) (decisions) and
[`ROADMAP.md`](ROADMAP.md) (what is left).

The objection to answer: *we train depth-only and RGB-D models from encoders pretrained on
RGB (ImageNet), by replacing the first convolution. Is that a legitimate thing to do, or
could the weak depth result and the near-zero RGB-D gain be artefacts of that choice?*

---

## 1. The exact thing we do, and who else does it

Our encoder is MobileNetV2 pretrained on ImageNet. Only the **first** convolution sees a
different number of channels; every later layer is untouched. Three cases:

| Input | first conv | what we do now |
|---|---|---|
| RGB, 3 ch | pretrained, unchanged | — |
| Depth, 1 ch | 3 pretrained filters collapsed to 1 | `segmentation_models_pytorch` sums them |
| RGB-D, 4 ch | 3 pretrained filters + 1 new | RGB kept exactly, depth filter zero-initialised |

Published precedent for each:

- **FuseNet** (Hazirbas, Ma, Domokos, Cremers, ACCV 2016) — same surgery on ImageNet VGG-16:
  "*i) averaged the weights along the channel for a single-channel depth input; ii) stacked the
  weights with their average for a four-channel RGB-D input; iii) duplicated the weights for a
  six-channel RGB-HHA input.*" Our depth arm is case (i) up to a constant factor (sum instead
  of mean); our RGB-D arm is case (ii) with a cleaner initialisation (below).
- **TernausNetV2** (Iglovikov, Seferbekov, Buslaev, Shvets, CVPR 2018 Workshops, DeepGlobe
  winner) — extends an ImageNet encoder from 3 to 11 channels: "*we replace the first
  convolutional layer by a larger one ... We copy weights of the original pre-trained
  WideResnet38 to the first three channels and initialize the remaining channels by zeros.*"
  This is exactly our RGB-D initialisation, including the reason: the expanded network is
  functionally identical to the RGB network at step 0, so anything it gains is learned from the
  new channels.
- **Temporal Segment Networks** (Wang et al., ECCV 2016), "cross modality pre-training" — the
  general version of the trick, for a modality much further from RGB than depth is (optical
  flow): "*we average the weights across the RGB channels and replicate this average by the
  channel number of temporal network input*", and it "*works pretty well ... and reduce[s] the
  effect of over-fitting*".
- **ESANet** (Seichter, Köhler, Lewandowski, Wengefeld, Gross, ICRA 2021) — the RGB-D
  segmentation model for mobile robots: both the RGB and the depth encoder are ResNets, and
  Fig. 4 reports exactly our three-way comparison (single-encoder RGB, single-encoder depth,
  two-encoder RGB-D) on NYUv2: "*neither processing depth data nor RGB data alone reach the
  segmentation performance of our proposed RGB-D network*".
- **RedNet** (Jiang, Zheng, Luo, Zhang, 2018) — same family; with ESANet it is one of the two
  RGB-D baselines the MOSTS authors run **on ISOD itself** (see §3).

So the answer to the objection is: this is the standard protocol, not a shortcut. Replacing
the first convolution of an ImageNet encoder and keeping the rest is how the RGB-D segmentation
literature builds its own single-modality baselines, and the specific zero-init we use has a
named precedent.

### TernausNetV2 also gives the one thing we are missing

They do not train the expanded network in one go. They freeze the encoder for the first epoch
so the decoder adapts while only the RGB part is active, then unfreeze: "*the network learns
how to go from three to a larger number of input channels in a delicate, careful manner, slowly
increasing weights of the multi-spectral part of the input.*" We train all 15 epochs jointly
from a zero depth filter, so the depth path *could* have stayed near zero. Nothing in the
current results shows whether it did — see check A1.

## 2. The RGB-D null result is a known result, not a bug

Our number: RGB 0.772 ± 0.001 vs RGB-D 0.772 ± 0.003 IoU (early fusion, 4-channel).

FuseNet, Table 2, SUN RGB-D, same architecture across rows:

| Input | Global | Mean | IoU |
|---|---|---|---|
| Depth (1 ch) | 69.06 | 42.80 | 28.49 |
| HHA (3 ch) | 69.21 | 43.23 | 28.88 |
| RGB (3 ch) | 72.14 | 47.14 | **32.47** |
| RGB-D (4 ch, early fusion) | 71.39 | 49.00 | **31.95** |
| FuseNet-SF5 (mid-level fusion) | 76.27 | 48.30 | **37.29** |

Early fusion is *below* RGB on IoU there too; mid-level fusion gains +4.8 IoU over RGB. Two
things follow:

1. Our null is the expected outcome for the fusion scheme we chose. It is evidence about
   *early fusion*, not about *depth*.
2. The fix is not a better depth encoding. HHA — the hand-crafted 3-channel geocentric
   encoding (Gupta et al., ECCV 2014) — beats raw depth by 0.4 IoU in that table. Where depth
   gets used is what matters, not how it is packaged.

This also means our claim must be worded as "early fusion does not help on ISOD", and that
adding one genuine fusion model is what turns a null into a result.

## 3. What ISOD itself says (MOSTS paper, Chen et al. 2023)

Facts worth citing, all from the MOSTS paper that published ISOD:

- The data comes from an **Intel RealSense D455** (active stereo), with RGB, depth, point cloud
  and IMU. So "stereo depth" in our writing is accurate.
- Their geometric ground-plane method is explicitly limited: "*Despite the algorithm's
  inability to identify small objects (≤ 5 cm in height), it achieved a mIoU of 82.7%*" on the
  drivable-area task. That is the published justification for our premise, and for why our
  ground-plane baseline scores 0.031 on the obstacle task.
- They benchmark **RedNet and ESANet on ISOD as RGB-D models**: 71.0 and 80.7 mIoU
  out of the box, 85.1 and 83.0 after fine-tuning, at 11.44 and 21.71 FPS on an SBC;
  MOSTS itself reaches 85.7 at 93 FPS.
  Careful: those are mIoU on the **binary drivable-area** task at 256×256, not on our
  obstacle-pixel task. The numbers are not comparable to ours and must not be put in the same
  table without that caveat.

The useful part: a two-branch RGB-D model is already known to work on this dataset. If we add
one and it also fails to beat RGB on *obstacles*, that is a real finding about low-lying
obstacles. If we never add one, a reader can dismiss our conclusion in one sentence.

## 4. Options to make the design defensible

Ordered by cost. A and D are free.

### A. Report what we did, and check that the depth path is alive (no new training)

- **A1 — does the depth filter learn anything?** Load the 3 RGB-D checkpoints and print the
  norm of the first conv's depth slice (`conv.weight[:, 3:]`) against the RGB slice. Zero-init
  plus a 15-epoch OneCycle schedule could leave it near zero, in which case "RGB-D ≈ RGB" is
  trivially true and says nothing about depth. One minute of work, and it decides whether the
  rest of the RGB-D section means anything. If it is near zero, the TernausNetV2 schedule
  (freeze encoder for epoch 1, then unfreeze) is the citable fix. Juan has to run this one: the
  checkpoints live in his home on shire and are not readable from Jakub's account.
- **A2 — write the conv1 surgery into the report**, with the citations in §1, and state that
  the depth encoder is ImageNet-initialised rather than trained from scratch.
- **A3 — state the fusion scheme in the claim**: "early 4-channel fusion", everywhere.

### B. Two cheap controls that answer the objection empirically (≈1 h GPU total)

- **B1 — test-time depth ablation (no training at all).** Run the trained RGB-D models on the
  test set three times: real depth, constant depth, and depth shuffled between frames. If the
  small/far-object gain survives a constant channel, it was never depth. If it disappears, we
  have shown the gain comes from the depth input. This is the single most informative hour in
  the whole list.
- **B2 — depth-only from scratch, 3 seeds (~15 min).** `encoder_weights=None`, same
  everything else. If it lands at or below 0.390, then ImageNet initialisation is not what
  holds the depth arm back, and the RGB/depth gap is about the modality. This is the direct
  answer to "you can't put depth into an RGB-pretrained net".
- **B3 — optional: depth replicated to 3 channels**, unmodified pretrained conv1, 1 seed.
  Removes the surgery from the depth arm entirely. If it matches 0.390, the surgery is not the
  explanation either. (FuseNet's HHA row predicts it will match.)

### C. One real fusion model (the reviewer's request)

Either of these, not both:

- **C1 — mid-level fusion U-Net**, our own: two MobileNetV2 encoders (RGB + depth), add or
  concatenate the depth features into the RGB skip connections at each stage, same decoder,
  same loss, same splits, 3 seeds. About 40 lines on top of `train_modal.py`, same training
  cost as now. This is FuseNet's design at MobileNetV2 scale, so it is directly citable.
- **C2 — ESANet off the shelf** (official PyTorch code, ResNet-34). More faithful to the
  literature and to the reviewer's comment, more integration work, and its published ISOD
  numbers are on a different task.

Recommendation: C1. It keeps everything else fixed, which is the whole point of the comparison,
and it is the cheapest way to say something about *fusion* rather than about *one fusion*.

### D. Statistics over objects instead of over 3 seeds (done, no compute)

Juan's claim "the small- and far-object gain is positive in all three seeds" uses n = 3 as the
sample size. The sampling unit is the object, not the seed. Recomputed from
`results/isod/objects_hit_0p5.csv`:

McNemar on paired object detections, per seed (RGB-only wins / RGB-D-only wins, exact
two-sided p):

| Objects | n | seed 0 | seed 1 | seed 2 |
|---|---|---|---|---|
| small (< 144 px) | 1151 | 31/53, p = 0.021 | 38/70, p = 0.0027 | 29/57, p = 0.0034 |
| large (≥ 490 px) | 1158 | 10/7, p = 0.63 | 4/15, p = 0.019 | 11/8, p = 0.65 |
| far (≥ 2900 mm) | 1157 | 33/60, p = 0.0067 | 43/76, p = 0.0032 | 29/53, p = 0.011 |
| near (< 1442 mm) | 1157 | 12/2, p = 0.013 | 13/10, p = 0.68 | 15/11, p = 0.56 |
| all | 3471 | 60/82, p = 0.078 | 66/114, p = 0.00043 | 53/89, p = 0.0032 |

Frame-level cluster bootstrap (10,000 resamples of the 300 test frames) of the mean per-object
recall difference, averaged over seeds:

| Objects | RGB-D − RGB | 95% CI |
|---|---|---|
| small | +0.024 | [+0.013, +0.035] |
| far | +0.024 | [+0.013, +0.036] |
| large | +0.001 | [−0.003, +0.006] |
| near | −0.005 | [−0.010, −0.001] |
| all | +0.010 | [+0.005, +0.015] |

So the small- and far-object gain is real and the near-object loss is real, both small. This
is worth saying properly in the report: it is the only place where depth measurably changes
anything, and it now has a test behind it instead of three numbers with the same sign.

Caveat to state: the objects come from 3 test rooms, so the interval covers frame-to-frame
variation, not room-to-room variation. With 3 rooms we cannot estimate the latter.

## 5. The architectures we are comparing, briefly

| Model | Family | Why it is in the comparison | Our IoU (RGB) |
|---|---|---|---|
| **U-Net** (Ronneberger et al. 2015) | encoder–decoder, skip connections | skips carry full-resolution detail across, which is what thin objects need | **0.776** |
| **DeepLabV3+** (Chen et al. 2018) | dilated convs + ASPP + light decoder | wide context per pixel; output stride 16, so fine detail is recovered by upsampling rather than by skips | 0.757 |
| **YOLOv8n-seg** | instance segmentation | real-time on weak hardware, and the model class the related work uses; trained on polygons, which costs accuracy on thin shapes | 0.744 |
| **MobileNetV2** (Sandler et al. 2018) | encoder for both semantic models | inverted residuals + depthwise separable convs; the embedded-scale backbone, same reason MOSTS uses MobileNetV3 and ESANet uses ResNet-34 | — |

The ordering we measured is what the designs predict: skip connections > dilated context for
3 cm objects, and polygon supervision costs YOLO a little more. That is a usable sentence for
the report, and it is the kind of architecture-level reasoning the reviewer asked for on
SegNet and DeepLabV3+.

**Missing from the comparison, in priority order:**

1. **A two-branch RGB-D model** (§4 C). Without it, "depth does not help" is really "a
   4-channel first layer does not help". This is also the reviewer's explicit comment.
2. **Nothing else, probably.** We have three RGB architectures that land within 0.03 IoU of
   each other and that fail on the same pixels (pairwise agreement 0.861, higher than any of
   them against the ground truth). A fourth RGB architecture would add a row and no insight.
   The axis with information left in it is the **input**, not the architecture.
3. **SegNet and Fast-SCNN** are cited in the bibliography report as candidates but not run.
   Either justify them by what they would test (SegNet: pooling-index upsampling; Fast-SCNN:
   the speed end) or demote them to background in the final report. Running them is low value:
   SegNet's unpooling is a weaker version of U-Net's skips, and we already have FPS numbers
   from YOLOv8n and MobileNetV2.
4. **MOSTS** as a reference method on our obstacle task would be interesting (it is the
   dataset's own method) but needs their reference-texture protocol, so only if time appears.

## 6. The live-data experiment

The models stay frozen: ISOD-trained, no fine-tuning, same threshold, same `depth_stats.json`.
So we are not collecting *training* data, and the volume we need is set by evaluation power,
not by what a network needs. Coverage of conditions matters, frame count does not.

### What it has to answer

- **RQ1 — transfer.** Does the ISOD ranking (RGB ≫ depth, RGB-D ≈ RGB) survive a different
  camera and different rooms?
- **RQ2 — the condition where depth should win.** ISOD cannot answer this: it has no controlled
  variation of how visible an object is in RGB. When an object is nearly the colour and texture
  of the floor, does depth (and so RGB-D) keep finding it while RGB loses it?
- **RQ3 — secondary.** Are SAM masks on our own frames good enough to replace hand labels?

RQ2 is the one that justifies the whole trip to the lab, and it drives the design below.

### The design: paired scenes, not 150 random frames

For every object, record it **twice**: once on a floor where it contrasts strongly, once on a
floor (or under a light) where it nearly disappears, at the same distance and pose. The
question then becomes a within-pair comparison — "did depth hold up where RGB dropped?" — which
is far stronger than the same number of unpaired frames, and it is readable straight off a
2×2 table.

Minimum useful set:

- 2–3 floor surfaces (one plain light, one patterned/dark, ideally one reflective).
- 3 object groups, our own classes: **cable** (thin, long), **latex glove** (flat, soft),
  **screw / small solid**. 2–3 instances each, chosen so that each has a high-contrast and a
  low-contrast pairing available.
- 3 distance bands: 0.5–1 m, 1–2 m, > 2 m. Mark them on the floor with tape so they are
  repeatable.
- ≈ 2 frames per (object group × contrast × distance × floor) → 40–70 frames with 3–6 objects
  each → 200–400 object instances. That is enough for the McNemar test in §4 D.
- **10% empty-floor frames.** No labelling needed: any prediction is a false positive. Cheapest
  informative data we can record.
- Optional and cheap: repeat one floor under dim light.

### What to get right while recording (this is where the experiment is won or lost)

- **Mount the camera like ISOD's.** Same approximate height and downward tilt, fixed for the
  whole session, written down. Depth features are geometry, and geometry does not transfer if
  the camera moves to a different height. If a robot mount is not available, a tripod at the
  robot's camera height is fine and more repeatable.
- **Depth aligned to colour, uint16 millimetres, 0 = invalid.** On a RealSense, align depth to
  the colour stream in the SDK; save 16-bit PNG or `.npy`, never a visualised colormap. Save
  the intrinsics and the camera model.
- **Record the raw depth, and note any post-processing.** ISOD's depth has under 1% invalid
  pixels, which suggests it was hole-filled. If our camera gives 10% invalid, the depth arm will
  degrade for reasons that have nothing to do with the research question. Measure the invalid
  rate in the pilot; if it is high, the honest fix is a 5th validity-mask channel (and that
  becomes a finding, not a confound).
- **Fix the exposure and white balance** per floor, so "low contrast" means the object, not the
  auto-exposure.
- **A capture log, written at capture time**, one row per frame: floor id, object ids, contrast
  (high/low, by eye, decided *before* seeing any prediction), distance band, lighting, camera
  height, tilt. Reconstructing this afterwards from the images does not work.

### Labelling and scoring

- Label the obstacles **directly** (SAM click-prompt plus hand correction), not the floor. ISOD
  forced us to derive obstacles from floor labels; our own data has no such constraint, and
  direct labels remove the "objects touching walls get dropped" limitation.
- Keep the same ignore convention: walls, furniture and everything not floor-level is ignore,
  not negative.
- Write the frames into the ISOD folder layout (`<root>/ISOD/data/<scene>/{rgb,depth,label}/`)
  so `eval.py` and `analyze_objects.py` run unchanged. One `predict.py` that loads a checkpoint
  and writes masks for an arbitrary folder is the only new code needed.
- Report: IoU / precision / recall per input; the drop from ISOD to our data per input; object
  recall split by **contrast**, size and distance; the paired RGB-D − RGB difference with the
  §4 D statistics; and the false-positive count on the empty-floor frames.

### Pilot first, one afternoon

One scene, ~10 frames: check alignment, dtype, invalid rate, resolution and aspect against
ISOD, and run the frozen models on it to be sure the numbers are not nonsense before recording
the full set.

---

## Open items

- `refs.bib` entries for the papers cited above (TernausNetV2, FuseNet, TSN, ESANet, RedNet,
  MobileNetV3, U-Net, DeepLabV3+, MobileNetV2) still have to be pulled exactly from Google
  Scholar, per the report workflow. Not written by hand.
- The bibliography-report feedback items that need a decision are listed in the chat, not here.
