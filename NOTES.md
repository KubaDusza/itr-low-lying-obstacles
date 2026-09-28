# ITR project notes

*Created: 2026-09-21 · Updated: 2026-09-22*

Record of the decisions made so far and the reasoning behind them. Report draft: `bibliography_report/REPORT.md`. GPU server access: `../GTE_SERVERS.md`.

## 1. Problem and objectives
- **Task:** detect low-lying obstacles on the floor for indoor and warehouse robots.
- **Why it's hard:** these objects are low, so depth sensors and ground-plane methods miss them. MOSTS reports that its geometric method only detects objects about 5 cm or taller. They also cover few pixels.
- **Main objective:** does adding stereo depth to RGB improve detection of these obstacles compared with RGB alone or depth alone?
- **Secondary objective:** can labels generated automatically by a foundation model (SAM) replace hand labels for training a lightweight model like YOLO?
  - Motivation: SAM is too slow to run on a robot. Hand-labeling a dataset this specific is expensive. If a SAM-labeled → YOLO pipeline performs close to one trained on hand labels, it's a very good tradeoff.
- **How to frame SAM → YOLO:** it's a way to *get labels*, not a competing detection method.
  - SAM masks are drawn on the RGB image. With aligned RGB-D data, the same masks can train any learned model: RGB (YOLO), RGB-D (ESANet), or depth-only.
  - It doesn't apply to RANSAC, which isn't learned and needs no labels.

## 2. Object classes (decided 2026-09-21)
Three classes, each with one representative object. Each fails differently for the sensors.

| Class | Representative | Other members | Why it's hard |
|---|---|---|---|
| Thin, long | **cable** | hoses, strapping | Depth barely registers it; few pixels, often low contrast. |
| Flat and soft | **latex glove** | rags, tape, cellophane, stretch-wrap | Almost zero height, so depth can't see it and only RGB can. Its shape varies. |
| Small and solid | **screw** | bolts, small parts, LEGO, debris | A few cm tall, so depth partly sees it. This is the middle case for measuring what RGB-D adds. |

- The classes range from "depth partly works" to "depth fails completely", which is what makes the RGB / depth / RGB-D comparison meaningful.
- Cables only are too narrow. The first draft overfitted on cables.
- Warehouse-only items (lowered forklift forks, flattened cardboard) aren't in any public dataset, so we'd have to record them ourselves.

## 3. Dataset search (2026-09-22)
We want one dataset to evaluate every method on, ideally with: (a) our object types, (b) **hand-labeled** masks, both as reliable ground truth and to grade SAM, and (c) **depth**. No dataset has all three.

| Dataset | Depth | Hand masks | Our objects | Problem |
|---|---|---|---|---|
| **ISOD** (Chen et al. 2023, MOSTS) | ✅ RGB-D + IMU, 2,000 images, 20 sites | ✅ | cables, gloves, rags, LEGO; objects ≤3 cm | Labels mark only drivable vs. not drivable, with no classes. CC BY-NC-SA. 1.8 GB on Kaggle. |
| ORG (reflective-ground obstacles) | ❌ | ✅ obstacle masks + per-object masks | 42 kinds of low obstacles | RGB only |
| GMRPD | ✅ D415, 3,896 images | ✅ | obstacles ≥5 cm | Mostly outdoor; objects too tall |
| Hazards&Robots | ❌ | ❌ one label per frame | Cable, Tape, Screws | No masks |
| OCID | ✅ | labels come from depth differences | household objects on floors and tables | Labels are biased toward depth-based methods |
| Hua et al. 2019 | ✅ | ✅ | objects 5–50 cm | Doesn't appear to be public |
| Airport-runway debris (FOD) | ❌ | bounding boxes | screws, bolts, wires | Runways, no depth |
| Synthetic (Isaac Sim / BlenderProc) | ✅ exact | ✅ exact | anything | Gap between simulated and real images |

**Choice: ISOD as the main dataset.**
- Its hand masks give the ground truth for the main comparison and for grading SAM.
- To get classes: each connected blob of "not drivable" pixels on the floor is one object. Tagging each blob as cable, glove or other takes about a click per object, with no mask drawing.
- Screws are missing. Use LEGO as the small solid class, or record a small set of our own.
- **To do:** download ISOD and check how the masks look, especially thin cables.

## 4. Models and how much data each needs
- Split ISOD **by site** (e.g. 14/3/3 sites ≈ 1,400 / 300 / 300 images) so test scenes are never seen in training. That's enough to fine-tune everything below.
- ResNet is a backbone, not a model on its own.

| Input | Model | Role | Labeled data needed |
|---|---|---|---|
| RGB | **YOLOv8n-seg / YOLO11n-seg** (pretrained on COCO) | main lightweight model; the "student" | about 100–300 images per class to work; Ultralytics recommends about 1,500 per class for best results |
| RGB | DeepLabV3+ (ResNet / MobileNetV3 backbone) | standard segmentation baseline (optional) | about 500–1,000 |
| RGB | **SAM / SAM 3** | zero-shot baseline and teacher | 0 |
| RGB | MOSTS | earlier result on ISOD (85.7% mIoU) | 0 target labels; needs a reference floor photo |
| Depth | **RANSAC ground plane** | geometry baseline | 0; a few images to tune the threshold |
| RGB-D | **ESANet** (two ResNet-34 branches, real-time) | standard efficient RGB-D segmentation model | about 500–1,000 |
| RGB-D | **YOLO mask + depth check** | simplest way to combine RGB and depth | 0 beyond YOLO |
| RGB-D | RedNet / DFormer | optional older baseline / newer, heavier model | about 1,000 |

**Minimum set:** YOLO-seg, RANSAC, ESANet, YOLO + depth check, SAM. Only YOLO and ESANet need training.

## 5. Evaluation plan
1. **Main comparison:** every method predicts on the hand-labeled test set. YOLO's per-object masks are merged into one obstacle-vs-floor mask to match ISOD's labels.
2. **Label quality (grades SAM):** SAM masks vs. hand masks on the training images. Does SAM miss thin cables?
3. **SAM labels vs. hand labels (secondary objective):** train the same YOLO on hand masks and on SAM masks, test both on the hand-labeled test set. Repeat at 100 / 250 / 500 / 1,000 / 1,400 training images. This shows what SAM labels cost and how much data is enough.

**Metrics:**
- **IoU** of the obstacle class (comparable to MOSTS)
- **Pixel recall** (missing an obstacle is worse than a mask that's too big)
- **Object detection rate** per class (cable / glove / small solid). Tiny objects barely move IoU, so this is the number that shows missed cables.
- **FPS** on the GPU server

## 6. Open items
- Download ISOD and check the labels (needs a Kaggle login or API token).
- Bibliography report: is 2 pages, limit is 1. State of the Art still thin.
- Decide whether to record our own data (screws, warehouse items) with a stereo camera.
