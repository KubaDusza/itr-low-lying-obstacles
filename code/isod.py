"""Shared ISOD helpers: paths, splits, ground-truth construction, metrics."""
import json
import os

import cv2
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "ISOD", "data")
GT = os.path.join(ROOT, "data", "ISOD_gt")
SPLITS = os.path.join(ROOT, "data", "splits.json")

MIN_AREA = 20  # px; below this a blob is sensor noise
DILATE_FOR_FLOOR_AREA = 41  # px; scoring region = floor plus a margin around it


def sites():
    return sorted(d for d in os.listdir(DATA) if os.path.isdir(os.path.join(DATA, d)))


def frame(site, i):
    b = os.path.join(DATA, site)
    rgb = np.array(Image.open(f"{b}/rgb/{i}.png"))
    depth = np.array(Image.open(f"{b}/depth/{i}.png"))  # uint16, mm, 0 = no reading
    label = np.array(Image.open(f"{b}/label/{i}.png"))  # 255 = drivable floor
    mask = np.array(Image.open(f"{b}/mask/{i}.png"))  # depth ground-plane output, black = not floor
    return rgb, depth, label, mask


def obstacles(label):
    """Small objects lying on the floor, as a binary mask.

    The label marks drivable floor only, so walls and furniture are "not drivable"
    exactly like the objects are. The walls, the furniture standing against them and
    everything touching that region form one big connected component; dropping it
    leaves the blobs enclosed by floor, which are the objects on the floor.
    """
    non = (label == 0).astype(np.uint8)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(non, 8)
    if n <= 1:
        return np.zeros_like(non)
    background = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(non)
    for k in range(1, n):
        if k != background and stats[k, cv2.CC_STAT_AREA] >= MIN_AREA:
            out[cc == k] = 1
    return out


def instances(obst):
    """Per-object masks; one connected component is one object."""
    n, cc = cv2.connectedComponents(obst, 8)
    return [(cc == k) for k in range(1, n)]


def depth_baseline(mask, label):
    """Obstacle prediction of the dataset's depth ground-plane method.

    A pixel is predicted as obstacle when the ground-plane method blacked it out
    while it lies inside the floor area. The floor area comes from the label, so
    every method is scored over the same region.
    """
    blacked = (mask.sum(2) == 0).astype(np.uint8)
    floor_area = cv2.dilate(
        (label == 255).astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (DILATE_FOR_FLOOR_AREA, DILATE_FOR_FLOOR_AREA)),
    )
    return (blacked & floor_area).astype(np.uint8)


def ignore_region(label, gt_mask):
    """Walls and furniture: labelled not-drivable but not one of our floor objects.

    Predictions there are neither right nor wrong for this task, so they are dropped
    before scoring, the way "void" pixels are handled in segmentation benchmarks.
    """
    return ((label == 0) & (gt_mask == 0)).astype(np.uint8)


def score(pred, gt_mask, ignore=None):
    """Pixel IoU, precision and recall for the obstacle class."""
    if ignore is not None:
        pred = (pred & ~ignore.astype(bool)).astype(np.uint8)
    inter = int((pred & gt_mask).sum())
    union = int((pred | gt_mask).sum())
    return {
        "inter": inter,
        "union": union,
        "pred": int(pred.sum()),
        "gt": int(gt_mask.sum()),
    }


def totals(rows):
    t = {k: sum(r[k] for r in rows) for k in ("inter", "union", "pred", "gt")}
    return {
        "IoU": t["inter"] / max(1, t["union"]),
        "precision": t["inter"] / max(1, t["pred"]),
        "recall": t["inter"] / max(1, t["gt"]),
        "frames": len(rows),
    }


def load_splits():
    with open(SPLITS) as f:
        return json.load(f)
