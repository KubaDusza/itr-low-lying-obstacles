"""Pairwise IoU between methods on the test split.

Off-diagonal: how much two methods agree with each other.
Diagonal: that method's IoU against the ground truth.
Prints a markdown table.
"""
import numpy as np
from PIL import Image

import isod

METHODS = [
    ("Ground plane (depth)", "depth"),
    ("YOLOv8n-seg", "yolo"),
    ("U-Net", "unet_mobilenet_v2"),
    ("DeepLabV3+", "deeplabv3plus_mobilenet_v2"),
]


def load(key, site, i, label, mask, ignore):
    if key == "depth":
        p = isod.depth_baseline(mask, label).astype(bool)
    else:
        p = np.array(Image.open(f"{isod.ROOT}/data/pred/{key}/{site}/{i}.png")) > 0
    return p & ~ignore


def main(n=100):
    keys = [k for _, k in METHODS]
    inter = np.zeros((len(keys), len(keys)))
    union = np.zeros_like(inter)
    for site in isod.load_splits()["test"]:
        for i in range(n):
            _, _, label, mask = isod.frame(site, i)
            g = np.array(Image.open(f"{isod.GT}/{site}/{i}.png")) > 0
            ignore = isod.ignore_region(label, g.astype(np.uint8)).astype(bool)
            preds = [load(k, site, i, label, mask, ignore) for k in keys]
            for a in range(len(keys)):
                inter[a, a] += (preds[a] & g).sum()
                union[a, a] += (preds[a] | g).sum()
                for b in range(a + 1, len(keys)):
                    inter[a, b] += (preds[a] & preds[b]).sum()
                    union[a, b] += (preds[a] | preds[b]).sum()
    iou = inter / np.maximum(union, 1)
    names = [n for n, _ in METHODS]
    print("| | " + " | ".join(names) + " |")
    print("|---" * (len(names) + 1) + "|")
    for a, na in enumerate(names):
        cells = []
        for b in range(len(names)):
            x, y = min(a, b), max(a, b)
            v = iou[x, y]
            cells.append(f"**{v:.3f}**" if a == b else f"{v:.3f}")
        print(f"| **{na}** | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
