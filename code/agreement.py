"""Pairwise IoU between the learned methods on the test split: how much they agree.

Prints a markdown table and writes figures/agreement.png (the diagonal is left empty —
a method always agrees with itself).
"""
import numpy as np
from PIL import Image

import isod

METHODS = [
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


def plot(iou, names, out=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = out or f"{isod.ROOT}/figures/agreement.png"
    n = len(names)
    shown = np.array([[np.nan if a == b else iou[a, b] for b in range(n)] for a in range(n)])
    fig, ax = plt.subplots(figsize=(5.4, 4.2), dpi=200)
    fig.patch.set_facecolor("#fcfcfb")
    # sequential: one hue, light -> dark; range chosen around the observed values
    im = ax.imshow(shown, cmap="Blues", vmin=0.6, vmax=0.9)
    for a in range(n):
        for b in range(n):
            if a == b:
                ax.text(b, a, "—", ha="center", va="center", color="#8a8a8a", fontsize=13)
            else:
                v = iou[a, b]
                ax.text(b, a, f"{v:.3f}", ha="center", va="center", fontsize=11,
                        color="#ffffff" if v > 0.8 else "#1a1a1a")
    ax.set_xticks(range(n), names, fontsize=9, color="#5c5c5c", rotation=20, ha="right")
    ax.set_yticks(range(n), names, fontsize=9, color="#5c5c5c")
    ax.set_title("Agreement between methods\nIoU of one model's mask against another's",
                 fontsize=10, color="#1a1a1a", pad=10, loc="left")
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=8, colors="#5c5c5c", length=0)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(out, facecolor=fig.get_facecolor())
    print("wrote", out)


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
    for a in range(len(names)):          # mirror, so the plot has both halves
        for b in range(a + 1, len(names)):
            iou[b, a] = iou[a, b]
    plot(iou, names)
    print("| | " + " | ".join(names) + " |")
    print("|---" * (len(names) + 1) + "|")
    for a, na in enumerate(names):
        cells = []
        for b in range(len(names)):
            x, y = min(a, b), max(a, b)
            v = iou[x, y]
            cells.append("—" if a == b else f"{v:.3f}")
        print(f"| **{na}** | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
