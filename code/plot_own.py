"""Figures for the own-recordings evaluation: metric panels and qualitative examples.

  python plot_own.py        # writes figures/own_results.png and figures/own_examples.jpg
"""
import csv
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, OWN, PRED = f"{ROOT}/results/own", f"{ROOT}/data/OWN", f"{ROOT}/data/OWN_pred"
FIG = f"{ROOT}/figures"
COLORS = {"YOLOv8n-seg": "#3b6fd4", "U-Net": "#c2410c", "DeepLabV3+": "#2f9e6e"}
INK, MUTED, GRID, BG = "#1a1a1a", "#5c5c5c", "#dcdcdc", "#fcfcfb"
MODELS = list(COLORS)
DIST = {"075": "0.87 m", "150": "1.56 m", "250": "2.60 m"}


def load():
    with open(f"{RES}/per_run.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("inter", "union", "pred", "gt", "frames"):
            r[k] = int(r[k])
        r["reliable"] = r["reliable"] == "True"
        r["is_empty"] = r["is_empty"] == "True"
    return rows


def iou(sel):
    if not sel:
        return np.nan
    return sum(r["inter"] for r in sel) / max(1, sum(r["union"] for r in sel))


def style(ax, title):
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.set_ylim(0, 0.55)


def panels(rows):
    good = [r for r in rows if r["reliable"] and not r["is_empty"]]
    groups = [
        ("By distance to the obstacle", lambda r: DIST[r["distance"]], [DIST[k] for k in ("075", "150", "250")]),
        ("By camera height", lambda r: "52 cm" if r["block"].startswith("h52") else "25 cm", ["52 cm", "25 cm"]),
        ("By background", lambda r: "bare floor" if "floor" in r["block"] else "white poster",
         ["bare floor", "white poster"]),
        ("By object set", lambda r: r["object_set"],
         ["tapes", "cables", "hexkeys", "screws", "rags", "random", "chess"]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), dpi=200,
                             gridspec_kw={"width_ratios": [1, 0.8, 0.8, 2]})
    fig.patch.set_facecolor(BG)
    for ax, (title, keyfn, order) in zip(axes, groups):
        x = np.arange(len(order))
        w = 0.26
        for j, m in enumerate(MODELS):
            vals = [iou([r for r in good if r["model"] == m and keyfn(r) == k]) for k in order]
            ax.bar(x + (j - 1) * w, vals, w, color=COLORS[m], label=m if ax is axes[0] else None)
        ax.set_xticks(x, order, fontsize=9)
        style(ax, title)
        if ax is axes[0]:
            ax.set_ylabel("IoU", color=MUTED, fontsize=10)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right")
    fig.suptitle("ISOD-trained models on our own recordings — IoU on the obstacle class",
                 color=INK, fontsize=13, x=0.007, ha="left", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{FIG}/own_results.png", facecolor=BG, bbox_inches="tight")
    print("wrote figures/own_results.png")


def examples(runs=(("h52_floor_tapes_075", 2), ("h25_floor_cables_075", 2),
                   ("h25_poster_rags_075", 2), ("h25_floor_empty", 2))):
    keys = {"YOLOv8n-seg": "yolo", "U-Net": "unet_mobilenet_v2", "DeepLabV3+": "deeplabv3plus_mobilenet_v2"}
    cols = ["image", "ground truth"] + MODELS
    tiles = []
    for run, i in runs:
        img = cv2.imread(f"{OWN}/{run}/rgb/{i}.png")
        gt = cv2.imread(f"{OWN}/{run}/gt/{i}.png", 0) > 0
        floor = cv2.imread(f"{OWN}/{run}/floor.png", 0) > 0
        row = [img.copy()]
        g = img.copy(); g[gt] = (0, 0, 255); row.append(g)
        for m in MODELS:
            p = cv2.imread(f"{PRED}/{keys[m]}/{run}/{i}.png", 0) > 0
            p &= floor
            v = img.copy()
            v[p & gt] = (80, 200, 80)      # hit
            v[gt & ~p] = (0, 0, 255)       # miss
            v[p & ~gt] = (0, 140, 255)     # false positive
            row.append(v)
        band = np.concatenate(row, 1)
        cv2.putText(band, run, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
        tiles.append(band)
    out = np.concatenate(tiles, 0)
    hdr = np.full((46, out.shape[1], 3), 252, np.uint8)
    for j, c in enumerate(cols):
        cv2.putText(hdr, c, (j * 640 + 12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (26, 26, 26), 2)
    cv2.imwrite(f"{FIG}/own_examples.jpg", np.concatenate([hdr, out], 0), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("wrote figures/own_examples.jpg  (green = hit, red = missed, orange = false positive)")


if __name__ == "__main__":
    rows = load()
    panels(rows)
    examples()
