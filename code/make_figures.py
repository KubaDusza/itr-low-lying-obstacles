"""Report figures for the ISOD RGB / Depth / RGB-D comparison, from the small result CSVs only
(no dataset, no GPU needed).

  python make_figures.py [results_dir]      # default: <repo>/results/isod

Reads  summary.csv, site_metrics.csv, objects_hit_0p5.csv  (copied from <data root>/runs/)
Writes figures/object_recall.png   recall per input, by object size and by distance
       figures/rgbd_minus_rgb.png  paired RGB-D − RGB differences, one dot per seed
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/results/isod"
OUT = f"{ROOT}/figures"
SEEDS = [0, 1, 2]
# same colourblind-safe palette and ink colours as training_curves.py; one fixed colour per input
STYLE = {"rgb": ("RGB", "#3b6fd4", "o"), "depth": ("Depth", "#c2410c", "^"), "rgbd": ("RGB-D", "#2f9e6e", "s")}
INK, MUTED, GRID, BG = "#1a1a1a", "#5c5c5c", "#dcdcdc", "#fcfcfb"


def read(name):
    with open(f"{RES}/{name}") as f:
        return list(csv.DictReader(f))


def style(ax, title):
    ax.set_title(title, color=INK, fontsize=10, loc="left", pad=8)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)


def terciles(values, unit):
    """Same equal-count bins as analyze_objects.py."""
    v = np.asarray(values, float)
    e = np.nanpercentile(v, [100 / 3, 200 / 3])
    idx = np.where(np.isnan(v), -1, np.digitize(v, e))
    return idx, [f"< {e[0]:.0f} {unit}", f"{e[0]:.0f}–{e[1]:.0f} {unit}", f"≥ {e[1]:.0f} {unit}"]


def recall(objs, idx, b, m, s):
    sel = [o for o, j in zip(objs, idx) if j == b]
    return np.mean([int(o[f"{m}_s{s}"]) for o in sel])


def object_recall(objs):
    groups = [("By object size", *terciles([float(o["area_px"]) for o in objs], "px")),
              ("By distance to the camera", *terciles([float(o["median_depth_mm"]) for o in objs], "mm"))]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), dpi=200, sharey=True)
    fig.patch.set_facecolor(BG)
    for ax, (title, idx, labels) in zip(axes, groups):
        x = np.arange(3)
        for m, (name, c, mk) in STYLE.items():
            y = [np.mean([recall(objs, idx, b, m, s) for s in SEEDS]) for b in range(3)]
            ax.plot(x, y, color=c, linewidth=2, marker=mk, markersize=7,
                    markeredgecolor=BG, markeredgewidth=1.5, label=name)
        ax.set_xticks(x, labels)
        ax.set_xlim(-0.3, 2.3)
        ax.set_ylim(0, 1)
        style(ax, title)
    axes[0].set_ylabel("object recall (≥ 50% of pixels found)", color=MUTED, fontsize=9)
    axes[1].legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower left")
    fig.suptitle("RGB and RGB-D overlap everywhere; depth alone fails on small and distant objects",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/object_recall.png", facecolor=BG)
    print("wrote", f"{OUT}/object_recall.png")


def differences(summary, sites_rows, objs):
    """Rows of (label, [per-seed RGB-D − RGB in percentage points])."""
    def val(rows, m, s, key, **kw):
        return next(float(r[key]) for r in rows if r["modality"] == m and int(r["seed"]) == s
                    and all(r[k] == v for k, v in kw.items()))
    pix = [("all test sites", [100 * (val(summary, "rgbd", s, "IoU") - val(summary, "rgb", s, "IoU")) for s in SEEDS])]
    for site in sorted({r["site"] for r in sites_rows}):
        pix.append((site, [100 * (val(sites_rows, "rgbd", s, "IoU", site=site) -
                                  val(sites_rows, "rgb", s, "IoU", site=site)) for s in SEEDS]))
    obj = [("all objects", [100 * np.mean([int(o[f"rgbd_s{s}"]) - int(o[f"rgb_s{s}"]) for o in objs]) for s in SEEDS])]
    for key, unit, names in (("area_px", "px", ("small", "medium", "large")),
                             ("median_depth_mm", "mm", ("near", "middle", "far"))):
        idx, labels = terciles([float(o[key]) for o in objs], unit)
        for b in range(3):
            obj.append((f"{names[b]} ({labels[b]})",
                        [100 * (recall(objs, idx, b, "rgbd", s) - recall(objs, idx, b, "rgb", s)) for s in SEEDS]))
    return pix, obj


def rgbd_minus_rgb(summary, sites_rows, objs):
    pix, obj = differences(summary, sites_rows, objs)
    fig, axes = plt.subplots(2, 1, figsize=(7, 5.4), dpi=200, sharex=True,
                             gridspec_kw={"height_ratios": [len(pix), len(obj)]})
    fig.patch.set_facecolor(BG)
    c = STYLE["rgbd"][1]
    lim = max(abs(v) for _, vs in pix + obj for v in vs) * 1.25 + 0.2
    for ax, rows, title in ((axes[0], pix, "Pixel IoU"), (axes[1], obj, "Object recall (≥ 50% of pixels found)")):
        for i, (label, vs) in enumerate(rows):
            y = len(rows) - 1 - i
            ax.scatter(vs, [y] * len(vs), s=22, color=c, alpha=0.45, edgecolor="none", zorder=3)
            ax.scatter([np.mean(vs)], [y], s=64, color=c, edgecolor=BG, linewidth=1.5, zorder=4)
            ax.text(lim, y, f"{np.mean(vs):+.1f}", va="center", ha="right", fontsize=8, color=INK)
        ax.axvline(0, color=MUTED, linewidth=1)
        ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
        ax.set_xlim(-lim, lim)
        style(ax, title)
        ax.grid(True, axis="x", color=GRID, linewidth=0.8)
        ax.grid(False, axis="y")
    axes[1].set_xlabel("RGB-D − RGB (percentage points); small dots = seeds, large dot = mean",
                       color=MUTED, fontsize=8)
    fig.suptitle("Fusion leaves pixel IoU unchanged but finds slightly more small and far objects",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/rgbd_minus_rgb.png", facecolor=BG)
    print("wrote", f"{OUT}/rgbd_minus_rgb.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    objs = read("objects_hit_0p5.csv")
    object_recall(objs)
    rgbd_minus_rgb(read("summary.csv"), read("site_metrics.csv"), objs)
