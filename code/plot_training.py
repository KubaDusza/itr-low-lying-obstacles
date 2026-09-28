"""Training curves for every model that has a history file.

YOLO writes runs/yolov8n_seg/results.csv; train_seg.py writes runs/<name>_history.csv.
Loss and IoU live on separate axes (never a dual-axis chart).
"""
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import isod

COLORS = ["#3b6fd4", "#c2410c", "#2f9e6e"]  # validated colourblind-safe, fixed order
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#dcdcdc"


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [{k.strip(): v for k, v in r.items()} for r in rows]


def series(runs_dir):
    """[(label, epochs, train_loss, val_metric, metric_name)] for every run found."""
    out = []
    y = f"{runs_dir}/yolov8n_seg/results.csv"
    if os.path.exists(y):
        r = read_csv(y)
        out.append(("YOLOv8n-seg",
                    [int(x["epoch"]) for x in r],
                    [float(x["train/seg_loss"]) for x in r],
                    [float(x["metrics/mAP50(M)"]) for x in r],
                    "val mask mAP50"))
    for h in sorted(glob.glob(f"{runs_dir}/*_history.csv")):
        r = read_csv(h)
        label = os.path.basename(h).replace("_history.csv", "").replace("_", " ")
        out.append((label,
                    [int(x["epoch"]) for x in r],
                    [float(x["train_loss"]) for x in r],
                    [float(x["val_IoU"]) for x in r],
                    "val IoU"))
    return out


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, color=INK, fontsize=12, pad=10, loc="left")
    ax.set_xlabel(xlabel, color=MUTED, fontsize=10)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=10)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def main(runs_dir=None, out=None):
    runs_dir = runs_dir or f"{isod.ROOT}/runs"
    out = out or f"{isod.ROOT}/explore/training_curves.png"
    data = series(runs_dir)
    if not data:
        print("no history files in", runs_dir)
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=160)
    fig.patch.set_facecolor("#fcfcfb")
    # keep end labels from colliding when two runs finish at a similar score
    order = sorted(range(len(data)), key=lambda k: data[k][3][-1])
    offsets = {k: (rank - (len(order) - 1) / 2) * 11 for rank, k in enumerate(order)}
    for i, (label, ep, loss, metric, _) in enumerate(data):
        c = COLORS[i % len(COLORS)]
        ax1.plot(ep, loss, color=c, linewidth=2, label=label)
        ax2.plot(ep, metric, color=c, linewidth=2, label=label)
        ax2.annotate(f"{metric[-1]:.3f}", (ep[-1], metric[-1]), color=c, fontsize=9,
                     xytext=(5, offsets[i]), textcoords="offset points", va="center")
    style(ax1, "Training loss", "epoch", "loss")
    style(ax2, "Validation score", "epoch", "IoU / mAP50")
    ax2.set_ylim(0, 1)
    ax1.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, facecolor=fig.get_facecolor())
    print("wrote", out)
    for label, ep, loss, metric, name in data:
        print(f"  {label}: {len(ep)} epochs, final {name} = {metric[-1]:.4f}")


if __name__ == "__main__":
    main()
