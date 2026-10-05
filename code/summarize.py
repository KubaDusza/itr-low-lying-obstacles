"""Evaluate the 9 runs with eval.py and summarise them.

Writes <root>/runs/summary.csv          (one row per run)
       <root>/runs/summary_mean_std.csv (one row per modality: mean and sample std over seeds)

  /opt/venv/jdusza_venv/bin/python summarize.py [root]     # default root: ~/datasets/isod_work
"""
import csv
import glob
import os
import sys

import numpy as np

import eval as ev
import isod

ROOT = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/datasets/isod_work")
MODALITIES = ["rgb", "depth", "rgbd"]
SEEDS = [0, 1, 2]
METRICS = ["IoU", "precision", "recall"]
N_TEST = 100 * len(isod.load_splits()["test"])

rows = []
for m in MODALITIES:
    for seed in SEEDS:
        d = f"{ROOT}/pred/unet_mobilenet_v2_{m}_s{seed}"
        n = len(glob.glob(f"{d}/*/*.png"))
        if n != N_TEST:
            print(f"SKIPPED {m} seed {seed}: {n}/{N_TEST} masks in {d}")
            continue
        r = ev.run("masks", "test", d)  # exactly what `python eval.py masks <d>` computes
        rows.append({"modality": m, "seed": seed, **{k: r[k] for k in METRICS}})
        print(f"{m:6s} seed {seed}: " + "  ".join(f"{k}={r[k]:.4f}" for k in METRICS))

with open(f"{ROOT}/runs/summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["modality", "seed", *METRICS])
    w.writeheader()
    w.writerows(rows)

print("\n| Input | n | IoU | Precision | Recall |\n|---|---|---|---|---|")
summary = []
for m in MODALITIES:
    vals = {k: np.array([r[k] for r in rows if r["modality"] == m]) for k in METRICS}
    n = len(vals["IoU"])
    if n == 0:
        continue
    # sample std (ddof=1); with a single seed there is no spread to report
    stats = {k: (v.mean(), v.std(ddof=1) if n > 1 else float("nan")) for k, v in vals.items()}
    summary.append({"modality": m, "n_seeds": n,
                    **{f"{k}_{s}": x for k, (mu, sd) in stats.items() for s, x in (("mean", mu), ("std", sd))}})
    print(f"| {m} | {n} | " + " | ".join(f"{mu:.3f} ± {sd:.3f}" for mu, sd in stats.values()) + " |")

with open(f"{ROOT}/runs/summary_mean_std.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0]))
    w.writeheader()
    w.writerows(summary)
print(f"\nwrote {ROOT}/runs/summary.csv and summary_mean_std.csv")
