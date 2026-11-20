"""Score every prediction directory against the ISOD ground truth and build the tables.

  ITR_DATA=/data/cs_courses/jdusza_itr python summarize.py --root /data/cs_courses/jdusza_itr

Scores identically to eval.py (same ignore region, same pooled pixel metrics), but over all
arms at once, and it keeps the per-frame counts so the paired statistics in objects.py and
the bootstrap do not have to recompute them.

Writes, into results/isod/:
  frames.csv            one row per (arm, site, frame): the four pixel counts
  summary.csv           pooled IoU / precision / recall per run
  summary_mean_std.csv  per arm, mean +- sample sd over seeds
and prints the tables as markdown.
"""
import argparse
import csv
import os
import re
from collections import defaultdict

import numpy as np
from PIL import Image

import isod

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results", "isod")


def arm_of(run):
    """Strip the seed suffix: rgbd_blind_s2 -> rgbd_blind."""
    return re.sub(r"_s\d+$", "", run)


def score_dir(pred_dir, split="test", n=100):
    rows = []
    for site in isod.load_splits()[split]:
        for i in range(n):
            p = f"{pred_dir}/{site}/{i}.png"
            if not os.path.exists(p):
                return None
            g = (np.array(Image.open(f"{isod.GT}/{site}/{i}.png")) > 0).astype(np.uint8)
            _, _, label, _ = isod.frame(site, i)
            pred = (np.array(Image.open(p)) > 0).astype(np.uint8)
            r = isod.score(pred, g, ignore=isod.ignore_region(label, g))
            rows.append(dict(site=site, frame=i, **r))
    return rows


def pooled(rows):
    t = {k: sum(r[k] for r in rows) for k in ("inter", "union", "pred", "gt")}
    return dict(IoU=t["inter"] / max(1, t["union"]),
                precision=t["inter"] / max(1, t["pred"]),
                recall=t["inter"] / max(1, t["gt"]))


def baseline_rows(split="test", n=100):
    rows = []
    for site in isod.load_splits()[split]:
        for i in range(n):
            g = (np.array(Image.open(f"{isod.GT}/{site}/{i}.png")) > 0).astype(np.uint8)
            _, _, label, mask = isod.frame(site, i)
            pred = isod.depth_baseline(mask, label)
            rows.append(dict(site=site, frame=i,
                             **isod.score(pred, g, ignore=isod.ignore_region(label, g))))
    return rows


def md(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("ITR_DATA", "/data/cs_courses/jdusza_itr"))
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)

    dirs = {"ground_plane_baseline": None}
    for d in sorted(os.listdir(f"{a.root}/pred/modal")):
        if os.path.isdir(f"{a.root}/pred/modal/{d}"):
            dirs[d] = f"{a.root}/pred/modal/{d}"
    for d in ("yolo", "unet_mobilenet_v2", "deeplabv3plus_mobilenet_v2"):
        if os.path.isdir(f"{a.root}/pred/{d}"):
            dirs[d] = f"{a.root}/pred/{d}"
    # ESANet writes one directory per run beside the others; same scoring, one table
    for d in sorted(os.listdir(f"{a.root}/pred")):
        if d.startswith("esanet_") and os.path.isdir(f"{a.root}/pred/{d}"):
            dirs[d] = f"{a.root}/pred/{d}"

    per_run, frames = {}, []
    for run, d in dirs.items():
        rows = baseline_rows(a.split) if d is None else score_dir(d, a.split)
        if rows is None:
            print(f"  {run}: incomplete, skipped")
            continue
        per_run[run] = pooled(rows)
        for r in rows:
            frames.append(dict(run=run, **r))
        print(f"  {run:26s} IoU {per_run[run]['IoU']:.3f}  P {per_run[run]['precision']:.3f}"
              f"  R {per_run[run]['recall']:.3f}")

    with open(f"{RES}/frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["run", "site", "frame", "inter", "union", "pred", "gt"])
        w.writeheader()
        w.writerows(frames)
    with open(f"{RES}/summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "arm", "IoU", "precision", "recall"])
        for run, m in per_run.items():
            w.writerow([run, arm_of(run)] + [f"{m[k]:.4f}" for k in
                                             ("IoU", "precision", "recall")])

    by_arm = defaultdict(list)
    for run, m in per_run.items():
        by_arm[arm_of(run)].append(m)
    out = []
    for arm, ms in sorted(by_arm.items()):
        row = {"arm": arm, "seeds": len(ms)}
        for k in ("IoU", "precision", "recall"):
            v = np.array([m[k] for m in ms])
            row[k] = float(v.mean())
            row[k + "_sd"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        out.append(row)
    with open(f"{RES}/summary_mean_std.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(out[0]))
        w.writeheader()
        w.writerows(out)

    print("\n" + md(["Arm", "Seeds", "IoU", "Precision", "Recall"],
                    [[r["arm"], str(r["seeds"])] +
                     [f"{r[k]:.3f} ± {r[k + '_sd']:.3f}" if r["seeds"] > 1 else f"{r[k]:.3f}"
                      for k in ("IoU", "precision", "recall")]
                     for r in sorted(out, key=lambda r: -r["IoU"])]))
    print(f"\n-> {RES}/")


if __name__ == "__main__":
    main()
