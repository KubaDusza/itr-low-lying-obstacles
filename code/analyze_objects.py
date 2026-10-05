"""Per-site IoU and per-object analysis of the 9 existing runs (no training).

Pixel IoU is dominated by large objects; this asks the research question per object:
which obstacles does each input detect, split by object size and by distance, and does
depth find objects that RGB misses?

An object (connected component of the GT, as in isod.instances) counts as DETECTED by a
run when at least --hit of its pixels are predicted. For each modality the object gets a
detection rate over its 3 seeds; for the overlap table, "detected" = in >= 2 of 3 seeds.

  /opt/venv/jdusza_venv/bin/python analyze_objects.py [--root ~/datasets/isod_work] [--hit 0.5]

Writes <root>/runs/objects_hit_<hit>.csv (one row per test object) and
<root>/runs/site_metrics.csv (pixel metrics per run and test site) and prints the tables.
"""
import argparse
import csv
import os

import numpy as np
from PIL import Image

import eval as ev
import isod

MODS = ["rgb", "depth", "rgbd"]
SEEDS = [0, 1, 2]


def collect(root, hit):
    rows, site_rows = [], {}
    for site in isod.load_splits()["test"]:
        for i in range(100):
            g = ev.gt(site, i)
            _, depth, label, _ = isod.frame(site, i)
            preds = {(m, s): np.array(Image.open(
                f"{root}/pred/unet_mobilenet_v2_{m}_s{s}/{site}/{i}.png")) > 0
                for m in MODS for s in SEEDS}
            ignore = isod.ignore_region(label, g)
            for key, p in preds.items():  # pixel scores per site, exactly as eval.py
                site_rows.setdefault((*key, site), []).append(
                    isod.score(p.astype(np.uint8), g, ignore=ignore))
            for k, inst in enumerate(isod.instances(g)):
                d = depth[inst]
                d = d[d > 0]
                row = {"site": site, "frame": i, "object": k, "area_px": int(inst.sum()),
                       "median_depth_mm": float(np.median(d)) if len(d) else float("nan")}
                for m in MODS:
                    for s in SEEDS:
                        row[f"{m}_s{s}"] = int(preds[m, s][inst].mean() >= hit)
                    row[f"{m}_hits"] = sum(row[f"{m}_s{s}"] for s in SEEDS)
                rows.append(row)
    return rows, site_rows


def per_site(site_rows, out_csv):
    sites = isod.load_splits()["test"]
    tot = {k: isod.totals(v) for k, v in site_rows.items()}
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["modality", "seed", "site", "IoU", "precision", "recall"])
        for (m, s, site), t in sorted(tot.items()):
            w.writerow([m, s, site, t["IoU"], t["precision"], t["recall"]])
    iou = {k: t["IoU"] for k, t in tot.items()}
    ms = lambda v: f"{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}"
    print("pixel IoU per test site (mean ± std over seeds)\n\n| Input | " + " | ".join(sites) +
          " |\n|---" * 1 + "|---" * len(sites) + "|")
    for m in MODS:
        print(f"| {m} | " + " | ".join(ms([iou[m, s, site] for s in SEEDS]) for site in sites) + " |")
    for m in ("depth", "rgbd"):
        print(f"| {m} − rgb (paired) | " + " | ".join(
            ms([iou[m, s, site] - iou["rgb", s, site] for s in SEEDS]) for site in sites) + " |")
    print()


def bins(values, name, unit):
    """Equal-count terciles, so every bin has enough objects."""
    v = np.asarray(values, float)
    e = np.nanpercentile(v, [100 / 3, 200 / 3])
    labels = [f"{name} < {e[0]:.0f} {unit}", f"{e[0]:.0f}–{e[1]:.0f} {unit}", f"{name} ≥ {e[1]:.0f} {unit}"]
    idx = np.where(np.isnan(v), -1, np.digitize(v, e))
    return idx, labels


def table(rows, idx, labels, title):
    print(f"\n{title}\n\n| Objects | n | RGB | Depth | RGB-D | RGB-D − RGB (paired by seed) |\n|---|---|---|---|---|---|")
    for b, lab in list(enumerate(labels)) + [(None, "all")]:
        sel = [r for r, j in zip(rows, idx) if b is None or j == b]
        if not sel:
            continue
        rec = {m: [r[f"{m}_hits"] / len(SEEDS) for r in sel] for m in MODS}
        diff = [np.mean([r[f"rgbd_s{s}"] - r[f"rgb_s{s}"] for r in sel]) for s in SEEDS]
        print(f"| {lab} | {len(sel)} | " + " | ".join(
            f"{np.mean(rec[m]):.3f}" for m in MODS) +
            f" | {np.mean(diff):+.3f} ± {np.std(diff, ddof=1):.3f} ({', '.join(f'{d:+.3f}' for d in diff)}) |")


def overlap(rows, a, b):
    da = np.array([r[f"{a}_hits"] >= 2 for r in rows])
    db = np.array([r[f"{b}_hits"] >= 2 for r in rows])
    n = len(rows)
    print(f"\n{a.upper()} vs {b.upper()} (detected = in >= 2 of 3 seeds), {n} objects:")
    for lab, m in (("both", da & db), (f"{a} only", da & ~db), (f"{b} only", ~da & db), ("neither", ~da & ~db)):
        print(f"  {lab:12s} {m.sum():5d}  ({m.mean():.1%})")


def depth_only_rescue(rows):
    """The direct question: of the objects Depth finds and RGB misses, does RGB-D find them?"""
    rgb, depth, rgbd = (np.array([r[f"{m}_hits"] >= 2 for r in rows]) for m in MODS)
    depth_only = depth & ~rgb
    n = int(depth_only.sum())
    print(f"\nobjects found by Depth but missed by RGB: {n}")
    if n == 0:
        return
    print(f"  recovered by RGB-D:    {(depth_only & rgbd).sum():5d} / {n} ({(depth_only & rgbd).mean() / depth_only.mean():.1%})")
    print(f"  still missed by RGB-D: {(depth_only & ~rgbd).sum():5d} / {n} ({(depth_only & ~rgbd).mean() / depth_only.mean():.1%})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/datasets/isod_work")
    ap.add_argument("--hit", type=float, default=0.5, help="fraction of object pixels that must be predicted")
    a = ap.parse_args()
    root = os.path.expanduser(a.root)

    rows, site_rows = collect(root, a.hit)
    per_site(site_rows, f"{root}/runs/site_metrics.csv")
    out_csv = f"{root}/runs/objects_hit_{str(a.hit).replace('.', 'p')}.csv"  # one file per --hit
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"object recall (fraction of objects with >= {a.hit:.0%} of pixels predicted, averaged over seeds)")
    table(rows, *bins([r["area_px"] for r in rows], "area", "px"), "by object size:")
    no_depth = sum(np.isnan(r["median_depth_mm"]) for r in rows)
    table(rows, *bins([r["median_depth_mm"] for r in rows], "dist", "mm"),
          f"by distance to the camera ({no_depth} objects without valid depth: only in 'all'):")
    overlap(rows, "rgb", "depth")   # does depth find objects RGB misses?
    overlap(rows, "rgb", "rgbd")    # overall gains / losses of fusion
    depth_only_rescue(rows)         # does fusion rescue the depth-only objects?
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()