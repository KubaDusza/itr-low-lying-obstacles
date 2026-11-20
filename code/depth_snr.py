"""How much of a low obstacle is visible in the depth geometry at all.

  ITR_DATA=... python depth_snr.py --root ...          # ISOD
  python depth_snr.py --own                            # our own recordings

This is the measurement that decides whether a weak depth result is our design or the
data. For every frame the floor plane is fitted and each pixel's gap to it is computed in
metres (modal.floor_gap). Then, over the obstacle and floor pixels:

  floor sd       the scatter of the floor's own measured gap: the noise
  obstacle gap   how far in front of the plane the obstacle pixels sit: the signal
  SNR            (obstacle mean - floor mean) / floor sd

An SNR near zero means a threshold on height above the floor cannot separate these
objects however it is tuned, and no amount of training or architecture fixes that, because
the quantity itself does not separate them. The plane is also fitted a second time on the
*labelled* floor pixels only ("oracle"), which removes the fit as an explanation: if the
oracle SNR is just as low, the floor's depth noise is the limit.

Our own recordings add the axis ISOD cannot test: the camera height. The geometric
prediction is that the gap for an object of height a at distance d with the camera at
height h is about a*d/h, so halving the height should double the signal.

Writes results/<isod|own>/depth_snr.csv and prints the tables.
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import cv2
import numpy as np

import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIN_OBST, MIN_FLOOR = 50, 2000


def stats_for(raw, obst, floor_hint, st, dump=None):
    """Blind and oracle gap statistics for one frame."""
    out = {}
    d = raw.astype(np.float32) / 1000.0
    valid = (raw > 0) & (d > st["near_m"]) & (d < st["far_m"])
    inv = st["near_m"] / np.maximum(d, 1e-3)
    h, w = raw.shape
    _, v = np.meshgrid(np.linspace(-1, 1, w, np.float32), np.linspace(-1, 1, h, np.float32))
    fits = {"blind": valid & (v > 0.1), "oracle": floor_hint & valid}
    for tag, mask in fits.items():
        pl = modal.fit_floor_plane(inv, mask)
        if pl is None:
            continue
        ok = valid & (pl > st["near_m"] / st["far_m"])
        gap = np.clip(st["near_m"] / np.maximum(pl, 1e-3) - d, -0.5, 0.5)
        f, o = floor_hint & ok, obst & ok
        if f.sum() < MIN_FLOOR or o.sum() < MIN_OBST:
            continue
        out[tag] = dict(floor_mean=float(gap[f].mean()), floor_sd=float(gap[f].std()),
                        obst_mean=float(gap[o].mean()), obst_px=int(o.sum()),
                        snr=float((gap[o].mean() - gap[f].mean()) / max(1e-9, gap[f].std())))
        if dump is not None and tag == "blind":
            dump["floor"].append(gap[f][::97].astype(np.float32))
            dump["obstacle"].append(gap[o][::7].astype(np.float32))
    return out


def isod_rows(root, st, every=5, dump=None):
    import isod
    rows = []
    for split, sites in isod.load_splits().items():
        for site in sites:
            for i in range(0, 100, every):
                raw = cv2.imread(f"{isod.DATA}/{site}/depth/{i}.png", cv2.IMREAD_UNCHANGED)
                lab = cv2.imread(f"{isod.DATA}/{site}/label/{i}.png", 0)
                gt = cv2.imread(f"{isod.GT}/{site}/{i}.png", 0) > 0
                if raw is None or lab is None:
                    continue
                s = stats_for(raw, gt, (lab == 255) & ~gt, st, dump)
                dz = raw[gt & (raw > 0)]
                for tag, v in s.items():
                    rows.append(dict(dataset="ISOD", split=split, site=site, frame=i,
                                     fit=tag, dist_mm=float(np.median(dz)) if dz.size else 0,
                                     **v))
    return rows


def own_rows(own, man, st, dump=None):
    rows = []
    meta = {r["run"]: r for r in json.load(open(f"{own}/gt_summary.json"))}
    for run, m in sorted(meta.items()):
        if run.endswith("_empty"):
            continue
        floor = cv2.imread(f"{own}/{run}/floor.png", 0)
        for p in sorted(glob.glob(f"{own}/{run}/gt/*.png")):
            i = int(os.path.basename(p)[:-4])
            gt = cv2.imread(f"{man}/{run}_{i}.png", cv2.IMREAD_UNCHANGED)
            source = "manual"
            if gt is None:
                if not m["reliable"]:
                    continue
                gt, source = cv2.imread(p, 0), "auto"
            if gt is None or floor is None:
                continue
            gt = (gt[..., 0] if gt.ndim == 3 else gt) > 0
            raw = cv2.imread(f"{own}/{run}/depth/{i}.png", cv2.IMREAD_UNCHANGED)
            s = stats_for(raw, gt, (floor > 0) & ~gt, st, dump)
            for tag, v in s.items():
                rows.append(dict(dataset="OWN", split=source, site=run, frame=i, fit=tag,
                                 height_cm=25 if run.startswith("h25") else 52,
                                 distance=run.split("_")[-1],
                                 background="poster" if "poster" in run else "floor",
                                 object_set=run.split("_")[2], dist_mm=0.0, **v))
    return rows


def table(rows, keyfn, keys, title):
    print(f"\n### {title}\n")
    body = []
    for k in keys:
        for fit in ("blind", "oracle"):
            sel = [r for r in rows if r["fit"] == fit and keyfn(r) == k]
            if not sel:
                continue
            body.append([str(k), fit, str(len(sel)),
                         f"{np.mean([r['floor_sd'] for r in sel]) * 100:.1f}",
                         f"{np.mean([r['obst_mean'] - r['floor_mean'] for r in sel]) * 100:+.1f}",
                         f"{np.mean([r['snr'] for r in sel]):+.2f}"])
    print("| " + " | ".join(["group", "plane fit", "frames", "floor sd (cm)",
                             "obstacle gap (cm)", "SNR"]) + " |")
    print("|" + "|".join(["---"] * 6) + "|")
    for r in body:
        print("| " + " | ".join(r) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("ITR_DATA", "/data/cs_courses/jdusza_itr"))
    ap.add_argument("--own", action="store_true")
    ap.add_argument("--own-dir", default=f"{ROOT}/data/OWN")
    ap.add_argument("--manual-dir", default=f"{ROOT}/data/OWN_manual")
    ap.add_argument("--every", type=int, default=5)
    a = ap.parse_args()

    st = modal.load_depth_stats(a.root)
    dump = {"floor": [], "obstacle": []}
    if a.own:
        rows = own_rows(a.own_dir, a.manual_dir, st, dump)
        out = f"{ROOT}/results/own/depth_snr.csv"
    else:
        rows = isod_rows(a.root, st, a.every, dump)
        out = f"{ROOT}/results/isod/depth_snr.csv"
    if not rows:
        print("no frames measured")
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    allk = sorted({k for r in rows for k in r})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, allk)
        w.writeheader()
        w.writerows(rows)

    np.savez_compressed(out.replace(".csv", "_gaps.npz"),
                        floor=np.concatenate(dump["floor"]),
                        obstacle=np.concatenate(dump["obstacle"]))
    print(f"{len(rows)} frame measurements -> {out} (+ _gaps.npz)")
    table(rows, lambda r: "all", ["all"], "Depth signal against depth noise, in metres")
    if a.own:
        table(rows, lambda r: f"{r['height_cm']} cm", ["25 cm", "52 cm"],
              "By camera height (the geometric prediction: half the height, twice the signal)")
        table(rows, lambda r: r["distance"], ["075", "150", "250"], "By tape distance")
        table(rows, lambda r: r["object_set"],
              sorted({r["object_set"] for r in rows}), "By object set")
    else:
        q = np.percentile([r["dist_mm"] for r in rows if r["dist_mm"] > 0], [33, 67])
        def band(r):
            return ("near" if r["dist_mm"] < q[0] else
                    "middle" if r["dist_mm"] < q[1] else "far") if r["dist_mm"] else "unknown"
        table(rows, band, ["near", "middle", "far"],
              f"By obstacle distance (tertiles at {q[0]:.0f} and {q[1]:.0f} mm)")
        table(rows, lambda r: r["split"], ["train", "val", "test"], "By split")


if __name__ == "__main__":
    main()
