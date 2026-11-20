"""Object-level recall on ISOD, and the paired statistics that go with it.

  ITR_DATA=... python objects.py --root ... --hit 0.5
  ITR_DATA=... python objects.py --compare rgb rgbd        # just the statistics, from the CSV

Pixel IoU is dominated by the few large objects in a frame, so each ground-truth object
(one connected component) is also scored on its own: it counts as detected when at least
`hit` of its pixels are predicted. Objects are grouped by size and by distance, which is
where depth should matter if it matters anywhere.

The statistics treat the *object* as the sampling unit, not the seed. Three seeds with the
same sign is not evidence of anything on its own; 3,471 paired objects are. Two tests:

  McNemar  per seed, on the paired detections (exact two-sided binomial on the discordant
           pairs). Answers: did this arm detect a different *set* of objects?
  bootstrap 10,000 resamples of the 300 test *frames*, not of the objects, because objects
           in one frame are not independent. Gives a confidence interval for the recall
           difference averaged over seeds.

Writes results/isod/objects_hit<h>.csv (one row per object per run) and prints the tables.
"""
import argparse
import csv
import math
import os
import re
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image

import isod

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results", "isod")


def arm_of(run):
    return re.sub(r"_s\d+$", "", run)


def seed_of(run):
    m = re.search(r"_s(\d+)$", run)
    return int(m.group(1)) if m else -1


def exact_mcnemar(b, c):
    """Two-sided exact binomial p on the discordant pairs (b wins vs c wins)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def collect(root, split, hit):
    """One row per (run, object): area, distance and whether it was detected."""
    runs = {}
    for d in sorted(os.listdir(f"{root}/pred/modal")):
        if os.path.isdir(f"{root}/pred/modal/{d}"):
            runs[d] = f"{root}/pred/modal/{d}"
    for d in ("yolo", "unet_mobilenet_v2", "deeplabv3plus_mobilenet_v2"):
        if os.path.isdir(f"{root}/pred/{d}"):
            runs[d] = f"{root}/pred/{d}"

    rows, oid = [], 0
    for site in isod.load_splits()[split]:
        for i in range(100):
            gt = (np.array(Image.open(f"{isod.GT}/{site}/{i}.png")) > 0).astype(np.uint8)
            if not gt.any():
                continue
            raw = cv2.imread(f"{isod.DATA}/{site}/depth/{i}.png", cv2.IMREAD_UNCHANGED)
            n, cc, stats, _ = cv2.connectedComponentsWithStats(gt, 8)
            preds = {}
            for run, d in runs.items():
                p = f"{d}/{site}/{i}.png"
                if os.path.exists(p):
                    preds[run] = np.array(Image.open(p)) > 0
            for k in range(1, n):
                sel = cc == k
                area = int(stats[k, cv2.CC_STAT_AREA])
                dz = raw[sel & (raw > 0)]
                dist = float(np.median(dz)) if dz.size else 0.0
                for run, pm in preds.items():
                    rows.append(dict(oid=oid, run=run, arm=arm_of(run), seed=seed_of(run),
                                     site=site, frame=i, area=area, dist_mm=dist,
                                     frac=round(float(pm[sel].mean()), 4),
                                     hit=int(pm[sel].mean() >= hit)))
                oid += 1
    return rows


def groups(rows):
    """Size and distance tertiles, taken over the objects themselves."""
    uniq = {r["oid"]: (r["area"], r["dist_mm"]) for r in rows}
    area = np.array([v[0] for v in uniq.values()])
    dist = np.array([v[1] for v in uniq.values() if v[1] > 0])
    a1, a2 = np.percentile(area, [33.3, 66.7])
    d1, d2 = np.percentile(dist, [33.3, 66.7])

    def g(r):
        out = ["all"]
        out.append("small" if r["area"] < a1 else "medium" if r["area"] < a2 else "large")
        if r["dist_mm"] > 0:
            out.append("near" if r["dist_mm"] < d1 else
                       "middle" if r["dist_mm"] < d2 else "far")
        return out
    return g, dict(area_tertiles=[float(a1), float(a2)], dist_tertiles=[float(d1), float(d2)])


def recall_table(rows, g):
    by = defaultdict(lambda: [0, 0])
    for r in rows:
        for grp in g(r):
            by[(r["arm"], grp)][0] += r["hit"]
            by[(r["arm"], grp)][1] += 1
    return {k: v[0] / max(1, v[1]) for k, v in by.items()}, by


def compare(rows, g, a_arm, b_arm, boots=10000, seed=0):
    """Paired object-level comparison of two arms: McNemar per seed, then a bootstrap."""
    idx = defaultdict(dict)
    for r in rows:
        if r["arm"] in (a_arm, b_arm):
            idx[(r["seed"], r["oid"])][r["arm"]] = r
    pairs = [(k, v) for k, v in idx.items() if len(v) == 2]
    if not pairs:
        return None
    seeds = sorted({k[0] for k, _ in pairs})
    out = {"groups": {}, "mcnemar": {}}

    for grp in ("all", "small", "medium", "large", "near", "middle", "far"):
        sel = [(k, v) for k, v in pairs if grp in g(v[a_arm])]
        if not sel:
            continue
        per_seed = []
        for s in seeds:
            ss = [v for k, v in sel if k[0] == s]
            b = sum(1 for v in ss if v[a_arm]["hit"] and not v[b_arm]["hit"])
            c = sum(1 for v in ss if v[b_arm]["hit"] and not v[a_arm]["hit"])
            per_seed.append((s, b, c, exact_mcnemar(b, c), len(ss)))
        out["mcnemar"][grp] = per_seed

        # frame-clustered bootstrap of the mean per-object recall difference
        by_frame = defaultdict(list)
        for k, v in sel:
            by_frame[(v[a_arm]["site"], v[a_arm]["frame"])].append(
                v[b_arm]["hit"] - v[a_arm]["hit"])
        keys = list(by_frame)
        diffs = [np.array(by_frame[k], float) for k in keys]
        rng = np.random.default_rng(seed)
        obs = float(np.concatenate(diffs).mean())
        draws = np.empty(boots)
        for t in range(boots):
            pick = rng.integers(0, len(keys), len(keys))
            draws[t] = np.concatenate([diffs[j] for j in pick]).mean()
        lo, hi = np.percentile(draws, [2.5, 97.5])
        out["groups"][grp] = dict(n_objects=len(sel) // len(seeds), diff=obs,
                                  lo=float(lo), hi=float(hi))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("ITR_DATA", "/data/cs_courses/jdusza_itr"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--hit", type=float, default=0.5)
    ap.add_argument("--compare", nargs=2, action="append", default=None,
                    help="two arm names; repeatable")
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    path = f"{RES}/objects_hit{str(a.hit).replace('.', 'p')}.csv"

    if os.path.exists(path):
        with open(path) as f:
            rows = [dict(r, oid=int(r["oid"]), seed=int(r["seed"]), area=int(r["area"]),
                         dist_mm=float(r["dist_mm"]), frac=float(r["frac"]), hit=int(r["hit"]))
                    for r in csv.DictReader(f)]
        print(f"{len(rows)} rows from {path}")
    else:
        rows = collect(a.root, a.split, a.hit)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"{len(rows)} rows -> {path}")

    g, cuts = groups(rows)
    print(f"cuts: {cuts}")
    rec, by = recall_table(rows, g)
    arms = sorted({r["arm"] for r in rows})
    cols = ["all", "small", "medium", "large", "near", "middle", "far"]
    print("\nobject recall at hit >= "
          f"{a.hit:.0%}\n{'arm':24s}" + "".join(f"{c:>9s}" for c in cols))
    for arm in sorted(arms, key=lambda x: -rec.get((x, "all"), 0)):
        print(f"{arm:24s}" + "".join(
            f"{rec[(arm, c)]:9.3f}" if (arm, c) in rec else f"{'-':>9s}" for c in cols))

    for pair in (a.compare or [["rgb", "rgbd"], ["rgb", "fuse"], ["rgb", "rgbgeo"],
                               ["rgb", "fusegeo"], ["rgbd_blind", "rgbd"],
                               ["fuse_blind", "fuse"], ["depth", "geo"]]):
        r = compare(rows, g, pair[0], pair[1])
        if r is None:
            continue
        print(f"\n=== {pair[1]} minus {pair[0]} (object level) ===")
        print(f"{'group':8s}{'n':>7s}{'diff':>9s}{'95% CI':>20s}   McNemar per seed "
              f"({pair[0]}-only / {pair[1]}-only, p)")
        for grp, d in r["groups"].items():
            mc = "  ".join(f"{b}/{c} p={p:.3g}" for _, b, c, p, _ in r["mcnemar"][grp])
            print(f"{grp:8s}{d['n_objects']:7d}{d['diff']:+9.4f}"
                  f"   [{d['lo']:+.4f}, {d['hi']:+.4f}]   {mc}")


if __name__ == "__main__":
    main()
