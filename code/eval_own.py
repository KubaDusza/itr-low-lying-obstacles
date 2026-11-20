"""Score the frozen ISOD-trained models on our own recordings, by condition.

  python eval_own.py                  # hand labels where they exist, derived ones elsewhere
  python eval_own.py --gt auto        # derived ground truth only
  python eval_own.py --gt manual      # hand-labelled frames only

Same metric as ISOD: pixel overlap on the obstacle class, with everything off the fitted
ground plane ignored. Two ground truths exist for these recordings:

  auto    median-background differencing across the runs of a group (own_gt.py). Covers all
          320 frames; 13 of 64 runs fail its reliability check and are excluded.
  manual  hand-drawn masks with one id per item (annotate.py). Covers fewer frames, but it
          carries *which* object each pixel belongs to, which is what makes the per-item
          analysis below possible, and it rescues runs the derived method could not do.

The default, --gt best, uses a hand label when there is one and the derived label otherwise,
and reports how much the two agree on the frames that have both. That agreement is the
honest way to report what the derived ground truth is worth.

The breakdowns are the point of the recording session: camera height, distance, background,
object class (from the item's shape) and RGB contrast (from the item's tone against the
background). The last one is the condition ISOD does not contain.

Writes results/own/{per_run.csv, per_item.csv, overall.json} and prints the tables.
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

import cv2
import numpy as np

import own_items

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = f"{ROOT}/results/own"
DIST = {"075": "0.87 m", "150": "1.56 m", "250": "2.60 m"}
HEIGHT = {"h25": "25 cm", "h52": "52 cm"}
SHAPE_CLASS = {"thin-linear": "thin (cable-like)", "flat-soft": "flat and soft",
               "compact": "small and solid", "raised": "raised"}


def arm_of(name):
    return re.sub(r"_s\d+$", "", name)


def seed_of(name):
    m = re.search(r"_s(\d+)$", name)
    return int(m.group(1)) if m else -1


def read_mask(path):
    if not os.path.exists(path):
        return None
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 3:
        m = m[..., 0]
    return m


def counts(pred, gt, floor):
    p = (pred > 0) & (floor > 0)
    g = (gt > 0) & (floor > 0)
    return dict(inter=int((p & g).sum()), union=int((p | g).sum()),
                pred=int(p.sum()), gt=int(g.sum()))


def pooled(rows, keys=("inter", "union", "pred", "gt")):
    if not rows:
        return None
    t = {k: sum(r[k] for r in rows) for k in keys}
    return dict(IoU=t["inter"] / max(1, t["union"]),
                precision=t["inter"] / max(1, t["pred"]),
                recall=t["inter"] / max(1, t["gt"]), **t)


def md(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |",
                      "|" + "|".join(["---"] * len(header)) + "|"] +
                     ["| " + " | ".join(r) + " |" for r in rows])


def mean_sd(vals):
    v = np.array(vals, float)
    if v.size == 0:
        return "-"
    if v.size == 1:
        return f"{v[0]:.3f}"
    return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--own", default=f"{ROOT}/data/OWN")
    ap.add_argument("--manual", default=f"{ROOT}/data/OWN_manual")
    ap.add_argument("--pred", default=f"{ROOT}/data/OWN_pred")
    ap.add_argument("--gt", default="best", choices=["best", "auto", "manual"])
    ap.add_argument("--hit", type=float, default=0.5, help="per-item detection threshold")
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)

    meta = {r["run"]: r for r in json.load(open(f"{a.own}/gt_summary.json"))}
    arms = sorted(d for d in os.listdir(a.pred) if os.path.isdir(f"{a.pred}/{d}"))
    sets = own_items.item_sets()
    print(f"{len(arms)} prediction sets, {len(meta)} runs, ground truth: {a.gt}")

    run_rows, item_rows, agree = [], [], []
    for run, m in sorted(meta.items()):
        floor = read_mask(f"{a.own}/{run}/floor.png")
        n = len(glob.glob(f"{a.own}/{run}/gt/*.png"))
        block, dist = m["block"], m["distance"]
        oset = run.split("_")[2]
        background = "poster" if "poster" in block else "floor"
        items = sets.get(oset, [])

        per_arm = defaultdict(list)
        per_item = defaultdict(lambda: defaultdict(list))
        used, src_count = 0, defaultdict(int)
        for i in range(n):
            g_auto = read_mask(f"{a.own}/{run}/gt/{i}.png")
            g_man = read_mask(f"{a.manual}/{run}_{i}.png")
            ids = read_mask(f"{a.manual}/{run}_{i}_id.png")
            if g_auto is not None and g_man is not None:
                # derived mask judged against the hand label: pred=derived, gt=hand
                agree.append(dict(run=run, reliable=m["reliable"], far=dist == "250",
                                  **counts(g_auto, g_man, floor)))
            if a.gt == "manual" or (a.gt == "best" and g_man is not None):
                gt, src = g_man, "manual"
            elif m["reliable"]:
                gt, src = g_auto, "auto"
            else:
                # the derived mask failed its reliability check on this run and there is no
                # hand label for this frame, so the frame has no usable ground truth
                continue
            if gt is None:
                continue
            used += 1
            src_count[src] += 1
            for arm in arms:
                p = read_mask(f"{a.pred}/{arm}/{run}/{i}.png")
                if p is None:
                    continue
                per_arm[arm].append(counts(p, gt, floor))
                if ids is None:
                    continue
                for it in items:
                    sel = (ids == it["idx"]) & (floor > 0)
                    if sel.sum() < 20:
                        continue
                    per_item[arm][it["idx"]].append(
                        dict(px=int(sel.sum()), found=int(((p > 0) & sel).sum())))

        if not used:
            continue
        for arm, rows in per_arm.items():
            t = pooled(rows)
            run_rows.append(dict(run=run, arm=arm_of(arm), seed=seed_of(arm), block=block,
                                 height=HEIGHT[block[:3]], distance=dist,
                                 background=background, object_set=oset,
                                 reliable=m["reliable"], is_empty=run.endswith("_empty"),
                                 has_manual=src_count["manual"] > 0,
                                 gt_source="manual" if src_count["manual"] == used else
                                 ("auto" if src_count["manual"] == 0 else "mixed"),
                                 frames=len(rows), **t))
        for arm, per in per_item.items():
            for idx, obs in per.items():
                it = next(x for x in items if x["idx"] == idx)
                px = sum(o["px"] for o in obs)
                found = sum(o["found"] for o in obs)
                hits = sum(1 for o in obs if o["found"] / max(1, o["px"]) >= a.hit)
                item_rows.append(dict(run=run, arm=arm_of(arm), seed=seed_of(arm),
                                      height=HEIGHT[block[:3]], distance=dist,
                                      background=background, object_set=oset,
                                      item=it["label"], tone=it["tone"], shape=it["shape"],
                                      klass=SHAPE_CLASS.get(it["shape"], it["shape"]),
                                      contrast=own_items.contrast(it["tone"], background),
                                      instances=len(obs), px=px, found=found,
                                      recall=found / max(1, px), hits=hits))

    for name, rows in (("per_run.csv", run_rows), ("per_item.csv", item_rows)):
        if rows:
            with open(f"{RES}/{name}", "w", newline="") as f:
                w = csv.DictWriter(f, list(rows[0]))
                w.writeheader()
                w.writerows(rows)

    # a run enters the aggregates if its derived ground truth passed the check, or if it was
    # hand-labelled, in which case the check does not apply
    good = [r for r in run_rows if not r["is_empty"] and (r["reliable"] or r["has_manual"])]
    arm_names = sorted({r["arm"] for r in good})
    order = sorted(arm_names, key=lambda x: -np.mean(
        [r["IoU"] for r in good if r["arm"] == x]))

    def table(title, keyfn, keys, rows=None, metric="IoU"):
        src = rows if rows is not None else good
        body = []
        for arm in order:
            cells = []
            for k in keys:
                sel = [r for r in src if r["arm"] == arm and keyfn(r) == k]
                by_seed = defaultdict(list)
                for r in sel:
                    by_seed[r["seed"]].append(r)
                vals = [pooled(v)[metric] for v in by_seed.values() if pooled(v)]
                cells.append(mean_sd(vals))
            body.append([arm] + cells)
        print(f"\n### {title}\n")
        print(md(["Arm"] + list(keys), body))

    print("\n## Overall, reliable or hand-labelled runs, obstacles present\n")
    body = []
    for arm in order:
        by_seed = defaultdict(list)
        for r in good:
            if r["arm"] == arm:
                by_seed[r["seed"]].append(r)
        cols = [mean_sd([pooled(v)[k] for v in by_seed.values()])
                for k in ("IoU", "precision", "recall")]
        body.append([arm, str(len(by_seed))] + cols +
                    [str(sum(r["frames"] for r in good if r["arm"] == arm) // len(by_seed))])
    print(md(["Arm", "Seeds", "IoU", "Precision", "Recall", "Frames"], body))

    table("IoU by camera height", lambda r: r["height"], ["25 cm", "52 cm"])
    table("IoU by distance", lambda r: DIST[r["distance"]],
          [DIST[k] for k in ("075", "150", "250")])
    table("IoU by background", lambda r: r["background"], ["floor", "poster"])
    table("IoU by object set", lambda r: r["object_set"],
          sorted({r["object_set"] for r in good}))

    if item_rows:
        print("\n## Per item, hand-labelled frames only\n")
        iorder = [arm for arm in order if any(r["arm"] == arm for r in item_rows)]

        def item_table(title, keyfn, keys, metric):
            body = []
            for arm in iorder:
                cells = []
                for k in keys:
                    sel = [r for r in item_rows if r["arm"] == arm and keyfn(r) == k]
                    by_seed = defaultdict(list)
                    for r in sel:
                        by_seed[r["seed"]].append(r)
                    vals = []
                    for v in by_seed.values():
                        if metric == "recall":
                            vals.append(sum(x["found"] for x in v) /
                                        max(1, sum(x["px"] for x in v)))
                        else:
                            vals.append(sum(x["hits"] for x in v) /
                                        max(1, sum(x["instances"] for x in v)))
                    cells.append(mean_sd(vals))
                body.append([arm] + cells)
            print(f"\n### {title}\n")
            print(md(["Arm"] + list(keys), body))

        item_table("Pixel recall by RGB contrast", lambda r: r["contrast"],
                   ["high", "medium", "low"], "recall")
        item_table(f"Share of item instances found (>= {a.hit:.0%} of pixels), by contrast",
                   lambda r: r["contrast"], ["high", "medium", "low"], "hit")
        item_table("Pixel recall by object class", lambda r: r["klass"],
                   sorted({r["klass"] for r in item_rows}), "recall")
        item_table(f"Share of item instances found, by class", lambda r: r["klass"],
                   sorted({r["klass"] for r in item_rows}), "hit")

    print("\n### False positives on the empty-floor runs (any prediction is wrong)\n")
    body = []
    for arm in order:
        sel = [r for r in run_rows if r["arm"] == arm and r["is_empty"]]
        if not sel:
            continue
        by_seed = defaultdict(list)
        for r in sel:
            by_seed[r["seed"]].append(r)
        px = [sum(r["pred"] for r in v) / max(1, sum(r["frames"] for r in v))
              for v in by_seed.values()]
        runs_any = [sum(1 for r in v if r["pred"] > 0) for v in by_seed.values()]
        body.append([arm, f"{np.mean(px):.0f}", f"{np.mean(runs_any):.1f}/{len(sel) // len(by_seed)}"])
    print(md(["Arm", "predicted px / frame", "runs with any prediction"], body))

    if agree:
        print(f"\n### What the derived ground truth is worth, on the {len(agree)} frames "
              "that also have hand labels\n")
        print("Hand labels are the reference. Recall is the share of hand-labelled obstacle\n"
              "pixels the derived mask found; precision is the share of its own pixels that\n"
              "are real. Split by whether own_gt.py's reliability check passed the run, and\n"
              "by distance, because that is where the two disagree.\n")
        body = []
        for label, sel in (
                ("check passed", [r for r in agree if r["reliable"]]),
                ("check passed, 0.87-1.56 m", [r for r in agree if r["reliable"] and not r["far"]]),
                ("check passed, 2.60 m", [r for r in agree if r["reliable"] and r["far"]]),
                ("check failed", [r for r in agree if not r["reliable"]])):
            t = pooled(sel)
            if t:
                body.append([label, str(len(sel)), f"{t['IoU']:.3f}",
                             f"{t['precision']:.3f}", f"{t['recall']:.3f}",
                             f"{t['pred'] / max(1, t['gt']):.2f}x"])
        print(md(["Runs", "Frames", "IoU", "Precision", "Recall", "px vs. hand"], body))

    with open(f"{RES}/overall.json", "w") as f:
        out = {}
        for arm in order:
            by_seed = defaultdict(list)
            for r in good:
                if r["arm"] == arm:
                    by_seed[r["seed"]].append(r)
            out[arm] = {k: float(np.mean([pooled(v)[k] for v in by_seed.values()]))
                        for k in ("IoU", "precision", "recall")}
        json.dump(out, f, indent=1)
    print(f"\n-> {RES}/")


if __name__ == "__main__":
    main()
